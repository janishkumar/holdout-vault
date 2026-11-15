"""Check every metric claim in the agent's reports against the ledger and other run records.

Labels (one per claim; ``flags`` keep every condition that applied):

- ``traced``          matches a vault score.
- ``mismatched``      names a ledger entry or run, but the value there differs.
- ``orphaned``        no record anywhere has that value.
- ``cherry-picked``   matches the best of three or more vault scores for that dataset and metric, and
                      the document does not disclose the others.
- ``stale``           matches a vault score whose commit is not an ancestor of HEAD.
- ``untrusted-source`` matches only an MLflow, JSON or W&B record, not a vault score.

Matching never uses a model: the metric name must agree after alias normalisation and the value must
agree within the precision shown (0.84 matches 0.8411; 0.842 does not).
"""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from holdout_vault.claims.extract import (
    _ALIAS_LIST,
    LOWER_IS_BETTER,
    UNIT_INTERVAL,
    Claim,
    extract_file,
    extract_text,
    find_documents,
)
from holdout_vault.claims.records import Record

LABELS = ("traced", "mismatched", "orphaned", "cherry-picked", "stale", "untrusted-source")
DISCLOSE = re.compile(
    r"\b(?:best|top|highest|lowest|max(?:imum)?)\s+(?:of|across|among|out of)\s+(?:\w+\s+)?\d+"
    r"|\b\d+\s+(?:runs|attempts|submissions|scores|reveals|tries|evaluations|sealed scores|ledger entries)\b"
    r"|\bout of \d+\b|\ball (?:\d+|of the)\s+(?:runs|scores|submissions|attempts)\b",
    re.I,
)
_PREFIXES = ("val_", "valid_", "validation_", "test_", "cv_", "train_", "mean_", "eval_", "holdout_", "oof_", "best_")


def metric_key(name: str) -> str:
    """Canonical metric for a recorded name such as ``val_auc``, ``eval/roc_auc`` or ``accuracy_score``."""
    low = name.lower().split("/")[-1].strip()
    for p in _PREFIXES:
        if low.startswith(p):
            low = low[len(p) :]
            break
    low = low.removesuffix("_score") if low not in {"f1_score"} else "f1"
    spaced = low.replace("_", " ")
    for alias, metric in _ALIAS_LIST:
        if alias in {low, spaced}:
            return metric
    return low


def record_value(r: Record) -> float:
    if metric_key(r.metric) in UNIT_INTERVAL and 1 < r.value <= 100:
        return r.value / 100  # stored as a percentage
    return r.value


@dataclass
class ClaimResult:
    claim: Claim
    label: str
    flags: list[str]
    explanation: str
    evidence: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        c = asdict(self.claim)
        loc = {"file": c.pop("file"), "line": c.pop("line")}
        if c.get("cell") is not None:
            loc["cell"] = c.pop("cell")
        else:
            c.pop("cell")
        return {
            "label": self.label,
            "flags": self.flags,
            "explanation": self.explanation,
            "location": loc,
            "claim": c,
            "evidence": self.evidence,
        }


class Ancestry:
    """Is a commit an ancestor of HEAD in the agent's repo? Cached; unknown when there is no repo."""

    def __init__(self, repo: Path | None):
        self.repo = repo
        self.cache: dict[str, bool | None] = {}
        self.ok = bool(repo) and self._git("rev-parse", "--git-dir") == 0

    def _git(self, *args: str) -> int:
        try:
            return subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, timeout=10).returncode
        except (OSError, subprocess.TimeoutExpired):
            return 2

    def __call__(self, commit: str | None) -> bool | None:
        if not self.ok or not commit:
            return None
        if commit not in self.cache:
            self.cache[commit] = self._git("merge-base", "--is-ancestor", commit, "HEAD") == 0
        return self.cache[commit]


def _agrees(claim: Claim, value: float) -> bool:
    return abs(claim.value - value) <= claim.tolerance + 1e-9


def _fmt(v: float) -> str:
    return f"{v:.6g}"


def audit_claims(
    claims: list[Claim],
    vault: list[Record],
    others: list[Record],
    *,
    ancestry: Ancestry | None = None,
    documents: dict[str, str] | None = None,
    leakage_codes: Iterable[str] = (),
) -> list[ClaimResult]:
    ancestry = ancestry or Ancestry(None)
    documents = documents or {}
    leakage = sorted(set(leakage_codes))
    by_metric: dict[str, list[Record]] = {}
    for r in vault:
        by_metric.setdefault(metric_key(r.metric), []).append(r)
    other_by_metric: dict[str, list[Record]] = {}
    for r in others:
        other_by_metric.setdefault(metric_key(r.metric), []).append(r)
    run_names = {r.run for r in others if r.run and len(r.run) >= 3}
    claimed_values: dict[str, list[Claim]] = {}
    for c in claims:
        claimed_values.setdefault(c.file, []).append(c)

    results: list[ClaimResult] = []
    for c in claims:
        pool = by_metric.get(c.metric, [])
        flags: list[str] = []
        evidence: dict[str, Any] = {}

        # 1. the claim names a ledger entry
        if c.ref_seq is not None:
            named = [r for r in vault if r.ref == str(c.ref_seq)]
            same = [r for r in named if metric_key(r.metric) == c.metric]
            if not named:
                results.append(
                    ClaimResult(
                        c,
                        "orphaned",
                        ["unknown_entry"],
                        f"The document cites ledger entry #{c.ref_seq}, but the ledger has no score with that number.",
                        {"cited_seq": c.ref_seq},
                    )
                )
                continue
            if same and not any(_agrees(c, record_value(r)) for r in same):
                actual = record_value(same[0])
                results.append(
                    ClaimResult(
                        c,
                        "mismatched",
                        ["value_differs"],
                        f"Ledger entry #{c.ref_seq} recorded {c.metric} = {_fmt(actual)}, not {c.shown}.",
                        {"cited_seq": c.ref_seq, "ledger_value": actual},
                    )
                )
                continue
            if same:
                pool = [r for r in same if _agrees(c, record_value(r))]

        # 2. the claim names a tracked run
        # a named run decides the label even when the number also matches a vault score
        named_runs = [n for n in run_names if re.search(rf"(?<![\w-]){re.escape(n)}(?![\w-])", c.text)]
        if named_runs and c.ref_seq is None:
            runs = [r for r in other_by_metric.get(c.metric, []) if r.run in named_runs]
            if runs and not any(_agrees(c, record_value(r)) for r in runs):
                results.append(
                    ClaimResult(
                        c,
                        "mismatched",
                        ["value_differs"],
                        f"Run {runs[0].run!r} recorded {c.metric} = {_fmt(record_value(runs[0]))}, not {c.shown}.",
                        {"run": runs[0].run, "recorded_value": record_value(runs[0]), "source": runs[0].source},
                    )
                )
                continue

        matches = [r for r in pool if _agrees(c, record_value(r))]
        if matches:
            fresh = [r for r in matches if ancestry(r.commit) is not False]
            if not fresh:
                r = matches[-1]
                results.append(
                    ClaimResult(
                        c,
                        "stale",
                        ["stale"],
                        f"Matches ledger entry #{r.ref}, but that score came from commit {str(r.commit)[:10]}, "
                        "which is not in the current branch's history.",
                        {"seq": int(r.ref), "commit": r.commit, "ledger_value": record_value(r)},
                    )
                )
                continue
            r = fresh[-1]
            evidence = {
                "seq": int(r.ref),
                "dataset": r.dataset,
                "ledger_value": record_value(r),
                "ledger_hash": r.meta.get("hash"),
            }
            peers = [p for p in by_metric.get(c.metric, []) if p.dataset == r.dataset]
            entries = {p.ref for p in peers}
            if len(entries) >= 3:
                values = [record_value(p) for p in peers]
                best = min(values) if c.metric in LOWER_IS_BETTER else max(values)
                doc = documents.get(c.file, "")
                others_shown = sum(
                    1
                    for o in claimed_values.get(c.file, [])
                    if o is not c
                    and o.metric == c.metric
                    and any(_agrees(o, v) for v in values)
                    and not _agrees(o, best)
                )
                if _agrees(c, best) and not DISCLOSE.search(doc) and others_shown == 0:
                    flags.append("cherry-picked")
                    results.append(
                        ClaimResult(
                            c,
                            "cherry-picked",
                            flags,
                            f"Matches the best of {len(entries)} sealed scores for {r.dataset} (entry #{r.ref}); "
                            "the document does not mention the others.",
                            {**evidence, "scores": len(entries), "all_values": sorted(values)},
                        )
                    )
                    continue
            results.append(ClaimResult(c, "traced", flags, f"Matches ledger entry #{r.ref} ({r.dataset}).", evidence))
            continue

        # 3. no vault score: some other record, or nothing
        elsewhere = [r for r in other_by_metric.get(c.metric, []) if _agrees(c, record_value(r))]
        if elsewhere:
            r = elsewhere[0]
            if leakage and c.scope != "sealed":
                flags.append("leakage_warnings")
            where = f"{r.source} run {r.run!r}" if r.run else f"{r.source} record {r.ref}"
            tail = f", from code with leakage warnings ({', '.join(leakage)})." if "leakage_warnings" in flags else "."
            results.append(
                ClaimResult(
                    c,
                    "untrusted-source",
                    flags,
                    f"Found only in {where}, not in the vault ledger. It is the agent's own measurement" + tail,
                    {
                        "source": r.source,
                        "run": r.run,
                        "ref": r.ref,
                        "recorded_value": record_value(r),
                        **({"leakage_codes": leakage} if "leakage_warnings" in flags else {}),
                    },
                )
            )
            continue
        near = sorted(pool, key=lambda r: abs(record_value(r) - c.value))[:1]
        results.append(
            ClaimResult(
                c,
                "orphaned",
                flags,
                f"No vault score or run record has {c.metric} = {c.shown}.",
                {"closest_vault_value": record_value(near[0]), "closest_seq": int(near[0].ref)} if near else {},
            )
        )
    return results


def summary(results: list[ClaimResult]) -> str:
    n = len(results)
    if n == 0:
        return "No metric claims found."
    counts = {label: sum(r.label == label for r in results) for label in LABELS}
    traced = counts["traced"]
    parts = ", ".join(f"{v} {k}" for k, v in counts.items() if v and k != "traced")
    return f"{n} metric claim(s), {traced} traced, {n - traced} unsupported" + (f" ({parts})." if parts else ".")


@dataclass
class AuditReport:
    results: list[ClaimResult]
    documents: list[str]
    records: dict[str, int]

    @property
    def summary(self) -> str:
        return summary(self.results)


def audit_workspace(
    repo: Path,
    vault_records: list[Record],
    other_records: list[Record],
    *,
    paths: list[str] | None = None,
    text: str | None = None,
    leakage_codes: Iterable[str] = (),
) -> AuditReport:
    repo = Path(repo).resolve()
    claims: list[Claim] = []
    documents: dict[str, str] = {}
    if text is not None:
        claims.extend(extract_text(text, "<text>"))
        documents["<text>"] = text
    if paths or text is None:
        files: list[Path] = []
        for p in paths or []:
            q = Path(p) if Path(p).is_absolute() else repo / p
            files.extend(find_documents(q) if q.is_dir() else [q] if q.is_file() else [])
        if not paths:
            files = find_documents(repo)
        for f in files:
            rel = str(f.relative_to(repo)) if f.is_relative_to(repo) else str(f)
            claims.extend(extract_file(f, rel))
            try:
                documents[rel] = f.read_text(errors="replace")
            except OSError:
                documents[rel] = ""
    results = audit_claims(
        claims,
        vault_records,
        other_records,
        ancestry=Ancestry(repo),
        documents=documents,
        leakage_codes=leakage_codes,
    )
    counts: dict[str, int] = {}
    for r in [*vault_records, *other_records]:
        counts[r.source] = counts.get(r.source, 0) + 1
    return AuditReport(results, sorted(documents), counts)
