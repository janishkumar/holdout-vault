"""The vault: score submissions against sealed labels under a reveal budget, and log everything.

The budget is derived from the ledger, never from a config file, so it cannot be raised without leaving
a trace: ``remaining = budget + granted - spent``, where ``budget`` is the last ``budget`` entry (or the
default), ``granted`` sums ``grant`` entries and ``spent`` counts ``score`` and ``evaluate`` entries.
Validation errors, tamper findings and blocking integrity findings never spend a reveal.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from holdout_vault import __version__
from holdout_vault.core.canary import canary_agreement
from holdout_vault.core.errors import BudgetExhausted, Refused, TamperDetected, ValidationError
from holdout_vault.core.findings import Finding
from holdout_vault.core.hashing import file_sha256, frame_sha256, normalise_value
from holdout_vault.core.ledger import Entry, VerifyResult
from holdout_vault.core.metrics import (
    MetricError,
    compute_metrics,
    load_custom_scorer,
    scorer_code_hash,
)
from holdout_vault.core.store import DEFAULT_BUDGET, DatasetRecord, Store

Mode = Literal["watch", "sealed"]


@dataclass(frozen=True)
class Approval:
    """A human decision to allow one more reveal past the budget."""

    by: str
    via: str
    note: str | None = None


@dataclass(frozen=True)
class BudgetState:
    budget: int
    granted: int
    spent: int

    @property
    def remaining(self) -> int:
        return self.budget + self.granted - self.spent


@dataclass
class ScoreResult:
    dataset: str
    metrics: dict[str, float]
    n_scored: int
    budget: int
    remaining: int
    entry_seq: int
    entry_hash: str
    mode: Mode
    stamp: dict[str, Any]
    findings: list[Finding] = field(default_factory=list)


def _label_str(s: pd.Series) -> pd.Series:
    return s.map(lambda v: str(normalise_value(v)))


class Vault:
    def __init__(self, home: str | Path | None, project: str, mode: Mode | None = None):
        self.store = Store.open(home, project)
        resolved = mode or os.environ.get("HOLDOUT_VAULT_MODE", "watch")
        if resolved not in ("watch", "sealed"):
            raise ValueError(f"mode must be 'watch' or 'sealed', not {resolved!r}")
        self.mode: Mode = resolved  # type: ignore[assignment]

    # budget -----------------------------------------------------------------------------------
    def budget_state(self) -> BudgetState:
        budget, granted, spent = DEFAULT_BUDGET, 0, 0
        for e in self.store.ledger.entries():
            if e.kind == "budget":
                budget = int(e.payload["budget"])
            elif e.kind == "grant":
                granted += int(e.payload["n"])
            elif e.kind in ("score", "evaluate"):
                spent += 1
        return BudgetState(budget, granted, spent)

    def set_budget(self, n: int, *, by: str) -> Entry:
        if n < 0:
            raise ValueError("budget must be zero or more")
        return self.store.ledger.append("budget", {"budget": int(n), "by": by})

    def grant(self, n: int, *, by: str, via: str, note: str | None = None) -> Entry:
        if n < 1:
            raise ValueError("grant at least one reveal")
        return self.store.ledger.append("grant", {"n": int(n), "by": by, "via": via, "note": note})

    def verify_ledger(self) -> VerifyResult:
        return self.store.ledger.verify()

    # integrity --------------------------------------------------------------------------------
    def _record(self, dataset: str) -> DatasetRecord:
        try:
            return self.store.read_record(dataset)
        except (KeyError, ValueError) as exc:
            raise ValidationError(
                f"dataset {dataset!r} is not registered in project {self.store.project!r}"
            ) from exc

    def integrity_findings(self, rec: DatasetRecord) -> list[Finding]:
        found: list[Finding] = []
        if rec.scorer_kind == "custom":
            current = file_sha256(self.store.custom_scorer_path(rec.name))
        else:
            current = scorer_code_hash()
        if current != rec.scorer_sha256:
            found.append(
                Finding(
                    code="HV-SCR-001",
                    severity="block",
                    title="The scorer changed since registration",
                    explanation=(
                        "The scoring code no longer matches the hash pinned when the dataset was registered. "
                        "Scores from a modified scorer cannot be trusted, so the vault refuses to score."
                    ),
                    location={"scorer": rec.scorer_kind},
                    evidence={"pinned": rec.scorer_sha256, "current": current},
                )
            )
        labels_now = file_sha256(self.store.labels_path(rec.name))
        if labels_now != rec.labels_sha256:
            found.append(
                Finding(
                    code="HV-LAB-001",
                    severity="block",
                    title="The sealed labels changed since registration",
                    explanation="The sealed label file does not match its registration hash.",
                    location={"dataset": rec.name},
                    evidence={"pinned": rec.labels_sha256, "current": labels_now},
                )
            )
        if self.mode == "watch":
            path = Path(rec.test_path)
            if not path.exists():
                found.append(
                    Finding(
                        code="HV-TST-002",
                        severity="warn",
                        title="The watched test file is missing",
                        explanation="The original test file was moved or deleted after registration.",
                        location={"path": rec.test_path},
                    )
                )
            elif file_sha256(path) != rec.test_file_sha256:
                found.append(
                    Finding(
                        code="HV-TST-001",
                        severity="block",
                        title="The watched test file changed since registration",
                        explanation=(
                            "In watch-only mode the original test file stays where it is, so the vault "
                            "checks its hash on every call. It no longer matches."
                        ),
                        location={"path": rec.test_path},
                        evidence={"pinned": rec.test_file_sha256, "current": file_sha256(path)},
                    )
                )
        return found

    # validation -------------------------------------------------------------------------------
    def _align(self, rec: DatasetRecord, labels: pd.DataFrame, preds: pd.DataFrame) -> pd.DataFrame:
        id_col = rec.id_col if rec.id_col in preds.columns else ("id" if "id" in preds.columns else None)
        if id_col is None:
            raise ValidationError(f"predictions need the id column {rec.id_col!r} (or 'id')")
        if "prediction" not in preds.columns:
            raise ValidationError("predictions need a 'prediction' column")
        ids = _label_str(preds[id_col])
        dup = ids[ids.duplicated()]
        if len(dup):
            raise ValidationError(f"{len(dup)} duplicate id(s) in predictions, e.g. {dup.iloc[0]}")
        expected = set(labels["id"])
        got = set(ids)
        missing, unknown = expected - got, got - expected
        if missing:
            raise ValidationError(
                f"{len(missing)} test id(s) missing from predictions, e.g. {sorted(missing)[0]}"
            )
        if unknown:
            raise ValidationError(f"{len(unknown)} unknown id(s) in predictions, e.g. {sorted(unknown)[0]}")
        if preds["prediction"].isna().any():
            raise ValidationError("predictions contain missing values")
        body = preds.drop(columns=[id_col]).assign(id=ids.to_numpy())
        return labels.merge(body, on="id", how="left", validate="one_to_one")

    # scoring ----------------------------------------------------------------------------------
    def score(
        self,
        dataset: str,
        predictions: pd.DataFrame,
        *,
        stamp: dict[str, Any] | None = None,
        approval: Approval | None = None,
        kind: Literal["score", "evaluate"] = "score",
    ) -> ScoreResult:
        rec = self._record(dataset)
        ledger = self.store.ledger

        findings = self.integrity_findings(rec)
        blocking = [f for f in findings if f.severity == "block"]
        if blocking:
            ledger.append("tamper", {"dataset": dataset, "findings": [f.to_dict() for f in blocking]})
            raise TamperDetected(blocking)

        labels = self.store.read_labels(dataset)
        merged = self._align(rec, labels, predictions)

        state = self.budget_state()
        if state.remaining <= 0:
            if approval is None:
                ledger.append("refused", {"dataset": dataset, "reason": "over_budget", "spent": state.spent})
                raise BudgetExhausted(dataset, state.budget + state.granted, state.spent)
            self.grant(1, by=approval.by, via=approval.via, note=approval.note)
            state = self.budget_state()

        real = merged[~merged["is_canary"]].reset_index(drop=True)
        canaries = merged[merged["is_canary"]].reset_index(drop=True)
        pred_cols = [c for c in merged.columns if c not in ("id", "label", "is_canary")]
        try:
            if rec.task == "classification":
                y_true = real["label"].astype(str)
                body = real[pred_cols].assign(prediction=_label_str(real["prediction"]))
            else:
                y_true = real["label"].astype(float)
                body = real[pred_cols]
            if rec.scorer_kind == "custom":
                fn = load_custom_scorer(self.store.custom_scorer_path(dataset))
                metrics = {str(k): float(v) for k, v in fn(y_true, body).items()}
            else:
                metrics = compute_metrics(rec.metrics, y_true, body)
        except MetricError as exc:
            raise ValidationError(str(exc)) from exc
        except (TypeError, ValueError) as exc:
            raise ValidationError(f"predictions could not be scored: {exc}") from exc

        can_pred = (
            _label_str(canaries["prediction"]) if rec.task == "classification" else canaries["prediction"]
        )
        cheat = canary_agreement(rec.task, canaries["label"], can_pred, len(rec.classes))
        if cheat is not None:
            ledger.append("refused", {"dataset": dataset, "findings": [cheat.to_dict()]})
            raise Refused([cheat])

        full_stamp = {
            **(stamp or {}),
            "dataset_sha256": rec.test_sha256,
            "scorer_sha256": rec.scorer_sha256,
            "mode": self.mode,
            "vault_version": __version__,
        }
        remaining_after = state.remaining - 1
        entry = ledger.append(
            kind,
            {
                "dataset": dataset,
                "metrics": metrics,
                "n_scored": len(real),
                "predictions_sha256": frame_sha256(predictions),
                "stamp": full_stamp,
                "remaining": remaining_after,
                "approved_by": approval.by if approval else None,
            },
        )
        full_stamp = {**full_stamp, "ledger_seq": entry.seq, "ledger_hash": entry.hash}
        return ScoreResult(
            dataset=dataset,
            metrics=metrics,
            n_scored=len(real),
            budget=state.budget + state.granted,
            remaining=remaining_after,
            entry_seq=entry.seq,
            entry_hash=entry.hash,
            mode=self.mode,
            stamp=full_stamp,
            findings=[f for f in findings if f.severity != "block"],
        )

    # status -----------------------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        state = self.budget_state()
        check = self.verify_ledger()
        datasets = []
        for name in self.store.datasets():
            r = self.store.read_record(name)
            datasets.append(
                {
                    "name": r.name,
                    "task": r.task,
                    "metrics": r.metrics,
                    "id_col": r.id_col,
                    "n_test_rows": r.n_test + r.n_canaries,
                    "export_dir": r.export_dir,
                    "scorer": r.scorer_kind,
                }
            )
        return {
            "project": self.store.project,
            "mode": self.mode,
            "budget": state.budget + state.granted,
            "spent": state.spent,
            "remaining": state.remaining,
            "ledger_ok": check.ok,
            "ledger_entries": check.count,
            "ledger_head": check.head_hash,
            "datasets": datasets,
        }
