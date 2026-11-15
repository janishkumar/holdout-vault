"""Precision and recall of the claim audit on hand-labelled reports.

Each ``caseNN.md`` in the corpus has a ``caseNN.labels.json`` listing its claims in order as
``{"metric", "shown", "label"}``. The reports are audited against the world in ``claims_world.py``.
Predicted and gold claims are paired by metric and the number as written. Reported:

- extraction precision and recall (did the audit find exactly the claims a person would mark);
- per-label precision and recall, end to end (a missed claim counts against its label);
- "unsupported" detection: anything not ``traced`` (what a reviewer acts on).

Run: PYTHONPATH=src uv run python benchmark/claim_checks.py [--corpus benchmark/corpus/claims_blind]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from claims_world import build_world

from holdout_vault.claims.audit import LABELS, audit_workspace
from holdout_vault.claims.records import vault_records, workspace_records
from holdout_vault.core.vault import Vault


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def norm(shown: str) -> str:
    s = str(shown).strip().lower().replace(" ", "").replace("percent", "%")
    return s.lstrip("+")


def pair(gold: list[dict], pred: list[dict]) -> list[tuple[dict | None, dict | None]]:
    """Greedy in-order pairing on (metric, number as written)."""
    left = list(range(len(pred)))
    pairs: list[tuple[dict | None, dict | None]] = []
    for g in gold:
        hit = next(
            (i for i in left if pred[i]["metric"] == g["metric"] and norm(pred[i]["shown"]) == norm(g["shown"])), None
        )
        if hit is None:
            pairs.append((g, None))
        else:
            left.remove(hit)
            pairs.append((g, pred[hit]))
    pairs.extend((None, pred[i]) for i in left)
    return pairs


def evaluate(corpus: Path) -> dict:
    cases = sorted(corpus.glob("case*.md"))
    rows = []
    with tempfile.TemporaryDirectory() as d:
        world = build_world(Path(d))
        vault = vault_records(Vault(world["home"], world["project"]).ledger_entries())
        others = workspace_records(world["ws"])
        for case in cases:
            gold = json.loads(case.with_suffix(".labels.json").read_text())
            report = audit_workspace(world["ws"], vault, others, text=case.read_text())
            pred = [{"metric": r.claim.metric, "shown": r.claim.shown, "label": r.label} for r in report.results]
            rows.append({"case": case.name, "pairs": pair(gold, pred)})
    allp = [p for r in rows for p in r["pairs"]]
    matched = [(g, p) for g, p in allp if g and p]
    n_gold = sum(1 for g, _ in allp if g)
    n_pred = sum(1 for _, p in allp if p)
    per_label = {}
    for label in LABELS:
        tp = sum(1 for g, p in matched if g["label"] == label and p["label"] == label)
        fp = sum(1 for g, p in allp if p and p["label"] == label and not (g and g["label"] == label))
        fn = sum(1 for g, p in allp if g and g["label"] == label and not (p and p["label"] == label))
        per_label[label] = {"tp": tp, "fp": fp, "fn": fn,
                            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
                            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
                            "recall_ci95": wilson(tp, tp + fn)}  # fmt: skip
    # unsupported = any label but traced; a missed gold claim counts as not flagged
    utp = sum(1 for g, p in allp if g and g["label"] != "traced" and p and p["label"] != "traced")
    ufp = sum(1 for g, p in allp if p and p["label"] != "traced" and not (g and g["label"] != "traced"))
    ufn = sum(1 for g, p in allp if g and g["label"] != "traced" and not (p and p["label"] != "traced"))
    errors = [
        {"case": r["case"], "gold": g, "predicted": p}
        for r in rows for g, p in r["pairs"] if not (g and p and g["label"] == p["label"])
    ]  # fmt: skip
    return {
        "corpus": corpus.name,
        "reports": len(cases),
        "gold_claims": n_gold,
        "predicted_claims": n_pred,
        "extraction_precision": round(len(matched) / n_pred, 3) if n_pred else None,
        "extraction_recall": round(len(matched) / n_gold, 3) if n_gold else None,
        "label_accuracy_on_matched": round(sum(g["label"] == p["label"] for g, p in matched) / len(matched), 3)
        if matched
        else None,
        "end_to_end_correct": sum(1 for g, p in matched if g["label"] == p["label"]),
        "unsupported": {
            "tp": utp,
            "fp": ufp,
            "fn": ufn,
            "precision": round(utp / (utp + ufp), 3) if utp + ufp else None,
            "precision_ci95": wilson(utp, utp + ufp),
            "recall": round(utp / (utp + ufn), 3) if utp + ufn else None,
            "recall_ci95": wilson(utp, utp + ufn),
        },
        "per_label": per_label,
        "errors": errors,
    }


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=here / "corpus" / "claims_blind")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    res = evaluate(args.corpus.resolve())
    out = args.out or here / "results" / f"claim_checks_{args.corpus.name}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"{res['reports']} reports, {res['gold_claims']} gold claims, {res['predicted_claims']} predicted")
    print(f"extraction precision {res['extraction_precision']} recall {res['extraction_recall']}")
    print(f"label accuracy on matched claims {res['label_accuracy_on_matched']}")
    u = res["unsupported"]
    print(f"unsupported: precision {u['precision']} {u['precision_ci95']} recall {u['recall']} {u['recall_ci95']}")
    for label, m in res["per_label"].items():
        print(f"  {label:17s} tp={m['tp']} fp={m['fp']} fn={m['fn']} precision={m['precision']} recall={m['recall']}")
    for e in res["errors"]:
        print(f"  ERR {e['case']}: gold {e['gold']} predicted {e['predicted']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
