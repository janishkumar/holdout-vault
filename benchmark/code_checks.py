"""Precision and recall of the code checks (HV-COD-001 to 006) on a labelled corpus.

Each case is a top-level .py or .ipynb file whose first line is ``# expect: <codes>`` (or ``none``), or a
folder with an ``EXPECT`` file. A case counts as a true positive for a code when the code is expected
and found anywhere in the case. ``corpus/code_blind`` was written from the finding definitions only,
without access to the checker.

Run: PYTHONPATH=src uv run python benchmark/code_checks.py [--corpus benchmark/corpus/code_blind]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from holdout_vault.checks.code import Context, code_findings

CODES = [f"HV-COD-00{i}" for i in range(1, 7)]
# the setting the corpus was written for
CTX = Context(
    targets={"churn"},
    labelled_test=["/home/me/project/raw/test.csv"],
    vault_home="~/.config/holdout-vault",
)


def parse_expect(text: str) -> set[str]:
    text = text.strip().removeprefix("# expect:").strip()
    return set() if text.lower() == "none" else {c.strip() for c in text.split(",") if c.strip()}


def expected(case: Path) -> set[str]:
    if case.is_dir():
        return parse_expect((case / "EXPECT").read_text())
    if case.suffix == ".ipynb":
        nb = json.loads(case.read_text())
        first = next(c for c in nb["cells"] if c["cell_type"] == "code")
        src = first["source"]
        return parse_expect(("".join(src) if isinstance(src, list) else src).splitlines()[0])
    return parse_expect(case.read_text().splitlines()[0])


def wilson(k: int, n: int, z: float = 1.96) -> list[float] | None:
    if n == 0:
        return None
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(c - h, 3), round(c + h, 3)]


def evaluate(corpus: Path) -> dict:
    cases = sorted(p for p in corpus.iterdir() if p.is_dir() or p.suffix in {".py", ".ipynb"})
    rows = []
    for case in cases:
        report = code_findings(case, CTX) if case.is_dir() else code_findings(corpus, CTX, paths=[case])
        found = {f.code for f in report.findings}
        rows.append({"case": case.name, "expected": sorted(expected(case)), "found": sorted(found)})
    per_code = {}
    tp_all = fp_all = fn_all = 0
    for code in CODES:
        tp = sum(code in r["expected"] and code in r["found"] for r in rows)
        fp = sum(code not in r["expected"] and code in r["found"] for r in rows)
        fn = sum(code in r["expected"] and code not in r["found"] for r in rows)
        tp_all, fp_all, fn_all = tp_all + tp, fp_all + fp, fn_all + fn
        per_code[code] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "recall_ci95": wilson(tp, tp + fn),
        }
    clean = [r for r in rows if not r["expected"]]
    return {
        "corpus": str(corpus.name),
        "cases": len(rows),
        "clean_cases": len(clean),
        "clean_cases_flagged": sum(bool(r["found"]) for r in clean),
        "precision": round(tp_all / (tp_all + fp_all), 3) if tp_all + fp_all else None,
        "precision_ci95": wilson(tp_all, tp_all + fp_all),
        "recall": round(tp_all / (tp_all + fn_all), 3) if tp_all + fn_all else None,
        "recall_ci95": wilson(tp_all, tp_all + fn_all),
        "exact_cases": sum(r["expected"] == r["found"] for r in rows),
        "per_code": per_code,
        "mismatches": [r for r in rows if r["expected"] != r["found"]],
    }


def main() -> None:
    here = Path(__file__).parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", type=Path, default=here / "corpus" / "code_blind")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    args.corpus = args.corpus.resolve()
    res = evaluate(args.corpus)
    out = args.out or here / "results" / f"code_checks_{args.corpus.name}.json"
    out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"{res['cases']} cases, {res['exact_cases']} exactly right")
    print(f"precision {res['precision']} {res['precision_ci95']}  recall {res['recall']} {res['recall_ci95']}")
    print(f"clean cases flagged: {res['clean_cases_flagged']}/{res['clean_cases']}")
    for code, m in res["per_code"].items():
        print(f"  {code}: tp={m['tp']} fp={m['fp']} fn={m['fn']} precision={m['precision']} recall={m['recall']}")
    for r in res["mismatches"]:
        print(f"  MISS {r['case']}: expected {r['expected']} found {r['found']}")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
