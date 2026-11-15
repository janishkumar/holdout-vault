"""The code checks on real, carefully written code: scikit-learn's own examples (BSD-3-Clause).

The examples are not copied into this repository. Fetch them at the pinned commit:

    git clone --depth 1 --filter=blob:none --sparse https://github.com/scikit-learn/scikit-learn.git skl
    git -C skl fetch --depth 1 origin ee56c8b5d809dc7b970a140ed501269b63c69cf8 && git -C skl checkout ee56c8b5d809
    git -C skl sparse-checkout set examples
    PYTHONPATH=src uv run python benchmark/real_code.py --examples skl/examples

Every finding is compared with the reviewed verdicts below. These files were used to fix false alarms,
so this is a development measurement of false alarms on real code, not a held-out score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from holdout_vault.checks.code import Context, code_findings

COMMIT = "ee56c8b5d809dc7b970a140ed501269b63c69cf8"
# (code, file, line) -> verdict after reading the code
REVIEWED = {
    (
        "HV-COD-001",
        "applications/plot_digits_denoising.py",
        41,
    ): "true: MinMaxScaler fitted on all of X, then train_test_split",
    (
        "HV-COD-001",
        "linear_model/plot_tweedie_regression_insurance_claims.py",
        257,
    ): "true: ColumnTransformer fitted on the whole frame, which is split afterwards",
    ("HV-COD-001", "neighbors/plot_digits_kde_sampling.py", 30): "true: PCA fitted before GridSearchCV over its output",
    (
        "HV-COD-001",
        "svm/plot_rbf_parameters.py",
        129,
    ): "true: scaler fitted on all data before GridSearchCV; the example's comment says it is 'cheating a bit'",
}
FIRST_CONTACT = {"findings": 43, "true": 4, "note": "before the fixes informed by these files"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--examples", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "real_code_sklearn.json")
    args = ap.parse_args()
    root = args.examples.resolve()
    ctx = Context(targets={"target"}, labelled_test=["/nonexistent/raw/test_labels.csv"], vault_home="~/.config/x")
    files = sorted(root.rglob("*.py"))
    found = []
    for f in files:
        for x in code_findings(root, ctx, paths=[f]).findings:
            key = (x.code, x.location["file"], x.location["line"])
            found.append(
                {
                    "code": x.code,
                    "file": key[1],
                    "line": key[2],
                    "severity": x.severity,
                    "verdict": REVIEWED.get(key, "unreviewed"),
                }
            )
    true = sum(v["verdict"].startswith("true") for v in found)
    res = {
        "source": "scikit-learn examples",
        "commit": COMMIT,
        "licence": "BSD-3-Clause",
        "files": len(files),
        "findings": len(found),
        "true_by_definition": true,
        "unreviewed": sum(v["verdict"] == "unreviewed" for v in found),
        "first_contact": FIRST_CONTACT,
        "detail": found,
    }
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    print(f"{len(files)} files, {len(found)} findings, {true} true by definition, {res['unreviewed']} unreviewed")


if __name__ == "__main__":
    main()
