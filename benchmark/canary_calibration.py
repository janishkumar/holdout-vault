"""Measure the canary check: false alarms on honest models, and detection when labels leak.

For each trial a synthetic binary task is generated with a fresh seed and registered in a temporary
vault. An honest logistic regression is scored, then "leaky" submissions copy the true sealed label for
a fraction of rows (canaries included at the same rate) and use the honest prediction elsewhere.

Run: PYTHONPATH=src uv run python benchmark/canary_calibration.py [--trials 200]
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from holdout_vault.core import registry
from holdout_vault.core.canary import canary_agreement
from holdout_vault.core.registry import register_dataset
from holdout_vault.core.store import Store

LEAK_FRACTIONS = [0.1, 0.25, 0.5, 1.0]


def synthetic(n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    logit = X @ rng.normal(size=5)
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    df["id"] = np.arange(n)
    df["y"] = y
    return df


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def trial(seed: int, n_test: int, tmp: Path) -> dict[str, bool]:
    df = synthetic(n_test * 3, seed)
    train, test = df.iloc[: n_test * 2], df.iloc[n_test * 2 :]
    d = tmp / f"t{seed}"
    d.mkdir()
    train.to_csv(d / "train.csv", index=False)
    test.to_csv(d / "test.csv", index=False)
    rec = register_dataset(
        home=d / "home", project="cal", name="t", train_path=d / "train.csv", test_path=d / "test.csv",
        target="y", id_col="id", metrics=["accuracy"], export_dir=d / "ws", seed=seed,
    )  # fmt: skip
    feats = pd.read_csv(d / "ws" / "t" / "test_features.csv")
    cols = [f"f{i}" for i in range(5)]
    model = LogisticRegression().fit(train[cols], train["y"])
    honest = pd.Series(model.predict(feats[cols]), index=feats["id"].astype(str))
    labels = Store.open(rec.home, "cal").read_labels("t").set_index("id")
    can = labels[labels["is_canary"]]
    rng = np.random.default_rng(10_000 + seed)
    out = {"honest": canary_agreement("classification", can["label"], honest[can.index].astype(str), 2) is not None}
    for f in LEAK_FRACTIONS:
        leaked = rng.random(len(can)) < f
        pred = np.where(leaked, can["label"].to_numpy(), honest[can.index].astype(str).to_numpy())
        out[f"leak_{f}"] = canary_agreement("classification", can["label"], pd.Series(pred), 2) is not None
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=200)
    ap.add_argument("--n-test", type=int, default=2000, help="test rows per trial")
    ap.add_argument("--min-canaries", type=int, default=None, help="override the canary floor (ablation)")
    ap.add_argument("--out", default="benchmark/results/canary_calibration.json")
    a = ap.parse_args(argv)
    if a.min_canaries is not None:
        registry.CANARY_MIN = a.min_canaries
    n_canaries = registry.canary_count(a.n_test)
    with tempfile.TemporaryDirectory() as tmp:
        rows = [trial(s, a.n_test, Path(tmp)) for s in range(a.trials)]
    keys = rows[0].keys()
    summary = {}
    for k in keys:
        hits = sum(r[k] for r in rows)
        lo, hi = wilson(hits, len(rows))
        summary[k] = {"flagged": hits, "trials": len(rows), "rate": hits / len(rows), "ci95": [lo, hi]}
    result = {
        "trials": a.trials,
        "n_test": a.n_test,
        "canaries_per_trial": n_canaries,
        "alpha": 1e-3,
        "summary": summary,
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(result, indent=2))
    print(f"{a.trials} trials, {n_canaries} canaries each, alpha 1e-3")
    print("| Submission | Flagged | Rate | 95% CI |\n|---|---|---|---|")
    for k, s in summary.items():
        label = "Honest model" if k == "honest" else f"{float(k.split('_')[1]):.0%} of labels leaked"
        print(
            f"| {label} | {s['flagged']}/{s['trials']} | {s['rate']:.1%} | {s['ci95'][0]:.1%} to {s['ci95'][1]:.1%} |"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
