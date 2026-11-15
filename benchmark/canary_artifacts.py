"""HV-CAN-001: is a workspace file built from the sealed labels caught, and are honest files left alone?

Each trial registers a fresh synthetic task. The honest workspace holds the exported train and test
features, honest test predictions and out-of-fold train predictions. A leak file copies the sealed
labels for a random fraction of test rows (canaries at the same rate), as an agent that read the label
store and kept a slice of it would.

Run: PYTHONPATH=src uv run python benchmark/canary_artifacts.py [--trials 100]
"""

from __future__ import annotations

import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict

from holdout_vault.checks.artifacts import canary_artifact_findings, collect_tables
from holdout_vault.core.registry import register_dataset
from holdout_vault.core.store import Store

LEAK_FRACTIONS = [0.05, 0.1, 0.25, 1.0]


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def trial(seed: int, root: Path) -> dict[str, bool]:
    rng = np.random.default_rng(seed)
    n = 2500
    X = rng.normal(size=(n, 5))
    y = (rng.random(n) < 1 / (1 + np.exp(-(X @ rng.normal(size=5))))).astype(int)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)]).assign(id=np.arange(n), y=y)
    df.iloc[:2000].to_csv(root / "train_src.csv", index=False)
    df.iloc[2000:].to_csv(root / "test_src.csv", index=False)
    ws = root / "ws"
    rec = register_dataset(
        home=root / "vault", project="b", name="d", train_path=root / "train_src.csv",
        test_path=root / "test_src.csv", target="y", id_col="id", metrics=["accuracy"],
        export_dir=ws / "data", seed=seed,
    )  # fmt: skip
    store = Store.open(rec.home, "b")
    labels, canaries = store.read_labels("d"), store.read_canaries("d")
    train = pd.read_csv(ws / "data" / "d" / "train.csv")
    test = pd.read_csv(ws / "data" / "d" / "test_features.csv")
    feats = [f"f{i}" for i in range(5)]
    model = LogisticRegression().fit(train[feats], train["y"])
    pd.DataFrame({"id": test["id"], "prediction": model.predict(test[feats])}).to_csv(ws / "preds.csv", index=False)
    oof = cross_val_predict(LogisticRegression(), train[feats], train["y"], cv=5)
    pd.DataFrame({"id": train["id"], "oof": oof}).to_csv(ws / "oof.csv", index=False)

    def flagged(tables) -> bool:
        return bool(canary_artifact_findings("classification", rec.classes, canaries, tables))

    out = {"honest": flagged(collect_tables(ws, ["id"]))}
    for frac in LEAK_FRACTIONS:
        take = labels.sample(frac=frac, random_state=seed)
        leak = pd.DataFrame({"id": take["id"].astype(int), "label_guess": take["label"]})
        out[f"leak_{frac}"] = flagged([("leak.csv", leak)])
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "canary_artifacts.json")
    args = ap.parse_args()
    hits: dict[str, int] = {}
    for t in range(args.trials):
        with tempfile.TemporaryDirectory() as d:
            for k, v in trial(t, Path(d)).items():
                hits[k] = hits.get(k, 0) + int(v)
    res = {
        "trials": args.trials,
        "n_test": 500,
        "n_canaries": 100,
        "settings": {
            k: {"flagged": v, "rate": round(v / args.trials, 3), "ci95": wilson(v, args.trials)}
            for k, v in hits.items()
        },
    }
    args.out.write_text(json.dumps(res, indent=2) + "\n")
    for k, v in res["settings"].items():
        print(f"{k:12s} {v['flagged']}/{args.trials}  CI {v['ci95']}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
