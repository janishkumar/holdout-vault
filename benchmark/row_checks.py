"""Detection and false-alarm rates of the row checks (HV-ROW-001 to 004) on seeded synthetic splits.

Every trial draws a fresh dataset (500 to 3000 rows, 3 to 12 numeric and 0 to 3 categorical features)
and builds one split per scenario. Positive scenarios inject one known problem; clean scenarios include
ones built to provoke false alarms (rounded features with natural repeats, all-categorical rows).

Run: PYTHONPATH=src uv run python benchmark/row_checks.py [--trials 100]
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

from holdout_vault.checks.rows import row_findings

# scenario -> the code it should raise (None: should raise nothing)
SCENARIOS: dict[str, str | None] = {
    "clean_random": None,
    "clean_rounded": None,
    "clean_categorical": None,
    "clean_temporal": None,
    "clean_grouped": None,
    "exact_dup_2pct": "HV-ROW-001",
    "exact_dup_10pct": "HV-ROW-001",
    "near_dup_2pct": "HV-ROW-002",
    "near_dup_10pct": "HV-ROW-002",
    "time_overlap": "HV-ROW-003",
    "group_overlap": "HV-ROW-004",
}


def wilson(k: int, n: int, z: float = 1.96) -> list[float]:
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [round(max(0.0, c - h), 3), round(min(1.0, c + h), 3)]


def dataset(rng: np.random.Generator, categorical_only: bool = False, rounded: bool = False) -> pd.DataFrame:
    n = int(rng.integers(500, 3001))
    cols: dict[str, np.ndarray] = {}
    if not categorical_only:
        k = int(rng.integers(3, 13)) if not rounded else 3
        mix = rng.normal(size=(n, k)) @ rng.normal(size=(k, k)) * rng.uniform(0.5, 20, size=k)
        for i in range(k):
            cols[f"num{i}"] = np.round(mix[:, i], 1) if rounded else mix[:, i]
    for i in range(int(rng.integers(0, 4)) if not categorical_only else 3):
        cols[f"cat{i}"] = rng.choice(list("abcde"[: int(rng.integers(2, 6))]), size=n)
    df = pd.DataFrame(cols)
    df["day"] = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 720, size=n), unit="D")
    df["group"] = rng.integers(0, n // 10, size=n)
    df["row"] = np.arange(n)
    return df


def build(name: str, rng: np.random.Generator) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    df = dataset(rng, categorical_only=name == "clean_categorical", rounded=name == "clean_rounded")
    feats = [c for c in df.columns if c.startswith(("num", "cat"))]
    kw: dict = {}
    if name in {"clean_temporal", "time_overlap"}:
        df = df.sort_values("day").reset_index(drop=True)
        kw["time_col"] = "day"
    if name in {"clean_grouped", "group_overlap"}:
        kw["group_col"] = "group"
    n = len(df)
    if name == "clean_temporal":
        cut = int(n * 0.8)
        train, test = df.iloc[:cut], df.iloc[cut:]
        test = test[test["day"] > train["day"].max()]
    elif name == "clean_grouped":
        groups = df["group"].unique()
        held = set(rng.choice(groups, size=len(groups) // 5, replace=False))
        test, train = df[df["group"].isin(held)], df[~df["group"].isin(held)]
    else:
        order = rng.permutation(n)
        cut = int(n * 0.8)
        train, test = df.iloc[order[:cut]], df.iloc[order[cut:]]
    train, test = train.reset_index(drop=True), test.reset_index(drop=True).copy()
    if name.startswith(("exact_dup", "near_dup")):
        frac = 0.02 if name.endswith("2pct") else 0.10
        k = max(1, int(len(test) * frac))
        src = train.sample(k, random_state=int(rng.integers(1 << 31)))
        rows = rng.choice(len(test), size=k, replace=False)
        for c in feats:
            vals = src[c].to_numpy()
            if name.startswith("near_dup") and c.startswith("num"):
                vals = vals + rng.normal(scale=0.01 * train[c].std(), size=k)
            test.loc[rows, c] = vals
    return train, test, {"feature_cols": [*feats, *kw.values()], **kw}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=100)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "results" / "row_checks.json")
    args = ap.parse_args()
    counts = {s: {} for s in SCENARIOS}
    for t in range(args.trials):
        for s in SCENARIOS:
            rng = np.random.default_rng([t, list(SCENARIOS).index(s)])
            train, test, kw = build(s, rng)
            feature_cols = kw.pop("feature_cols")
            found = row_findings(train, test, feature_cols, test_ids=test["row"], seed=t, **kw)
            for code in {f.code for f in found}:
                counts[s][code] = counts[s].get(code, 0) + 1
    results = {"trials": args.trials, "scenarios": {}}
    for s, expect in SCENARIOS.items():
        hit = counts[s].get(expect, 0) if expect else 0
        other = sum(v for k, v in counts[s].items() if k != expect)
        results["scenarios"][s] = {
            "expected": expect,
            "detected": hit if expect else None,
            "detection_rate": round(hit / args.trials, 3) if expect else None,
            "detection_ci95": wilson(hit, args.trials) if expect else None,
            "false_alarms": other,
            "false_alarm_ci95": wilson(other, args.trials),
            "codes_seen": counts[s],
        }
    clean = [s for s, e in SCENARIOS.items() if e is None]
    flagged = sum(sum(counts[s].values()) > 0 for s in clean)  # scenario-level, not trial-level
    results["clean_trials_with_any_finding"] = sum(sum(counts[s].values()) for s in clean)
    results["clean_trials"] = len(clean) * args.trials
    results["clean_scenarios_ever_flagged"] = flagged
    args.out.write_text(json.dumps(results, indent=2) + "\n")
    for s, r in results["scenarios"].items():
        print(
            f"{s:20s} expect={r['expected'] or '-':11s} detected={r['detected']} "
            f"fa={r['false_alarms']} {r['codes_seen']}"
        )
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
