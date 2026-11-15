"""Row checks on a registered split: is the test set really unseen?

Run once at registration on the real test rows (canaries excluded), stored with the dataset record and
shown to the agent through ``check_leakage``. They describe the dataset, not the agent, so they warn
rather than block; ``register --strict`` turns them into a refusal.

- HV-ROW-001 exact duplicate feature rows across train and test
- HV-ROW-002 near-duplicate rows (nearest neighbour under a threshold calibrated on train itself)
- HV-ROW-003 train rows later than the earliest test row
- HV-ROW-004 the same group id in train and test
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from holdout_vault.core.findings import Finding
from holdout_vault.core.hashing import normalise_value, row_key

MAX_ROWS = 20_000  # rows sampled per split for the neighbour search
CALIBRATION_ROWS = 2_000
NEAR_FRACTION = 0.05  # near-duplicate radius as a fraction of the median train-to-train neighbour distance
TOP_CATEGORIES = 50
EXAMPLES = 5
MIN_NEAR = 3  # fewest near-copies worth reporting (5 caught fewer small leaks, same false alarms)


def _examples(ids: pd.Series) -> list:
    return [normalise_value(v) for v in ids.head(EXAMPLES).tolist()]


def _row_hashes(df: pd.DataFrame, cols: list[str]) -> pd.Series:
    return pd.Series(
        [row_key([normalise_value(v) for v in row]) for row in df[cols].itertuples(index=False, name=None)],
        index=df.index,
    )


def exact_duplicates(
    train: pd.DataFrame, test: pd.DataFrame, cols: list[str], test_ids: pd.Series
) -> tuple[Finding | None, pd.Series]:
    """HV-ROW-001. Returns the finding and a mask of test rows that exactly match a train row."""
    if not cols:
        return None, pd.Series(False, index=test.index)
    tr, te = _row_hashes(train, cols), _row_hashes(test, cols)
    mask = te.isin(set(tr))
    n = int(mask.sum())
    # how often train rows repeat among themselves: cross-split repeats at that rate are expected
    train_rate = float(tr.duplicated(keep=False).mean()) if len(tr) else 0.0
    test_rate = n / max(len(te), 1)
    if n == 0 or test_rate <= 2 * train_rate + 0.001:
        return None, mask
    return (
        Finding(
            code="HV-ROW-001",
            severity="warn",
            title=f"{n} test row(s) also appear in train",
            explanation=(
                "These test rows have exactly the same feature values as a training row. A model can "
                "memorise them, so the sealed score overstates how it does on genuinely new rows."
            ),
            location={"split": "test", "row_ids": _examples(test_ids[mask])},
            evidence={"count": n, "test_rate": round(test_rate, 4), "train_self_duplicate_rate": round(train_rate, 4)},
        ),
        mask,
    )


def _encode(train: pd.DataFrame, test: pd.DataFrame, cols: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    numeric = [c for c in cols if pd.api.types.is_numeric_dtype(train[c]) and not pd.api.types.is_bool_dtype(train[c])]
    if not numeric:
        return None  # all-categorical rows: a near duplicate is an exact duplicate
    parts_tr, parts_te = [], []
    for c in numeric:
        a = pd.to_numeric(train[c], errors="coerce").to_numpy(dtype=float)
        b = pd.to_numeric(test[c], errors="coerce").to_numpy(dtype=float)
        mu, sd = np.nanmean(a), np.nanstd(a)
        sd = sd if sd > 0 else 1.0
        parts_tr.append(np.nan_to_num((a - mu) / sd))
        parts_te.append(np.nan_to_num((b - mu) / sd))
    for c in cols:
        if c in numeric:
            continue
        top = train[c].astype(str).value_counts().index[:TOP_CATEGORIES]
        for cat in top:
            # scaled so that a category mismatch adds 1 to the squared distance
            parts_tr.append((train[c].astype(str) == cat).to_numpy(dtype=float) / np.sqrt(2))
            parts_te.append((test[c].astype(str) == cat).to_numpy(dtype=float) / np.sqrt(2))
    return np.column_stack(parts_tr), np.column_stack(parts_te)


def near_duplicates(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cols: list[str],
    test_ids: pd.Series,
    exact: pd.Series,
    rng: np.random.Generator,
) -> Finding | None:
    """HV-ROW-002, calibrated on train: the radius is a small fraction of the typical train-to-train
    nearest-neighbour distance, and test is flagged only when it sits that close to train far more
    often than train rows sit that close to each other."""
    if len(train) < 10 or not cols:
        return None
    tr = train.sample(min(len(train), MAX_ROWS), random_state=int(rng.integers(1 << 31)))
    keep = ~exact
    te = test[keep]
    ids = test_ids[keep]
    if len(te) > MAX_ROWS:
        te = te.sample(MAX_ROWS, random_state=int(rng.integers(1 << 31)))
        ids = ids.loc[te.index]
    if te.empty:
        return None
    enc = _encode(tr, te, cols)
    if enc is None:
        return None
    a, b = enc
    nn = NearestNeighbors(n_neighbors=2).fit(a)
    cal = a[rng.choice(len(a), size=min(len(a), CALIBRATION_ROWS), replace=False)]
    d_self = nn.kneighbors(cal, n_neighbors=2)[0][:, 1]
    positive = d_self[d_self > 0]
    if positive.size == 0:
        return None
    radius = NEAR_FRACTION * float(np.median(positive))
    base_rate = float(((d_self > 0) & (d_self < radius)).mean())
    d_test = nn.kneighbors(b, n_neighbors=1)[0][:, 0]
    near = (d_test > 0) & (d_test < radius)
    n = int(near.sum())
    rate = n / len(te)
    if n < MIN_NEAR or rate <= 3 * base_rate + 0.005:
        return None
    return Finding(
        code="HV-ROW-002",
        severity="warn",
        title=f"{n} test row(s) are near-copies of training rows",
        explanation=(
            "These test rows sit much closer to a training row than training rows usually sit to each "
            "other, which is what jittered or re-rounded copies look like. The sealed score may be "
            "optimistic."
        ),
        location={"split": "test", "row_ids": _examples(ids[near])},
        evidence={
            "count": n,
            "test_rate": round(rate, 4),
            "train_base_rate": round(base_rate, 4),
            "radius": round(radius, 6),
            "rows_checked": len(te),
        },
    )


def _as_time(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s):
        return s
    return pd.to_datetime(s, errors="coerce", utc=True, format="mixed")


def time_order(train: pd.DataFrame, test: pd.DataFrame, col: str) -> Finding | None:
    """HV-ROW-003."""
    tr, te = _as_time(train[col]), _as_time(test[col])
    if te.dropna().empty or tr.dropna().empty:
        return None
    first_test = te.min()
    late = tr > first_test
    n = int(late.sum())
    if n == 0:
        return None
    return Finding(
        code="HV-ROW-003",
        severity="warn",
        title=f"{n} training row(s) are later than the earliest test row",
        explanation=(
            f"Train and test overlap in time on {col!r}. A model trained on the future can predict the "
            "past, so the sealed score overstates real forecasting performance."
        ),
        location={"split": "train", "column": col},
        evidence={
            "count": n,
            "fraction": round(n / len(train), 4),
            "earliest_test": str(first_test),
            "latest_train": str(tr.max()),
        },
    )


def group_overlap(train: pd.DataFrame, test: pd.DataFrame, col: str, test_ids: pd.Series) -> Finding | None:
    """HV-ROW-004."""
    shared = set(train[col].dropna().map(normalise_value)) & set(test[col].dropna().map(normalise_value))
    if not shared:
        return None
    mask = test[col].map(normalise_value).isin(shared)
    return Finding(
        code="HV-ROW-004",
        severity="warn",
        title=f"{len(shared)} group(s) appear in both train and test",
        explanation=(
            f"Rows sharing a {col!r} value are related. With the same groups on both sides, the sealed "
            "score measures performance on familiar groups, not new ones."
        ),
        location={"split": "test", "column": col, "row_ids": _examples(test_ids[mask])},
        evidence={
            "groups_shared": len(shared),
            "test_rows_affected": int(mask.sum()),
            "example_groups": sorted(map(str, shared))[:EXAMPLES],
        },
    )


def row_findings(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_cols: list[str],
    *,
    test_ids: pd.Series,
    time_col: str | None = None,
    group_col: str | None = None,
    seed: int = 0,
) -> list[Finding]:
    """All row checks for one train/test pair. ``test`` holds the real test rows only."""
    rng = np.random.default_rng(seed)
    train, test = train.reset_index(drop=True), test.reset_index(drop=True)
    test_ids = pd.Series(test_ids).reset_index(drop=True)
    # time and group columns identify rows; comparing them as features would hide real duplicates
    cols = [c for c in feature_cols if c not in {time_col, group_col}]
    found: list[Finding] = []
    exact, mask = exact_duplicates(train, test, cols, test_ids)
    if exact:
        found.append(exact)
    near = near_duplicates(train, test, cols, test_ids, mask, rng)
    if near:
        found.append(near)
    if time_col:
        t = time_order(train, test, time_col)
        if t:
            found.append(t)
    if group_col:
        g = group_overlap(train, test, group_col, test_ids)
        if g:
            found.append(g)
    return found
