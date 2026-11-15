"""Canary rows: synthetic test rows with random labels.

Features are drawn column by column from the training data, so each canary looks like a plausible
row but carries no joint signal, and its label is drawn uniformly at random from the classes (or from
the training targets, for regression). An honest model predicts canary labels at chance. Agreement far
above chance means the predictions were made with the sealed labels in hand.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from holdout_vault.core.findings import Finding

ALPHA = 1e-3


def make_canaries(
    train: pd.DataFrame, feature_cols: list[str], target: str, task: str, n: int, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.Series]:
    feats = {}
    for col in feature_cols:
        values = train[col].dropna().to_numpy()
        if len(values) == 0:
            feats[col] = [None] * n
            continue
        draw = rng.choice(values, size=n, replace=True)
        if pd.api.types.is_float_dtype(train[col]):
            scale = float(np.nanstd(values)) or 1.0
            draw = draw + rng.normal(0, 0.01 * scale, size=n)
        feats[col] = draw
    features = pd.DataFrame(feats, columns=feature_cols)
    if task == "classification":
        classes = np.array(sorted(train[target].dropna().unique().tolist(), key=str), dtype=object)
        labels = pd.Series(rng.choice(classes, size=n, replace=True))
    else:
        labels = pd.Series(rng.permutation(train[target].dropna().to_numpy())[:n])
        if len(labels) < n:
            labels = pd.Series(rng.choice(train[target].dropna().to_numpy(), size=n, replace=True))
    return features, labels


def new_ids(existing: pd.Series, n: int, rng: np.random.Generator) -> list:
    """Ids for canaries that blend in with the real ones and collide with none of them."""
    if pd.api.types.is_integer_dtype(existing):
        taken = set(existing.astype(int).tolist())
        lo, hi = int(existing.min()), int(existing.max())
        span = max(hi - lo + 1, len(taken) + n)
        pool = np.setdiff1d(np.arange(lo, lo + span + n), np.fromiter(taken, dtype=np.int64))
        return rng.choice(pool, size=n, replace=False).astype(int).tolist()
    width = max(8, int(existing.astype(str).str.len().median()))
    taken_s = set(existing.astype(str))
    out: list[str] = []
    while len(out) < n:
        cand = "".join(rng.choice(list("0123456789abcdef"), size=width))
        if cand not in taken_s:
            taken_s.add(cand)
            out.append(cand)
    return out


def canary_agreement(task: str, labels: pd.Series, predictions: pd.Series, n_classes: int) -> Finding | None:
    """Return a blocking finding when predictions match canary labels far above chance."""
    n = len(labels)
    if n == 0:
        return None
    if task == "classification":
        hits = int((labels.astype(str).to_numpy() == predictions.astype(str).to_numpy()).sum())
        chance = 1.0 / max(n_classes, 2)
        p = float(stats.binomtest(hits, n, chance, alternative="greater").pvalue)
        evidence = {"canaries": n, "matched": hits, "chance_rate": chance, "p_value": p}
    else:
        y = pd.to_numeric(labels, errors="coerce").to_numpy(dtype=float)
        yp = pd.to_numeric(predictions, errors="coerce").to_numpy(dtype=float)
        if np.nanstd(yp) == 0:
            return None
        rho, p_two = stats.spearmanr(y, yp)
        p = float(p_two / 2) if rho > 0 else 1.0
        evidence = {"canaries": n, "spearman_rho": float(rho), "p_value": p}
    if p >= ALPHA:
        return None
    return Finding(
        code="HV-CAN-002",
        severity="block",
        title="Predictions match the sealed canary labels far above chance",
        explanation=(
            "Some test rows are canaries whose labels were drawn at random and exist only inside the "
            "vault. An honest model cannot predict them better than chance. These predictions do, "
            "which means they were produced with access to the sealed labels."
        ),
        location={"rows": "canary rows (identities withheld)"},
        evidence=evidence,
    )
