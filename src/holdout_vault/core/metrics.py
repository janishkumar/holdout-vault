"""Built-in metrics and custom scorer loading.

Predictions are a frame with a ``prediction`` column. Probability metrics (``roc_auc``,
``log_loss``) need either ``proba`` (positive-class probability, binary) or one
``proba_<class>`` column per class.
"""

from __future__ import annotations

import importlib.util
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import metrics as skm

from holdout_vault.core.hashing import file_sha256

Scorer = Callable[[pd.Series, pd.DataFrame], dict[str, float]]

PROBA_METRICS = frozenset({"roc_auc", "log_loss"})
CLASSIFICATION_METRICS = frozenset({"accuracy", "balanced_accuracy", "f1", "roc_auc", "log_loss"})
REGRESSION_METRICS = frozenset({"rmse", "mae", "r2"})
BUILTIN_METRICS = CLASSIFICATION_METRICS | REGRESSION_METRICS
HIGHER_IS_BETTER = {
    "accuracy": True,
    "balanced_accuracy": True,
    "f1": True,
    "roc_auc": True,
    "log_loss": False,
    "rmse": False,
    "mae": False,
    "r2": True,
}


class MetricError(ValueError):
    """Predictions or metric configuration are invalid. Never spends a reveal."""


def _proba(y_true: pd.Series, preds: pd.DataFrame) -> tuple[np.ndarray, list | None]:
    cols = [c for c in preds.columns if str(c).startswith("proba_")]
    if cols:
        classes = [str(c)[len("proba_") :] for c in cols]
        mat = preds[cols].to_numpy(dtype=float)
        _check_range(mat)
        return mat, classes
    if "proba" in preds.columns:
        vec = preds["proba"].to_numpy(dtype=float)
        _check_range(vec)
        return vec, None
    raise MetricError("probability metrics need a 'proba' column or 'proba_<class>' columns")


def _check_range(a: np.ndarray) -> None:
    if np.isnan(a).any() or (a < 0).any() or (a > 1).any():
        raise MetricError("probabilities must be in [0, 1] with no missing values")


def _one(name: str, y_true: pd.Series, preds: pd.DataFrame) -> float:
    y_pred = preds["prediction"]
    multiclass = y_true.nunique() > 2
    if name == "accuracy":
        return float(skm.accuracy_score(y_true, y_pred))
    if name == "balanced_accuracy":
        return float(skm.balanced_accuracy_score(y_true, y_pred))
    if name == "f1":
        if multiclass:
            return float(skm.f1_score(y_true, y_pred, average="macro"))
        pos = sorted(y_true.unique().tolist())[-1]
        return float(skm.f1_score(y_true, y_pred, pos_label=pos))
    if name in PROBA_METRICS:
        proba, classes = _proba(y_true, preds)
        if classes is not None:
            labels = [str(v) for v in y_true]
            if name == "roc_auc":
                return float(skm.roc_auc_score(labels, proba, multi_class="ovr", labels=classes))
            return float(skm.log_loss(labels, proba, labels=classes))
        if name == "roc_auc":
            return float(skm.roc_auc_score(y_true, proba))
        return float(skm.log_loss(y_true, proba, labels=sorted(y_true.unique().tolist())))
    y = pd.to_numeric(y_true, errors="raise").to_numpy(dtype=float)
    try:
        yp = pd.to_numeric(y_pred, errors="raise").to_numpy(dtype=float)
    except (ValueError, TypeError) as exc:
        raise MetricError(f"regression predictions must be numeric: {exc}") from exc
    if np.isnan(yp).any():
        raise MetricError("predictions contain missing values")
    if name == "rmse":
        return float(np.sqrt(skm.mean_squared_error(y, yp)))
    if name == "mae":
        return float(skm.mean_absolute_error(y, yp))
    return float(skm.r2_score(y, yp))


def compute_metrics(names: list[str], y_true: pd.Series, preds: pd.DataFrame) -> dict[str, float]:
    unknown = [n for n in names if n not in BUILTIN_METRICS]
    if unknown:
        raise MetricError(f"unknown metric(s): {', '.join(unknown)}")
    if "prediction" not in preds.columns:
        raise MetricError("predictions need a 'prediction' column")
    y_true = y_true.reset_index(drop=True)
    preds = preds.reset_index(drop=True)
    return {n: _one(n, y_true, preds) for n in names}


def scorer_code_hash() -> str:
    """Hash of this module's source: built-in scoring changes are visible in every stamp."""
    return file_sha256(Path(__file__))


def load_custom_scorer(path: str | Path) -> Scorer:
    path = Path(path)
    spec = importlib.util.spec_from_file_location(f"holdout_vault_custom_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise MetricError(f"cannot load scorer from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "score", None)
    if not callable(fn):
        raise MetricError(f"{path} must define score(y_true, predictions) -> dict[str, float]")
    return fn
