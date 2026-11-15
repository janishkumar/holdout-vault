"""Turn a model into a predictions file. Runs inside the runner container, which has no labels.

A model is either a joblib/pickle file holding an estimator with ``predict`` (and optionally
``predict_proba``), or a directory containing ``predict.py`` that defines
``predict(features: pandas.DataFrame) -> pandas.DataFrame | array``. Features are passed without the
id column. The output has the id column, ``prediction`` and, when available, ``proba`` (binary) or
``proba_<class>`` columns.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd


class HarnessError(RuntimeError):
    pass


def _from_estimator(model, X: pd.DataFrame) -> pd.DataFrame:
    if not hasattr(model, "predict"):
        raise HarnessError(f"loaded object of type {type(model).__name__} has no predict()")
    out = pd.DataFrame({"prediction": np.asarray(model.predict(X))})
    if hasattr(model, "predict_proba"):
        try:
            proba = np.asarray(model.predict_proba(X))
        except Exception:
            proba = None
        classes = list(getattr(model, "classes_", []))
        if proba is not None and proba.ndim == 2 and len(classes) == proba.shape[1]:
            if len(classes) == 2:
                out["proba"] = proba[:, 1]
            else:
                for i, c in enumerate(classes):
                    out[f"proba_{c}"] = proba[:, i]
    return out


def _from_script(directory: Path, X: pd.DataFrame) -> pd.DataFrame:
    script = directory / "predict.py"
    if not script.is_file():
        raise HarnessError(f"{directory} has no predict.py")
    sys.path.insert(0, str(directory))
    spec = importlib.util.spec_from_file_location("agent_predict", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = getattr(module, "predict", None)
    if not callable(fn):
        raise HarnessError("predict.py must define predict(features)")
    result = fn(X.copy())
    if isinstance(result, pd.DataFrame):
        if "prediction" not in result.columns:
            raise HarnessError("predict() returned a frame without a 'prediction' column")
        return result.reset_index(drop=True)
    return pd.DataFrame({"prediction": np.asarray(result)})


def predict_file(model_path: Path, features: pd.DataFrame, id_col: str) -> pd.DataFrame:
    if id_col not in features.columns:
        raise HarnessError(f"features have no id column {id_col!r}")
    X = features.drop(columns=[id_col])
    if model_path.is_dir():
        preds = _from_script(model_path, X)
    else:
        import joblib

        preds = _from_estimator(joblib.load(model_path), X)
    if len(preds) != len(features):
        raise HarnessError(f"model returned {len(preds)} predictions for {len(features)} rows")
    preds.insert(0, id_col, features[id_col].to_numpy())
    return preds


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="holdout-vault-runner")
    p.add_argument("--model", required=True)
    p.add_argument("--features", required=True)
    p.add_argument("--id-col", required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)
    try:
        features = pd.read_csv(a.features)
        predict_file(Path(a.model), features, a.id_col).to_csv(a.out, index=False)
    except Exception as exc:
        print(f"runner error: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
