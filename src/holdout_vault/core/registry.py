"""Dataset registration: hash the splits, seal the labels and scorer, export what the agent may see."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from holdout_vault.checks.rows import row_findings
from holdout_vault.core.canary import make_canaries, new_ids
from holdout_vault.core.errors import RegistrationError
from holdout_vault.core.hashing import file_sha256, frame_sha256
from holdout_vault.core.metrics import (
    BUILTIN_METRICS,
    CLASSIFICATION_METRICS,
    REGRESSION_METRICS,
    load_custom_scorer,
    scorer_code_hash,
)
from holdout_vault.core.store import DatasetRecord, Store, check_name

CANARY_FRACTION = 0.01
CANARY_MIN = 100
CANARY_MAX = 500
MAX_CLASSES_FOR_INT_TARGET = 20


def canary_count(n_test: int) -> int:
    return int(min(CANARY_MAX, max(CANARY_MIN, round(CANARY_FRACTION * n_test))))


def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if path.suffix.lower() in {".csv", ".txt"}:
        return pd.read_csv(path)
    raise RegistrationError(f"unsupported file type {path.suffix!r} for {path.name}: use .csv or .parquet")


def infer_task(y: pd.Series) -> str:
    if pd.api.types.is_bool_dtype(y) or pd.api.types.is_object_dtype(y) or isinstance(y.dtype, pd.CategoricalDtype):
        return "classification"
    if pd.api.types.is_integer_dtype(y) and y.nunique() <= MAX_CLASSES_FOR_INT_TARGET:
        return "classification"
    return "regression"


def _is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def register_dataset(
    *,
    home: str | Path | None,
    project: str,
    name: str,
    train_path: str | Path,
    test_path: str | Path,
    target: str,
    metrics: list[str],
    export_dir: str | Path,
    id_col: str | None = None,
    time_col: str | None = None,
    group_col: str | None = None,
    scorer_path: str | Path | None = None,
    seed: int = 0,
    replace: bool = False,
    by: str | None = None,
    agent_export_dir: str | None = None,
    strict: bool = False,
) -> DatasetRecord:
    try:
        check_name("dataset", name)
        store = Store.open(home, project)
    except ValueError as exc:
        raise RegistrationError(str(exc)) from exc
    export_dir = Path(export_dir)
    if _is_inside(export_dir, store.home):
        raise RegistrationError("export_dir is inside the vault store; export to the agent's workspace instead")
    if store.has_dataset(name) and not replace:
        raise RegistrationError(f"dataset {name!r} is already registered; pass replace=True to re-register")

    train = read_table(train_path)
    test = read_table(test_path)
    for label, df in (("train", train), ("test", test)):
        if target not in df.columns:
            raise RegistrationError(f"target column {target!r} missing from the {label} split")
    for col in (time_col, group_col):
        if col and (col not in train.columns or col not in test.columns):
            raise RegistrationError(f"column {col!r} must exist in both splits")
    if test[target].isna().any():
        raise RegistrationError("the test split has missing target values")

    task = infer_task(train[target])
    unknown = [m for m in metrics if m not in BUILTIN_METRICS]
    if unknown:
        raise RegistrationError(f"unknown metric(s): {', '.join(unknown)}")
    allowed = CLASSIFICATION_METRICS if task == "classification" else REGRESSION_METRICS
    wrong = [m for m in metrics if m not in allowed]
    if wrong:
        raise RegistrationError(f"metric(s) {', '.join(wrong)} do not apply to a {task} target")
    if not metrics and scorer_path is None:
        raise RegistrationError("give at least one metric or a custom scorer")

    rng = np.random.default_rng(seed)
    if id_col is None:
        id_col = "row_id"
        if id_col in test.columns:
            raise RegistrationError("test split already has a 'row_id' column; pass it as id_col")
    elif id_col not in test.columns:
        raise RegistrationError(f"id column {id_col!r} missing from the test split")
    elif not test[id_col].is_unique:
        raise RegistrationError(f"id column {id_col!r} must be unique in the test split")

    feature_cols = [c for c in test.columns if c not in {target, id_col}]
    missing = [c for c in feature_cols if c not in train.columns]
    if missing:
        raise RegistrationError(f"test columns missing from train: {', '.join(missing)}")

    rows = row_findings(
        train,
        test,
        feature_cols,
        test_ids=test[id_col] if id_col in test.columns else pd.Series(range(len(test))),
        time_col=time_col,
        group_col=group_col,
        seed=seed,
    )
    if strict and rows:
        listed = "; ".join(f"{f.code} {f.title}" for f in rows)
        raise RegistrationError(f"row checks failed (--strict): {listed}")

    n_can = canary_count(len(test))
    can_feats, can_labels = make_canaries(train, feature_cols, target, task, n_can, rng)

    real = test[feature_cols].copy()
    real_labels = test[target].reset_index(drop=True)
    synthetic_ids = id_col == "row_id"
    if synthetic_ids:
        real_ids = pd.Series(np.zeros(len(test), dtype=int))
        can_ids: list = [0] * n_can
    else:
        real_ids = test[id_col].reset_index(drop=True)
        can_ids = new_ids(pd.concat([real_ids, train[id_col]]) if id_col in train.columns else real_ids, n_can, rng)

    combined = pd.concat([real.reset_index(drop=True), can_feats], ignore_index=True)
    ids = pd.concat([real_ids, pd.Series(can_ids)], ignore_index=True)
    labels = pd.concat([real_labels, can_labels], ignore_index=True)
    is_canary = np.r_[np.zeros(len(test), dtype=bool), np.ones(n_can, dtype=bool)]

    order = rng.permutation(len(combined))
    combined, ids, labels = combined.iloc[order], ids.iloc[order], labels.iloc[order]
    is_canary = is_canary[order]
    if synthetic_ids:
        ids = pd.Series(np.arange(len(combined)))
    combined.insert(0, id_col, ids.to_numpy())

    ddir = store.dataset_dir(name)
    if ddir.exists():
        shutil.rmtree(ddir)
    ddir.mkdir(parents=True, mode=0o700)
    sealed = pd.DataFrame(
        {"id": ids.astype(str).to_numpy(), "label": labels.astype(str).to_numpy(), "is_canary": is_canary}
    )
    if task == "regression":
        sealed["label"] = pd.to_numeric(labels).to_numpy(dtype=float)
    sealed.to_parquet(store.labels_path(name), index=False)
    combined.to_parquet(store.features_path(name), index=False)

    if scorer_path is not None:
        load_custom_scorer(scorer_path)  # fail early on a broken scorer
        shutil.copyfile(scorer_path, store.custom_scorer_path(name))
        scorer_kind, scorer_sha = "custom", file_sha256(store.custom_scorer_path(name))
    else:
        scorer_kind, scorer_sha = "builtin", scorer_code_hash()

    out = export_dir / name
    out.mkdir(parents=True, exist_ok=True)
    train.to_csv(out / "train.csv", index=False)
    combined.to_csv(out / "test_features.csv", index=False)

    classes = sorted({str(v) for v in train[target].dropna()}) if task == "classification" else []
    record = DatasetRecord(
        home=str(store.home),
        project=project,
        name=name,
        task=task,
        target=target,
        id_col=id_col,
        metrics=list(metrics),
        feature_cols=feature_cols,
        n_train=len(train),
        n_test=len(test),
        n_canaries=n_can,
        train_sha256=frame_sha256(train),
        test_sha256=frame_sha256(test),
        labels_sha256=file_sha256(store.labels_path(name)),
        scorer_kind=scorer_kind,
        scorer_sha256=scorer_sha,
        test_path=str(Path(test_path).resolve()),
        test_file_sha256=file_sha256(test_path),
        export_dir=agent_export_dir or str(out.resolve()),
        registered_at=datetime.now(UTC).isoformat(timespec="seconds"),
        time_col=time_col,
        group_col=group_col,
        classes=classes,
        row_findings=[f.to_dict() for f in rows],
    )
    store.write_record(record)
    store.ledger.append(
        "register",
        {
            "dataset": name,
            "task": task,
            "metrics": record.metrics,
            "n_train": record.n_train,
            "n_test": record.n_test,
            "n_canaries": n_can,
            "train_sha256": record.train_sha256,
            "test_sha256": record.test_sha256,
            "labels_sha256": record.labels_sha256,
            "scorer_kind": scorer_kind,
            "scorer_sha256": scorer_sha,
            "replace": replace,
            "row_findings": [f.code for f in rows],
            "by": by,
        },
    )
    return record
