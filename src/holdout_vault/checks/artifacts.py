"""HV-CAN-001: sealed canary labels found in a file in the agent's workspace.

Canary labels exist only inside the vault. A table in the workspace that carries canary ids next to a
column that agrees with those labels far above chance can only have come from the sealed store.

The front door collects candidate tables (it can see the workspace, the vault cannot); the vault does
the comparison (it holds the canary labels, the front door does not). Only plain data files are read.
Model files and pickles are never opened, because unpickling runs arbitrary code.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from holdout_vault.core.canary import canary_agreement
from holdout_vault.core.findings import Finding
from holdout_vault.core.hashing import normalise_value

TABLE_SUFFIXES = {".csv", ".tsv", ".parquet", ".pq", ".jsonl", ".json", ".feather"}
SKIP_DIRS = {".git", ".venv", "venv", "env", "node_modules", "__pycache__", "site-packages", ".tox", ".mypy_cache"}
MAX_FILES = 200
MAX_FILE_BYTES = 50 * 1024 * 1024
MAX_COLUMNS = 40
MIN_CANARY_ROWS = 10


def _read(path: Path) -> pd.DataFrame | None:
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t", low_memory=False)
        if suffix in {".parquet", ".pq"}:
            return pd.read_parquet(path)
        if suffix == ".feather":
            return pd.read_feather(path)
        if suffix == ".jsonl":
            return pd.read_json(path, lines=True)
        if suffix == ".json":
            df = pd.read_json(path)
            return df if isinstance(df, pd.DataFrame) else None
    except Exception:
        return None
    return None


def iter_files(root: Path, suffixes: set[str], max_files: int = MAX_FILES) -> list[Path]:
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if len(out) >= max_files:
            break
        if any(part in SKIP_DIRS for part in path.relative_to(root).parts[:-1]):
            continue
        if path.is_file() and path.suffix.lower() in suffixes:
            out.append(path)
    return out


def collect_tables(
    root: str | Path, id_cols: list[str], exclude: list[str | Path] = ()
) -> list[tuple[str, pd.DataFrame]]:
    """Tables under ``root`` that carry one of ``id_cols``, renamed so the id column is ``id``."""
    root = Path(root).resolve()
    skip = {Path(p).resolve() for p in exclude}
    tables: list[tuple[str, pd.DataFrame]] = []
    for path in iter_files(root, TABLE_SUFFIXES):
        if path.resolve() in skip or path.stat().st_size > MAX_FILE_BYTES:
            continue
        df = _read(path)
        if df is None or df.empty:
            continue
        id_col = next((c for c in id_cols if c in df.columns), None)
        if id_col is None:
            continue
        others = [c for c in df.columns if c != id_col][:MAX_COLUMNS]
        body = df[[id_col, *others]].rename(columns={id_col: "id"})
        tables.append((str(path.relative_to(root)), body.loc[:, ~body.columns.duplicated()]))
    return tables


def canary_artifact_findings(
    task: str, classes: list[str], canaries: pd.DataFrame, tables: list[tuple[str, pd.DataFrame]]
) -> list[Finding]:
    """Vault side. ``canaries`` has columns id (str) and label."""
    labels = canaries.set_index("id")["label"]
    found: list[Finding] = []
    class_set = set(classes)
    for path, df in tables:
        ids = df["id"].map(lambda v: str(normalise_value(v)))
        hit = ids.isin(labels.index)
        if int(hit.sum()) < MIN_CANARY_ROWS:
            continue
        rows = df[hit.to_numpy()]
        truth = labels.loc[ids[hit]].reset_index(drop=True)
        for col in [c for c in rows.columns if c != "id"]:
            values = rows[col].reset_index(drop=True)
            if values.isna().mean() > 0.5:
                continue
            if task == "classification":
                as_str = values.map(lambda v: str(normalise_value(v)))
                if as_str.isin(class_set).mean() < 0.5:
                    continue
                verdict = canary_agreement(task, truth, as_str, len(classes))
            else:
                if not pd.api.types.is_numeric_dtype(values):
                    continue
                verdict = canary_agreement(task, truth, values, 0)
            if verdict is None:
                continue
            found.append(
                Finding(
                    code="HV-CAN-001",
                    severity="block",
                    title=f"Sealed canary labels found in {path}",
                    explanation=(
                        f"Column {col!r} of {path} matches the labels of the hidden canary rows far above "
                        "chance. Those labels exist only in the vault, so this file was built from the "
                        "sealed labels."
                    ),
                    location={"file": path, "column": str(col)},
                    evidence=verdict.evidence,
                )
            )
            break  # one finding per file is enough
    return found
