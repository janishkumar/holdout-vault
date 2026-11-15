"""Run records that a claimed number can be traced to.

- ``vault``: score and evaluate entries in the hash-chained ledger. The only trusted source.
- ``mlflow``: local file stores (``mlruns/``) and SQLite tracking databases, read directly; MLflow
  itself is not needed.
- ``json``: metric files matched by glob, every numeric leaf read as ``key = value``.
- ``wandb``: offline run summaries (``wandb/*/files/wandb-summary.json``), no account involved.
"""

from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from holdout_vault.checks.artifacts import SKIP_DIRS

DEFAULT_JSON_GLOBS = ("**/metrics*.json", "**/*results*.json", "**/*scores*.json", "**/*.metrics.jsonl")
MAX_RECORDS = 50_000


@dataclass(frozen=True)
class Record:
    source: str  # vault, mlflow, json, wandb
    metric: str  # as recorded
    value: float
    ref: str  # ledger seq, run id or file path
    run: str | None = None  # run name, when the source has one
    dataset: str | None = None
    commit: str | None = None
    meta: dict[str, Any] = field(default_factory=dict, compare=False, hash=False)


def vault_records(entries: list[Any]) -> list[Record]:
    """Ledger entries (objects with seq, kind, payload, hash, or their dict form) to records."""
    out: list[Record] = []
    for e in entries:
        d = e if isinstance(e, dict) else {"seq": e.seq, "kind": e.kind, "payload": e.payload, "hash": e.hash}
        if d["kind"] not in ("score", "evaluate"):
            continue
        p = d["payload"]
        stamp = p.get("stamp") or {}
        for name, value in (p.get("metrics") or {}).items():
            out.append(
                Record(
                    source="vault",
                    metric=name,
                    value=float(value),
                    ref=str(d["seq"]),
                    dataset=p.get("dataset"),
                    commit=stamp.get("git_commit"),
                    meta={"hash": d.get("hash"), "kind": d["kind"], "ts": d.get("ts")},
                )
            )
    return out


def _skipped(path: Path, root: Path) -> bool:
    return any(part in SKIP_DIRS for part in path.relative_to(root).parts[:-1])


def mlflow_file_records(root: Path) -> list[Record]:
    out: list[Record] = []
    for store in [p for p in root.rglob("mlruns") if p.is_dir() and not _skipped(p, root)]:
        for metric_file in store.glob("*/*/metrics/**/*"):
            if not metric_file.is_file():
                continue
            run_dir = metric_file.parents[len(metric_file.relative_to(store).parts) - 3]
            name = str(metric_file.relative_to(run_dir / "metrics"))
            run_name = None
            tag = run_dir / "tags" / "mlflow.runName"
            if tag.is_file():
                run_name = tag.read_text().strip()
            commit = None
            ctag = run_dir / "tags" / "mlflow.source.git.commit"
            if ctag.is_file():
                commit = ctag.read_text().strip()
            for line in metric_file.read_text().splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    try:
                        value = float(parts[1])
                    except ValueError:
                        continue
                    out.append(
                        Record(
                            "mlflow",
                            name,
                            value,
                            ref=run_dir.name,
                            run=run_name,
                            commit=commit,
                            meta={"path": str(run_dir.relative_to(root))},
                        )
                    )
    return out


def mlflow_sqlite_records(db: Path, root: Path) -> list[Record]:
    out: list[Record] = []
    try:
        con = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    except sqlite3.Error:
        return out
    try:
        rows = con.execute(
            "SELECT m.key, m.value, m.run_uuid, r.name FROM metrics m LEFT JOIN runs r ON r.run_uuid = m.run_uuid"
        ).fetchall()
    except sqlite3.Error:
        rows = []
    finally:
        con.close()
    for key, value, run_uuid, name in rows[:MAX_RECORDS]:
        if value is not None:
            rel = str(db.relative_to(root))
            out.append(Record("mlflow", key, float(value), ref=run_uuid, run=name, meta={"path": rel}))
    return out


def _leaves(obj: Any, prefix: str = "") -> list[tuple[str, float]]:
    out: list[tuple[str, float]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.extend(_leaves(v, f"{prefix}.{k}" if prefix else str(k)))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out.extend(_leaves(v, f"{prefix}[{i}]"))
    elif isinstance(obj, int | float) and not isinstance(obj, bool):
        out.append((prefix, float(obj)))
    return out


def json_records(root: Path, globs: tuple[str, ...] = DEFAULT_JSON_GLOBS, source: str = "json") -> list[Record]:
    out: list[Record] = []
    seen: set[Path] = set()
    for pattern in globs:
        for path in sorted(root.glob(pattern)):
            if path in seen or not path.is_file() or _skipped(path, root) or path.stat().st_size > 20_000_000:
                continue
            seen.add(path)
            rel = str(path.relative_to(root))
            try:
                if path.suffix == ".jsonl":
                    docs = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
                else:
                    docs = [json.loads(path.read_text())]
            except (ValueError, OSError):
                continue
            for i, doc in enumerate(docs):
                for key, value in _leaves(doc):
                    metric = re.split(r"[.\[]", key)[-1] if key else key
                    out.append(
                        Record(
                            source, metric, value, ref=rel if len(docs) == 1 else f"{rel}:{i + 1}", meta={"key": key}
                        )
                    )
    return out


def workspace_records(root: str | Path, json_globs: tuple[str, ...] = DEFAULT_JSON_GLOBS) -> list[Record]:
    """Every non-vault record under the agent's workspace."""
    root = Path(root).resolve()
    out = mlflow_file_records(root)
    for db in [*root.glob("*.db"), *root.glob("mlruns/*.db"), *root.glob("**/mlflow.db")]:
        if not _skipped(db, root):
            out.extend(mlflow_sqlite_records(db, root))
    out.extend(json_records(root, json_globs))
    out.extend(json_records(root, ("wandb/*/files/wandb-summary.json",), source="wandb"))
    return list(dict.fromkeys(out))[:MAX_RECORDS]
