"""On-disk layout of the vault store.

``$HOLDOUT_VAULT_HOME`` (default ``~/.config/holdout-vault``)::

    projects/<project>/project.json
    projects/<project>/ledger.jsonl
    projects/<project>/datasets/<name>/record.json
    projects/<project>/datasets/<name>/labels.parquet     # id, label, is_canary
    projects/<project>/datasets/<name>/test_features.parquet  # what the agent and runner see
    projects/<project>/datasets/<name>/scorer.py          # only for custom scorers
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from holdout_vault.core.ledger import Ledger

DEFAULT_BUDGET = 20
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


def default_home() -> Path:
    env = os.environ.get("HOLDOUT_VAULT_HOME")
    return Path(env).expanduser() if env else Path.home() / ".config" / "holdout-vault"


def check_name(kind: str, value: str) -> str:
    if not _NAME.match(value):
        raise ValueError(f"invalid {kind} name {value!r}: use letters, digits, '.', '_' or '-'")
    return value


@dataclass
class DatasetRecord:
    home: str
    project: str
    name: str
    task: str
    target: str
    id_col: str
    metrics: list[str]
    feature_cols: list[str]
    n_train: int
    n_test: int
    n_canaries: int
    train_sha256: str
    test_sha256: str
    labels_sha256: str
    scorer_kind: str
    scorer_sha256: str
    test_path: str
    test_file_sha256: str
    export_dir: str
    registered_at: str
    time_col: str | None = None
    group_col: str | None = None
    classes: list[str] = field(default_factory=list)
    row_findings: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Store:
    def __init__(self, home: Path, project: str):
        self.home = Path(home)
        self.project = check_name("project", project)
        self.root = self.home / "projects" / project

    @classmethod
    def open(cls, home: str | Path | None, project: str) -> Store:
        store = cls(Path(home) if home else default_home(), project)
        store.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not store.config_path.exists():
            store.write_config({"project": project, "budget": DEFAULT_BUDGET})
        return store

    @property
    def config_path(self) -> Path:
        return self.root / "project.json"

    @property
    def ledger(self) -> Ledger:
        return Ledger(self.root / "ledger.jsonl")

    def read_config(self) -> dict[str, Any]:
        return json.loads(self.config_path.read_text())

    def write_config(self, cfg: dict[str, Any]) -> None:
        self.config_path.write_text(json.dumps(cfg, indent=2, sort_keys=True))

    def dataset_dir(self, name: str) -> Path:
        return self.root / "datasets" / check_name("dataset", name)

    def datasets(self) -> list[str]:
        d = self.root / "datasets"
        return sorted(p.name for p in d.iterdir() if (p / "record.json").exists()) if d.exists() else []

    def has_dataset(self, name: str) -> bool:
        return (self.dataset_dir(name) / "record.json").exists()

    def read_record(self, name: str) -> DatasetRecord:
        path = self.dataset_dir(name) / "record.json"
        if not path.exists():
            raise KeyError(f"dataset {name!r} is not registered in project {self.project!r}")
        return DatasetRecord(**json.loads(path.read_text()))

    def write_record(self, rec: DatasetRecord) -> None:
        (self.dataset_dir(rec.name) / "record.json").write_text(json.dumps(rec.to_dict(), indent=2))

    def labels_path(self, name: str) -> Path:
        return self.dataset_dir(name) / "labels.parquet"

    def read_labels(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.labels_path(name))

    def read_canaries(self, name: str) -> pd.DataFrame:
        labels = self.read_labels(name)
        return labels[labels["is_canary"]].reset_index(drop=True)

    def features_path(self, name: str) -> Path:
        return self.dataset_dir(name) / "test_features.parquet"

    def read_features(self, name: str) -> pd.DataFrame:
        return pd.read_parquet(self.features_path(name))

    def custom_scorer_path(self, name: str) -> Path:
        return self.dataset_dir(name) / "scorer.py"
