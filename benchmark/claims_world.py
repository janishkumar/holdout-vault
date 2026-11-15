"""A fixed world of run records for testing the claim audit.

``build_world(root)`` creates an agent workspace (a git repo whose HEAD is on ``main``, plus an abandoned
``experiment`` branch), a vault ledger with known scores stamped with those commits, an MLflow file store,
an MLflow SQLite database, a JSON metrics file and a W&B offline summary. ``WORLD_MD`` describes the same
facts in prose, for people writing labelled reports against it.
"""

from __future__ import annotations

import json
import sqlite3
import subprocess
from pathlib import Path

from holdout_vault.core.store import Store

PROJECT = "world"

# (kind, dataset, metrics, commit label)
LEDGER = [
    ("budget", None, None, None),
    ("register", "churn", None, None),
    ("score", "churn", {"accuracy": 0.8125, "roc_auc": 0.8840}, "A"),
    ("score", "churn", {"accuracy": 0.8203, "roc_auc": 0.8911}, "B"),
    ("score", "churn", {"accuracy": 0.8411, "roc_auc": 0.9012}, "C"),
    ("evaluate", "churn", {"accuracy": 0.8350, "roc_auc": 0.8975}, "D"),
    ("register", "fraud", None, None),
    ("score", "fraud", {"f1": 0.6620, "log_loss": 0.3120}, "C"),
    ("score", "fraud", {"f1": 0.6710, "log_loss": 0.3011}, "C"),
    ("register", "price", None, None),
    ("score", "price", {"rmse": 12.41, "mae": 8.93, "r2": 0.7120}, "C"),
    ("refused", "churn", None, None),
]

WORLD_MD = """\
# The world the reports are written against

Project with three registered datasets. The vault ledger (numbered entries) holds these sealed scores:

| Entry | Kind | Dataset | Metrics | Commit |
|---|---|---|---|---|
| #0 | budget | - | (reveal budget set to 20) | - |
| #1 | register | churn | - | - |
| #2 | score | churn | accuracy 0.8125, roc_auc 0.8840 | A (on main) |
| #3 | score | churn | accuracy 0.8203, roc_auc 0.8911 | B (on main) |
| #4 | score | churn | accuracy 0.8411, roc_auc 0.9012 | C (on main, current HEAD) |
| #5 | evaluate | churn | accuracy 0.8350, roc_auc 0.8975 | D (on an abandoned branch, NOT in main's history) |
| #6 | register | fraud | - | - |
| #7 | score | fraud | f1 0.6620, log_loss 0.3120 | C |
| #8 | score | fraud | f1 0.6710, log_loss 0.3011 | C |
| #9 | register | price | - | - |
| #10 | score | price | rmse 12.41, mae 8.93, r2 0.7120 | C |
| #11 | refused | churn | - (a refused submission, no metrics) | - |

The agent's own run records in the workspace (not vault scores):
- MLflow run "lgbm-v3": val_auc 0.9420, val_accuracy 0.8710
- MLflow run "baseline": accuracy 0.7900
- MLflow (SQLite database) run "xgb-tuned": val_f1 0.7015, val_log_loss 0.2890
- outputs/metrics.json: {"cv": {"roc_auc": 0.9310, "f1": 0.6990}}
- W&B offline run summary: val_rmse 11.20, val_r2 0.7590

The fraud dataset has only 2 sealed scores, price has 1, churn has 4 (#2 to #5).
"""


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True, text=True).stdout.strip()


def build_world(root: Path) -> dict:
    root = Path(root)
    ws = root / "ws"
    ws.mkdir(parents=True)
    _git(ws, "init", "-q", "-b", "main")
    _git(ws, "config", "user.email", "test@example.invalid")
    _git(ws, "config", "user.name", "Test")
    commits: dict[str, str] = {}

    def commit(label: str) -> None:
        (ws / "train.py").write_text(f"# version {label}\n")
        _git(ws, "add", "train.py")
        _git(ws, "commit", "-qm", label)
        commits[label] = _git(ws, "rev-parse", "HEAD")

    commit("A")
    commit("B")
    _git(ws, "checkout", "-q", "-b", "experiment")
    commit("D")
    _git(ws, "checkout", "-q", "main")
    commit("C")
    _git(ws, "branch", "-D", "experiment")  # abandoned: D is now unreachable from HEAD

    store = Store.open(root / "vault", PROJECT)
    for kind, dataset, metrics, commit in LEDGER:
        payload: dict = {"dataset": dataset} if dataset else {"budget": 20, "by": "human"}
        if metrics:
            payload |= {"metrics": metrics, "stamp": {"git_commit": commits[commit]}}
        store.ledger.append(kind, payload)

    def mlflow_run(run_id: str, name: str, metrics: dict[str, float]) -> None:
        run = ws / "mlruns" / "0" / run_id
        (run / "metrics").mkdir(parents=True)
        (run / "tags").mkdir()
        (run / "tags" / "mlflow.runName").write_text(name)
        for k, v in metrics.items():
            (run / "metrics" / k).write_text(f"1700000000000 {v} 0\n")

    mlflow_run("a1b2c3", "lgbm-v3", {"val_auc": 0.9420, "val_accuracy": 0.8710})
    mlflow_run("d4e5f6", "baseline", {"accuracy": 0.7900})
    con = sqlite3.connect(ws / "mlflow.db")
    con.execute("CREATE TABLE runs (run_uuid TEXT, name TEXT)")
    con.execute("CREATE TABLE metrics (key TEXT, value REAL, run_uuid TEXT, step INT, timestamp INT)")
    con.execute("INSERT INTO runs VALUES ('9z9z', 'xgb-tuned')")
    con.executemany("INSERT INTO metrics VALUES (?, ?, '9z9z', 0, 0)", [("val_f1", 0.7015), ("val_log_loss", 0.2890)])
    con.commit()
    con.close()
    (ws / "outputs").mkdir()
    (ws / "outputs" / "metrics.json").write_text(json.dumps({"cv": {"roc_auc": 0.9310, "f1": 0.6990}}))
    wb = ws / "wandb" / "offline-run-20261115_000000-abc" / "files"
    wb.mkdir(parents=True)
    (wb / "wandb-summary.json").write_text(json.dumps({"val_rmse": 11.20, "val_r2": 0.7590, "_step": 12}))
    return {"ws": ws, "home": root / "vault", "project": PROJECT, "commits": commits}
