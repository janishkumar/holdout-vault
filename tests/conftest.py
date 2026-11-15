from __future__ import annotations

import subprocess

import numpy as np
import pandas as pd
import pytest

from holdout_vault.core.registry import register_dataset


def make_classification(n: int = 600, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    cat = rng.choice(["red", "green", "blue"], size=n)
    logit = 1.5 * x1 - x2 + (cat == "red") * 0.5
    y = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    return pd.DataFrame(
        {"customer_id": np.arange(1000, 1000 + n), "x1": x1, "x2": x2, "colour": cat, "churn": y}
    )


@pytest.fixture
def home(tmp_path, monkeypatch):
    h = tmp_path / "vault-home"
    monkeypatch.setenv("HOLDOUT_VAULT_HOME", str(h))
    return h


@pytest.fixture
def split_files(tmp_path):
    df = make_classification()
    train, test = df.iloc[:400], df.iloc[400:]
    src = tmp_path / "src"
    src.mkdir()
    train.to_csv(src / "train.csv", index=False)
    test.to_csv(src / "test.csv", index=False)
    return src / "train.csv", src / "test.csv"


@pytest.fixture
def registered(home, split_files, tmp_path):
    train, test = split_files
    record = register_dataset(
        home=home,
        project="demo",
        name="churn",
        train_path=train,
        test_path=test,
        target="churn",
        id_col="customer_id",
        metrics=["accuracy", "roc_auc"],
        export_dir=tmp_path / "workspace" / "data",
        seed=7,
    )
    return record


@pytest.fixture
def git_repo(tmp_path):
    repo = tmp_path / "agent-repo"
    repo.mkdir()

    def git(*args):
        subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)

    git("init", "-q")
    git("config", "user.email", "test@example.invalid")
    git("config", "user.name", "Test")
    (repo / "train.py").write_text("print('hi')\n")
    (repo / "uv.lock").write_text("lock\n")
    git("add", "-A")
    git("commit", "-qm", "init")
    return repo


def honest_predictions(export_dir) -> pd.DataFrame:
    """Fit a plain logistic regression on the exported files, as an honest agent would."""
    from pathlib import Path

    from sklearn.linear_model import LogisticRegression

    out = Path(export_dir)
    train = pd.read_csv(out / "train.csv")
    test = pd.read_csv(out / "test_features.csv")
    X = pd.get_dummies(train[["x1", "x2", "colour"]], dtype=float)
    Xt = pd.get_dummies(test[["x1", "x2", "colour"]], dtype=float).reindex(columns=X.columns, fill_value=0)
    model = LogisticRegression().fit(X, train["churn"])
    return pd.DataFrame(
        {
            "customer_id": test["customer_id"],
            "prediction": model.predict(Xt),
            "proba": model.predict_proba(Xt)[:, 1],
        }
    )
