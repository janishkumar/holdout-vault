"""End-to-end sealed mode against a real Docker daemon. Skipped when Docker is not running."""

import asyncio
import os

import joblib
import pandas as pd
import pytest
from mcp import Client
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from conftest import make_classification
from holdout_vault import sealed
from holdout_vault.cli import main
from holdout_vault.mcp.server import build_server
from holdout_vault.runner.launcher import docker_available

pytestmark = [pytest.mark.docker, pytest.mark.skipif(not docker_available(), reason="Docker is not running")]


@pytest.fixture
def sealed_vault(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setenv("HOLDOUT_VAULT_HOME", str(home))
    monkeypatch.setenv("HOLDOUT_VAULT_COMPOSE_PROJECT", "hv-test")
    monkeypatch.setenv("HOLDOUT_VAULT_PORT", "7349")
    sealed.up(home)
    yield home
    sealed.down(home, purge=True)


def test_sealed_register_evaluate_and_isolation(sealed_vault, tmp_path):
    df = make_classification()
    df.iloc[:400].to_csv(tmp_path / "train.csv", index=False)
    df.iloc[400:].to_csv(tmp_path / "test.csv", index=False)
    ws = tmp_path / "agent"
    ws.mkdir()
    code = main(
        ["register", "--sealed", "--project", "demo", "--name", "churn", "--train", str(tmp_path / "train.csv"),
         "--test", str(tmp_path / "test.csv"), "--target", "churn", "--id-col", "customer_id",
         "--metrics", "accuracy,roc_auc", "--export-dir", str(ws / "data")]
    )  # fmt: skip
    assert code == 0
    assert (ws / "data" / "churn" / "test_features.csv").exists()
    assert "churn" not in pd.read_csv(ws / "data" / "churn" / "test_features.csv").columns
    assert not (sealed_vault / "projects").exists()  # nothing sealed on the host

    train = pd.read_csv(ws / "data" / "churn" / "train.csv")
    model = make_pipeline(
        make_column_transformer((OneHotEncoder(), ["colour"]), remainder="passthrough"), LogisticRegression()
    ).fit(train[["x1", "x2", "colour"]], train["churn"])
    joblib.dump(model, ws / "model.joblib")

    # a model that tries to read the vault store and phone home
    sneaky = ws / "sneaky"
    sneaky.mkdir()
    (sneaky / "predict.py").write_text(
        "import os, socket\n"
        "def predict(X):\n"
        "    # the runner shares the vault image, so /vault exists, but it must be an empty folder\n"
        "    assert not os.listdir('/vault'), 'vault store visible'\n"
        "    for root, _, files in os.walk('/'):\n"
        "        if root.startswith('/proc'):\n"
        "            continue\n"
        "        assert not {'labels.parquet', 'ledger.jsonl', 'api_token'} & set(files), 'vault store visible'\n"
        "    try:\n"
        "        socket.create_connection(('1.1.1.1', 80), timeout=2)\n"
        "        raise AssertionError('network reachable')\n"
        "    except OSError:\n"
        "        pass\n"
        "    return [0] * len(X)\n"
    )

    server = build_server("demo", repo=ws)

    async def go():
        async with Client(server) as c:
            good = (
                await c.call_tool("evaluate", {"dataset": "churn", "model_path": "model.joblib"})
            ).structured_content
            probe = (await c.call_tool("evaluate", {"dataset": "churn", "model_path": "sneaky"})).structured_content
            status = (await c.call_tool("vault_status", {})).structured_content
        return good, probe, status

    good, probe, status = asyncio.run(go())
    assert good["status"] == "scored", good["message"]
    assert good["mode"] == "sealed" and good["stamp"]["runner"] == "container"
    assert 0.6 < good["metrics"]["accuracy"] <= 1
    # the probe model ran to completion: no vault store, no network, so it only predicts zeros
    assert probe["status"] in {"scored", "invalid"}, probe["message"]
    assert "vault store visible" not in probe["message"] and "network reachable" not in probe["message"]
    assert status["mode"] == "sealed" and status["datasets"][0]["export_dir"].endswith("data/churn")
    assert main(["verify-ledger", "--project", "demo", "--sealed"]) == 0
    assert os.stat(sealed_vault / "sealed.json").st_mode & 0o777 == 0o600
