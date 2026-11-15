import socket
import threading
import time

import pandas as pd
import pytest
import uvicorn

from conftest import honest_predictions
from holdout_vault.api import create_app
from holdout_vault.core.errors import BudgetExhausted, Refused, ValidationError, VaultError
from holdout_vault.core.vault import Approval, Vault
from holdout_vault.remote import RemoteVault

TOKEN = "test-token"


@pytest.fixture
def live(registered):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(
        uvicorn.Config(create_app(registered.home, token=TOKEN), host="127.0.0.1", port=port, log_level="error")
    )
    t = threading.Thread(target=server.run, daemon=True)
    t.start()
    for _ in range(100):
        if server.started:
            break
        time.sleep(0.05)
    yield f"http://127.0.0.1:{port}", registered
    server.should_exit = True
    t.join(5)


def test_remote_score_and_status(live):
    url, rec = live
    rv = RemoteVault(url, TOKEN, "demo")
    r = rv.score("churn", honest_predictions(rec.export_dir), stamp={"seed": 1})
    assert r.mode == "sealed" and r.remaining == 19
    assert rv.budget_state().spent == 1
    assert rv.status()["mode"] == "sealed"
    assert rv.dataset_info("churn")["id_col"] == "customer_id"
    feats = rv.test_features("churn")
    assert "churn" not in feats.columns and len(feats) == 200 + rec.n_canaries


def test_remote_errors_map_to_exceptions(live):
    url, rec = live
    rv = RemoteVault(url, TOKEN, "demo")
    preds = honest_predictions(rec.export_dir)
    with pytest.raises(ValidationError, match="missing"):
        rv.score("churn", preds.iloc[3:])
    Vault(rec.home, "demo").set_budget(0, by="test")
    with pytest.raises(BudgetExhausted):
        rv.score("churn", preds)
    assert rv.score("churn", preds, approval=Approval(by="human", via="test")).remaining == 0
    labels = Vault(rec.home, "demo").store.read_labels("churn")
    y = labels["label"].astype(int)
    cheat = pd.DataFrame({"customer_id": labels["id"].astype(int), "prediction": y, "proba": y.astype(float)})
    Vault(rec.home, "demo").grant(1, by="t", via="t")
    with pytest.raises(Refused):
        rv.score("churn", cheat)
    with pytest.raises(ValidationError, match="not registered"):
        rv.dataset_info("nope")


def test_bad_token_rejected(live):
    url, _ = live
    with pytest.raises(VaultError, match="token"):
        RemoteVault(url, "wrong", "demo").status()


def test_unreachable_vault():
    with pytest.raises(VaultError, match="running"):
        RemoteVault("http://127.0.0.1:9", TOKEN, "demo", timeout=1).status()


def test_api_has_no_admin_routes():
    app = create_app("/nonexistent-home", token=TOKEN)
    paths = {r.path for r in app.routes}
    assert not any(w in p for p in paths for w in ("grant", "register", "budget/set", "unseal"))


def test_token_file_created_with_private_permissions(tmp_path, monkeypatch):
    from holdout_vault.api import load_or_create_token

    monkeypatch.delenv("HOLDOUT_VAULT_TOKEN", raising=False)
    tok = load_or_create_token(tmp_path)
    assert (tmp_path / "api_token").stat().st_mode & 0o777 == 0o600
    assert load_or_create_token(tmp_path) == tok
