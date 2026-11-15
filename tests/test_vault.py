import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from holdout_vault.core.errors import BudgetExhausted, Refused, TamperDetected, ValidationError
from holdout_vault.core.store import Store
from holdout_vault.core.vault import Approval, Vault


def honest_predictions(registered, tmp_path) -> pd.DataFrame:
    out = tmp_path / "workspace" / "data" / "churn"
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


def test_score_excludes_canaries_and_matches_sklearn(registered, tmp_path):
    preds = honest_predictions(registered, tmp_path)
    vault = Vault(registered.home, "demo")
    result = vault.score("churn", preds)
    labels = Store.open(registered.home, "demo").read_labels("churn")
    real = labels[~labels["is_canary"]]
    merged = real.merge(preds.assign(id=preds["customer_id"].astype(str)), on="id")
    expected = accuracy_score(merged["label"].astype(int), merged["prediction"])
    assert result.metrics["accuracy"] == pytest.approx(expected)
    assert result.n_scored == 200
    assert result.findings == []


def test_score_spends_and_logs(registered, tmp_path):
    vault = Vault(registered.home, "demo")
    before = vault.budget_state()
    result = vault.score("churn", honest_predictions(registered, tmp_path), stamp={"seed": 3})
    after = vault.budget_state()
    assert after.spent == before.spent + 1
    assert result.remaining == after.remaining == 19
    entry = vault.store.ledger.head()
    assert entry.kind == "score" and entry.seq == result.entry_seq and entry.hash == result.entry_hash
    assert entry.payload["metrics"] == result.metrics
    assert entry.payload["stamp"]["seed"] == 3
    assert entry.payload["stamp"]["scorer_sha256"] == registered.scorer_sha256
    assert vault.verify_ledger().ok


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda p: p.iloc[5:], "missing"),
        (lambda p: pd.concat([p, p.iloc[:1]]), "duplicate"),
        (lambda p: pd.concat([p, p.iloc[:1].assign(customer_id=-1)]), "unknown"),
        (lambda p: p.drop(columns=["prediction"]), "prediction"),
        (lambda p: p.rename(columns={"customer_id": "cid"}), "id column"),
        (lambda p: p.assign(proba=p["proba"] + 2), r"\[0, 1\]"),
    ],
)
def test_bad_submissions_do_not_spend(registered, tmp_path, mutate, match):
    vault = Vault(registered.home, "demo")
    preds = mutate(honest_predictions(registered, tmp_path))
    with pytest.raises(ValidationError, match=match):
        vault.score("churn", preds)
    assert vault.budget_state().spent == 0


def test_id_column_named_id_is_accepted(registered, tmp_path):
    preds = honest_predictions(registered, tmp_path).rename(columns={"customer_id": "id"})
    assert Vault(registered.home, "demo").score("churn", preds).n_scored == 200


def test_unknown_dataset(registered):
    with pytest.raises(ValidationError, match="not registered"):
        Vault(registered.home, "demo").score("nope", pd.DataFrame())


def test_budget_exhausted_then_human_approval(registered, tmp_path):
    vault = Vault(registered.home, "demo")
    vault.set_budget(2, by="test")
    preds = honest_predictions(registered, tmp_path)
    vault.score("churn", preds)
    vault.score("churn", preds)
    with pytest.raises(BudgetExhausted):
        vault.score("churn", preds)
    assert vault.store.ledger.head().kind == "refused"
    result = vault.score("churn", preds, approval=Approval(by="human", via="test"))
    kinds = [e.kind for e in vault.store.ledger.entries()]
    assert kinds[-2:] == ["grant", "score"]
    assert result.remaining == 0
    assert vault.budget_state().granted == 1


def test_grant_adds_reveals(registered):
    vault = Vault(registered.home, "demo")
    vault.grant(5, by="human", via="cli")
    assert vault.budget_state().remaining == 25


def test_set_budget_rejects_negative(registered):
    with pytest.raises(ValueError):
        Vault(registered.home, "demo").set_budget(-1, by="x")


def test_budget_is_derived_from_ledger_not_config(registered, tmp_path):
    vault = Vault(registered.home, "demo")
    vault.score("churn", honest_predictions(registered, tmp_path))
    cfg = vault.store.read_config()
    cfg["budget"] = 999
    vault.store.write_config(cfg)
    assert vault.budget_state().budget == 20


def test_label_cheat_caught_by_canaries(registered, tmp_path):
    vault = Vault(registered.home, "demo")
    labels = vault.store.read_labels("churn")
    cheat = pd.DataFrame({"customer_id": labels["id"].astype(int), "prediction": labels["label"].astype(int)})
    cheat["proba"] = cheat["prediction"].astype(float)
    with pytest.raises(Refused) as exc:
        vault.score("churn", cheat)
    assert exc.value.findings[0].code == "HV-CAN-002"
    assert vault.budget_state().spent == 0
    assert vault.store.ledger.head().kind == "refused"


def test_tampered_custom_scorer_detected(home, split_files, tmp_path):
    from holdout_vault.core.registry import register_dataset

    train, test = split_files
    scorer = tmp_path / "scorer.py"
    scorer.write_text("def score(y_true, predictions):\n    return {'n': float(len(y_true))}\n")
    register_dataset(
        home=home,
        project="demo",
        name="c2",
        train_path=train,
        test_path=test,
        target="churn",
        id_col="customer_id",
        metrics=[],
        scorer_path=scorer,
        export_dir=tmp_path / "ws",
    )
    vault = Vault(home, "demo")
    sealed = vault.store.custom_scorer_path("c2")
    test_ids = pd.read_csv(tmp_path / "ws" / "c2" / "test_features.csv")["customer_id"]
    preds = pd.DataFrame({"customer_id": test_ids, "prediction": 0})
    assert vault.score("c2", preds).metrics == {"n": 200.0}
    sealed.write_text("def score(y_true, predictions):\n    return {'n': 1.0}\n")
    with pytest.raises(TamperDetected) as exc:
        vault.score("c2", preds)
    assert exc.value.findings[0].code == "HV-SCR-001"
    assert vault.store.ledger.head().kind == "tamper"


def test_tampered_labels_detected(registered, tmp_path):
    vault = Vault(registered.home, "demo")
    path = vault.store.labels_path("churn")
    df = pd.read_parquet(path)
    df.loc[0, "label"] = "1" if df.loc[0, "label"] == "0" else "0"
    df.to_parquet(path, index=False)
    with pytest.raises(TamperDetected, match="HV-LAB-001"):
        vault.score("churn", honest_predictions(registered, tmp_path))


def test_watch_mode_detects_test_file_change(registered, tmp_path, split_files):
    _, test = split_files
    test.write_text(test.read_text() + "\n")
    with pytest.raises(TamperDetected, match="HV-TST-001"):
        Vault(registered.home, "demo", mode="watch").score("churn", honest_predictions(registered, tmp_path))


def test_sealed_mode_does_not_watch_host_test_file(registered, tmp_path, split_files):
    _, test = split_files
    test.unlink()
    result = Vault(registered.home, "demo", mode="sealed").score(
        "churn", honest_predictions(registered, tmp_path)
    )
    assert result.mode == "sealed"


def test_status(registered, tmp_path):
    vault = Vault(registered.home, "demo")
    vault.score("churn", honest_predictions(registered, tmp_path))
    s = vault.status()
    assert s["mode"] == "watch"
    assert s["remaining"] == 19
    assert s["ledger_ok"] is True
    assert s["datasets"][0]["name"] == "churn"
    assert "test_path" not in s["datasets"][0]
