import pandas as pd
import pytest

from holdout_vault.core.registry import RegistrationError, canary_count, register_dataset
from holdout_vault.core.store import Store


def test_canary_count_bounds():
    assert canary_count(100) == 100
    assert canary_count(20_000) == 200
    assert canary_count(100_000) == 500


def test_register_exports_features_without_labels(registered, tmp_path):
    out = tmp_path / "workspace" / "data" / "churn"
    train = pd.read_csv(out / "train.csv")
    test = pd.read_csv(out / "test_features.csv")
    assert "churn" in train.columns
    assert "churn" not in test.columns
    assert len(train) == 400
    assert len(test) == 200 + registered.n_canaries
    assert test["customer_id"].is_unique


def test_canaries_look_like_real_rows(registered, tmp_path):
    test = pd.read_csv(tmp_path / "workspace" / "data" / "churn" / "test_features.csv")
    assert set(test.columns) == {"customer_id", "x1", "x2", "colour"}
    assert set(test["colour"]) <= {"red", "green", "blue"}
    # canaries are not appended at the end
    store = Store.open(registered.home, "demo")
    canary_ids = set(store.read_canaries("churn")["id"])
    last_ids = set(test["customer_id"].tail(registered.n_canaries))
    assert canary_ids != last_ids


def test_labels_live_only_in_vault(registered, tmp_path):
    store = Store.open(registered.home, "demo")
    labels = store.read_labels("churn")
    assert len(labels) == 200 + registered.n_canaries
    workspace = tmp_path / "workspace"
    assert not any(p.suffix == ".parquet" for p in workspace.rglob("*"))
    assert not str(store.root).startswith(str(workspace))


def test_register_logs_ledger_entry(registered):
    store = Store.open(registered.home, "demo")
    entries = store.ledger.entries()
    assert entries[0].kind == "register"
    p = entries[0].payload
    assert p["dataset"] == "churn"
    assert p["train_sha256"] == registered.train_sha256
    assert p["scorer_sha256"] == registered.scorer_sha256
    # the ledger may be shown to the agent, so it must never list which rows are canaries
    assert p["n_canaries"] == registered.n_canaries
    assert "canary_ids" not in p


def test_register_twice_refused(registered, split_files, tmp_path):
    train, test = split_files
    with pytest.raises(RegistrationError, match="already registered"):
        register_dataset(
            home=registered.home,
            project="demo",
            name="churn",
            train_path=train,
            test_path=test,
            target="churn",
            id_col="customer_id",
            metrics=["accuracy"],
            export_dir=tmp_path / "x",
        )


def test_missing_target_column(home, split_files, tmp_path):
    train, test = split_files
    with pytest.raises(RegistrationError, match="target"):
        register_dataset(
            home=home,
            project="demo",
            name="d",
            train_path=train,
            test_path=test,
            target="nope",
            metrics=["accuracy"],
            export_dir=tmp_path / "x",
        )


def test_unknown_metric_refused(home, split_files, tmp_path):
    train, test = split_files
    with pytest.raises(RegistrationError, match="metric"):
        register_dataset(
            home=home,
            project="demo",
            name="d",
            train_path=train,
            test_path=test,
            target="churn",
            metrics=["bleu"],
            export_dir=tmp_path / "x",
        )


def test_duplicate_test_ids_refused(home, tmp_path, split_files):
    train, test = split_files
    df = pd.read_csv(test)
    df.loc[1, "customer_id"] = df.loc[0, "customer_id"]
    df.to_csv(test, index=False)
    with pytest.raises(RegistrationError, match="unique"):
        register_dataset(
            home=home,
            project="demo",
            name="d",
            train_path=train,
            test_path=test,
            target="churn",
            id_col="customer_id",
            metrics=["accuracy"],
            export_dir=tmp_path / "x",
        )


def test_export_inside_store_refused(home, split_files):
    train, test = split_files
    with pytest.raises(RegistrationError, match="inside the vault"):
        register_dataset(
            home=home,
            project="demo",
            name="d",
            train_path=train,
            test_path=test,
            target="churn",
            metrics=["accuracy"],
            export_dir=home / "projects",
        )


def test_no_id_column_gets_row_id(home, split_files, tmp_path):
    train, test = split_files
    rec = register_dataset(
        home=home,
        project="demo",
        name="noid",
        train_path=train,
        test_path=test,
        target="churn",
        metrics=["accuracy"],
        export_dir=tmp_path / "ws",
    )
    exported = pd.read_csv(tmp_path / "ws" / "noid" / "test_features.csv")
    assert rec.id_col == "row_id"
    assert exported["row_id"].is_unique
