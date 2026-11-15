import numpy as np
import pandas as pd

from holdout_vault.core.canary import canary_agreement, make_canaries, new_ids
from holdout_vault.core.registry import register_dataset
from holdout_vault.core.vault import Vault


def test_new_int_ids_blend_and_never_collide():
    rng = np.random.default_rng(0)
    existing = pd.Series(np.arange(100, 300, 2))
    ids = new_ids(existing, 30, rng)
    assert len(set(ids)) == 30
    assert not set(ids) & set(existing)
    assert min(ids) < 300  # not all parked after the real range


def test_new_string_ids():
    rng = np.random.default_rng(0)
    existing = pd.Series([f"cust-{i:05d}" for i in range(50)])
    ids = new_ids(existing, 10, rng)
    assert len(set(ids)) == 10 and not set(ids) & set(existing)


def test_canary_features_follow_train_columns_with_missing_column():
    rng = np.random.default_rng(1)
    train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [None, None, None], "y": [0, 1, 0]})
    feats, labels = make_canaries(train, ["a", "b"], "y", "classification", 5, rng)
    assert list(feats.columns) == ["a", "b"]
    assert feats["b"].isna().all()
    assert set(labels) <= {0, 1}


def test_regression_canary_agreement():
    rng = np.random.default_rng(2)
    y = pd.Series(rng.normal(size=50))
    assert canary_agreement("regression", y, y * 2 + 1, 0).code == "HV-CAN-002"
    assert canary_agreement("regression", y, pd.Series(rng.normal(size=50)), 0) is None
    assert canary_agreement("regression", y, pd.Series(np.zeros(50)), 0) is None
    assert canary_agreement("classification", pd.Series([], dtype=str), pd.Series([], dtype=str), 2) is None


def test_regression_dataset_end_to_end(home, tmp_path):
    rng = np.random.default_rng(3)
    n = 500
    x = rng.normal(size=n)
    df = pd.DataFrame({"x": x, "price": 3 * x + rng.normal(scale=0.5, size=n)})
    df.iloc[:350].to_csv(tmp_path / "tr.csv", index=False)
    df.iloc[350:].to_csv(tmp_path / "te.csv", index=False)
    register_dataset(
        home=home,
        project="reg",
        name="price",
        train_path=tmp_path / "tr.csv",
        test_path=tmp_path / "te.csv",
        target="price",
        metrics=["rmse", "r2"],
        export_dir=tmp_path / "ws",
    )
    test = pd.read_csv(tmp_path / "ws" / "price" / "test_features.csv")
    preds = pd.DataFrame({"row_id": test["row_id"], "prediction": 3 * test["x"]})
    r = Vault(home, "reg").score("price", preds)
    assert r.metrics["r2"] > 0.9 and r.n_scored == 150
