import numpy as np
import pandas as pd
import pytest

from conftest import make_classification
from holdout_vault.checks.rows import row_findings
from holdout_vault.core.errors import RegistrationError
from holdout_vault.core.registry import register_dataset

FEATS = ["x1", "x2", "colour"]


def split(seed=0, n=600):
    df = make_classification(n=n, seed=seed)
    return df.iloc[:400].reset_index(drop=True), df.iloc[400:].reset_index(drop=True)


def codes(found):
    return sorted(f.code for f in found)


def test_clean_split_has_no_findings():
    for seed in range(5):
        train, test = split(seed)
        assert row_findings(train, test, FEATS, test_ids=test["customer_id"], seed=seed) == []


def test_exact_duplicates_found():
    train, test = split()
    test.loc[:19, FEATS] = train.loc[:19, FEATS].to_numpy()
    found = row_findings(train, test, FEATS, test_ids=test["customer_id"])
    assert codes(found) == ["HV-ROW-001"]
    assert found[0].evidence["count"] == 20
    assert found[0].location["row_ids"][0] == int(test["customer_id"][0])


def test_categorical_only_repeats_are_expected():
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {"a": rng.choice(["x", "y"], 600), "b": rng.choice(["p", "q"], 600), "y": rng.integers(0, 2, 600)}
    )
    train, test = df.iloc[:400], df.iloc[400:]
    assert row_findings(train, test, ["a", "b"], test_ids=pd.Series(range(200))) == []


def test_near_duplicates_found():
    train, test = split()
    rng = np.random.default_rng(1)
    copy = train.loc[:29, FEATS].copy()
    copy["x1"] += rng.normal(scale=1e-3, size=30)
    copy["x2"] += rng.normal(scale=1e-3, size=30)
    test.loc[:29, FEATS] = copy.to_numpy()
    found = row_findings(train, test, FEATS, test_ids=test["customer_id"])
    assert codes(found) == ["HV-ROW-002"]
    assert found[0].evidence["count"] >= 25


def test_time_overlap():
    train, test = split()
    train["day"] = pd.date_range("2024-01-01", periods=400, freq="D").astype(str)
    test["day"] = pd.date_range("2025-03-01", periods=200, freq="D").astype(str)
    assert row_findings(train, test, FEATS, test_ids=test["customer_id"], time_col="day") == []
    train.loc[:9, "day"] = "2025-06-01"
    found = row_findings(train, test, [*FEATS, "day"], test_ids=test["customer_id"], time_col="day")
    assert codes(found) == ["HV-ROW-003"]
    assert found[0].evidence["count"] == 10


def test_group_overlap():
    train, test = split()
    train["store"] = np.arange(400) % 40
    test["store"] = 100 + np.arange(200) % 20
    assert row_findings(train, test, FEATS, test_ids=test["customer_id"], group_col="store") == []
    test.loc[:4, "store"] = 3
    found = row_findings(train, test, FEATS, test_ids=test["customer_id"], group_col="store")
    assert codes(found) == ["HV-ROW-004"]
    assert found[0].evidence["groups_shared"] == 1


def test_registration_stores_row_findings_and_strict_refuses(home, tmp_path):
    train, test = split()
    test.loc[:19, FEATS] = train.loc[:19, FEATS].to_numpy()
    train.to_csv(tmp_path / "tr.csv", index=False)
    test.to_csv(tmp_path / "te.csv", index=False)
    kw = dict(
        home=home,
        project="demo",
        train_path=tmp_path / "tr.csv",
        test_path=tmp_path / "te.csv",
        target="churn",
        id_col="customer_id",
        metrics=["accuracy"],
        export_dir=tmp_path / "ws",
    )
    rec = register_dataset(name="d", **kw)
    assert [f["code"] for f in rec.row_findings] == ["HV-ROW-001"]
    with pytest.raises(RegistrationError, match="HV-ROW-001"):
        register_dataset(name="e", strict=True, **kw)
