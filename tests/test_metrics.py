import pandas as pd
import pytest

from holdout_vault.core.metrics import (
    BUILTIN_METRICS,
    MetricError,
    compute_metrics,
    load_custom_scorer,
    scorer_code_hash,
)


def test_builtin_names():
    assert set(BUILTIN_METRICS) == {
        "accuracy",
        "balanced_accuracy",
        "f1",
        "roc_auc",
        "log_loss",
        "rmse",
        "mae",
        "r2",
    }


def test_classification_metrics():
    y = pd.Series([0, 1, 1, 0])
    preds = pd.DataFrame({"prediction": [0, 1, 0, 0], "proba": [0.1, 0.9, 0.4, 0.2]})
    m = compute_metrics(["accuracy", "f1", "roc_auc", "log_loss"], y, preds)
    assert m["accuracy"] == pytest.approx(0.75)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["roc_auc"] == pytest.approx(1.0)
    assert 0 < m["log_loss"] < 1


def test_multiclass_f1_is_macro_and_auc_uses_proba_columns():
    y = pd.Series(["a", "b", "c", "a"])
    preds = pd.DataFrame(
        {
            "prediction": ["a", "b", "c", "b"],
            "proba_a": [0.8, 0.1, 0.1, 0.4],
            "proba_b": [0.1, 0.8, 0.1, 0.5],
            "proba_c": [0.1, 0.1, 0.8, 0.1],
        }
    )
    m = compute_metrics(["f1", "roc_auc"], y, preds)
    assert 0 < m["f1"] < 1
    assert 0.5 < m["roc_auc"] <= 1


def test_regression_metrics():
    y = pd.Series([1.0, 2.0, 3.0])
    preds = pd.DataFrame({"prediction": [1.0, 2.0, 4.0]})
    m = compute_metrics(["rmse", "mae", "r2"], y, preds)
    assert m["mae"] == pytest.approx(1 / 3)
    assert m["rmse"] == pytest.approx((1 / 3) ** 0.5)


def test_auc_without_proba_is_an_error():
    with pytest.raises(MetricError, match="proba"):
        compute_metrics(["roc_auc"], pd.Series([0, 1]), pd.DataFrame({"prediction": [0, 1]}))


def test_proba_out_of_range_is_an_error():
    with pytest.raises(MetricError, match=r"\[0, 1\]"):
        compute_metrics(
            ["log_loss"], pd.Series([0, 1]), pd.DataFrame({"prediction": [0, 1], "proba": [0.2, 1.4]})
        )


def test_unknown_metric():
    with pytest.raises(MetricError, match="unknown"):
        compute_metrics(["nope"], pd.Series([0]), pd.DataFrame({"prediction": [0]}))


def test_scorer_code_hash_stable():
    assert scorer_code_hash() == scorer_code_hash()
    assert len(scorer_code_hash()) == 64


def test_custom_scorer(tmp_path):
    p = tmp_path / "scorer.py"
    p.write_text(
        "def score(y_true, predictions):\n"
        "    return {'hits': float((y_true.values == predictions['prediction'].values).sum())}\n"
    )
    fn = load_custom_scorer(p)
    assert fn(pd.Series([1, 0]), pd.DataFrame({"prediction": [1, 1]})) == {"hits": 1.0}


def test_custom_scorer_must_define_score(tmp_path):
    p = tmp_path / "scorer.py"
    p.write_text("x = 1\n")
    with pytest.raises(MetricError, match="score"):
        load_custom_scorer(p)
