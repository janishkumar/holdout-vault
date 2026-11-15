import joblib
import pandas as pd
import pytest
from sklearn.compose import make_column_transformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder

from holdout_vault.runner.harness import HarnessError, predict_file
from holdout_vault.runner.launcher import RunnerConfig, RunnerError, docker_command, run_model


def fit_pipeline(export_dir):
    train = pd.read_csv(export_dir / "train.csv")
    pre = make_column_transformer((OneHotEncoder(), ["colour"]), remainder="passthrough")
    model = make_pipeline(pre, LogisticRegression())
    model.fit(train[["x1", "x2", "colour"]], train["churn"])
    return model


@pytest.fixture
def export_dir(registered):
    from pathlib import Path

    return Path(registered.export_dir)


def test_harness_estimator(export_dir, tmp_path):
    joblib.dump(fit_pipeline(export_dir), tmp_path / "m.joblib")
    feats = pd.read_csv(export_dir / "test_features.csv")
    out = predict_file(tmp_path / "m.joblib", feats, "customer_id")
    assert list(out.columns) == ["customer_id", "prediction", "proba"]
    assert len(out) == len(feats)


def test_harness_script_dir(export_dir, tmp_path):
    d = tmp_path / "model"
    d.mkdir()
    joblib.dump(fit_pipeline(export_dir), d / "m.joblib")
    (d / "predict.py").write_text(
        "import joblib, pathlib\n"
        "M = joblib.load(pathlib.Path(__file__).parent / 'm.joblib')\n"
        "def predict(X):\n"
        "    return M.predict(X)\n"
    )
    feats = pd.read_csv(export_dir / "test_features.csv")
    out = predict_file(d, feats, "customer_id")
    assert list(out.columns) == ["customer_id", "prediction"]


def test_harness_errors(tmp_path):
    feats = pd.DataFrame({"id": [1], "x": [0.0]})
    with pytest.raises(HarnessError, match="id column"):
        predict_file(tmp_path, feats, "nope")
    with pytest.raises(HarnessError, match=r"predict\.py"):
        predict_file(tmp_path, feats, "id")
    joblib.dump({"not": "a model"}, tmp_path / "x.joblib")
    with pytest.raises(HarnessError, match="no predict"):
        predict_file(tmp_path / "x.joblib", feats, "id")


def test_launcher_subprocess(export_dir, tmp_path):
    joblib.dump(fit_pipeline(export_dir), tmp_path / "m.joblib")
    feats = pd.read_csv(export_dir / "test_features.csv")
    seen = []
    out = run_model(tmp_path / "m.joblib", feats, "customer_id", RunnerConfig(), progress=seen.append)
    assert len(out) == len(feats) and "proba" in out.columns
    assert any("not isolated" in s for s in seen)


def test_launcher_reports_model_failure(tmp_path):
    d = tmp_path / "bad"
    d.mkdir()
    (d / "predict.py").write_text("def predict(X):\n    raise RuntimeError('boom')\n")
    with pytest.raises(RunnerError, match="boom"):
        run_model(d, pd.DataFrame({"id": [1], "x": [1.0]}), "id", RunnerConfig())


def test_launcher_missing_model(tmp_path):
    with pytest.raises(RunnerError, match="not found"):
        run_model(tmp_path / "nope.joblib", pd.DataFrame({"id": [1]}), "id", RunnerConfig())


def test_docker_command_is_locked_down(tmp_path):
    (tmp_path / "m.joblib").write_bytes(b"")
    cmd = docker_command(tmp_path / "m.joblib", tmp_path, "id", RunnerConfig(use_docker=True))
    joined = " ".join(cmd)
    for flag in [
        "--network none",
        "--read-only",
        "--cap-drop ALL",
        "no-new-privileges",
        "--user 65534:65534",
    ]:
        assert flag in joined
    assert ":/model:ro" in joined
    assert "vault" not in joined.replace("holdout-vault:local", "").replace("holdout_vault", "")
