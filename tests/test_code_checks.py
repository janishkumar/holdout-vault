import json
import textwrap

import pytest

from holdout_vault.checks.code import Context, code_findings

CTX = Context(targets={"churn"}, labelled_test=["/data/raw/test.csv"], vault_home="~/.config/holdout-vault")

HEAD = """
import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
"""


def run(tmp_path, body, files=None):
    (tmp_path / "main.py").write_text(HEAD + textwrap.dedent(body))
    for name, text in (files or {}).items():
        (tmp_path / name).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / name).write_text(textwrap.dedent(text))
    return code_findings(tmp_path, CTX)


def codes(report):
    return sorted({f.code for f in report.findings})


CASES = {
    "scaler_before_split": (
        """
        df = pd.read_csv("data/churn/train.csv")
        X, y = df.drop(columns=["churn"]), df["churn"]
        X = StandardScaler().fit_transform(X)
        X_tr, X_va, y_tr, y_va = train_test_split(X, y)
        """,
        ["HV-COD-001"],
    ),
    "scaler_after_split_is_fine": (
        """
        df = pd.read_csv("data/churn/train.csv")
        X, y = df.drop(columns=["churn"]), df["churn"]
        X_tr, X_va, y_tr, y_va = train_test_split(X, y)
        sc = StandardScaler().fit(X_tr)
        m = LogisticRegression().fit(sc.transform(X_tr), y_tr)
        print(m.score(sc.transform(X_va), y_va))
        """,
        [],
    ),
    "final_fit_on_all_train_is_fine": (
        """
        df = pd.read_csv("data/churn/train.csv")
        test = pd.read_csv("data/churn/test_features.csv")
        X, y = df.drop("churn", axis=1), df["churn"]
        sc = StandardScaler()
        Xs = sc.fit_transform(X)
        m = LogisticRegression().fit(Xs, y)
        pd.DataFrame({"prediction": m.predict(sc.transform(test))}).to_csv("predictions.csv")
        """,
        [],
    ),
    "fitted_scaler_applied_to_split": (
        """
        df = pd.read_csv("data/churn/train.csv")
        X, y = df.drop(columns=["churn"]), df["churn"]
        sc = StandardScaler()
        sc.fit(X)
        X_tr, X_va, y_tr, y_va = train_test_split(X, y)
        m = LogisticRegression().fit(sc.transform(X_tr), y_tr)
        """,
        ["HV-COD-001"],
    ),
    "fit_in_helper_before_split": (
        """
        def prep(frame):
            sel = SelectKBest(k=5)
            return sel.fit_transform(frame.drop(columns=["churn"]), frame["churn"])

        df = pd.read_csv("data/churn/train.csv")
        X = prep(df)
        scores = cross_val_score(LogisticRegression(), X, df["churn"], cv=5)
        """,
        ["HV-COD-001"],
    ),
    "pipeline_in_cv_is_fine": (
        """
        df = pd.read_csv("data/churn/train.csv")
        X, y = df.drop(columns=["churn"]), df["churn"]
        print(cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), X, y, cv=5))
        """,
        [],
    ),
    "kfold_loop_scaler_outside": (
        """
        df = pd.read_csv("data/churn/train.csv")
        X = StandardScaler().fit_transform(df.drop(columns=["churn"]))
        y = df["churn"].values
        for tr, va in KFold(5).split(X):
            LogisticRegression().fit(X[tr], y[tr]).score(X[va], y[va])
        """,
        ["HV-COD-001"],
    ),
    "kfold_fit_on_validation_fold": (
        """
        df = pd.read_csv("data/churn/train.csv")
        X, y = df.drop(columns=["churn"]), df["churn"]
        for i, (tr, va) in enumerate(KFold(5).split(X)):
            sc = StandardScaler().fit(X.iloc[va])
        """,
        ["HV-COD-002"],
    ),
    "fit_on_test_features": (
        """
        train = pd.read_csv("data/churn/train.csv")
        test = pd.read_csv("data/churn/test_features.csv")
        sc = StandardScaler().fit(pd.concat([train.drop(columns=["churn"]), test]))
        """,
        ["HV-COD-002"],
    ),
    "refit_on_train_plus_valid_is_fine": (
        """
        df = pd.read_csv("data/churn/train.csv")
        X, y = df.drop(columns=["churn"]), df["churn"]
        X_tr, X_va, y_tr, y_va = train_test_split(X, y)
        LogisticRegression().fit(pd.concat([X_tr, X_va]), pd.concat([y_tr, y_va]))
        """,
        [],
    ),
    "target_left_in_features": (
        """
        df = pd.read_csv("data/churn/train.csv")
        y = df["churn"]
        X_tr, X_va, y_tr, y_va = train_test_split(df, y)
        LogisticRegression().fit(X_tr, y_tr)
        """,
        ["HV-COD-003"],
    ),
    "target_derived_feature": (
        """
        TARGET = "churn"
        df = pd.read_csv("data/churn/train.csv")
        y = df.pop(TARGET)
        df["leak"] = y * 2
        LogisticRegression().fit(df, y)
        """,
        ["HV-COD-003"],
    ),
    "pop_target_is_fine": (
        """
        df = pd.read_csv("data/churn/train.csv")
        y = df.pop("churn")
        LogisticRegression().fit(df, y)
        """,
        [],
    ),
    "reads_labelled_test_joined_path": (
        """
        raw = os.path.join("raw", "test.csv")
        truth = pd.read_csv(raw)
        """,
        ["HV-COD-004"],
    ),
    "reads_vault_labels_fstring": (
        """
        home = Path.home()
        labels = pd.read_parquet(f"{home}/.config/holdout-vault/projects/churn/datasets/churn/labels.parquet")
        """,
        ["HV-COD-004"],
    ),
    "smote_before_split": (
        """
        df = pd.read_csv("data/churn/train.csv")
        X, y = df.drop(columns=["churn"]), df["churn"]
        X_res, y_res = SMOTE().fit_resample(X, y)
        X_tr, X_va, y_tr, y_va = train_test_split(X_res, y_res)
        """,
        ["HV-COD-005"],
    ),
    "smote_after_split_is_fine": (
        """
        df = pd.read_csv("data/churn/train.csv")
        X, y = df.drop(columns=["churn"]), df["churn"]
        X_tr, X_va, y_tr, y_va = train_test_split(X, y)
        X_res, y_res = SMOTE().fit_resample(X_tr, y_tr)
        """,
        [],
    ),
    "time_filter_split": (
        """
        df = pd.read_csv("data/churn/train.csv")
        df[["x1", "x2"]] = StandardScaler().fit_transform(df[["x1", "x2"]])
        train_df = df[df.signup_date < "2024-01-01"]
        valid_df = df[df.signup_date >= "2024-01-01"]
        """,
        ["HV-COD-001"],
    ),
    "writes_ledger": (
        """
        with open(os.path.expanduser("~/.config/holdout-vault/projects/churn/ledger.jsonl"), "a") as fh:
            fh.write("{}")
        """,
        ["HV-COD-006"],
    ),
    "monkeypatch_metric": (
        """
        import sklearn.metrics
        sklearn.metrics.accuracy_score = lambda *a, **k: 1.0
        """,
        ["HV-COD-006"],
    ),
    "docker_exec_into_vault": (
        """
        import subprocess
        subprocess.run(["docker", "exec", "holdout-vault-vault-1", "cat", "/vault/projects/churn/ledger.jsonl"])
        """,
        ["HV-COD-006"],
    ),
    "derived_column_selected_by_name": (
        """
        df = pd.read_csv("data/churn/train.csv")
        df["lag_rate"] = df.groupby("plan")["churn"].transform("sum") / 10
        model = LogisticRegression().fit(df[["x1", "lag_rate"]], df["churn"])
        """,
        ["HV-COD-003"],
    ),
    "plain_engineered_column_is_fine": (
        """
        df = pd.read_csv("data/churn/train.csv")
        df["ratio"] = df["x1"] / (df["x2"] + 1)
        model = LogisticRegression().fit(df[["x1", "ratio"]], df["churn"])
        """,
        [],
    ),
    "columns_comprehension_keeps_target": (
        """
        df = pd.read_csv("data/churn/train.csv")
        cols = [c for c in df.columns if c != "customer_id"]
        LogisticRegression().fit(df[cols], df["churn"])
        """,
        ["HV-COD-003"],
    ),
    "columns_comprehension_excludes_target": (
        """
        TARGET = "churn"
        df = pd.read_csv("data/churn/train.csv")
        cols = [c for c in df.columns if c not in {TARGET, "customer_id"}]
        other = df.columns.drop(TARGET)
        LogisticRegression().fit(df[cols], df[TARGET])
        LogisticRegression().fit(df[other], df[TARGET])
        """,
        [],
    ),
    "test_path_under_script_folder": (
        """
        here = Path(__file__).parent
        frame = pd.read_csv(here.parent / "raw" / "test.csv")
        """,
        ["HV-COD-004"],
    ),
    "vault_read_inside_comprehension": (
        """
        import json
        store = Path.home() / ".config" / "holdout-vault"
        entries = [json.loads(x) for x in (store / "projects" / "p" / "ledger.jsonl").open()]
        """,
        ["HV-COD-004"],
    ),
    "writing_predictions_is_fine": (
        """
        test = pd.read_csv("data/churn/test_features.csv")
        test_size = 0.2
        pd.DataFrame({"customer_id": test["customer_id"], "prediction": 0}).to_csv("predictions.csv", index=False)
        """,
        [],
    ),
}


@pytest.mark.parametrize("name", list(CASES))
def test_case(tmp_path, name):
    body, expected = CASES[name]
    assert codes(run(tmp_path, body)) == expected


def test_cross_module_helper(tmp_path):
    report = run(
        tmp_path,
        """
        from features import build
        df = pd.read_csv("data/churn/train.csv")
        X = build(df)
        X_tr, X_va = train_test_split(X)
        """,
        files={
            "features.py": """
                from sklearn.preprocessing import StandardScaler
                def build(df):
                    return StandardScaler().fit_transform(df.drop(columns=["churn"]))
            """
        },
    )
    assert codes(report) == ["HV-COD-001"]
    f = report.findings[0]
    assert f.location == {"file": "features.py", "line": 4}
    assert "StandardScaler" in f.evidence["code"]


def test_notebook_locations(tmp_path):
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {},
        "cells": [
            {"cell_type": "markdown", "metadata": {}, "source": ["# notes"]},
            {"cell_type": "code", "metadata": {}, "source": ["%matplotlib inline\n", "import pandas as pd"]},
            {"cell_type": "code", "metadata": {}, "source": ["x = 1\n", "t = pd.read_csv('raw/test.csv')"]},
        ],
    }
    (tmp_path / "explore.ipynb").write_text(json.dumps(nb))
    report = code_findings(tmp_path, CTX)
    assert codes(report) == ["HV-COD-004"]
    assert report.findings[0].location == {"file": "explore.ipynb", "cell": 2, "line": 2}
    assert report.findings[0].severity == "block"


def test_syntax_errors_are_reported_not_raised(tmp_path):
    (tmp_path / "broken.py").write_text("def (:\n")
    report = code_findings(tmp_path, CTX)
    assert report.findings == [] and report.unparsed == ["broken.py: SyntaxError"]


def test_virtualenvs_are_skipped(tmp_path):
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "x.py").write_text("import pandas as pd\npd.read_csv('raw/test.csv')\n")
    assert code_findings(tmp_path, CTX).files == 0


def test_narrowing_column_filter_is_not_target(tmp_path):
    report = run(
        tmp_path,
        """
        df = pd.read_csv("data/churn/train.csv")
        num = [c for c in df.columns if c.startswith("x")]
        LogisticRegression().fit(df[num], df["churn"])
        """,
    )
    assert codes(report) == []


MORE = {
    "instance_state_across_methods": (
        """
        class Job:
            def __init__(self, path):
                self.path = path
                self.enc = SelectKBest(k=3)

            def load(self):
                self.data = pd.read_csv(self.path)

            def features(self):
                self.X = self.enc.fit_transform(self.data.drop(columns=["churn"]), self.data["churn"])

            def evaluate(self):
                self.load()
                self.features()
                return cross_val_score(LogisticRegression(), self.X, self.data["churn"])

        Job("data/churn/train.csv").evaluate()
        """,
        ["HV-COD-001"],
    ),
    "instance_clean_flow": (
        """
        class Job:
            def __init__(self):
                self.data = pd.read_csv("data/churn/train.csv")

            def evaluate(self):
                X, y = self.data.drop(columns=["churn"]), self.data["churn"]
                return cross_val_score(make_pipeline(StandardScaler(), LogisticRegression()), X, y)

        Job().evaluate()
        """,
        [],
    ),
    "config_dict_path": (
        """
        PATHS = {"fit": "data/churn/train.csv", "audit": "raw/test.csv"}

        def read(which):
            return pd.read_csv(PATHS[which])

        frame = read("audit")
        """,
        ["HV-COD-004"],
    ),
    "config_module_constant": (
        """
        import settings
        frame = pd.read_csv(getattr(settings, "EVAL_FILE"))
        """,
        ["HV-COD-004"],
    ),
    "dataclass_feature_list_with_target": (
        """
        from dataclasses import dataclass

        @dataclass
        class Cols:
            use: tuple = ("x1", "x2", "churn")

        df = pd.read_csv("data/churn/train.csv")
        LogisticRegression().fit(df[list(Cols().use)], df["churn"])
        """,
        ["HV-COD-003"],
    ),
    "partial_fit_on_validation": (
        """
        from functools import partial
        df = pd.read_csv("data/churn/train.csv")
        a, b = train_test_split(df)
        sc = StandardScaler()
        refit = partial(sc.fit)
        refit(b[["x1", "x2"]])
        """,
        ["HV-COD-002"],
    ),
    "numpy_all_columns_includes_target": (
        """
        df = pd.read_csv("data/churn/train.csv")[["x1", "x2", "churn"]]
        M = df.to_numpy()
        LogisticRegression().fit(M[:, :], M[:, 2])
        """,
        ["HV-COD-003"],
    ),
    "numpy_all_but_last_column_is_fine": (
        """
        df = pd.read_csv("data/churn/train.csv")[["x1", "x2", "churn"]]
        M = df.to_numpy()
        LogisticRegression().fit(M[:, :-1], M[:, -1])
        """,
        [],
    ),
    "starred_path_parts": (
        """
        bits = ("raw", "test.csv")
        frame = pd.read_csv(Path(__file__).parent.joinpath(*bits))
        """,
        ["HV-COD-004"],
    ),
    "docker_read_only_is_fine": (
        """
        import subprocess
        subprocess.run(["docker", "compose", "-p", "holdout-vault", "logs", "vault"])
        subprocess.run("docker inspect holdout-vault-vault-1", shell=True)
        """,
        [],
    ),
}


@pytest.mark.parametrize("name", list(MORE))
def test_more_cases(tmp_path, name):
    body, expected = MORE[name]
    files = {"settings.py": "EVAL_FILE = 'raw/test.csv'\n"} if name == "config_module_constant" else None
    assert codes(run(tmp_path, body, files)) == expected


def test_package_reexport_clean(tmp_path):
    report = run(
        tmp_path,
        """
        from lib import prep
        df = pd.read_csv("data/churn/train.csv")
        print(cross_val_score(LogisticRegression(), prep(df), df["churn"]))
        """,
        files={
            "lib/__init__.py": "from .cols import prep\n",
            "lib/cols.py": "KEEP = ['x1', 'x2']\ndef prep(df):\n    return df[KEEP]\n",
        },
    )
    assert codes(report) == []


def test_package_reexport_leak(tmp_path):
    report = run(
        tmp_path,
        """
        from lib import prep
        df = pd.read_csv("data/churn/train.csv")
        X_tr, X_va = train_test_split(prep(df))
        """,
        files={
            "lib/__init__.py": "from .cols import prep\n",
            "lib/cols.py": (
                "from sklearn.decomposition import PCA\n"
                "def prep(df):\n    return PCA(2).fit_transform(df[['x1', 'x2', 'x3']])\n"
            ),
        },
    )
    assert codes(report) == ["HV-COD-001"]
    assert report.findings[0].location["file"] == "lib/cols.py"


EXTRA = {
    "assign_lambda_from_target": (
        """
        df = pd.read_csv("data/churn/train.csv").assign(hint=lambda d: d["churn"].rolling(3).mean())
        LogisticRegression().fit(df[["x1", "hint"]], df["churn"])
        """,
        ["HV-COD-003"],
    ),
    "assign_lambda_clean": (
        """
        df = pd.read_csv("data/churn/train.csv").assign(r=lambda d: d["x1"] - d["x2"])
        LogisticRegression().fit(df[["x1", "r"]], df["churn"])
        """,
        [],
    ),
    "json_config_paths": (
        """
        import json
        cfg = json.loads('{"data": {"eval": "raw/test.csv"}}')
        frame = pd.read_csv(cfg["data"]["eval"])
        """,
        ["HV-COD-004"],
    ),
    "yaml_config_columns": (
        """
        import yaml
        cfg = yaml.safe_load("cols: [x1, x2, churn]")
        df = pd.read_csv("data/churn/train.csv")
        LogisticRegression().fit(df[cfg["cols"]], df["churn"])
        """,
        ["HV-COD-003"],
    ),
    "mock_patch_metric": (
        """
        from unittest import mock
        with mock.patch("sklearn.metrics.roc_auc_score", return_value=1.0):
            pass
        """,
        ["HV-COD-006"],
    ),
    "patch_object_vault": (
        """
        from unittest import mock
        import holdout_vault.core.metrics as hm
        mock.patch.object(hm, "compute_metrics", lambda *a: {}).start()
        """,
        ["HV-COD-006"],
    ),
    "patching_own_code_is_fine": (
        """
        from unittest import mock
        import helpers
        with mock.patch.object(helpers, "load", return_value=None):
            pass
        """,
        [],
    ),
}


@pytest.mark.parametrize("name", list(EXTRA))
def test_extra_cases(tmp_path, name):
    body, expected = EXTRA[name]
    assert codes(run(tmp_path, body)) == expected


@pytest.mark.parametrize(
    ("remainder", "expected"),
    [("'drop'", []), ("'passthrough'", ["HV-COD-003"])],
)
def test_column_transformer_drops_unlisted_columns(tmp_path, remainder, expected):
    body = f"""
    from sklearn.compose import ColumnTransformer
    df = pd.read_csv("data/churn/train.csv")
    ct = ColumnTransformer([("s", StandardScaler(), ["x1", "x2"])], remainder={remainder})
    LogisticRegression().fit(ct.fit_transform(df), df["churn"])
    """
    assert codes(run(tmp_path, body)) == expected


@pytest.mark.parametrize(
    ("body", "expected"),
    [
        (
            """
        from sklearn.datasets import load_breast_cancer
        X, y = load_breast_cancer(return_X_y=True)
        X = StandardScaler().fit_transform(X)
        X_tr, X_te, y_tr, y_te = train_test_split(X, y)
        """,
            ["HV-COD-001"],
        ),
        (
            """
        from sklearn import datasets
        bunch = datasets.load_wine()
        X_tr, X_te, y_tr, y_te = train_test_split(bunch.data, bunch.target)
        model = make_pipeline(StandardScaler(), LogisticRegression()).fit(X_tr, y_tr)
        """,
            [],
        ),
        (
            """
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=200)
        X_res, y_res = SMOTE().fit_resample(X, y)
        print(cross_val_score(LogisticRegression(), X_res, y_res))
        """,
            ["HV-COD-005"],
        ),
    ],
)
def test_sklearn_loaders(tmp_path, body, expected):
    assert codes(run(tmp_path, body)) == expected
