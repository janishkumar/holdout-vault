"""The checks inside the scoring flow: front door, vault, ledger, MCP tools, and the sealed API."""

import pandas as pd
import pytest
from mcp import Client

from conftest import honest_predictions
from holdout_vault.checks.artifacts import collect_tables
from holdout_vault.checks.run import full_checks, workspace_checks
from holdout_vault.core.store import Store
from holdout_vault.core.vault import Vault
from holdout_vault.mcp.server import build_server
from holdout_vault.remote import RemoteVault
from test_api_remote import TOKEN, live  # noqa: F401  (fixture)

LEAKY_SCALER = """
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
df = pd.read_csv("data/churn/train.csv")
X = StandardScaler().fit_transform(df.drop(columns=["churn"]))
X_tr, X_va = train_test_split(X)
"""


@pytest.fixture
def repo(registered, git_repo):
    honest_predictions(registered.export_dir).to_csv(git_repo / "preds.csv", index=False)
    return git_repo


def canary_table(registered) -> pd.DataFrame:
    can = Store.open(registered.home, "demo").read_canaries("churn")
    return pd.DataFrame({"customer_id": can["id"].astype(int), "churn_guess": can["label"]})


async def tool(server, name, args):
    async with Client(server) as c:
        return (await c.call_tool(name, args)).structured_content


async def test_reading_the_labelled_test_file_refuses_the_score(registered, repo):
    (repo / "peek.py").write_text("import pandas as pd\nt = pd.read_csv('../src/test.csv')\n")
    server = build_server("demo", home=registered.home, repo=repo)
    out = await tool(server, "score", {"dataset": "churn", "predictions_path": "preds.csv"})
    assert out["status"] == "refused"
    assert [f["code"] for f in out["findings"]] == ["HV-COD-004"]
    assert out["findings"][0]["location"] == {"file": "peek.py", "line": 2}
    vault = Vault(registered.home, "demo")
    assert vault.budget_state().spent == 0
    head = vault.store.ledger.head()
    assert head.kind == "refused" and head.payload["source"] == "front-door"


async def test_warnings_still_score_and_are_logged(registered, repo):
    (repo / "prep.py").write_text(LEAKY_SCALER)
    server = build_server("demo", home=registered.home, repo=repo)
    out = await tool(server, "score", {"dataset": "churn", "predictions_path": "preds.csv"})
    assert out["status"] == "scored"
    assert "HV-COD-001" in {f["code"] for f in out["findings"]}
    assert "Warnings: HV-COD-001" in out["message"]
    assert Vault(registered.home, "demo").store.ledger.head().payload["findings"] == ["HV-COD-001"]


async def test_canary_labels_in_a_workspace_file_refuse(registered, repo):
    canary_table(registered).to_csv(repo / "scratch.csv", index=False)
    server = build_server("demo", home=registered.home, repo=repo)
    out = await tool(server, "score", {"dataset": "churn", "predictions_path": "preds.csv"})
    assert out["status"] == "refused"
    assert out["findings"][0]["code"] == "HV-CAN-001"
    assert out["findings"][0]["location"] == {"file": "scratch.csv", "column": "churn_guess"}


async def test_honest_workspace_files_are_not_flagged(registered, repo):
    # exported train and test features, and the submitted predictions, are all in the workspace
    vault = Vault(registered.home, "demo")
    result = workspace_checks(vault, "churn", registered.export_dir.rsplit("/data", 1)[0])
    assert result.tables_scanned >= 1
    assert [f for f in result.findings if f.code.startswith("HV-CAN")] == []


async def test_check_leakage_tool_is_free(registered, repo):
    (repo / "prep.py").write_text(LEAKY_SCALER)
    (repo / "peek.py").write_text("import pandas as pd\nt = pd.read_csv('src/test.csv')\n")
    server = build_server("demo", home=registered.home, repo=repo)
    out = await tool(server, "check_leakage", {"dataset": "churn"})
    assert out["blocking"] == 1 and out["warnings"] == 1
    assert out["files_scanned"] == 3
    assert "refused until" in out["message"]
    vault = Vault(registered.home, "demo")
    assert vault.budget_state().spent == 0 and vault.store.ledger.head().kind == "register"


async def test_check_leakage_clean_and_unknown_dataset(registered, repo):
    server = build_server("demo", home=registered.home, repo=repo)
    out = await tool(server, "check_leakage", {"dataset": "churn"})
    assert out["findings"] == [] and out["message"].startswith("No problems")
    bad = await tool(server, "check_leakage", {"dataset": "nope"})
    assert "not registered" in bad["message"]


def test_row_findings_reach_check_leakage(home, tmp_path):
    from conftest import make_classification
    from holdout_vault.core.registry import register_dataset

    df = make_classification()
    train, test = df.iloc[:400].copy(), df.iloc[400:].copy()
    test.iloc[:20, 1:4] = train.iloc[:20, 1:4].to_numpy()
    train.to_csv(tmp_path / "tr.csv", index=False)
    test.to_csv(tmp_path / "te.csv", index=False)
    register_dataset(
        home=home, project="demo", name="d", train_path=tmp_path / "tr.csv", test_path=tmp_path / "te.csv",
        target="churn", id_col="customer_id", metrics=["accuracy"], export_dir=tmp_path / "ws",
    )  # fmt: skip
    result = full_checks(Vault(home, "demo"), "d", tmp_path / "ws")
    assert [f.code for f in result.findings] == ["HV-ROW-001"]


def test_collect_tables_skips_excluded_and_unrelated(tmp_path):
    pd.DataFrame({"customer_id": [1], "y": [0]}).to_csv(tmp_path / "a.csv", index=False)
    pd.DataFrame({"other": [1]}).to_csv(tmp_path / "b.csv", index=False)
    pd.DataFrame({"customer_id": [1]}).to_parquet(tmp_path / "c.parquet")
    (tmp_path / "d.json").write_text("not json")
    (tmp_path / ".venv").mkdir()
    pd.DataFrame({"customer_id": [1]}).to_csv(tmp_path / ".venv" / "e.csv", index=False)
    got = collect_tables(tmp_path, ["customer_id"], exclude=[tmp_path / "c.parquet"])
    assert [p for p, _ in got] == ["a.csv"]
    assert list(got[0][1].columns) == ["id", "y"]


def test_sealed_api_runs_the_vault_side_checks(live, tmp_path):  # noqa: F811
    url, rec = live
    rv = RemoteVault(url, TOKEN, "demo")
    ctx = rv.check_context("churn")
    assert ctx["target"] == "churn" and ctx["test_path"].endswith("test.csv")
    assert rv.row_findings("churn") == []
    found = rv.artifact_findings("churn", [("x.csv", canary_table(rec).rename(columns={"customer_id": "id"}))])
    assert [f.code for f in found] == ["HV-CAN-001"]
    rv.refuse("churn", found, source="front-door")
    head = Vault(rec.home, "demo").store.ledger.head()
    assert head.kind == "refused" and head.payload["findings"][0]["code"] == "HV-CAN-001"
    r = rv.score("churn", honest_predictions(rec.export_dir), extra_findings=found)
    assert r.findings == []  # blocking findings are never attached to a score
