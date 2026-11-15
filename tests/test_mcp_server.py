import pandas as pd
import pytest
from mcp import Client
from mcp_types import ElicitResult

from conftest import honest_predictions
from holdout_vault.core.vault import Vault
from holdout_vault.mcp.server import build_server


@pytest.fixture
def setup(registered, git_repo):
    honest_predictions(registered.export_dir).to_csv(git_repo / "preds.csv", index=False)
    server = build_server("demo", home=registered.home, repo=git_repo)
    return server, Vault(registered.home, "demo"), git_repo


class Answers:
    def __init__(self, *responses):
        self.responses = list(responses)
        self.asked = []

    async def __call__(self, ctx, params):
        self.asked.append(params.message)
        return self.responses.pop(0)


async def call(server, args, answers=None, mode="auto"):
    kwargs = {"elicitation_callback": answers} if answers else {}
    async with Client(server, mode=mode, **kwargs) as c:
        r = await c.call_tool("score", args)
        return r.structured_content


async def test_only_agent_tools_are_exposed(setup):
    server, _, _ = setup
    async with Client(server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
    assert set(tools) == {"score", "evaluate", "vault_status"}
    assert all(t.output_schema for t in tools.values())
    assert "gate" not in tools["score"].input_schema["properties"]


async def test_score_happy_path(setup):
    server, vault, _ = setup
    answers = Answers()
    out = await call(server, {"dataset": "churn", "predictions_path": "preds.csv", "seed": 1}, answers)
    assert out["status"] == "scored"
    assert 0.5 < out["metrics"]["accuracy"] <= 1
    assert out["remaining"] == 19
    assert len(out["stamp"]["git_commit"]) == 40
    assert out["stamp"]["git_dirty"] is True  # preds.csv is untracked
    assert out["stamp"]["seed"] == 1
    assert answers.asked == []
    assert vault.store.ledger.head().kind == "score"


async def test_missing_file_is_invalid_and_free(setup):
    server, vault, _ = setup
    out = await call(server, {"dataset": "churn", "predictions_path": "nope.csv"})
    assert out["status"] == "invalid" and "not found" in out["message"]
    assert vault.budget_state().spent == 0


async def test_over_budget_human_approves(setup):
    server, vault, _ = setup
    vault.set_budget(1, by="test")
    args = {"dataset": "churn", "predictions_path": "preds.csv"}
    assert (await call(server, args))["status"] == "scored"
    answers = Answers(ElicitResult(action="accept", content={"approve": True}))
    out = await call(server, args, answers)
    assert out["status"] == "scored" and out["remaining"] == 0
    assert "budget" in answers.asked[0]
    grant = [e for e in vault.store.ledger.entries() if e.kind == "grant"][-1]
    assert grant.payload["via"] == "mcp-elicitation"


@pytest.mark.parametrize(
    "answer",
    [
        ElicitResult(action="accept", content={"approve": False}),
        ElicitResult(action="decline"),
        ElicitResult(action="cancel"),
    ],
)
async def test_over_budget_not_approved(setup, answer):
    server, vault, _ = setup
    vault.set_budget(0, by="test")
    out = await call(server, {"dataset": "churn", "predictions_path": "preds.csv"}, Answers(answer))
    assert out["status"] == "over_budget"
    assert "holdout-vault grant" in out["message"]
    assert vault.budget_state().spent == 0


async def test_over_budget_legacy_client(setup):
    server, vault, _ = setup
    vault.set_budget(0, by="test")
    answers = Answers(ElicitResult(action="accept", content={"approve": True}))
    out = await call(server, {"dataset": "churn", "predictions_path": "preds.csv"}, answers, mode="legacy")
    assert out["status"] == "scored"


async def test_label_cheat_refused(setup):
    server, vault, repo = setup
    labels = vault.store.read_labels("churn")
    y = labels["label"].astype(int)
    cheat = pd.DataFrame({"customer_id": labels["id"].astype(int), "prediction": y, "proba": y.astype(float)})
    cheat.to_csv(repo / "cheat.csv", index=False)
    out = await call(server, {"dataset": "churn", "predictions_path": "cheat.csv"})
    assert out["status"] == "refused"
    assert out["findings"][0]["code"] == "HV-CAN-002"
    assert out["metrics"] is None


async def test_tamper_reported(setup, split_files):
    server, _, _ = setup
    _, test = split_files
    test.write_text(test.read_text() + "\n")
    out = await call(server, {"dataset": "churn", "predictions_path": "preds.csv"})
    assert out["status"] == "tamper" and out["findings"][0]["code"] == "HV-TST-001"


async def test_vault_status_tool(setup):
    server, _, _ = setup
    async with Client(server) as c:
        r = await c.call_tool("vault_status", {})
    s = r.structured_content
    assert s["mode"] == "watch" and "detects" in s["note"]
    assert s["datasets"][0]["id_col"] == "customer_id"


async def test_tools_list_order_is_deterministic(setup):
    server, _, _ = setup
    async with Client(server) as c:
        first = [t.name for t in (await c.list_tools()).tools]
        second = [t.name for t in (await c.list_tools()).tools]
    assert first == second == ["score", "evaluate", "vault_status"]


async def test_evaluate_runs_model_and_scores(setup):
    import joblib
    from sklearn.compose import make_column_transformer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    server, vault, repo = setup
    train = pd.read_csv(vault.dataset_info("churn")["export_dir"] + "/train.csv")
    model = make_pipeline(
        make_column_transformer((OneHotEncoder(), ["colour"]), remainder="passthrough"), LogisticRegression()
    ).fit(train[["x1", "x2", "colour"]], train["churn"])
    joblib.dump(model, repo / "model.joblib")
    async with Client(server) as c:
        r = await c.call_tool("evaluate", {"dataset": "churn", "model_path": "model.joblib"})
    out = r.structured_content
    assert out["status"] == "scored", out["message"]
    assert out["stamp"]["runner"] == "subprocess"
    assert len(out["stamp"]["model_sha256"]) == 64
    assert vault.store.ledger.head().kind == "evaluate"


async def test_evaluate_missing_model_and_broken_model(setup):
    server, vault, repo = setup
    async with Client(server) as c:
        r = await c.call_tool("evaluate", {"dataset": "churn", "model_path": "nope.joblib"})
        assert r.structured_content["status"] == "invalid"
        bad = repo / "bad"
        bad.mkdir()
        (bad / "predict.py").write_text("def predict(X):\n    raise ValueError('nope')\n")
        r = await c.call_tool("evaluate", {"dataset": "churn", "model_path": "bad"})
    assert r.structured_content["status"] == "invalid" and "nope" in r.structured_content["message"]
    assert vault.budget_state().spent == 0


async def test_evaluate_model_cannot_read_labels_through_canaries(setup):
    """A model that smuggles the sealed labels in is refused by the canary check."""
    server, vault, repo = setup
    labels = vault.store.read_labels("churn")
    d = repo / "sneaky"
    d.mkdir()
    labels.to_csv(d / "stolen.csv", index=False)
    (d / "predict.py").write_text(
        "import pandas as pd, pathlib\n"
        "L = pd.read_csv(pathlib.Path(__file__).parent / 'stolen.csv')\n"
        "def predict(X):\n"
        "    return pd.DataFrame({'prediction': L['label'].values, 'proba': L['label'].astype(float).values})\n"
    )
    async with Client(server) as c:
        r = await c.call_tool("evaluate", {"dataset": "churn", "model_path": "sneaky"})
    assert r.structured_content["status"] == "refused"
