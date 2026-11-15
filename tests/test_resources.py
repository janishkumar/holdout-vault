"""Resources with cache hints, the dashboard tool and its MCP App view."""

import json

import pytest
from mcp import Client

from conftest import honest_predictions
from holdout_vault.mcp.dashboard import DASHBOARD_URI, score_gaps
from holdout_vault.mcp.server import build_server


@pytest.fixture
async def scored(registered, git_repo):
    honest_predictions(registered.export_dir).to_csv(git_repo / "preds.csv", index=False)
    run = git_repo / "mlruns" / "0" / "r1"
    (run / "metrics").mkdir(parents=True)
    (run / "tags").mkdir()
    (run / "tags" / "mlflow.runName").write_text("my-val")
    (run / "metrics" / "val_accuracy").write_text("1 0.99 0\n")
    server = build_server("demo", home=registered.home, repo=git_repo)
    async with Client(server) as c:
        out = (await c.call_tool("score", {"dataset": "churn", "predictions_path": "preds.csv"})).structured_content
    assert out["status"] == "scored"
    return server, out


async def test_ledger_and_provenance_resources(scored):
    server, out = scored
    seq = out["stamp"]["ledger_seq"]
    async with Client(server) as c:
        entries = await c.read_resource("ledger://demo/entries")
        one = await c.read_resource(f"ledger://demo/entry/{seq}")
        prov = await c.read_resource(f"provenance://demo/{seq}")
    assert entries.ttl_ms == 5000
    assert [e["kind"] for e in json.loads(entries.contents[0].text)] == ["register", "score"]
    assert json.loads(one.contents[0].text)["hash"] == out["stamp"]["ledger_hash"]
    p = json.loads(prov.contents[0].text)
    assert p["ledger_ok"] is True and p["entry_hash"] == out["stamp"]["ledger_hash"]
    assert p["stamp"]["dataset_sha256"] == out["stamp"]["dataset_sha256"]


async def test_resource_errors(scored):
    server, _ = scored
    async with Client(server) as c:
        for uri in ("ledger://other/entries", "ledger://demo/entry/99", "provenance://demo/0"):
            with pytest.raises(Exception):  # noqa: B017  (the SDK wraps the ValueError)
                await c.read_resource(uri)


async def test_findings_resource(scored, git_repo):
    server, _ = scored
    (git_repo / "peek.py").write_text("import pandas as pd\nt = pd.read_csv('src/test.csv')\n")
    async with Client(server) as c:
        found = json.loads((await c.read_resource("findings://demo/latest")).contents[0].text)
    assert [f["code"] for f in found] == ["HV-COD-004"]


async def test_dashboard_tool_and_app(scored):
    server, _ = scored
    async with Client(server) as c:
        tools = {t.name: t for t in (await c.list_tools()).tools}
        data = (await c.call_tool("dashboard", {})).structured_content
        html = await c.read_resource(DASHBOARD_URI)
    assert tools["dashboard"].meta["ui"]["resourceUri"] == DASHBOARD_URI
    assert data["spent"] == 1 and data["remaining"] == 19 and data["ledger_ok"]
    assert data["scores"][0]["dataset"] == "churn"
    gap = next(g for g in data["gaps"] if g["metric"] == "accuracy")
    assert gap["own"] == 0.99 and gap["own_source"] == "mlflow my-val" and gap["optimism"] > 0
    assert "Own measurement vs sealed score" in data["summary"] and "optimism" in data["summary"]
    assert html.contents[0].mime_type == "text/html;profile=mcp-app"
    assert "ui/initialize" in html.contents[0].text


def test_score_gaps_lower_is_better(tmp_path):
    (tmp_path / "metrics.json").write_text(json.dumps({"val_rmse": 10.0}))
    entries = [{"seq": 3, "kind": "score", "payload": {"dataset": "p", "metrics": {"rmse": 12.0}}}]
    (gap,) = score_gaps(entries, tmp_path)
    assert gap["optimism"] == 2.0 and gap["sealed_entry"] == 3
