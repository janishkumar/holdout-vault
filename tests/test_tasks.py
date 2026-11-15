"""The Tasks extension (SEP-2663): evaluate as a task, polling, cancellation, and the synchronous fallback."""

import asyncio
import threading
import time

import joblib
import pandas as pd
import pytest
from mcp import Client
from mcp.shared.exceptions import MCPError
from mcp_types import ElicitResult, Result
from sklearn.linear_model import LogisticRegression

from holdout_vault.core.vault import Vault
from holdout_vault.mcp.server import build_server
from holdout_vault.mcp.tasks import (
    EXTENSION_ID,
    Cancelled,
    TaskIdParams,
    TasksClient,
    TasksExtension,
    TasksGetRequest,
    TaskState,
)


@pytest.fixture
def setup(registered, git_repo):
    from sklearn.compose import make_column_transformer
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import OneHotEncoder

    train = pd.read_csv(registered.export_dir + "/train.csv")
    model = make_pipeline(
        make_column_transformer((OneHotEncoder(), ["colour"]), remainder="passthrough"), LogisticRegression()
    ).fit(train[["x1", "x2", "colour"]], train["churn"])
    joblib.dump(model, git_repo / "model.joblib")
    server = build_server("demo", home=registered.home, repo=git_repo)
    return server, Vault(registered.home, "demo")


@pytest.fixture
def model_dir(setup):
    return "model.joblib"


async def test_evaluate_runs_as_a_task(setup, model_dir):
    server, vault = setup
    seen = []
    async with Client(server, extensions=[TasksClient(on_status=lambda s: seen.append(s.status))]) as c:
        r = await c.call_tool("evaluate", {"dataset": "churn", "model_path": model_dir})
    assert r.structured_content["status"] == "scored", r.structured_content["message"]
    assert seen[-1] == "completed"
    tasks = list(server.tasks.tasks.values())
    assert len(tasks) == 1 and tasks[0].status == "completed"
    assert vault.budget_state().spent == 1


async def test_client_without_extension_gets_a_plain_result(setup, model_dir):
    server, _ = setup
    async with Client(server) as c:
        r = await c.call_tool("evaluate", {"dataset": "churn", "model_path": model_dir})
    assert r.structured_content["status"] == "scored"
    assert server.tasks.tasks == {}


async def test_over_budget_stays_synchronous_so_the_human_can_answer(setup, model_dir):
    server, vault = setup
    vault.set_budget(0, by="test")
    asked = []

    async def answer(ctx, params):
        asked.append(params.message)
        return ElicitResult(action="accept", content={"approve": True})

    async with Client(server, extensions=[TasksClient()], elicitation_callback=answer) as c:
        r = await c.call_tool("evaluate", {"dataset": "churn", "model_path": model_dir})
    assert r.structured_content["status"] == "scored"
    assert asked and server.tasks.tasks == {}


async def test_task_methods_need_the_capability_and_a_known_id(setup):
    server, _ = setup
    async with Client(server) as c:
        with pytest.raises(MCPError) as no_cap:
            await c.session.send_request(TasksGetRequest(params=TaskIdParams(taskId="x")), Result)
    assert no_cap.value.error.code == -32021
    async with Client(server, extensions=[TasksClient()]) as c:
        with pytest.raises(MCPError) as unknown:
            await c.session.send_request(TasksGetRequest(params=TaskIdParams(taskId="nope")), Result)
    assert unknown.value.error.code == -32602 and "not found" in unknown.value.error.message


# the extension on its own ------------------------------------------------------------------------
async def wait_terminal(ext: TasksExtension, task: TaskState, timeout: float = 5) -> TaskState:
    end = time.monotonic() + timeout
    while task.status not in {"completed", "failed", "cancelled"} and time.monotonic() < end:
        await asyncio.sleep(0.01)
    return task


async def test_cancel_before_completion():
    gate = threading.Event()

    def slow(args, status, cancel):
        status("Working.")
        gate.wait(5)
        if cancel.is_set():
            raise Cancelled()
        return {"content": [], "isError": False}

    ext = TasksExtension({"t": slow})
    task = ext.create("t", {})
    await asyncio.sleep(0.05)
    assert task.wire()["statusMessage"] == "Working."
    task.cancel.set()
    gate.set()
    await wait_terminal(ext, task)
    assert task.status == "cancelled" and "result" not in task.wire()


async def test_runner_error_fails_the_task():
    def broken(args, status, cancel):
        raise RuntimeError("boom")

    ext = TasksExtension({"t": broken})
    task = await wait_terminal(ext, ext.create("t", {}))
    wire = task.wire()
    assert wire["status"] == "failed" and wire["error"] == {"code": -32603, "message": "boom"}


async def test_terminal_status_is_final_and_wire_shape():
    ext = TasksExtension({"t": lambda a, s, c: {"content": [], "isError": False}}, ttl_ms=None, poll_interval_ms=250)
    task = await wait_terminal(ext, ext.create("t", {}))
    ext._update(task, status="failed")
    wire = task.wire()
    assert wire["status"] == "completed" and wire["ttlMs"] is None and wire["pollIntervalMs"] == 250
    assert set(wire) == {"taskId", "status", "createdAt", "lastUpdatedAt", "ttlMs", "pollIntervalMs",
                         "statusMessage", "result"}  # fmt: skip
    assert len(wire["taskId"]) == 32 and EXTENSION_ID == "io.modelcontextprotocol/tasks"
