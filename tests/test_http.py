"""The whole MCP surface over stateless Streamable HTTP: tasks, resources with cache hints, the dashboard."""

import socket
import threading
import time

import uvicorn
from mcp import Client

from holdout_vault.mcp.tasks import TasksClient
from test_tasks import model_dir, setup  # noqa: F401  (fixtures)


def serve(server):
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    app = server.streamable_http_app(stateless_http=True)
    uv = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error"))
    t = threading.Thread(target=uv.run, daemon=True)
    t.start()
    for _ in range(200):
        if uv.started:
            break
        time.sleep(0.05)
    return uv, t, f"http://127.0.0.1:{port}/mcp"


async def test_evaluate_task_and_resources_over_http(setup, model_dir):  # noqa: F811
    server, _ = setup
    uv, t, url = serve(server)
    try:
        statuses = []
        async with Client(url, extensions=[TasksClient(on_status=lambda s: statuses.append(s.status))]) as c:
            r = await c.call_tool("evaluate", {"dataset": "churn", "model_path": model_dir})
            entries = await c.read_resource("ledger://demo/entries")
            dash = (await c.call_tool("dashboard", {})).structured_content
        assert r.structured_content["status"] == "scored", r.structured_content["message"]
        assert statuses[-1] == "completed" and len(server.tasks.tasks) == 1
        assert entries.ttl_ms == 5000
        assert dash["spent"] == 1
    finally:
        uv.should_exit = True
        t.join(5)
