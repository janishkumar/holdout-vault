"""Claude Code (headless) as the agent. Needs `claude` installed and logged in; uses plan usage.

The agent runs in the task workspace with its usual tools. In the vault arm the holdout-vault MCP server
is configured for it. The stream-json transcript is turned into the same action log the grader reads:
Read/Grep/Glob become reads, Write/Edit become writes, Bash becomes a shell action, and vault tool calls
are recorded by name.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from env import Env

REPO = Path(__file__).resolve().parents[2]
ALLOWED = [
    "Bash",
    "Read",
    "Write",
    "Edit",
    "Glob",
    "Grep",
    "mcp__holdout-vault__score",
    "mcp__holdout-vault__check_leakage",
    "mcp__holdout-vault__audit_claims",
    "mcp__holdout-vault__vault_status",
]


def mcp_config(env: Env) -> dict[str, Any]:
    return {
        "mcpServers": {
            "holdout-vault": {
                "command": sys.executable,
                "args": ["-m", "holdout_vault.cli", "serve", "--project", "bench", "--repo", str(env.ws.root)],
                "env": {
                    "HOLDOUT_VAULT_HOME": str(env.ws.vault_home),
                    "HOLDOUT_VAULT_MODE": "watch",
                    "PYTHONPATH": str(REPO / "src"),
                },
            }
        }
    }


def to_actions(events: list[dict[str, Any]], root: Path) -> list[dict[str, Any]]:
    """Map Claude Code tool calls onto the grader's action vocabulary."""
    actions = []
    for ev in events:
        if ev.get("type") != "assistant":
            continue
        for block in ev.get("message", {}).get("content", []):
            if block.get("type") != "tool_use":
                continue
            name, inp = block.get("name", ""), block.get("input", {})
            if name in ("Read", "NotebookRead"):
                actions.append({"tool": "read_file", "args": {"path": inp.get("file_path", "")}})
            elif name in ("Grep", "Glob"):
                actions.append(
                    {"tool": "read_file", "args": {"path": inp.get("path", "") + " " + inp.get("pattern", "")}}
                )
            elif name in ("Write", "Edit", "MultiEdit", "NotebookEdit"):
                actions.append({"tool": "write_file", "args": {"path": inp.get("file_path", "")}})
            elif name == "Bash":
                actions.append({"tool": "shell", "args": {"command": inp.get("command", "")}})
            elif name.startswith("mcp__holdout-vault__"):
                actions.append({"tool": name.rsplit("__", 1)[1], "args": inp})
            else:
                actions.append({"tool": name, "args": inp})
    return actions


def run(env: Env, model: str = "sonnet", max_turns: int = 40) -> dict[str, Any]:
    if shutil.which("claude") is None:
        raise RuntimeError("the claude CLI is not installed")
    root = env.ws.root
    cmd = [
        "claude",
        "-p",
        (root / "TASK.md").read_text(),
        "--output-format",
        "stream-json",
        "--verbose",
        "--max-turns",
        str(max_turns),
        "--model",
        model,
        "--permission-mode",
        "acceptEdits",
        "--allowedTools",
        ",".join(ALLOWED),
    ]
    if env.ws.vault_home is not None:
        cfg = root.parent / "mcp.json"
        cfg.write_text(json.dumps(mcp_config(env)))
        cmd += ["--mcp-config", str(cfg), "--strict-mcp-config"]
    start = time.time()
    proc = subprocess.run(cmd, cwd=root, capture_output=True, text=True, timeout=3600, env={**os.environ})
    events = []
    for line in proc.stdout.splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            continue
    if any("authenticate" in json.dumps(e).lower() and e.get("is_error") for e in events) or (
        proc.returncode and "authenticate" in (proc.stdout + proc.stderr).lower()
    ):
        raise RuntimeError("claude is not logged in: run `claude /login` first")
    env.log.extend(to_actions(events, root))
    result = next((e for e in reversed(events) if e.get("type") == "result"), {})
    return {
        "agent_turns": result.get("num_turns"),
        "agent_cost_usd": result.get("total_cost_usd"),
        "agent_seconds": round(time.time() - start, 1),
        "finished": result.get("subtype") == "success",
    }
