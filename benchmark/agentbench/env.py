"""The tools an agent acts through, with every call logged for grading.

The same tools back the scripted policies and the local-model agent: list a directory, read a file,
write a file, run Python, and (vault arm) the holdout-vault MCP tools, called through a real MCP client
against the real server. Paths may point anywhere on disk, as a shell would allow.
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from tasks import Workspace

MAX_OUTPUT = 6000


def cell(code: str) -> str:
    """Notebook semantics: a trailing bare expression is printed, as agents expect from a code cell."""
    import ast

    try:
        tree = ast.parse(code)
    except SyntaxError:
        return code
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = tree.body[-1]
        call = ast.Call(func=ast.Name("print", ast.Load()), args=[last.value], keywords=[])
        tree.body[-1] = ast.copy_location(ast.Expr(call), last)
        return ast.unparse(ast.fix_missing_locations(tree))
    return code


class Env:
    def __init__(self, ws: Workspace):
        self.ws = ws
        self.log: list[dict[str, Any]] = []
        self.vault_seconds = 0.0
        self._server = None
        if ws.vault_home is not None:
            from holdout_vault.mcp.server import build_server

            self._server = build_server("bench", home=ws.vault_home, repo=ws.root, mode="watch")

    def _path(self, p: str) -> Path:
        q = Path(os.path.expanduser(p))
        return q if q.is_absolute() else self.ws.root / q

    def _record(self, tool: str, args: dict[str, Any], output: str) -> str:
        self.log.append({"tool": tool, "args": args, "output": output[:MAX_OUTPUT], "t": time.time()})
        return output[:MAX_OUTPUT]

    # file and code tools ------------------------------------------------------------------
    def list_dir(self, path: str = ".") -> str:
        p = self._path(path)
        try:
            items = sorted(f"{c.name}/" if c.is_dir() else c.name for c in p.iterdir())
            out = "\n".join(items) or "(empty)"
        except OSError as exc:
            out = f"error: {exc}"
        return self._record("list_dir", {"path": path}, out)

    def read_file(self, path: str) -> str:
        p = self._path(path)
        try:
            if p.suffix == ".parquet":
                import pandas as pd

                out = pd.read_parquet(p).to_csv(index=False)
            else:
                out = p.read_text(errors="replace")
        except (OSError, ValueError) as exc:
            out = f"error: {exc}"
        return self._record("read_file", {"path": path}, out)

    def write_file(self, path: str, content: str) -> str:
        p = self._path(path)
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(content)
            out = f"wrote {len(content)} characters to {path}"
        except OSError as exc:
            out = f"error: {exc}"
        return self._record("write_file", {"path": path, "content": content[:4000]}, out)

    def run_python(self, code: str, timeout: int = 300) -> str:
        script = self.ws.root / ".agent_run.py"
        script.write_text(cell(code))
        try:
            proc = subprocess.run(
                [sys.executable, script.name],
                cwd=self.ws.root,
                capture_output=True,
                text=True,
                timeout=timeout,
                env={**os.environ, "PYTHONWARNINGS": "ignore"},
            )
            out = (proc.stdout + ("\n" + proc.stderr if proc.returncode else "")).strip() or (
                "(ran without output; print values to see them)"
            )
        except subprocess.TimeoutExpired:
            out = f"error: timed out after {timeout}s"
        finally:
            script.unlink(missing_ok=True)
        return self._record("run_python", {"code": code[:6000]}, out)

    # vault tools ----------------------------------------------------------------------------
    def _mcp(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        from mcp import Client

        async def go():
            async with Client(self._server) as c:
                return (await c.call_tool(name, args)).structured_content

        start = time.time()
        try:
            return asyncio.run(go())
        finally:
            self.vault_seconds += time.time() - start

    def score(self, dataset: str, predictions_path: str = "predictions.csv") -> str:
        if self._server is None:
            return self._record("score", {"dataset": dataset}, "error: no vault in this arm")
        out = self._mcp("score", {"dataset": dataset, "predictions_path": predictions_path})
        return self._record("score", {"dataset": dataset, "predictions_path": predictions_path}, json.dumps(out))

    def check_leakage(self, dataset: str) -> str:
        out = self._mcp("check_leakage", {"dataset": dataset}) if self._server else {"error": "no vault"}
        return self._record("check_leakage", {"dataset": dataset}, json.dumps(out))

    def audit_claims(self) -> str:
        out = self._mcp("audit_claims", {}) if self._server else {"error": "no vault"}
        return self._record("audit_claims", {}, json.dumps(out))
