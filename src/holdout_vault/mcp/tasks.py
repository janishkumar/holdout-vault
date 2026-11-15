"""The MCP Tasks extension (``io.modelcontextprotocol/tasks``, SEP-2663), server and client sides.

The Python SDK ships the extension hooks but not Tasks itself, so this module implements it:

- :class:`TasksExtension` (server) turns a ``tools/call`` for a task-capable tool into a task when the
  request's client capabilities declare the extension, runs the work in a thread, and serves
  ``tasks/get``, ``tasks/update`` and ``tasks/cancel``. A client that does not declare the extension
  gets an ordinary synchronous result, as the SEP requires.
- :class:`TasksClient` (client) claims ``resultType: "task"`` and resolves it by polling ``tasks/get``
  until the task is terminal, returning the ordinary ``CallToolResult``.

Progress is reported through ``statusMessage``; cancellation is cooperative.
"""

from __future__ import annotations

import asyncio
import threading
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any, Literal

from mcp.client.extension import ClaimContext, ClientExtension, ResultClaim
from mcp.server.extension import Extension, MethodBinding
from mcp.shared.exceptions import MCPError
from mcp_types import CallToolRequestParams, CallToolResult, Request, RequestParams, Result
from pydantic import ConfigDict, Field

EXTENSION_ID = "io.modelcontextprotocol/tasks"
CAPS_META_KEY = "io.modelcontextprotocol/clientCapabilities"
MISSING_CAPABILITY = -32021
TERMINAL = {"completed", "failed", "cancelled"}

Status = Literal["working", "input_required", "completed", "failed", "cancelled"]
# A runner gets the tool arguments, a status callback and a cancel flag, and returns a CallToolResult
# as a dict, or None when it stopped because of the cancel flag.
Runner = Callable[[dict[str, Any], Callable[[str], None], threading.Event], dict[str, Any] | None]


class Cancelled(Exception):
    """Raised by a runner that noticed the cancel flag."""


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds").replace("+00:00", "Z")


@dataclass
class TaskState:
    task_id: str
    created_at: str
    ttl_ms: int | None
    poll_interval_ms: int
    status: Status = "working"
    status_message: str | None = None
    last_updated_at: str = ""
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None
    cancel: threading.Event = field(default_factory=threading.Event)
    created_mono: float = 0.0

    def wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "taskId": self.task_id,
            "status": self.status,
            "createdAt": self.created_at,
            "lastUpdatedAt": self.last_updated_at or self.created_at,
            "ttlMs": self.ttl_ms,
            "pollIntervalMs": self.poll_interval_ms,
        }
        if self.status_message:
            out["statusMessage"] = self.status_message
        if self.status == "completed" and self.result is not None:
            out["result"] = self.result
        if self.status == "failed" and self.error is not None:
            out["error"] = self.error
        return out


class TaskIdParams(RequestParams):
    task_id: str = Field(alias="taskId")


class TaskUpdateParams(RequestParams):
    task_id: str = Field(alias="taskId")
    input_responses: dict[str, Any] = Field(default_factory=dict, alias="inputResponses")


def declares_tasks(ctx: Any) -> bool:
    """Did this request's client capabilities declare the Tasks extension?"""
    meta = getattr(ctx, "meta", None)
    caps = None
    if meta is not None:
        caps = (
            meta.get(CAPS_META_KEY) if isinstance(meta, dict) else getattr(meta, "model_extra", {}).get(CAPS_META_KEY)
        )
    if isinstance(caps, dict) and EXTENSION_ID in (caps.get("extensions") or {}):
        return True
    session = getattr(ctx, "session", None)
    client_caps = getattr(session, "client_capabilities", None) if session is not None else None
    extensions = getattr(client_caps, "extensions", None) or {}
    return EXTENSION_ID in extensions


def _missing_capability() -> MCPError:
    return MCPError(
        code=MISSING_CAPABILITY,
        message="Missing required client capability",
        data={"requiredCapabilities": {"extensions": {EXTENSION_ID: {}}}},
    )


class TasksExtension(Extension):
    identifier = EXTENSION_ID

    def __init__(
        self,
        runners: dict[str, Runner],
        *,
        eligible: Callable[[str, dict[str, Any]], bool] | None = None,
        ttl_ms: int | None = 3_600_000,
        poll_interval_ms: int = 1000,
    ):
        self.runners = runners
        self.eligible = eligible or (lambda name, args: True)
        self.ttl_ms = ttl_ms
        self.poll_interval_ms = poll_interval_ms
        self.tasks: dict[str, TaskState] = {}
        self._lock = threading.Lock()
        self._background: set[asyncio.Task] = set()

    # task lifecycle -----------------------------------------------------------------------
    def _update(self, task: TaskState, **changes: Any) -> None:
        with self._lock:
            if task.status in TERMINAL:
                return
            for k, v in changes.items():
                setattr(task, k, v)
            task.last_updated_at = _now()

    def _expired(self, task: TaskState) -> bool:
        if task.ttl_ms is None:
            return False
        loop_time = asyncio.get_event_loop().time() if task.created_mono else 0
        return bool(task.created_mono) and (loop_time - task.created_mono) * 1000 > task.ttl_ms

    async def _run(self, task: TaskState, runner: Runner, arguments: dict[str, Any]) -> None:
        def status(msg: str) -> None:
            self._update(task, status_message=msg)

        try:
            result = await asyncio.to_thread(runner, arguments, status, task.cancel)
        except Cancelled:
            result = None
        except Exception as exc:  # the task fails; the server keeps running
            self._update(task, status="failed", error={"code": -32603, "message": str(exc)[:500]},
                         status_message="The tool raised an error.")  # fmt: skip
            return
        if result is None:
            self._update(task, status="cancelled", status_message="Cancelled before anything was revealed.")
        else:
            self._update(task, status="completed", result=result, status_message="Done.")

    def create(self, name: str, arguments: dict[str, Any]) -> TaskState:
        now = _now()
        task = TaskState(uuid.uuid4().hex, now, self.ttl_ms, self.poll_interval_ms, last_updated_at=now)
        task.created_mono = asyncio.get_event_loop().time()
        with self._lock:
            self.tasks[task.task_id] = task
        job = asyncio.get_event_loop().create_task(self._run(task, self.runners[name], arguments))
        self._background.add(job)
        job.add_done_callback(self._background.discard)
        return task

    def _get(self, task_id: str) -> TaskState:
        task = self.tasks.get(task_id)
        if task is None:
            raise MCPError(code=-32602, message="Failed to retrieve task: Task not found")
        if self._expired(task):
            self.tasks.pop(task_id, None)
            raise MCPError(code=-32602, message="Failed to retrieve task: Task has expired")
        return task

    # extension surface --------------------------------------------------------------------
    async def intercept_tool_call(self, params: CallToolRequestParams, ctx: Any, call_next: Any) -> Any:
        args = dict(params.arguments or {})
        if params.name not in self.runners or not declares_tasks(ctx) or not self.eligible(params.name, args):
            return await call_next(ctx)
        task = self.create(params.name, args)
        return {"resultType": "task", **task.wire()}

    def methods(self) -> list[MethodBinding]:
        async def get(ctx: Any, params: TaskIdParams) -> dict[str, Any]:
            if not declares_tasks(ctx):
                raise _missing_capability()
            with self._lock:
                return {"resultType": "complete", **self._get(params.task_id).wire()}

        async def update(ctx: Any, params: TaskUpdateParams) -> dict[str, Any]:
            if not declares_tasks(ctx):
                raise _missing_capability()
            self._get(params.task_id)  # holdout-vault tasks never ask for input; responses are ignored
            return {"resultType": "complete"}

        async def cancel(ctx: Any, params: TaskIdParams) -> dict[str, Any]:
            if not declares_tasks(ctx):
                raise _missing_capability()
            task = self._get(params.task_id)
            task.cancel.set()
            self._update(task, status_message="Cancelling.")
            return {"resultType": "complete"}

        return [
            MethodBinding("tasks/get", TaskIdParams, get),
            MethodBinding("tasks/update", TaskUpdateParams, update),
            MethodBinding("tasks/cancel", TaskIdParams, cancel),
        ]


# client side -------------------------------------------------------------------------------
class CreateTaskResult(Result):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    result_type: Literal["task"] = Field("task", alias="resultType")
    task_id: str = Field(alias="taskId")
    status: str
    poll_interval_ms: int | None = Field(None, alias="pollIntervalMs")


class TaskResult(Result):
    model_config = ConfigDict(extra="allow", populate_by_name=True)
    task_id: str = Field(alias="taskId")
    status: str
    status_message: str | None = Field(None, alias="statusMessage")
    poll_interval_ms: int | None = Field(None, alias="pollIntervalMs")
    result: dict[str, Any] | None = None
    error: dict[str, Any] | None = None


class TasksGetRequest(Request[TaskIdParams, Literal["tasks/get"]]):
    method: Literal["tasks/get"] = "tasks/get"
    params: TaskIdParams
    name_param = "taskId"


class TasksCancelRequest(Request[TaskIdParams, Literal["tasks/cancel"]]):
    method: Literal["tasks/cancel"] = "tasks/cancel"
    params: TaskIdParams
    name_param = "taskId"


class TasksClient(ClientExtension):
    """Declare the Tasks extension and resolve task results by polling ``tasks/get``."""

    identifier = EXTENSION_ID

    def __init__(self, on_status: Callable[[TaskResult], None] | None = None, max_wait_s: float = 3600):
        self.on_status = on_status
        self.max_wait_s = max_wait_s

    def claims(self) -> list[ResultClaim[Any]]:
        return [ResultClaim(result_type="task", model=CreateTaskResult, resolve=self._resolve)]

    async def _resolve(self, created: CreateTaskResult, ctx: ClaimContext) -> CallToolResult:
        waited, interval = 0.0, (created.poll_interval_ms or 1000) / 1000
        while True:
            state = await ctx.session.send_request(
                TasksGetRequest(params=TaskIdParams(taskId=created.task_id)), TaskResult
            )
            if self.on_status:
                self.on_status(state)
            if state.status == "completed" and state.result is not None:
                return CallToolResult.model_validate(state.result)
            if state.status in {"failed", "cancelled"}:
                msg = (state.error or {}).get("message") or state.status_message or state.status
                return CallToolResult(content=[{"type": "text", "text": f"Task {state.status}: {msg}"}], is_error=True)
            if waited >= self.max_wait_s:
                await ctx.session.send_request(TasksCancelRequest(params=TaskIdParams(taskId=created.task_id)), Result)
                return CallToolResult(content=[{"type": "text", "text": "Task timed out and was cancelled."}],
                                      is_error=True)  # fmt: skip
            await asyncio.sleep(interval)
            waited += interval
            interval = (state.poll_interval_ms or 1000) / 1000
