"""Phase 4 web UI: start tasks, watch the plan, steps and tool calls live, answer approvals,
and see tokens and cost per task.

    task-agent ui            then open http://127.0.0.1:8765

Local use only. It binds 127.0.0.1, accepts only JSON POSTs (a form on another website
cannot post here without a CORS preflight, which this server never grants) and only
localhost Host headers (blocks DNS rebinding). There are no accounts; the phase 6 API adds
keys. Tasks are kept in memory and are gone when the server stops.

ponytail: one static page with plain JavaScript instead of the planned Next.js app; no build
step, served by this process. Move to Next.js when the phase 6 API needs a real frontend.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from pydantic import ValidationError
from starlette.applications import Starlette
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.middleware.trustedhost import TrustedHostMiddleware
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from task_agent.core.approval import ApprovalDecision, ApprovalRequest, Approver, Verdict
from task_agent.core.events import listen
from task_agent.core.runner import Runner
from task_agent.core.types import Budget, RunResult, TaskRequest
from task_agent.observability.logging import get_logger

log = get_logger(__name__)

PAGE = Path(__file__).with_name("ui.html")
RUNNERS = ("baseline", "orchestrator")
MakeRunner = Callable[[str, Approver], Runner]


class TaskRecord:
    """One task's live state: its event history, open streams and pending approvals."""

    def __init__(self, request: TaskRequest, runner: str) -> None:
        self.request = request
        self.runner = runner
        self.status = "running"
        self.events: list[dict[str, Any]] = []
        self.streams: set[asyncio.Queue[dict[str, Any]]] = set()
        self.approvals: dict[str, asyncio.Future[ApprovalDecision]] = {}
        self.result: RunResult | None = None
        self.task: asyncio.Task[None] | None = None
        self.created = time.time()

    def push(self, event: dict[str, Any]) -> None:
        self.events.append(event)
        for queue in self.streams:
            queue.put_nowait(event)

    @property
    def finished(self) -> bool:
        return bool(self.events) and self.events[-1]["type"] == "done"

    def start(self, runner: Runner) -> asyncio.Task[None]:
        """Run the task in the background; the event history ends with one 'done' event."""

        async def go() -> None:
            req = self.request
            self.push(
                {"type": "start", "ts": time.time(), "prompt": req.prompt, "runner": self.runner}
            )
            with listen(self.push):
                try:
                    result = await runner.run(req)
                except asyncio.CancelledError:
                    self.status = "cancelled"
                    self.push({"type": "done", "ts": time.time(), "status": "cancelled"})
                    raise
                except Exception as exc:  # a runner bug must still close the stream
                    log.exception("ui.task_crashed")
                    self.status = "failed"
                    self.push(
                        {"type": "done", "ts": time.time(), "status": "failed", "error": repr(exc)}
                    )
                    return
            self.result = result
            self.status = result.status.value
            self.push({"type": "done", "ts": time.time(), **result.model_dump(mode="json")})

        self.task = asyncio.create_task(go())
        return self.task

    async def follow(self) -> AsyncIterator[dict[str, Any]]:
        """Every event so far, then live ones, until 'done'."""
        # No await between subscribing and copying the history, so no event is lost or doubled.
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self.streams.add(queue)
        history = list(self.events)
        try:
            for ev in history:
                yield ev
            if any(ev["type"] == "done" for ev in history):
                return
            while True:
                ev = await queue.get()
                yield ev
                if ev["type"] == "done":
                    return
        finally:
            self.streams.discard(queue)

    def summary(self) -> dict[str, Any]:
        usage: dict[str, Any] = {}
        steps = 0
        for ev in self.events:
            if ev["type"] == "model":
                usage, steps = ev["usage"], ev["step"]
        if self.result is not None:
            usage, steps = self.result.usage.model_dump(), self.result.steps
        return {
            "task_id": self.request.task_id,
            "prompt": self.request.prompt,
            "runner": self.runner,
            "status": self.status,
            "steps": steps,
            "cost_usd": usage.get("cost_usd", 0.0),
            "pending_approvals": list(self.approvals),
            "created": self.created,
        }


class WebApprover:
    """Parks an approval request until someone answers it, or rejects it after ``timeout_s``."""

    def __init__(self, record: TaskRecord, timeout_s: float | None = None) -> None:
        self.record = record
        self.timeout_s = timeout_s

    async def request(self, req: ApprovalRequest) -> ApprovalDecision:
        approval_id = uuid.uuid4().hex[:8]
        future: asyncio.Future[ApprovalDecision] = asyncio.get_running_loop().create_future()
        self.record.approvals[approval_id] = future
        self.record.status = "waiting_approval"
        self.record.push(
            {
                "type": "approval",
                "ts": time.time(),
                "id": approval_id,
                "tool": req.tool,
                "side_effect": req.side_effect.value,
                "description": req.description,
                "args": req.args,
            }
        )
        try:
            decision = await asyncio.wait_for(future, self.timeout_s)
        except TimeoutError:
            decision = ApprovalDecision(
                verdict=Verdict.REJECT, reason=f"no answer within {self.timeout_s:.0f}s"
            )
        finally:
            self.record.approvals.pop(approval_id, None)
            self.record.status = "running"
        self.record.push(
            {
                "type": "approval_done",
                "ts": time.time(),
                "id": approval_id,
                "verdict": decision.verdict.value,
            }
        )
        return decision


def sse(record: TaskRecord) -> Response:
    """A task's events as server-sent events (history first, then live, until 'done')."""
    from sse_starlette.sse import EventSourceResponse

    async def events() -> AsyncIterator[dict[str, str]]:
        async for ev in record.follow():
            yield {"data": json.dumps(ev, default=str)}

    return EventSourceResponse(events())


async def _json_body(request: Request) -> dict[str, Any]:
    if request.headers.get("content-type", "").split(";")[0].strip() != "application/json":
        raise HTTPException(415, "send application/json")
    try:
        body = await request.json()
    except ValueError as exc:
        raise HTTPException(400, "body is not valid JSON") from exc
    if not isinstance(body, dict):
        raise HTTPException(400, "body must be a JSON object")
    return body


def create_app(make_runner: MakeRunner) -> Starlette:
    tasks: dict[str, TaskRecord] = {}
    running: set[asyncio.Task[None]] = set()

    def get_task(request: Request) -> TaskRecord:
        record = tasks.get(request.path_params["task_id"])
        if record is None:
            raise HTTPException(404, "no such task")
        return record

    async def index(request: Request) -> Response:
        return HTMLResponse(PAGE.read_text(encoding="utf-8"))

    async def list_tasks(request: Request) -> Response:
        return JSONResponse([t.summary() for t in reversed(tasks.values())])

    async def start_task(request: Request) -> Response:
        body = await _json_body(request)
        runner_name = str(body.get("runner") or "baseline")
        if runner_name not in RUNNERS:
            raise HTTPException(400, f"runner must be one of {', '.join(RUNNERS)}")
        try:
            max_steps = max(1, min(int(body.get("max_steps") or 20), 60))
            req = TaskRequest(
                prompt=str(body.get("prompt") or "").strip(), budget=Budget(max_steps=max_steps)
            )
        except (ValueError, ValidationError) as exc:
            raise HTTPException(400, f"invalid task: {exc}") from exc
        record = TaskRecord(req, runner_name)
        tasks[req.task_id] = record
        task = record.start(make_runner(runner_name, WebApprover(record)))
        running.add(task)
        task.add_done_callback(running.discard)
        return JSONResponse({"task_id": req.task_id}, status_code=201)

    async def stream(request: Request) -> Response:
        return sse(get_task(request))

    async def answer(request: Request) -> Response:
        record = get_task(request)
        future = record.approvals.get(request.path_params["approval_id"])
        if future is None or future.done():
            raise HTTPException(409, "this approval is no longer waiting")
        try:
            decision = ApprovalDecision.model_validate(await _json_body(request))
        except ValidationError as exc:
            raise HTTPException(400, f"invalid decision: {exc}") from exc
        future.set_result(decision)
        return JSONResponse({"ok": True})

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/tasks", list_tasks, methods=["GET"]),
            Route("/api/tasks", start_task, methods=["POST"]),
            Route("/api/tasks/{task_id}/events", stream),
            Route("/api/tasks/{task_id}/approvals/{approval_id}", answer, methods=["POST"]),
        ],
        middleware=[Middleware(TrustedHostMiddleware, allowed_hosts=["127.0.0.1", "localhost"])],
    )


async def serve(make_runner: MakeRunner, port: int) -> None:
    import uvicorn

    config = uvicorn.Config(
        create_app(make_runner), host="127.0.0.1", port=port, log_level="warning"
    )
    await uvicorn.Server(config).serve()
