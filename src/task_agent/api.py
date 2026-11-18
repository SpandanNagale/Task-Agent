"""Phase 6 HTTP API for programs: ``/v1``. (``ui.py`` is the page for a person on this machine.)

    task-agent keys create my-app --scope tasks:read --scope tasks:write
    task-agent serve --port 8000          then  Authorization: Bearer ta_...

- **API keys** with scopes (``tasks:read``, ``tasks:write``, ``approvals:write``) and an
  optional tool-group allow list. Only a SHA-256 of each key is stored. A key sees only its
  own tasks; anything else is a 404, so task ids cannot be probed.
- **Limits per key**: requests per minute, tasks running at once, tasks per day and spend
  per day (UTC); per-task budgets are clamped to server maximums.
- **Errors** are RFC 9457 ``application/problem+json``.
- **Idempotency-Key** on ``POST /v1/tasks``: a retry with the same key and body returns the
  first task; the same key with a different body is a 422.
- Approvals: keys with ``approvals:write`` get them over the API (rejected after
  ``approval_timeout_s`` with no answer); other keys' side effects are rejected outright.

ponytail: tasks, rate windows and the running set live in this process (one server); keys,
spend and idempotency records are in SQLite. The Redis queue and worker pool replace the
in-process part for several workers.
"""

# No `from __future__ import annotations`: FastAPI must evaluate the Depends(auth(...))
# annotations, which name a closure inside create_api.

import hashlib
import secrets
import sqlite3
import time
from collections import defaultdict, deque
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any, Literal

from fastapi import Depends, FastAPI, Header, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field
from starlette.exceptions import HTTPException
from starlette.middleware.trustedhost import TrustedHostMiddleware

from task_agent.core.approval import ApprovalDecision, Approver, RejectAll
from task_agent.core.runner import Runner
from task_agent.core.types import Budget, TaskRequest
from task_agent.ui import TaskRecord, WebApprover, sse

SCOPES = frozenset({"tasks:read", "tasks:write", "approvals:write"})
MakeRunner = Callable[[str, Approver], Runner]

# Server-wide ceilings for any one task, whatever the request or key asks for.
MAX_BUDGET = Budget(max_steps=60, max_cost_usd=1.0, max_wall_s=1800, max_tokens=1_000_000)
FINISHED_TTL_S = 24 * 3600  # finished tasks are dropped from memory after this

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY, hash TEXT UNIQUE NOT NULL, name TEXT NOT NULL, scopes TEXT NOT NULL,
    groups TEXT, rate_per_min INTEGER NOT NULL, max_running INTEGER NOT NULL,
    daily_tasks INTEGER NOT NULL, daily_cost_usd REAL NOT NULL, created TEXT NOT NULL,
    revoked INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS spend (
    key_id TEXT NOT NULL, day TEXT NOT NULL, tasks INTEGER NOT NULL DEFAULT 0,
    cost_usd REAL NOT NULL DEFAULT 0, PRIMARY KEY (key_id, day));
CREATE TABLE IF NOT EXISTS idempotency (
    key_id TEXT NOT NULL, idem_key TEXT NOT NULL, body_hash TEXT NOT NULL,
    task_id TEXT NOT NULL, created REAL NOT NULL, PRIMARY KEY (key_id, idem_key));
"""


def _hash(secret: str) -> str:
    return hashlib.sha256(secret.encode()).hexdigest()


def _today() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%d")


@dataclass(frozen=True)
class ApiKey:
    id: str
    name: str
    scopes: frozenset[str]
    groups: frozenset[str] | None  # None: every tool group
    rate_per_min: int
    max_running: int
    daily_tasks: int
    daily_cost_usd: float


class KeyStore:
    """API keys, daily spend and idempotency records in one SQLite file."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        con = sqlite3.connect(self.path)
        try:
            with con:
                con.executescript(_SCHEMA)
                yield con
        finally:
            con.close()

    def create(
        self,
        name: str,
        scopes: set[str],
        *,
        groups: set[str] | None = None,
        rate_per_min: int = 30,
        max_running: int = 3,
        daily_tasks: int = 200,
        daily_cost_usd: float = 1.0,
    ) -> tuple[str, str]:
        """Returns (key id, the secret key). The secret is shown once and never stored."""
        unknown = set(scopes) - SCOPES
        if unknown or not scopes:
            raise ValueError(f"scopes must be some of {sorted(SCOPES)}, got {sorted(scopes)}")
        key_id = secrets.token_hex(4)
        secret = f"ta_{key_id}_{secrets.token_urlsafe(32)}"
        with self._connect() as con:
            con.execute(
                "INSERT INTO api_keys VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)",
                (
                    key_id,
                    _hash(secret),
                    name,
                    " ".join(sorted(scopes)),
                    None if groups is None else " ".join(sorted(groups)),
                    rate_per_min,
                    max_running,
                    daily_tasks,
                    daily_cost_usd,
                    datetime.now(UTC).isoformat(timespec="seconds"),
                ),
            )
        return key_id, secret

    def lookup(self, secret: str) -> ApiKey | None:
        with self._connect() as con:
            row = con.execute(
                "SELECT id, name, scopes, groups, rate_per_min, max_running, daily_tasks, "
                "daily_cost_usd FROM api_keys WHERE hash = ? AND revoked = 0",
                (_hash(secret),),
            ).fetchone()
        if row is None:
            return None
        groups = None if row[3] is None else frozenset(row[3].split())
        return ApiKey(row[0], row[1], frozenset(row[2].split()), groups, *row[4:])

    def list(self) -> list[dict[str, Any]]:
        with self._connect() as con:
            con.row_factory = sqlite3.Row
            rows = con.execute(
                "SELECT id, name, scopes, groups, rate_per_min, max_running, daily_tasks, "
                "daily_cost_usd, created, revoked FROM api_keys ORDER BY created"
            ).fetchall()
        return [dict(r) for r in rows]

    def revoke(self, key_id: str) -> bool:
        with self._connect() as con:
            return (
                con.execute("UPDATE api_keys SET revoked = 1 WHERE id = ?", (key_id,)).rowcount > 0
            )

    def usage_today(self, key_id: str) -> tuple[int, float]:
        with self._connect() as con:
            row = con.execute(
                "SELECT tasks, cost_usd FROM spend WHERE key_id = ? AND day = ?",
                (key_id, _today()),
            ).fetchone()
        return (row[0], row[1]) if row else (0, 0.0)

    def charge(self, key_id: str, *, tasks: int = 0, cost_usd: float = 0.0) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT INTO spend (key_id, day, tasks, cost_usd) VALUES (?, ?, ?, ?) "
                "ON CONFLICT (key_id, day) DO UPDATE SET tasks = tasks + excluded.tasks, "
                "cost_usd = cost_usd + excluded.cost_usd",
                (key_id, _today(), tasks, cost_usd),
            )

    def idempotent(self, key_id: str, idem_key: str) -> tuple[str, str] | None:
        """(body hash, task id) stored for this Idempotency-Key, if any (kept 24 h)."""
        with self._connect() as con:
            con.execute("DELETE FROM idempotency WHERE created < ?", (time.time() - 86400,))
            row = con.execute(
                "SELECT body_hash, task_id FROM idempotency WHERE key_id = ? AND idem_key = ?",
                (key_id, idem_key),
            ).fetchone()
        return (row[0], row[1]) if row else None

    def remember(self, key_id: str, idem_key: str, body_hash: str, task_id: str) -> None:
        with self._connect() as con:
            con.execute(
                "INSERT OR REPLACE INTO idempotency VALUES (?, ?, ?, ?, ?)",
                (key_id, idem_key, body_hash, task_id, time.time()),
            )


# --- request and response bodies -----------------------------------------------------------


class BudgetIn(BaseModel):
    max_steps: int | None = Field(default=None, ge=1)
    max_wall_s: float | None = Field(default=None, gt=0)
    max_tokens: int | None = Field(default=None, ge=1)
    max_cost_usd: float | None = Field(default=None, ge=0)


class TaskIn(BaseModel):
    prompt: str = Field(min_length=1, max_length=20_000)
    runner: Literal["baseline", "orchestrator"] = "baseline"
    budget: BudgetIn = Field(default_factory=BudgetIn)
    tool_groups: list[str] | None = Field(
        default=None, description="Narrow the tool groups; the key's allow list still applies."
    )
    metadata: dict[str, str] = Field(default_factory=dict, max_length=20)


def _problem(status: int, title: str, detail: str = "", **extra: Any) -> JSONResponse:
    headers = {"Retry-After": str(extra.pop("retry_after"))} if "retry_after" in extra else None
    body = {"type": "about:blank", "title": title, "status": status, "detail": detail, **extra}
    return JSONResponse(body, status, headers=headers, media_type="application/problem+json")


class Problem(Exception):
    def __init__(self, status: int, title: str, detail: str = "", **extra: Any) -> None:
        super().__init__(title)
        self.status, self.title, self.detail, self.extra = status, title, detail, extra


def _clamp(asked: float | None, default: float | None, ceiling: float | None) -> Any:
    value = asked if asked is not None else default
    if value is None or ceiling is None:
        return value if value is not None else ceiling
    return min(value, ceiling)


def create_api(
    make_runner: MakeRunner,
    keys: KeyStore,
    *,
    allowed_hosts: list[str] | None = None,
    approval_timeout_s: float = 600,
) -> FastAPI:
    tasks: dict[str, tuple[str, TaskRecord]] = {}  # task id -> (owner key id, record)
    windows: dict[str, deque[float]] = defaultdict(deque)  # key id -> recent request times
    running: set[Any] = set()

    app = FastAPI(
        title="Task Agent API",
        version="1",
        description="Run Task Agent tasks, stream their progress and answer approvals.",
    )
    app.add_middleware(
        TrustedHostMiddleware, allowed_hosts=allowed_hosts or ["127.0.0.1", "localhost"]
    )

    @app.exception_handler(Problem)
    async def on_problem(request: Request, exc: Problem) -> Response:
        return _problem(exc.status, exc.title, exc.detail, **exc.extra)

    @app.exception_handler(HTTPException)
    async def on_http(request: Request, exc: HTTPException) -> Response:
        return _problem(exc.status_code, str(exc.detail), "")

    @app.exception_handler(RequestValidationError)
    async def on_invalid(request: Request, exc: RequestValidationError) -> Response:
        errors = [{"loc": list(e["loc"]), "msg": e["msg"]} for e in exc.errors()]
        return _problem(422, "Invalid request", "see errors", errors=errors)

    def auth(scope: str) -> Callable[..., ApiKey]:
        def check(authorization: Annotated[str | None, Header()] = None) -> ApiKey:
            secret = (authorization or "").removeprefix("Bearer ").strip()
            key = keys.lookup(secret) if secret else None
            if key is None:
                raise Problem(401, "Missing or invalid API key", "send Authorization: Bearer <key>")
            if scope not in key.scopes:
                raise Problem(403, "Missing scope", f"this key lacks '{scope}'")
            now = time.monotonic()
            window = windows[key.id]
            while window and window[0] <= now - 60:
                window.popleft()
            if len(window) >= key.rate_per_min:
                retry = int(window[0] + 60 - now) + 1
                raise Problem(
                    429, "Rate limit", f"{key.rate_per_min} requests/minute", retry_after=retry
                )
            window.append(now)
            return key

        return check

    def own(task_id: str, key: ApiKey) -> TaskRecord:
        entry = tasks.get(task_id)
        if entry is None or entry[0] != key.id:
            raise Problem(404, "No such task", task_id)
        return entry[1]

    def view(record: TaskRecord) -> dict[str, Any]:
        out = record.summary()
        if record.result is not None:
            out["output"] = record.result.output
            out["error"] = record.result.error
            out["usage"] = record.result.usage.model_dump()
        return out

    def prune() -> None:
        cutoff = time.time() - FINISHED_TTL_S
        for task_id in [t for t, (_, r) in tasks.items() if r.finished and r.created < cutoff]:
            del tasks[task_id]

    @app.get("/v1/health", tags=["meta"])
    async def health() -> dict[str, Any]:
        return {"ok": True, "tasks_in_memory": len(tasks)}

    @app.post("/v1/tasks", status_code=202, tags=["tasks"])
    async def create_task(
        body: TaskIn,
        key: Annotated[ApiKey, Depends(auth("tasks:write"))],
        idempotency_key: Annotated[str | None, Header(max_length=200)] = None,
    ) -> Response:
        body_hash = _hash(body.model_dump_json())
        if idempotency_key:
            seen = keys.idempotent(key.id, idempotency_key)
            if seen and seen[0] != body_hash:
                raise Problem(422, "Idempotency-Key reused", "this key was used with another body")
            if seen and seen[1] in tasks:
                return JSONResponse(view(tasks[seen[1]][1]), 200)
        prune()
        if (
            sum(1 for owner, r in tasks.values() if owner == key.id and not r.finished)
            >= key.max_running
        ):
            raise Problem(429, "Too many running tasks", f"limit {key.max_running}", retry_after=10)
        done_today, spent = keys.usage_today(key.id)
        if done_today >= key.daily_tasks or spent >= key.daily_cost_usd:
            raise Problem(
                429,
                "Daily limit reached",
                f"{done_today}/{key.daily_tasks} tasks, "
                f"${spent:.4f}/${key.daily_cost_usd} today (UTC)",
                retry_after=3600,
            )
        groups = key.groups
        if body.tool_groups is not None:  # narrows, never widens, the key's allow list
            asked = frozenset(body.tool_groups)
            groups = asked if groups is None else asked & groups
        default = Budget()
        budget = Budget(
            max_steps=_clamp(body.budget.max_steps, default.max_steps, MAX_BUDGET.max_steps),
            max_wall_s=_clamp(body.budget.max_wall_s, default.max_wall_s, MAX_BUDGET.max_wall_s),
            max_tokens=_clamp(body.budget.max_tokens, default.max_tokens, MAX_BUDGET.max_tokens),
            max_cost_usd=_clamp(
                body.budget.max_cost_usd,
                default.max_cost_usd,
                min(MAX_BUDGET.max_cost_usd or 0, key.daily_cost_usd - spent),
            ),
        )
        req = TaskRequest(
            prompt=body.prompt,
            budget=budget,
            allowed_tool_groups=groups,
            metadata={**body.metadata, "api_key": key.id},
        )
        record = TaskRecord(req, body.runner)
        tasks[req.task_id] = (key.id, record)
        keys.charge(key.id, tasks=1)
        if idempotency_key:
            keys.remember(key.id, idempotency_key, body_hash, req.task_id)
        approver: Approver = (
            WebApprover(record, timeout_s=approval_timeout_s)
            if "approvals:write" in key.scopes
            else RejectAll("this API key cannot approve actions")
        )
        task = record.start(make_runner(body.runner, approver))

        def finished(_: Any) -> None:
            running.discard(task)
            if record.result is not None:
                keys.charge(key.id, cost_usd=record.result.usage.cost_usd)

        running.add(task)
        task.add_done_callback(finished)
        return JSONResponse(view(record), 202, headers={"Location": f"/v1/tasks/{req.task_id}"})

    @app.get("/v1/tasks", tags=["tasks"])
    async def list_tasks(
        key: Annotated[ApiKey, Depends(auth("tasks:read"))],
    ) -> list[dict[str, Any]]:
        return [view(r) for owner, r in reversed(tasks.values()) if owner == key.id]

    @app.get("/v1/tasks/{task_id}", tags=["tasks"])
    async def get_task(
        task_id: str, key: Annotated[ApiKey, Depends(auth("tasks:read"))]
    ) -> dict[str, Any]:
        return view(own(task_id, key))

    @app.get("/v1/tasks/{task_id}/events", tags=["tasks"])
    async def events(task_id: str, key: Annotated[ApiKey, Depends(auth("tasks:read"))]) -> Response:
        """Server-sent events: the history so far, then live events, ending with 'done'."""
        return sse(own(task_id, key))

    @app.post("/v1/tasks/{task_id}/cancel", tags=["tasks"])
    async def cancel(
        task_id: str, key: Annotated[ApiKey, Depends(auth("tasks:write"))]
    ) -> dict[str, Any]:
        record = own(task_id, key)
        if record.finished or record.task is None:
            raise Problem(409, "Task already finished", record.status)
        record.task.cancel()
        return {"ok": True}

    @app.post("/v1/tasks/{task_id}/approvals/{approval_id}", tags=["approvals"])
    async def answer(
        task_id: str,
        approval_id: str,
        decision: ApprovalDecision,
        key: Annotated[ApiKey, Depends(auth("approvals:write"))],
    ) -> dict[str, Any]:
        future = own(task_id, key).approvals.get(approval_id)
        if future is None or future.done():
            raise Problem(409, "Approval no longer waiting", approval_id)
        future.set_result(decision)
        return {"ok": True}

    return app
