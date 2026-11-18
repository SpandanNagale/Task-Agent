from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

from task_agent.api import KeyStore, create_api
from task_agent.core.approval import Approver
from task_agent.core.baseline import BaselineRunner
from task_agent.core.runner import Runner
from task_agent.core.tool_executor import ToolExecutor
from task_agent.core.types import TaskRequest
from task_agent.models.gateway import Completion
from task_agent.tools.base import SideEffect, ToolContext, ToolOutput, ToolSpec
from task_agent.tools.registry import ToolRegistry
from tests.fakes import reply, tool_call


class ActThenAnswer:
    """Calls the 'act' tool once, then answers with what it returned. Waits on ``gate``."""

    def __init__(self) -> None:
        self.gate = asyncio.Event()
        self.gate.set()

    async def complete(self, role: str, messages: list[dict[str, Any]], **kw: Any) -> Completion:
        await self.gate.wait()
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        out = (
            reply(f"FINAL ANSWER: {tool_msgs[-1]['content'][:80]}")
            if tool_msgs
            else reply("", [tool_call("act", {})])
        )
        if kw.get("budget") is not None:
            kw["budget"].charge(out.usage)
        return out


async def act(args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
    return ToolOutput(content="acted")


class Harness:
    def __init__(self, tmp_path: Any, side_effect: SideEffect = SideEffect.READ) -> None:
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="act",
                description="Do the one thing this test needs. Nothing else.",
                group="code",
                side_effect=side_effect,
            ),
            act,
        )
        self.model = ActThenAnswer()
        self.requests: list[TaskRequest] = []
        self.keys = KeyStore(tmp_path / "api.db")

        def make_runner(kind: str, approver: Approver) -> Runner:
            runner = BaselineRunner(self.model, ToolExecutor(reg, approver=approver))  # type: ignore[arg-type]
            original = runner.run

            async def run(req: TaskRequest) -> Any:
                self.requests.append(req)
                return await original(req)

            runner.run = run  # type: ignore[method-assign]
            return runner

        self.app = create_api(make_runner, self.keys, approval_timeout_s=5)

    def client(self, secret: str | None = None, host: str = "127.0.0.1") -> httpx.AsyncClient:
        headers = {"Authorization": f"Bearer {secret}"} if secret else {}
        return httpx.AsyncClient(
            transport=httpx.ASGITransport(app=self.app), base_url=f"http://{host}", headers=headers
        )

    def key(self, *scopes: str, **limits: Any) -> str:
        return self.keys.create("t", set(scopes or ("tasks:read", "tasks:write")), **limits)[1]


async def events(client: httpx.AsyncClient, task_id: str) -> list[dict[str, Any]]:
    out = []
    async with client.stream("GET", f"/v1/tasks/{task_id}/events") as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                out.append(json.loads(line[5:]))
    return out


async def test_task_runs_and_streams_to_done(tmp_path) -> None:
    h = Harness(tmp_path)
    async with h.client(h.key()) as c:
        r = await c.post("/v1/tasks", json={"prompt": "do it"})
        assert r.status_code == 202 and r.headers["location"].endswith(r.json()["task_id"])
        task_id = r.json()["task_id"]
        evs = await events(c, task_id)
        assert next(e["type"] for e in evs) == "start" and evs[-1]["type"] == "done"
        got = (await c.get(f"/v1/tasks/{task_id}")).json()
        assert got["status"] == "succeeded" and "acted" in got["output"]
        assert [t["task_id"] for t in (await c.get("/v1/tasks")).json()] == [task_id]


async def test_auth_errors_are_problem_json(tmp_path) -> None:
    h = Harness(tmp_path)
    async with h.client() as c:
        r = await c.get("/v1/tasks")
        assert r.status_code == 401
        assert r.headers["content-type"] == "application/problem+json"
        assert r.json()["status"] == 401 and r.json()["title"]
    async with h.client("ta_wrong") as c:
        assert (await c.get("/v1/tasks")).status_code == 401
    async with h.client(h.key("tasks:read")) as c:
        r = await c.post("/v1/tasks", json={"prompt": "x"})
        assert r.status_code == 403 and "tasks:write" in r.json()["detail"]
    async with h.client(h.key()) as c:
        r = await c.post("/v1/tasks", json={"prompt": ""})
        assert r.status_code == 422 and r.json()["errors"][0]["loc"][-1] == "prompt"
    async with h.client(h.key(), host="evil.example") as c:
        assert (await c.get("/v1/health")).status_code == 400  # DNS rebinding guard


async def test_revoked_key_and_other_keys_tasks(tmp_path) -> None:
    h = Harness(tmp_path)
    mine, theirs = h.key(), h.key()
    async with h.client(mine) as c:
        task_id = (await c.post("/v1/tasks", json={"prompt": "x"})).json()["task_id"]
    async with h.client(theirs) as c:
        assert (await c.get(f"/v1/tasks/{task_id}")).status_code == 404
        assert (await c.get("/v1/tasks")).json() == []
    key_id = h.keys.list()[0]["id"]
    assert h.keys.revoke(key_id)
    async with h.client(mine) as c:
        assert (await c.get("/v1/tasks")).status_code == 401


async def test_idempotency_key(tmp_path) -> None:
    h = Harness(tmp_path)
    async with h.client(h.key()) as c:
        hdr = {"Idempotency-Key": "abc"}
        first = await c.post("/v1/tasks", json={"prompt": "x"}, headers=hdr)
        again = await c.post("/v1/tasks", json={"prompt": "x"}, headers=hdr)
        assert first.status_code == 202 and again.status_code == 200
        assert first.json()["task_id"] == again.json()["task_id"]
        other = await c.post("/v1/tasks", json={"prompt": "y"}, headers=hdr)
        assert other.status_code == 422
    assert len(h.requests) == 1


async def test_rate_running_and_daily_limits(tmp_path) -> None:
    h = Harness(tmp_path)
    async with h.client(h.key(rate_per_min=2)) as c:
        await c.get("/v1/tasks")
        await c.get("/v1/tasks")
        r = await c.get("/v1/tasks")
        assert r.status_code == 429 and int(r.headers["retry-after"]) >= 1

    h.model.gate.clear()  # tasks hang until released
    async with h.client(h.key(max_running=1)) as c:
        assert (await c.post("/v1/tasks", json={"prompt": "a"})).status_code == 202
        r = await c.post("/v1/tasks", json={"prompt": "b"})
        assert r.status_code == 429 and r.json()["title"] == "Too many running tasks"
    h.model.gate.set()

    async with h.client(h.key(daily_tasks=1)) as c:
        task_id = (await c.post("/v1/tasks", json={"prompt": "a"})).json()["task_id"]
        await events(c, task_id)
        r = await c.post("/v1/tasks", json={"prompt": "b"})
        assert r.status_code == 429 and r.json()["title"] == "Daily limit reached"


async def test_budget_clamped_and_groups_narrowed(tmp_path) -> None:
    h = Harness(tmp_path)
    async with h.client(h.key("tasks:read", "tasks:write", groups={"code", "search"})) as c:
        body = {"prompt": "x", "budget": {"max_steps": 999}, "tool_groups": ["search", "comms"]}
        task_id = (await c.post("/v1/tasks", json=body)).json()["task_id"]
        await events(c, task_id)
    req = h.requests[0]
    assert req.budget.max_steps == 60
    assert req.allowed_tool_groups == frozenset({"search"})  # comms is not on the key


@pytest.mark.parametrize("can_approve", [True, False])
async def test_approvals(tmp_path, can_approve: bool) -> None:
    h = Harness(tmp_path, side_effect=SideEffect.WRITE)
    scopes = ("tasks:read", "tasks:write") + (("approvals:write",) if can_approve else ())
    async with h.client(h.key(*scopes)) as c:
        task_id = (await c.post("/v1/tasks", json={"prompt": "x"})).json()["task_id"]
        if can_approve:
            for _ in range(100):
                pending = (await c.get(f"/v1/tasks/{task_id}")).json()["pending_approvals"]
                if pending:
                    break
                await asyncio.sleep(0.01)
            r = await c.post(
                f"/v1/tasks/{task_id}/approvals/{pending[0]}", json={"verdict": "approve"}
            )
            assert r.status_code == 200
        evs = await events(c, task_id)
    tool = next(e for e in evs if e["type"] == "tool")
    assert (tool["output"] or "").startswith("<tool_output") is can_approve


async def test_cancel(tmp_path) -> None:
    h = Harness(tmp_path)
    h.model.gate.clear()
    async with h.client(h.key()) as c:
        task_id = (await c.post("/v1/tasks", json={"prompt": "x"})).json()["task_id"]
        await asyncio.sleep(0.01)
        assert (await c.post(f"/v1/tasks/{task_id}/cancel")).status_code == 200
        evs = await events(c, task_id)
        assert evs[-1] == {**evs[-1], "type": "done", "status": "cancelled"}
        assert (await c.post(f"/v1/tasks/{task_id}/cancel")).status_code == 409


def test_unknown_scope_is_refused(tmp_path) -> None:
    with pytest.raises(ValueError, match="scopes"):
        KeyStore(tmp_path / "k.db").create("x", {"admin"})
