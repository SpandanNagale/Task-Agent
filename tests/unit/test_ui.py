from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx

from task_agent.core.approval import Approver, Policy, Rule
from task_agent.core.baseline import BaselineRunner
from task_agent.core.runner import Runner
from task_agent.core.tool_executor import ToolExecutor
from task_agent.tools.base import SideEffect, ToolContext, ToolOutput, ToolSpec
from task_agent.tools.registry import ToolRegistry
from task_agent.ui import create_app
from tests.fakes import ScriptedModel, reply, tool_call


class Send:
    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        self.sent.append(args)
        return ToolOutput(content="sent")


def client_for(model: ScriptedModel, send: Send) -> httpx.AsyncClient:
    def make_runner(kind: str, approver: Approver) -> Runner:
        reg = ToolRegistry()
        reg.register(
            ToolSpec(
                name="send_note",
                description="Send a note to someone. Do not use unless asked to send one.",
                group="comms",
                side_effect=SideEffect.IRREVERSIBLE,
            ),
            send,
        )
        executor = ToolExecutor(
            reg, policy=Policy(tools={"send_note": Rule.ASK}), approver=approver
        )
        return BaselineRunner(model, executor)

    transport = httpx.ASGITransport(app=create_app(make_runner))
    return httpx.AsyncClient(transport=transport, base_url="http://127.0.0.1")


async def wait_for(client: httpx.AsyncClient, cond: Any) -> dict[str, Any]:
    for _ in range(200):
        tasks = (await client.get("/api/tasks")).json()
        if tasks and cond(tasks[0]):
            return tasks[0]  # type: ignore[no-any-return]
        await asyncio.sleep(0.01)
    raise AssertionError(f"condition never met: {tasks}")


async def test_task_with_an_edited_approval_end_to_end() -> None:
    send = Send()
    model = ScriptedModel(
        reply("sending", [tool_call("send_note", {"to": "bob", "text": "hi"})]),
        reply("FINAL ANSWER: sent"),
    )
    async with client_for(model, send) as client:
        assert "Task Agent" in (await client.get("/")).text
        r = await client.post("/api/tasks", json={"prompt": "send bob a note", "max_steps": 5})
        assert r.status_code == 201
        task_id = r.json()["task_id"]

        task = await wait_for(client, lambda t: t["pending_approvals"])
        assert task["status"] == "waiting_approval"
        (approval_id,) = task["pending_approvals"]
        url = f"/api/tasks/{task_id}/approvals/{approval_id}"
        edited = {"verdict": "edit", "args": {"to": "bob", "text": "hello"}}
        assert (await client.post(url, json=edited)).status_code == 200
        assert (await client.post(url, json=edited)).status_code == 409  # already answered

        done = await wait_for(client, lambda t: t["status"] == "succeeded")
        assert done["steps"] == 2
        assert send.sent == [{"to": "bob", "text": "hello"}]

        body = (await client.get(f"/api/tasks/{task_id}/events")).text
        events = [json.loads(line[5:]) for line in body.splitlines() if line.startswith("data:")]
        kinds = [e["type"] for e in events]
        assert kinds[0] == "start" and kinds[-1] == "done"
        assert {"model", "approval", "approval_done", "tool"} <= set(kinds)
        assert events[-1]["output"] == "sent"


async def test_rejects_cross_site_and_foreign_hosts() -> None:
    async with client_for(ScriptedModel(), Send()) as client:
        form = await client.post(
            "/api/tasks", content="prompt=x", headers={"content-type": "text/plain"}
        )
        assert form.status_code == 415  # simple cross-site requests can't start tasks
        rebind = await client.get("/api/tasks", headers={"host": "evil.example"})
        assert rebind.status_code == 400
        bad = await client.post("/api/tasks", json={"prompt": "x", "runner": "nope"})
        assert bad.status_code == 400
        assert (await client.get("/api/tasks/missing/events")).status_code == 404
