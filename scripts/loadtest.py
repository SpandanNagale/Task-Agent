"""Release check: load test of the task server (HTTP + server-sent events), no model quota.

    uv run python -m scripts.loadtest --tasks 200 --model-latency 0.2

Starts the UI server in its own process on a free local port with a fake model (fixed
latency, one tool call then an answer, so each task makes 2 model calls and 1 tool call),
submits every task at once over real HTTP, follows each task's event stream to its end,
and reports throughput, latency percentiles and the server's overhead on top of the model
time. Exits 1 on any error or lost event.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import socket
import statistics
import subprocess
import sys
import time
from typing import Any

import httpx
import uvicorn
from tests.fakes import reply, tool_call

from task_agent.core.approval import Approver
from task_agent.core.baseline import BaselineRunner
from task_agent.core.runner import Runner
from task_agent.core.tool_executor import ToolExecutor
from task_agent.models.gateway import Completion
from task_agent.tools.base import SideEffect, ToolContext, ToolOutput, ToolSpec
from task_agent.tools.registry import ToolRegistry
from task_agent.ui import create_app


class SlowModel:
    """Each task: one tool call, then an answer. Every call waits ``latency`` seconds."""

    def __init__(self, latency: float) -> None:
        self.latency = latency

    async def complete(self, role: str, messages: list[dict[str, Any]], **kw: Any) -> Completion:
        await asyncio.sleep(self.latency)
        done = any(m.get("role") == "tool" for m in messages)
        out = reply("FINAL ANSWER: ok") if done else reply("", [tool_call("echo", {"x": 1})])
        if kw.get("budget") is not None:
            kw["budget"].charge(out.usage)
        return out


async def echo(args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
    return ToolOutput(content="echo")


def make_runner_factory(latency: float):  # type: ignore[no-untyped-def]
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="echo",
            description="Echo the input back. Do not use for anything real.",
            group="code",
            side_effect=SideEffect.READ,
        ),
        echo,
    )
    model = SlowModel(latency)

    def make_runner(kind: str, approver: Approver) -> Runner:
        return BaselineRunner(model, ToolExecutor(reg, approver=approver))  # type: ignore[arg-type]

    return make_runner


async def one_task(base_url: str, i: int) -> tuple[float, int, str]:
    # One client per simulated user: a single shared httpx pool scans all its connections
    # on every request, which made the load generator, not the server, the bottleneck.
    async with httpx.AsyncClient(base_url=base_url, timeout=120) as client:
        return await _one_task(client, i)


async def _one_task(client: httpx.AsyncClient, i: int) -> tuple[float, int, str]:
    t0 = time.perf_counter()
    r = await client.post("/api/tasks", json={"prompt": f"load test task {i}"})
    r.raise_for_status()
    task_id = r.json()["task_id"]
    events, status = 0, "no done event"
    async with client.stream("GET", f"/api/tasks/{task_id}/events") as resp:
        async for line in resp.aiter_lines():
            if line.startswith("data:"):
                events += 1
                ev = json.loads(line[5:])
                if ev["type"] == "done":
                    status = ev["status"]
    return time.perf_counter() - t0, events, status


def pct(values: list[float], p: float) -> float:
    return sorted(values)[min(len(values) - 1, round(p * (len(values) - 1)))]


def serve(port: int, latency: float) -> None:
    app = create_app(make_runner_factory(latency))
    uvicorn.run(app, host="127.0.0.1", port=port, log_level="warning")


async def main(tasks: int, latency: float) -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    server = subprocess.Popen(  # noqa: ASYNC220, S603 - our own interpreter and module
        [
            sys.executable,
            "-m",
            "scripts.loadtest",
            "--serve",
            str(port),
            "--model-latency",
            str(latency),
        ],
        stdout=subprocess.DEVNULL,
    )
    try:
        for _ in range(200):
            try:
                httpx.get(f"{base_url}/api/tasks", timeout=1)  # noqa: ASYNC210 - startup poll
                break
            except httpx.TransportError:
                await asyncio.sleep(0.1)
        t0 = time.perf_counter()
        results = await asyncio.gather(
            *(one_task(base_url, i) for i in range(tasks)), return_exceptions=True
        )
        wall = time.perf_counter() - t0
    finally:
        server.kill()
        server.wait()

    errors = [r for r in results if isinstance(r, BaseException)]
    ok = [r for r in results if not isinstance(r, BaseException)]
    bad = [r for r in ok if r[2] != "succeeded"]
    # start, model, tool, model, done: every task must stream exactly these 5 events
    lost = [r for r in ok if r[1] != 5]
    lat = [r[0] for r in ok]
    model_time = 2 * latency
    print(f"tasks: {tasks} concurrent, fake model latency {latency * 1000:.0f} ms x 2 calls")
    print(
        f"finished: {len(ok) - len(bad)} succeeded, {len(bad)} other status, {len(errors)} errors"
    )
    print(f"events lost: {len(lost)} task(s)")
    if lat:
        print(f"wall: {wall:.2f} s   throughput: {len(ok) / wall:.1f} tasks/s")
        print(
            f"latency p50 {statistics.median(lat):.3f} s  p95 {pct(lat, 0.95):.3f} s  "
            f"max {max(lat):.3f} s  (model time {model_time:.3f} s)"
        )
        print(f"server overhead p95: {(pct(lat, 0.95) - model_time) * 1000:.0f} ms")
    for e in errors[:3]:
        print("error:", repr(e))
    return 1 if errors or bad or lost else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=200)
    ap.add_argument("--model-latency", type=float, default=0.2)
    ap.add_argument("--serve", type=int, help=argparse.SUPPRESS)  # internal: server process
    args = ap.parse_args()
    if args.serve:
        serve(args.serve, args.model_latency)
        sys.exit(0)
    sys.exit(asyncio.run(main(args.tasks, args.model_latency)))
