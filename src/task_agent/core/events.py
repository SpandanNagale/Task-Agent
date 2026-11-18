"""Live progress events (plan, steps, model calls, tool calls) for whoever is watching.

A run emits events into the sink bound to its context; nothing is bound by default, so the
CLI and evals pay nothing. The web UI binds a sink per task. Context variables are copied
into child asyncio tasks, so parallel sub-agents report into their task's sink.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any

Sink = Callable[[dict[str, Any]], None]

_sink: ContextVar[Sink | None] = ContextVar("task_agent_event_sink", default=None)


def emit(kind: str, **data: Any) -> None:
    sink = _sink.get()
    if sink is not None:
        sink({"type": kind, "ts": time.time(), **data})


@contextmanager
def listen(sink: Sink) -> Iterator[None]:
    token = _sink.set(sink)
    try:
        yield
    finally:
        _sink.reset(token)
