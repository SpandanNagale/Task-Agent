"""Test doubles shared across test modules."""

from __future__ import annotations

import json
from typing import Any

from task_agent.core.budget import BudgetTracker
from task_agent.core.types import Usage
from task_agent.models.gateway import Completion


def tool_call(name: str, args: dict[str, Any] | str, call_id: str = "c1") -> dict[str, Any]:
    arguments = args if isinstance(args, str) else json.dumps(args)
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": arguments}}


def reply(content: str | None = None, calls: list[dict[str, Any]] | None = None) -> Completion:
    return Completion(
        role="planner",
        model="fake/model",
        content=content,
        tool_calls=calls or [],
        usage=Usage(prompt_tokens=10, completion_tokens=5, calls=1),
        latency_s=0.0,
    )


class ScriptedModel:
    """A ModelClient that replays scripted completions and records what it was sent."""

    def __init__(self, *script: Completion) -> None:
        self.script = list(script)
        self.calls: list[dict[str, Any]] = []

    async def complete(
        self,
        role: str,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        budget: BudgetTracker | None = None,
        timeout_s: float | None = None,
        **params: Any,
    ) -> Completion:
        self.calls.append({"role": role, "messages": [dict(m) for m in messages], "tools": tools})
        if not self.script:
            raise AssertionError("ScriptedModel ran out of replies")
        completion = self.script.pop(0)
        if budget is not None:
            budget.charge(completion.usage)
        return completion
