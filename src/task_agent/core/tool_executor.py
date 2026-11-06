"""Runs tool calls on behalf of an agent: policy check, approval, timeouts, retries, tracing.

This is the only path from a model's tool call to a side effect, so every safety rule lives
here rather than in individual tools:

- unknown tools, malformed arguments and missing required fields come back as errors the
  model can read and correct, never as crashes
- the policy decides allow / ask / deny; ``ask`` goes to the Approver, whose edits replace
  the arguments
- read-only tools retry transient failures with exponential backoff and jitter; tools with
  side effects never retry, so nothing is sent twice
- outputs are size-capped and wrapped as untrusted data before the model sees them
"""

from __future__ import annotations

import asyncio
import json
import random
from typing import Any

from pydantic import BaseModel

from task_agent.core.approval import (
    ApprovalRequest,
    Approver,
    Policy,
    RejectAll,
    Rule,
    Verdict,
)
from task_agent.core.errors import ToolError, TransientToolError
from task_agent.core.types import ToolStats
from task_agent.observability.logging import get_logger
from task_agent.observability.tracing import record_error, set_io, tool_span
from task_agent.tools.base import SideEffect, ToolContext, ToolOutput
from task_agent.tools.registry import ToolRegistry

log = get_logger(__name__)

_READ_RETRIES = 2
_BACKOFF_BASE_S = 0.5


class ToolCallResult(BaseModel):
    name: str
    content: str  # what goes back to the model (already wrapped)
    is_error: bool = False
    files: list[str] = []


def wrap_untrusted(name: str, text: str) -> str:
    """Mark tool output as data. The closing tag is escaped so content cannot break out."""
    safe = text.replace("</tool_output", "<\\/tool_output")
    return f'<tool_output tool="{name}" trust="untrusted">\n{safe}\n</tool_output>'


def _truncate(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[:limit] + f"\n...[truncated {len(text) - limit} characters]"


def _missing_required(schema: dict[str, Any], args: dict[str, Any]) -> list[str]:
    required = schema.get("required") or []
    return [key for key in required if key not in args]


class ToolExecutor:
    def __init__(
        self,
        registry: ToolRegistry,
        *,
        policy: Policy | None = None,
        approver: Approver | None = None,
        max_output_chars: int = 12_000,
    ) -> None:
        self.registry = registry
        self.policy = policy or Policy()
        self.approver: Approver = approver or RejectAll()
        self.max_output_chars = max_output_chars

    async def execute(
        self,
        name: str,
        raw_args: str | dict[str, Any] | None,
        ctx: ToolContext,
        stats: ToolStats,
        *,
        allowed_groups: frozenset[str] | None = None,
    ) -> ToolCallResult:
        stats.calls += 1
        result = await self._execute(name, raw_args, ctx, stats, allowed_groups)
        if result.is_error:
            stats.errors += 1
        return result

    def _error(self, name: str, message: str) -> ToolCallResult:
        return ToolCallResult(name=name, content=f"ERROR: {message}", is_error=True)

    async def _execute(
        self,
        name: str,
        raw_args: str | dict[str, Any] | None,
        ctx: ToolContext,
        stats: ToolStats,
        allowed_groups: frozenset[str] | None,
    ) -> ToolCallResult:
        if name not in self.registry:
            known = ", ".join(s.name for s in self.registry.list_tools(allowed_groups))
            return self._error(name, f"unknown tool '{name}'. Available tools: {known}")
        spec = self.registry.get(name)
        if allowed_groups is not None and spec.group not in allowed_groups:
            return self._error(name, f"tool '{name}' is not allowed for this task")

        try:
            args = json.loads(raw_args or "{}") if isinstance(raw_args, str) else raw_args or {}
        except json.JSONDecodeError as exc:
            return self._error(name, f"arguments are not valid JSON ({exc}); send a JSON object")
        if not isinstance(args, dict):
            return self._error(name, "arguments must be a JSON object")
        missing = _missing_required(spec.input_schema, args)
        if missing:
            return self._error(name, f"missing required arguments: {', '.join(missing)}")

        rule = self.policy.decide(spec)
        if rule is Rule.DENY:
            stats.blocked_by_policy += 1
            return self._error(name, "blocked by policy. Do not retry; tell the user instead.")
        if rule is Rule.ASK:
            stats.approvals_requested += 1
            decision = await self.approver.request(
                ApprovalRequest(
                    task_id=ctx.task_id,
                    tool=name,
                    side_effect=spec.side_effect,
                    description=spec.description,
                    args=args,
                )
            )
            log.info("tool.approval", tool=name, verdict=decision.verdict.value)
            if decision.verdict is Verdict.REJECT:
                stats.approvals_rejected += 1
                why = f" Reason: {decision.reason}" if decision.reason else ""
                return self._error(
                    name,
                    "the user rejected this action." + why + " Do not retry the same action; "
                    "adjust the plan or report what you would have done.",
                )
            if decision.verdict is Verdict.EDIT and decision.args is not None:
                stats.approvals_edited += 1
                args = decision.args

        handler = self.registry.handler(name)
        attempts = 1 + (_READ_RETRIES if spec.side_effect is SideEffect.READ else 0)
        with tool_span(name, spec.side_effect.value, args) as span:
            for attempt in range(1, attempts + 1):
                try:
                    output: ToolOutput = await asyncio.wait_for(
                        handler(args, ctx), timeout=spec.timeout_s
                    )
                    break
                except (TransientToolError, TimeoutError) as exc:
                    if attempt < attempts:
                        delay = _BACKOFF_BASE_S * 2 ** (attempt - 1) * (1 + random.random())  # noqa: S311
                        log.info("tool.retry", tool=name, attempt=attempt, error=str(exc))
                        await asyncio.sleep(delay)
                        continue
                    record_error(span, exc)
                    msg = str(exc) or f"timed out after {spec.timeout_s:.0f}s"
                    return self._error(name, msg)
                except ToolError as exc:
                    record_error(span, exc)
                    return self._error(name, str(exc))
                except Exception as exc:  # a tool bug must not kill the task
                    record_error(span, exc)
                    log.exception("tool.crashed", tool=name)
                    return self._error(name, f"tool failed unexpectedly ({type(exc).__name__})")

            content = _truncate(output.content, self.max_output_chars)
            set_io(span, output=content)
        return ToolCallResult(name=name, content=wrap_untrusted(name, content), files=output.files)
