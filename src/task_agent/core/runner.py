"""Runners turn a TaskRequest into a RunResult.

Everything that executes tasks — the CLI, the eval harness, and (phase 6) queue workers —
depends only on the ``Runner`` protocol, so the agent behind it can change phase by phase
(direct answer -> single-agent tool loop -> orchestrator) without touching its callers, and
the single-agent baseline stays runnable for comparisons.
"""

from __future__ import annotations

import re
import time
from typing import Protocol, runtime_checkable

from opentelemetry.trace import Span

from task_agent import __version__
from task_agent.core.budget import BudgetTracker
from task_agent.core.errors import BudgetExceeded, TaskAgentError
from task_agent.core.types import (
    RunResult,
    RunVersions,
    TaskRequest,
    TaskStatus,
    ToolStats,
    utcnow,
)
from task_agent.models.gateway import ModelClient
from task_agent.observability.logging import bind_task, get_logger, unbind_task
from task_agent.observability.tracing import current_trace_id, record_error, set_io, task_span

log = get_logger(__name__)

_FINAL_ANSWER = re.compile(r"FINAL ANSWER:", re.IGNORECASE)


@runtime_checkable
class Runner(Protocol):
    name: str

    async def run(self, request: TaskRequest) -> RunResult: ...


def extract_final_answer(text: str | None) -> str | None:
    """Take what follows the last 'FINAL ANSWER:' marker, or the whole text if absent."""
    if text is None:
        return None
    matches = list(_FINAL_ANSWER.finditer(text))
    last = matches[-1] if matches else None
    return (text[last.end() :] if last else text).strip()


class RunState:
    """Mutable per-run bookkeeping handed to a runner's ``_execute``."""

    def __init__(self, request: TaskRequest, versions: RunVersions, span: Span) -> None:
        self.request = request
        self.budget = BudgetTracker(request.budget)
        self.versions = versions
        self.tools = ToolStats()
        self.span = span


class BaseRunner:
    """Shared lifecycle: trace + log context, budget, error mapping, and the RunResult.

    Subclasses implement ``_execute`` and return the model's final text.
    """

    name = "base"
    prompt_version = "none"

    async def _execute(self, state: RunState) -> str | None:
        raise NotImplementedError

    async def run(self, request: TaskRequest) -> RunResult:
        started_at = utcnow()
        t0 = time.perf_counter()
        versions = RunVersions(
            agent_version=__version__, runner=self.name, prompt_version=self.prompt_version
        )
        status, output, raw, error = TaskStatus.FAILED, None, None, None
        state: RunState | None = None
        trace_id = None

        bind_task(request.task_id)
        try:
            with task_span(
                request.task_id, request.prompt, metadata={"runner": self.name, **request.metadata}
            ) as span:
                trace_id = current_trace_id()
                state = RunState(request, versions, span)
                log.info("task.started", runner=self.name)
                try:
                    raw = await self._execute(state)
                    output = extract_final_answer(raw)
                    status = TaskStatus.SUCCEEDED
                except BudgetExceeded as exc:
                    status, error = TaskStatus.BUDGET_EXCEEDED, str(exc)
                    record_error(span, exc)
                except TaskAgentError as exc:
                    error = str(exc)
                    record_error(span, exc)
                set_io(span, output=output if error is None else {"error": error}, trace_level=True)
                span.set_attribute("task_agent.status", status.value)
        finally:
            log.info("task.finished", status=status.value, steps=state.budget.steps if state else 0)
            unbind_task()

        assert state is not None
        return RunResult(
            task_id=request.task_id,
            status=status,
            output=output,
            raw_output=raw,
            error=error,
            usage=state.budget.usage,
            tools=state.tools,
            steps=state.budget.steps,
            wall_s=time.perf_counter() - t0,
            trace_id=trace_id,
            versions=versions,
            started_at=started_at,
            finished_at=utcnow(),
        )


DIRECT_SYSTEM_PROMPT = """\
You are a careful assistant completing a task for a user. You have no tools in this mode.
Reason step by step as briefly as the task allows. Then finish with a line of the form
FINAL ANSWER: <answer>
where <answer> is as short as possible: a number, a few words, or the requested text.
Do not put units or thousands separators on numbers unless the task asks for them.
If the task cannot be done without tools or live data you do not have, say so in the
FINAL ANSWER line instead of guessing."""


class DirectRunner(BaseRunner):
    """Phase 0 runner: one model call, no tools.

    Kept as the 'no tools' floor that every later agent must beat on the eval suite.
    """

    name = "direct"
    prompt_version = "direct-v1"

    def __init__(self, gateway: ModelClient, *, role: str = "planner") -> None:
        self.gateway = gateway
        self.role = role

    async def _execute(self, state: RunState) -> str | None:
        state.budget.start_step()
        completion = await self.gateway.complete(
            self.role,
            [
                {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
                {"role": "user", "content": state.request.prompt},
            ],
            budget=state.budget,
        )
        state.versions.models[self.role] = completion.model
        return completion.content
