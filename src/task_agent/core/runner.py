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

from task_agent import __version__
from task_agent.core.budget import BudgetTracker
from task_agent.core.errors import BudgetExceeded, TaskAgentError
from task_agent.core.types import RunResult, RunVersions, TaskRequest, TaskStatus, utcnow
from task_agent.models.gateway import ModelGateway
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


DIRECT_PROMPT_VERSION = "direct-v1"
DIRECT_SYSTEM_PROMPT = """\
You are a careful assistant completing a task for a user. You have no tools in this mode.
Reason step by step as briefly as the task allows. Then finish with a line of the form
FINAL ANSWER: <answer>
where <answer> is as short as possible: a number, a few words, or the requested text.
Do not put units or thousands separators on numbers unless the task asks for them.
If the task cannot be done without tools or live data you do not have, say so in the
FINAL ANSWER line instead of guessing."""


class DirectRunner:
    """Phase 0 runner: one model call, no tools.

    It exists to prove the plumbing end to end — gateway, budgets, tracing, eval harness —
    and to give the eval suite a 'no tools' floor that every later agent must beat.
    """

    name = "direct"

    def __init__(self, gateway: ModelGateway, *, role: str = "planner") -> None:
        self.gateway = gateway
        self.role = role

    async def run(self, request: TaskRequest) -> RunResult:
        started_at = utcnow()
        t0 = time.perf_counter()
        budget = BudgetTracker(request.budget)
        versions = RunVersions(
            agent_version=__version__, runner=self.name, prompt_version=DIRECT_PROMPT_VERSION
        )
        status, output, raw, error = TaskStatus.FAILED, None, None, None

        bind_task(request.task_id)
        try:
            with task_span(
                request.task_id, request.prompt, metadata={"runner": self.name, **request.metadata}
            ) as span:
                trace_id = current_trace_id()
                log.info("task.started", runner=self.name)
                try:
                    budget.start_step()
                    completion = await self.gateway.complete(
                        self.role,
                        [
                            {"role": "system", "content": DIRECT_SYSTEM_PROMPT},
                            {"role": "user", "content": request.prompt},
                        ],
                        budget=budget,
                    )
                    versions.models[self.role] = completion.model
                    raw = completion.content
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
            log.info("task.finished", status=status.value, steps=budget.steps)
            unbind_task()

        return RunResult(
            task_id=request.task_id,
            status=status,
            output=output,
            raw_output=raw,
            error=error,
            usage=budget.usage,
            steps=budget.steps,
            wall_s=time.perf_counter() - t0,
            trace_id=trace_id,
            versions=versions,
            started_at=started_at,
            finished_at=utcnow(),
        )
