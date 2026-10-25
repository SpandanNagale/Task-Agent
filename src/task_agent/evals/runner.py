"""Run a set of eval tasks through a Runner, grade them, and summarise.

Tasks run concurrently up to a limit. A task whose required tool groups are not available
to the runner is recorded as skipped, not failed, so the same suite serves every phase.
"""

from __future__ import annotations

import asyncio
import statistics
import uuid
from collections import defaultdict
from collections.abc import Iterable, Sequence
from enum import StrEnum

from pydantic import BaseModel, Field

from task_agent import __version__
from task_agent.core.runner import Runner
from task_agent.core.types import Budget, RunResult, TaskRequest, TaskStatus, utcnow
from task_agent.evals.checkers import CheckResult, check
from task_agent.evals.schema import EvalTask
from task_agent.models.gateway import ModelGateway
from task_agent.observability.logging import get_logger

log = get_logger(__name__)


class Outcome(StrEnum):
    PASSED = "passed"
    FAILED = "failed"  # ran, wrong answer
    ERROR = "error"  # the run itself failed (model error, budget, crash)
    SKIPPED = "skipped"  # needs tool groups this runner does not have


class TaskOutcome(BaseModel):
    task_id: str
    category: str
    outcome: Outcome
    detail: str = ""
    answer: str | None = None
    run: RunResult | None = None
    check: CheckResult | None = None


class Metrics(BaseModel):
    total: int = 0
    passed: int = 0
    failed: int = 0
    errors: int = 0
    skipped: int = 0
    success_rate: float | None = None  # passed / (total - skipped)
    cost_usd_total: float = 0.0
    cost_usd_mean: float | None = None
    steps_mean: float | None = None
    wall_s_mean: float | None = None
    wall_s_p95: float | None = None


class EvalReport(BaseModel):
    run_id: str
    label: str = ""
    agent_version: str = __version__
    runner: str
    started_at: str
    finished_at: str
    available_groups: list[str]
    overall: Metrics
    by_category: dict[str, Metrics]
    judge_cost_usd: float = 0.0
    results: list[TaskOutcome] = Field(default_factory=list)


def summarise(outcomes: Sequence[TaskOutcome]) -> Metrics:
    m = Metrics(total=len(outcomes))
    for o in outcomes:
        match o.outcome:
            case Outcome.PASSED:
                m.passed += 1
            case Outcome.FAILED:
                m.failed += 1
            case Outcome.ERROR:
                m.errors += 1
            case Outcome.SKIPPED:
                m.skipped += 1
    attempted = m.total - m.skipped
    runs = [o.run for o in outcomes if o.run is not None]
    if attempted:
        m.success_rate = m.passed / attempted
    if runs:
        costs = [r.usage.cost_usd for r in runs]
        walls = sorted(r.wall_s for r in runs)
        m.cost_usd_total = sum(costs)
        m.cost_usd_mean = statistics.fmean(costs)
        m.steps_mean = statistics.fmean(r.steps for r in runs)
        m.wall_s_mean = statistics.fmean(walls)
        m.wall_s_p95 = walls[min(len(walls) - 1, round(0.95 * (len(walls) - 1)))]
    return m


async def _run_one(
    task: EvalTask,
    runner: Runner,
    gateway: ModelGateway,
    available: frozenset[str],
    default_budget: Budget,
    run_id: str,
    sem: asyncio.Semaphore,
) -> TaskOutcome:
    base = {"task_id": task.id, "category": task.category.value}
    missing = task.requires - available
    if missing:
        return TaskOutcome(
            **base, outcome=Outcome.SKIPPED, detail=f"needs groups {sorted(missing)}"
        )

    async with sem:
        request = TaskRequest(
            prompt=task.prompt,
            budget=task.budget or default_budget,
            allowed_tool_groups=available,
            metadata={"eval_task_id": task.id, "eval_run_id": run_id},
        )
        try:
            result = await runner.run(request)
        except Exception as exc:  # a crash in one task must not sink the whole run
            log.exception("eval.task_crashed", eval_task_id=task.id)
            return TaskOutcome(**base, outcome=Outcome.ERROR, detail=f"crash: {exc!r}")

        if result.status is not TaskStatus.SUCCEEDED:
            return TaskOutcome(
                **base,
                outcome=Outcome.ERROR,
                detail=f"{result.status.value}: {result.error or ''}".strip(),
                run=result,
            )
        try:
            graded = await check(task, result.output, gateway)
        except Exception as exc:
            log.warning("eval.check_failed", eval_task_id=task.id, error=repr(exc))
            return TaskOutcome(
                **base,
                outcome=Outcome.ERROR,
                detail=f"checker error: {exc!r}",
                answer=result.output,
                run=result,
            )
        return TaskOutcome(
            **base,
            outcome=Outcome.PASSED if graded.passed else Outcome.FAILED,
            detail=graded.detail,
            answer=result.output,
            run=result,
            check=graded,
        )


async def run_eval(
    tasks: Iterable[EvalTask],
    runner: Runner,
    gateway: ModelGateway,
    *,
    available_groups: Iterable[str] = (),
    concurrency: int = 4,
    default_budget: Budget | None = None,
    label: str = "",
) -> EvalReport:
    run_id = f"eval_{utcnow():%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:6]}"
    available = frozenset(available_groups)
    sem = asyncio.Semaphore(max(1, concurrency))
    started = utcnow()
    task_list = list(tasks)
    log.info("eval.started", run_id=run_id, tasks=len(task_list), runner=runner.name)

    outcomes = await asyncio.gather(
        *(
            _run_one(t, runner, gateway, available, default_budget or Budget(), run_id, sem)
            for t in task_list
        )
    )

    by_cat: dict[str, list[TaskOutcome]] = defaultdict(list)
    for o in outcomes:
        by_cat[o.category].append(o)

    report = EvalReport(
        run_id=run_id,
        label=label,
        runner=runner.name,
        started_at=started.isoformat(),
        finished_at=utcnow().isoformat(),
        available_groups=sorted(available),
        overall=summarise(outcomes),
        by_category={c: summarise(v) for c, v in sorted(by_cat.items())},
        judge_cost_usd=sum(o.check.judge_cost_usd for o in outcomes if o.check),
        results=list(outcomes),
    )
    log.info(
        "eval.finished",
        run_id=run_id,
        passed=report.overall.passed,
        attempted=report.overall.total - report.overall.skipped,
    )
    return report
