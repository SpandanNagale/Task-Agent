"""Shared data contracts.

These models cross process boundaries (API -> queue -> worker -> event stream), so they are
plain, JSON-serialisable pydantic models with no live handles in them. Keep them stable:
changing a field here is an API change.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

BudgetKind = Literal["steps", "cost_usd", "wall_s", "tokens"]


def new_task_id() -> str:
    return f"task_{uuid.uuid4().hex}"


def utcnow() -> datetime:
    return datetime.now(UTC)


class Budget(BaseModel):
    """Hard per-task limits. Any limit set to None is unbounded."""

    model_config = ConfigDict(frozen=True)

    # Defaults are sized for free-tier models: cost is ~0, so tokens and wall time are the
    # limits that matter (free tiers cap tokens per minute/day). The cost cap is a guard in
    # case a fallback lands on a paid model.
    max_steps: int | None = Field(default=20, ge=1)
    max_cost_usd: float | None = Field(default=0.10, ge=0)
    max_wall_s: float | None = Field(default=300.0, gt=0)
    max_tokens: int | None = Field(default=300_000, ge=1)


class Usage(BaseModel):
    """Token and cost accounting, additive across calls."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0  # what was actually spent (free tiers count as 0)
    list_cost_usd: float = 0.0  # the same usage at the provider's paid list price
    calls: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: Usage) -> Usage:
        return Usage(
            prompt_tokens=self.prompt_tokens + other.prompt_tokens,
            completion_tokens=self.completion_tokens + other.completion_tokens,
            cost_usd=self.cost_usd + other.cost_usd,
            list_cost_usd=self.list_cost_usd + other.list_cost_usd,
            calls=self.calls + other.calls,
        )


class TaskStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    BUDGET_EXCEEDED = "budget_exceeded"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {
            TaskStatus.SUCCEEDED,
            TaskStatus.FAILED,
            TaskStatus.BUDGET_EXCEEDED,
            TaskStatus.CANCELLED,
        }


class TaskRequest(BaseModel):
    """What a caller submits. Mirrors the body of POST /v1/tasks."""

    model_config = ConfigDict(frozen=True)

    task_id: str = Field(default_factory=new_task_id)
    prompt: str = Field(min_length=1)
    budget: Budget = Field(default_factory=Budget)
    # None: every tool group the runner has. A set narrows it (API key scopes, evals).
    allowed_tool_groups: frozenset[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RunVersions(BaseModel):
    """What a run executed with, so any result can be reproduced."""

    agent_version: str
    runner: str
    prompt_version: str
    models: dict[str, str] = Field(default_factory=dict)  # role -> model actually served


class ToolStats(BaseModel):
    """Per-run tool accounting; feeds the tool-error-rate and approval metrics."""

    calls: int = 0
    errors: int = 0
    approvals_requested: int = 0
    approvals_edited: int = 0
    approvals_rejected: int = 0
    blocked_by_policy: int = 0


class RunResult(BaseModel):
    task_id: str
    status: TaskStatus
    output: str | None = None  # the final answer
    raw_output: str | None = None  # full model text the answer was extracted from
    error: str | None = None
    usage: Usage = Field(default_factory=Usage)
    tools: ToolStats = Field(default_factory=ToolStats)
    steps: int = 0
    wall_s: float = 0.0
    trace_id: str | None = None
    versions: RunVersions
    started_at: datetime
    finished_at: datetime
