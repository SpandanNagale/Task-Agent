"""Per-task budget enforcement."""

from __future__ import annotations

import time
from collections.abc import Callable

from task_agent.core.errors import BudgetExceeded
from task_agent.core.types import Budget, Usage


class BudgetTracker:
    """Tracks steps, spend, tokens and wall time for one task and raises when a limit is hit.

    Not shared across tasks or threads; each run owns one. ``check()`` is called before
    each step and after each model call so an overrun is caught at the next boundary.
    """

    def __init__(self, budget: Budget, *, clock: Callable[[], float] = time.monotonic) -> None:
        self.budget = budget
        self._monotonic = clock
        self._started = self._monotonic()
        self.steps = 0
        self.usage = Usage()

    @property
    def elapsed_s(self) -> float:
        return self._monotonic() - self._started

    @property
    def remaining_s(self) -> float | None:
        if self.budget.max_wall_s is None:
            return None
        return max(0.0, self.budget.max_wall_s - self.elapsed_s)

    def start_step(self) -> None:
        """Count a new step. Raises if it would exceed max_steps."""
        b = self.budget
        if b.max_steps is not None and self.steps >= b.max_steps:
            raise BudgetExceeded("steps", b.max_steps, self.steps + 1)
        self.check()
        self.steps += 1

    def charge(self, usage: Usage) -> None:
        self.usage = self.usage + usage
        self.check()

    def check(self) -> None:
        b = self.budget
        if b.max_cost_usd is not None and self.usage.cost_usd > b.max_cost_usd:
            raise BudgetExceeded("cost_usd", b.max_cost_usd, self.usage.cost_usd)
        if b.max_tokens is not None and self.usage.total_tokens > b.max_tokens:
            raise BudgetExceeded("tokens", b.max_tokens, self.usage.total_tokens)
        if b.max_wall_s is not None and self.elapsed_s > b.max_wall_s:
            raise BudgetExceeded("wall_s", b.max_wall_s, self.elapsed_s)
