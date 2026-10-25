import pytest

from task_agent.core import Budget, BudgetExceeded, BudgetTracker, Usage


class FakeClock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


def test_steps_limit() -> None:
    t = BudgetTracker(Budget(max_steps=2))
    t.start_step()
    t.start_step()
    with pytest.raises(BudgetExceeded) as exc:
        t.start_step()
    assert exc.value.kind == "steps"
    assert t.steps == 2


def test_cost_limit() -> None:
    t = BudgetTracker(Budget(max_cost_usd=0.10))
    t.charge(Usage(cost_usd=0.06, calls=1))
    with pytest.raises(BudgetExceeded) as exc:
        t.charge(Usage(cost_usd=0.06, calls=1))
    assert exc.value.kind == "cost_usd"
    assert t.usage.calls == 2


def test_token_limit() -> None:
    t = BudgetTracker(Budget(max_tokens=100))
    with pytest.raises(BudgetExceeded) as exc:
        t.charge(Usage(prompt_tokens=80, completion_tokens=30))
    assert exc.value.kind == "tokens"


def test_wall_limit_and_remaining() -> None:
    clock = FakeClock()
    t = BudgetTracker(Budget(max_wall_s=10), clock=clock)
    clock.now += 4
    assert t.remaining_s == pytest.approx(6)
    clock.now += 7
    with pytest.raises(BudgetExceeded) as exc:
        t.check()
    assert exc.value.kind == "wall_s"


def test_unbounded() -> None:
    t = BudgetTracker(Budget(max_steps=None, max_cost_usd=None, max_wall_s=None, max_tokens=None))
    for _ in range(1000):
        t.start_step()
    t.charge(Usage(cost_usd=1e6, prompt_tokens=10**9))
    assert t.remaining_s is None
