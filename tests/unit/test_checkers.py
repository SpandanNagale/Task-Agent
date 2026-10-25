import pytest

from task_agent.evals.checkers import check, check_static, first_number, normalize
from task_agent.evals.schema import (
    Category,
    ContainsChecker,
    EvalTask,
    ExactChecker,
    LLMJudgeChecker,
    NumericChecker,
    OneOfChecker,
    RegexChecker,
)
from tests.conftest import make_gateway


def test_normalize() -> None:
    assert normalize('  "Example   Domain." ') == "example domain"
    assert normalize("Paris", case_sensitive=True) == "Paris"


@pytest.mark.parametrize(
    ("text", "num"),
    [("about 3,812 m", 3812.0), ("-2.5", -2.5), ("1e3 things", 1000.0), ("none", None)],
)
def test_first_number(text: str, num: float | None) -> None:
    assert first_number(text) == num


@pytest.mark.parametrize(
    ("checker", "answer", "ok"),
    [
        (ExactChecker(expected="Example Domain"), "example domain.", True),
        (ExactChecker(expected="Example Domain", case_sensitive=True), "example domain", False),
        (OneOfChecker(options=["BD", "B and D"]), "b and d", True),
        (ContainsChecker(all_of=["Guido"], none_of=["Larry"]), "Guido van Rossum", True),
        (ContainsChecker(all_of=["Guido"], none_of=["Larry"]), "Guido or Larry", False),
        (ContainsChecker(any_of=["a", "b"]), "xyz", False),
        (RegexChecker(pattern=r"\b2606\b"), "RFC 2606", True),
        (NumericChecker(expected=62.5, abs_tol=0.001), "$62.50", True),
        (NumericChecker(expected=3812, abs_tol=15), "3,800 m", True),
        (NumericChecker(expected=190), "two hundred", False),
    ],
)
def test_static_checkers(checker, answer: str, ok: bool) -> None:
    assert check_static(checker, answer).passed is ok


def _judge_task() -> EvalTask:
    return EvalTask(
        id="judge-001",
        category=Category.COMMUNICATION,
        prompt="Write a haiku about rain.",
        checker=LLMJudgeChecker(rubric="PASS if it is a haiku about rain."),
    )


async def test_llm_judge_pass_and_fail() -> None:
    task = _judge_task()
    assert (await check(task, "drops", make_gateway("x", judge="good\n**PASS**"))).passed
    assert not (await check(task, "drops", make_gateway("x", judge="bad\nFAIL"))).passed


async def test_empty_answer_fails_without_calling_judge() -> None:
    result = await check(_judge_task(), "   ", make_gateway("x"))
    assert not result.passed
    assert result.detail == "empty answer"
