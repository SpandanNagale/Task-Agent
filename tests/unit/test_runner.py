import pytest

from task_agent.core import Budget, TaskRequest, TaskStatus
from task_agent.core.runner import DirectRunner, Runner, extract_final_answer
from task_agent.models import ModelGateway
from tests.conftest import make_gateway


@pytest.mark.parametrize(
    ("text", "answer"),
    [
        ("reasoning\nFINAL ANSWER: 1991", "1991"),
        ("final answer: a\nmore\nFINAL ANSWER:  b ", "b"),
        ("  just text  ", "just text"),
        (None, None),
    ],
)
def test_extract_final_answer(text: str | None, answer: str | None) -> None:
    assert extract_final_answer(text) == answer


async def test_direct_runner_success_is_traced(mock_gateway: ModelGateway, spans) -> None:
    runner = DirectRunner(mock_gateway)
    assert isinstance(runner, Runner)
    req = TaskRequest(prompt="What is 6 x 7?", metadata={"source": "test"})
    result = await runner.run(req)

    assert result.status is TaskStatus.SUCCEEDED
    assert result.output == "42"
    assert result.steps == 1
    assert result.versions.models == {"planner": "openai/mock-planner"}

    finished = {s.name: s for s in spans.get_finished_spans()}
    assert set(finished) == {"task", "llm.planner"}
    root, gen = finished["task"], finished["llm.planner"]
    assert gen.parent is not None
    assert gen.parent.span_id == root.context.span_id
    assert format(root.context.trace_id, "032x") == result.trace_id
    assert root.attributes["task_agent.task_id"] == req.task_id
    assert root.attributes["langfuse.trace.output"] == "42"
    assert root.attributes["langfuse.trace.metadata.source"] == "test"


async def test_direct_runner_model_failure_reports_failed(spans) -> None:
    runner = DirectRunner(make_gateway("x"), role="does-not-exist")
    result = await runner.run(TaskRequest(prompt="hello there"))
    assert result.status is TaskStatus.FAILED
    assert "does-not-exist" in (result.error or "")


async def test_direct_runner_budget(mock_gateway: ModelGateway) -> None:
    runner = DirectRunner(mock_gateway)
    result = await runner.run(TaskRequest(prompt="hello there", budget=Budget(max_tokens=1)))
    assert result.status is TaskStatus.BUDGET_EXCEEDED
    assert "tokens" in (result.error or "")
