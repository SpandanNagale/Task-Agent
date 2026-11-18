from __future__ import annotations

import json
from typing import Any

from task_agent.core.orchestrator import (
    OrchestratorRunner,
    Step,
    load_request,
    parse_json_object,
)
from task_agent.core.tool_executor import ToolExecutor
from task_agent.core.types import Budget, TaskRequest, TaskStatus
from task_agent.tools.base import SideEffect, ToolContext, ToolOutput, ToolSpec
from task_agent.tools.registry import ToolRegistry
from tests.fakes import ScriptedModel, reply, tool_call


class Calc:
    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return ToolOutput(content=str(eval(args["expr"], {"__builtins__": {}})))


def make_runner(model: ScriptedModel, **kw: Any) -> OrchestratorRunner:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="calc",
            description="Evaluate arithmetic. Do not use for anything but plain arithmetic.",
            group="code",
            side_effect=SideEffect.READ,
            input_schema={"type": "object", "required": ["expr"]},
        ),
        Calc(),
    )
    return OrchestratorRunner(model, ToolExecutor(reg), **kw)


def plan(*steps: dict[str, Any]) -> Any:
    return reply(json.dumps({"steps": list(steps)}))


def final(answer: str) -> Any:
    return reply(json.dumps({"final_answer": answer}))


def test_parse_json_object_tolerates_wrappers() -> None:
    text = '<think>{"no": 1}</think>Sure:\n```json\n{"final_answer": "x"}\n```'
    assert parse_json_object(text) == {"final_answer": "x"}


async def test_planner_can_answer_directly() -> None:
    model = ScriptedModel(final("42"))
    result = await make_runner(model).run(TaskRequest(prompt="6*7?"))
    assert result.status is TaskStatus.SUCCEEDED
    assert result.output == "42"
    assert result.steps == 1


async def test_fan_out_then_dependent_step_then_finish() -> None:
    model = ScriptedModel(
        plan(
            {"id": "a", "role": "coder", "goal": "compute 6*7"},
            {"id": "b", "role": "writer", "goal": "say hi"},
            {"id": "c", "role": "writer", "goal": "combine", "depends_on": ["a", "b"]},
        ),
        reply("using calc", [tool_call("calc", {"expr": "6*7"})]),  # a (has tools)
        reply("FINAL ANSWER: 42"),  # a
        reply("FINAL ANSWER: hi"),  # b
        reply("FINAL ANSWER: hi 42"),  # c
        final("hi 42"),
    )
    result = await make_runner(model).run(TaskRequest(prompt="task"))
    assert result.status is TaskStatus.SUCCEEDED, result.error
    assert result.output == "hi 42"
    assert result.tools.calls == 1
    planner, a, *_ = model.calls
    assert planner["role"] == "planner" and planner["tools"] is None
    # the planner sees each group's tools, so groups no preset covers still get used
    assert "- code: calc: Evaluate arithmetic" in planner["messages"][0]["content"]
    assert a["role"] == "worker" and [t["function"]["name"] for t in a["tools"]] == ["calc"]
    writer_c = model.calls[-2]
    assert writer_c["tools"] is None  # writer gets no tools
    brief = writer_c["messages"][1]["content"]
    assert '<step_result id="a"' in brief and "42" in brief
    assert "task" not in brief  # sub-agents never see the original task


async def test_invalid_plan_gets_one_corrective_retry() -> None:
    model = ScriptedModel(reply("I think the answer is 42"), final("42"))
    result = await make_runner(model).run(TaskRequest(prompt="q"))
    assert result.output == "42"
    assert "not valid" in model.calls[1]["messages"][-1]["content"]


async def test_planner_tool_call_is_told_to_make_a_step() -> None:
    # gpt-oss answers the planner with its built-in browser call though it has no tools
    model = ScriptedModel(reply("", [tool_call("browser.open", {"url": "u"})]), final("42"))
    result = await make_runner(model).run(TaskRequest(prompt="q"))
    assert result.output == "42"
    assert "you tried to call browser.open" in model.calls[1]["messages"][-1]["content"]


def test_nested_role_object_is_flattened() -> None:
    step = Step.model_validate(
        {"id": "s1", "goal": "g", "role": {"name": "reader", "instructions": "i", "tools": ["x"]}}
    )
    assert (step.role, step.instructions, step.tools) == ("reader", "i", ["x"])


async def test_failed_step_triggers_replan() -> None:
    model = ScriptedModel(
        plan({"id": "s1", "role": "writer", "goal": "x"}),
        reply(""),  # empty
        reply(""),  # empty again -> step fails
        plan({"id": "s1", "role": "writer", "goal": "x again"}),  # id clash gets renamed
        reply("FINAL ANSWER: done"),
        final("done"),
    )
    result = await make_runner(model).run(TaskRequest(prompt="q"))
    assert result.status is TaskStatus.SUCCEEDED, result.error
    assert result.output == "done"
    review = model.calls[3]["messages"][-1]["content"]
    assert 'status="failed"' in review


async def test_replan_limit_forces_an_answer() -> None:
    model = ScriptedModel(
        plan({"id": "s1", "role": "writer", "goal": "x"}),
        reply("FINAL ANSWER: partial"),
        plan({"id": "s2", "role": "writer", "goal": "more"}),
    )
    result = await make_runner(model, max_replans=0).run(TaskRequest(prompt="q"))
    assert result.status is TaskStatus.FAILED
    assert "replan limit" in (result.error or "")


async def test_resume_skips_finished_steps(tmp_path) -> None:
    db = tmp_path / "cp.db"
    steps = (
        {"id": "a", "role": "writer", "goal": "first"},
        {"id": "b", "role": "writer", "goal": "second", "depends_on": ["a"]},
    )
    request = TaskRequest(prompt="q", budget=Budget(max_steps=2))
    model = ScriptedModel(plan(*steps), reply("FINAL ANSWER: A"))
    first = await make_runner(model, checkpoint_db=db).run(request)
    assert first.status is TaskStatus.BUDGET_EXCEEDED  # crashed before step b

    saved = await load_request(db, request.task_id)
    assert saved is not None and saved.prompt == "q"

    model = ScriptedModel(reply("FINAL ANSWER: B"), final("A B"))  # no re-plan, no step a
    retry = saved.model_copy(update={"budget": Budget(max_steps=5)})
    second = await make_runner(model, checkpoint_db=db).run(retry)
    assert second.status is TaskStatus.SUCCEEDED, second.error
    assert second.output == "A B"
    assert '<step_result id="a"' in model.calls[0]["messages"][1]["content"]


async def test_runtime_roles_without_tools_still_get_tools() -> None:
    model = ScriptedModel(
        plan(
            {"id": "a", "role": "code", "goal": "role named after a group"},
            {"id": "b", "role": "analyst", "goal": "invented role, no tools listed"},
        ),
        reply("FINAL ANSWER: a"),
        reply("FINAL ANSWER: b"),
        final("done"),
    )
    result = await make_runner(model).run(TaskRequest(prompt="q"))
    assert result.status is TaskStatus.SUCCEEDED, result.error
    for call in model.calls[1:3]:
        assert [t["function"]["name"] for t in call["tools"]] == ["calc"]


async def test_live_events_cover_plan_steps_and_tools() -> None:
    from task_agent.core.events import listen

    model = ScriptedModel(
        plan({"id": "a", "role": "coder", "goal": "compute 6*7"}),
        reply("using calc", [tool_call("calc", {"expr": "6*7"})]),
        reply("FINAL ANSWER: 42"),
        final("42"),
    )
    events: list[dict[str, Any]] = []
    with listen(events.append):
        await make_runner(model).run(TaskRequest(prompt="task"))
    seen = [(e["type"], e.get("id") or e.get("agent") or e.get("status")) for e in events]
    assert ("plan", None) in seen
    assert ("step", "a") in seen and ("tool", "a") in seen  # through LangGraph's tasks
    assert [e["status"] for e in events if e["type"] == "step"] == ["running", "done"]
    assert events[-1] == {**events[-1], "type": "answer", "text": "42"}
