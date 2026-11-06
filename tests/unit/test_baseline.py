from contextlib import AsyncExitStack
from typing import Any

from task_agent.core import Budget, TaskRequest, TaskStatus
from task_agent.core.approval import Policy, Rule
from task_agent.core.baseline import BaselineRunner
from task_agent.core.runner import Runner
from task_agent.core.tool_executor import ToolExecutor
from task_agent.tools import SideEffect, ToolRegistry, ToolSpec
from task_agent.tools.base import ToolContext, ToolOutput
from tests.fakes import ScriptedModel, reply, tool_call


class Calc:
    """A 'sandbox' tool that also proves per-task resources get cleaned up."""

    def __init__(self) -> None:
        self.closed = 0

    async def _factory(self, stack: AsyncExitStack) -> str:
        async def close() -> None:
            self.closed += 1

        stack.push_async_callback(close)
        return "session"

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        await ctx.resource("sandbox", self._factory)
        return ToolOutput(content=str(eval(args["expr"], {"__builtins__": {}})))


def _runner(model: ScriptedModel, calc: Calc | None = None, **policy: Rule) -> BaselineRunner:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="calc",
            description="Evaluate arithmetic. Do not use for anything else.",
            group="code",
            side_effect=SideEffect.READ,
            input_schema={"type": "object", "required": ["expr"]},
        ),
        calc or Calc(),
    )
    reg.register(
        ToolSpec(
            name="send_email",
            description="Send an email. Do not use without the user's request.",
            group="comms",
            side_effect=SideEffect.IRREVERSIBLE,
        ),
        Calc(),
    )
    return BaselineRunner(model, ToolExecutor(reg, policy=Policy(tools=policy)))


async def test_tool_loop_to_final_answer(spans) -> None:
    calc = Calc()
    model = ScriptedModel(
        reply("let me compute", [tool_call("calc", {"expr": "6*7"})]),
        reply("It is 42.\nFINAL ANSWER: 42"),
    )
    runner = _runner(model, calc)
    assert isinstance(runner, Runner)
    result = await runner.run(TaskRequest(prompt="what is 6 times 7"))

    assert result.status is TaskStatus.SUCCEEDED
    assert result.output == "42"
    assert result.steps == 2
    assert result.tools.calls == 1 and result.tools.errors == 0
    assert calc.closed == 1, "per-task resources must be torn down"

    # The model saw both tools, then the tool result as untrusted data.
    first, second = model.calls
    assert {t["function"]["name"] for t in first["tools"]} == {"calc", "send_email"}
    tool_msg = second["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert 'trust="untrusted"' in tool_msg["content"] and "42" in tool_msg["content"]

    names = [s.name for s in spans.get_finished_spans()]
    assert {"task", "step.1", "step.2", "tool.calc"} <= set(names)


async def test_allowed_groups_limit_tools() -> None:
    model = ScriptedModel(reply("FINAL ANSWER: ok"))
    await _runner(model).run(TaskRequest(prompt="hello there", allowed_tool_groups={"code"}))
    assert [t["function"]["name"] for t in model.calls[0]["tools"]] == ["calc"]


async def test_rejected_side_effect_is_reported_not_run() -> None:
    model = ScriptedModel(
        reply(None, [tool_call("send_email", {"to": "a@b.c"})]),
        reply("I could not send it: approval was not given.\nFINAL ANSWER: not sent"),
    )
    result = await _runner(model).run(TaskRequest(prompt="email a@b.c"))
    assert result.status is TaskStatus.SUCCEEDED
    assert result.tools.approvals_rejected == 1
    assert "rejected" in model.calls[1]["messages"][-1]["content"]


async def test_repeated_identical_calls_stop_the_task() -> None:
    same = [reply(None, [tool_call("calc", {"expr": "1+1"}, f"c{i}")]) for i in range(5)]
    model = ScriptedModel(*same)
    calc = Calc()
    result = await _runner(model, calc).run(TaskRequest(prompt="loop forever please"))
    assert result.status is TaskStatus.FAILED
    assert "same arguments" in (result.error or "")
    warned = model.calls[3]["messages"][-1]["content"]
    assert "Repeating it again will stop the task" in warned
    assert calc.closed == 1


async def test_last_step_withdraws_tools() -> None:
    model = ScriptedModel(
        reply(None, [tool_call("calc", {"expr": "1"})]),
        reply("FINAL ANSWER: 1"),
    )
    result = await _runner(model).run(
        TaskRequest(prompt="count to one", budget=Budget(max_steps=2))
    )
    assert result.status is TaskStatus.SUCCEEDED
    assert model.calls[1]["tools"] is None
    assert "step limit" in model.calls[1]["messages"][-1]["content"]


async def test_empty_reply_gets_one_nudge() -> None:
    model = ScriptedModel(reply(""), reply("FINAL ANSWER: fine"))
    result = await _runner(model).run(TaskRequest(prompt="say something"))
    assert result.output == "fine"

    model = ScriptedModel(reply(""), reply(""))
    result = await _runner(model).run(TaskRequest(prompt="say something"))
    assert result.status is TaskStatus.FAILED


async def test_bad_tool_arguments_go_back_to_the_model() -> None:
    model = ScriptedModel(
        reply(None, [tool_call("calc", "{oops")]),
        reply(None, [tool_call("calc", {"expr": "2+2"}, "c2")]),
        reply("FINAL ANSWER: 4"),
    )
    result = await _runner(model).run(TaskRequest(prompt="2+2"))
    assert result.output == "4"
    assert result.tools.calls == 2 and result.tools.errors == 1
    assert "not valid JSON" in model.calls[1]["messages"][-1]["content"]
