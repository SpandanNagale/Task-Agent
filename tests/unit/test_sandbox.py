from contextlib import AsyncExitStack
from types import SimpleNamespace
from typing import Any

import pytest

from task_agent.core import ToolError
from task_agent.tools.base import ToolContext
from task_agent.tools.builtin.sandbox import E2BSession, RunPython, format_execution


class FakeSession:
    def __init__(self) -> None:
        self.codes: list[str] = []

    async def run_python(self, code: str, timeout_s: float) -> dict[str, Any]:
        self.codes.append(code)
        return {"stdout": "hi\n", "stderr": "", "error": None, "result": "42"}


async def test_one_sandbox_per_task_and_cleanup() -> None:
    created, killed = [], []

    async def factory(stack: AsyncExitStack) -> FakeSession:
        session = FakeSession()
        created.append(session)

        async def kill() -> None:
            killed.append(session)

        stack.push_async_callback(kill)
        return session

    tool = RunPython(factory)
    ctx = ToolContext("t")
    out1 = await tool({"code": "print('hi')"}, ctx)
    await tool({"code": "x = 1"}, ctx)
    assert len(created) == 1 and created[0].codes == ["print('hi')", "x = 1"]
    assert "stdout:\nhi" in out1.content and "result:\n42" in out1.content
    await ctx.aclose()
    assert killed == created


async def test_empty_code_rejected() -> None:
    with pytest.raises(ToolError):
        await RunPython(None)({"code": "  "}, ToolContext("t"))


def test_format_execution_error_and_empty() -> None:
    text = format_execution({"stdout": "", "stderr": "warn", "error": "ZeroDivisionError: x"})
    assert "stderr:\nwarn" in text and "error:\nZeroDivisionError" in text
    assert "no output" in format_execution({})


async def test_e2b_session_maps_execution() -> None:
    execution = SimpleNamespace(
        logs=SimpleNamespace(stdout=["a\n", "b\n"], stderr=[]),
        error=SimpleNamespace(name="ValueError", value="bad", traceback="tb"),
        results=[SimpleNamespace(text="7"), SimpleNamespace(text=None)],
    )

    class FakeSbx:
        async def run_code(self, code: str, timeout: float) -> Any:
            return execution

    res = await E2BSession(FakeSbx()).run_python("x", 5)
    assert res == {"stdout": "a\nb\n", "stderr": "", "error": "ValueError: bad\ntb", "result": "7"}
