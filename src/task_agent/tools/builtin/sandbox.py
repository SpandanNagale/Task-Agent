"""Code execution in an E2B cloud sandbox.

One sandbox per task, created on first use and killed when the task ends (plus a hard
lifetime, so a crashed worker cannot leak one). Variables and files persist across calls
within the task, like a notebook. Nothing on the host is reachable from it, and internet
access is off unless TASK_AGENT_SANDBOX_INTERNET=true.

A local Docker backend can implement the same ``SandboxSession`` protocol later.
"""

from __future__ import annotations

from contextlib import AsyncExitStack
from typing import Any, Protocol

from task_agent.core.errors import ToolError, TransientToolError
from task_agent.tools.base import SideEffect, ToolContext, ToolGroup, ToolOutput, ToolSpec

RUN_PYTHON = ToolSpec(
    name="run_python",
    description=(
        "Run Python code in an isolated sandbox (pandas, numpy, matplotlib and other common "
        "libraries installed) and get stdout, errors and the last expression's value; state "
        "and files in the working directory persist for the rest of the task. "
        "Use it for any calculation, data analysis or file work instead of doing it in your "
        "head; do not use it to reach the user's accounts or local machine."
    ),
    group=ToolGroup.CODE,
    side_effect=SideEffect.WRITE,  # writes only inside the throwaway sandbox
    timeout_s=180,
    input_schema={
        "type": "object",
        "properties": {"code": {"type": "string", "description": "Python source to execute."}},
        "required": ["code"],
    },
)


class SandboxSession(Protocol):
    async def run_python(self, code: str, timeout_s: float) -> dict[str, Any]: ...


class E2BSession:
    def __init__(self, sandbox: Any) -> None:
        self._sbx = sandbox

    async def run_python(self, code: str, timeout_s: float) -> dict[str, Any]:
        execution = await self._sbx.run_code(code, timeout=timeout_s)
        error = None
        if execution.error is not None:
            err = execution.error
            error = f"{err.name}: {err.value}\n{err.traceback}"
        texts = [r.text for r in execution.results if getattr(r, "text", None)]
        return {
            "stdout": "".join(execution.logs.stdout),
            "stderr": "".join(execution.logs.stderr),
            "error": error,
            "result": "\n".join(texts) if texts else None,
        }


class E2BSandboxFactory:
    def __init__(self, api_key: str, *, internet: bool = False, lifetime_s: int = 600) -> None:
        self._api_key = api_key
        self._internet = internet
        self._lifetime_s = lifetime_s

    async def __call__(self, stack: AsyncExitStack) -> SandboxSession:
        from e2b_code_interpreter import AsyncSandbox

        try:
            sbx = await AsyncSandbox.create(
                api_key=self._api_key,
                timeout=self._lifetime_s,
                allow_internet_access=self._internet,
                metadata={"app": "task-agent"},
            )
        except Exception as exc:
            raise TransientToolError(f"could not start the sandbox: {exc}") from exc
        stack.push_async_callback(sbx.kill)
        return E2BSession(sbx)


def format_execution(res: dict[str, Any]) -> str:
    parts = []
    if res.get("stdout"):
        parts.append(f"stdout:\n{res['stdout'].rstrip()}")
    if res.get("result"):
        parts.append(f"result:\n{res['result']}")
    if res.get("stderr"):
        parts.append(f"stderr:\n{res['stderr'].rstrip()}")
    if res.get("error"):
        parts.append(f"error:\n{res['error'].rstrip()}")
    return "\n\n".join(parts) or "(no output; print() what you need to see)"


class RunPython:
    def __init__(self, factory: Any, *, exec_timeout_s: float = 120.0) -> None:
        self._factory = factory
        self._timeout = exec_timeout_s

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        code = str(args["code"])
        if not code.strip():
            raise ToolError("code is empty")
        session: SandboxSession = await ctx.resource("sandbox", self._factory)
        res = await session.run_python(code, self._timeout)
        return ToolOutput(content=format_execution(res), data=res)
