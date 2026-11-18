"""Tools from MCP servers, configured in ``config/mcp.yaml``.

Each server runs as its own process over stdio, in its own environment, so projects with
conflicting dependencies (DataSleuth on LangGraph 0.2, the CrewAI research pipeline, ...)
plug in without sharing this venv. Every tool a server lists is registered with the
server's group and side-effect level, so it goes through the same policy and approval gate
as any built-in tool.

A server is started per call and stopped afterwards: these are long-running agent tools,
where a second of startup is noise, and nothing stays alive between tasks.
"""

from __future__ import annotations

import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from task_agent.core.errors import ConfigError, ToolError
from task_agent.tools.base import SideEffect, ToolContext, ToolOutput, ToolSpec

LOCAL_NOTE = "Runs on the user's computer, so it can open local file paths."


class McpServer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    command: str
    args: list[str] = Field(default_factory=list)
    cwd: str | None = None
    env: dict[str, str] | None = None  # None: the MCP SDK's minimal default environment
    group: str
    # Default to asking: a server must be marked read-only explicitly to skip approval.
    side_effect: SideEffect = SideEffect.WRITE
    timeout_s: float = Field(default=600.0, gt=0)

    @model_validator(mode="after")
    def _resolve_paths(self) -> McpServer:
        # cwd is relative to where task-agent runs; a relative command path (one with a
        # slash, like .venv/Scripts/python.exe) is relative to cwd. Bare names use PATH.
        if self.cwd:
            self.cwd = str(Path(self.cwd).resolve())
        if self.cwd and ("/" in self.command or "\\" in self.command):
            self.command = str((Path(self.cwd) / self.command).resolve())
        return self


def load_mcp_config(path: Path) -> dict[str, McpServer]:
    if not path.is_file():
        return {}
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return {
            name: McpServer.model_validate(cfg) for name, cfg in (raw.get("servers") or {}).items()
        }
    except (yaml.YAMLError, ValidationError, AttributeError) as exc:
        raise ConfigError(f"invalid MCP config {path}: {exc}") from exc


@asynccontextmanager
async def _session(server: McpServer) -> AsyncIterator[Any]:
    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=server.command, args=server.args, cwd=server.cwd, env=server.env
    )
    # Server logs go to the real process stderr: a captured sys.stderr (pytest, a wrapped
    # console) has no fileno, and the subprocess needs one.
    errlog = sys.__stderr__ or sys.stderr
    async with (
        stdio_client(params, errlog=errlog) as (read, write),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        yield session


class McpTool:
    def __init__(self, server: McpServer, tool: str) -> None:
        self.server = server
        self.tool = tool

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        async with _session(self.server) as session:
            result = await session.call_tool(self.tool, args)
        text = "\n".join(c.text for c in getattr(result, "content", []) if getattr(c, "text", None))
        if getattr(result, "is_error", False):
            raise ToolError(text or f"MCP tool {self.tool} failed")
        return ToolOutput(
            content=text or "(no output)", data=getattr(result, "structured_content", None)
        )


async def list_mcp_tools(
    servers: dict[str, McpServer],
) -> tuple[list[tuple[ToolSpec, McpTool]], list[str]]:
    """Start each server once to read its tool list. Failures become notes, not crashes."""
    tools: list[tuple[ToolSpec, McpTool]] = []
    notes: list[str] = []
    for name, server in servers.items():
        try:
            async with _session(server) as session:
                listed = (await session.list_tools()).tools
        except Exception as exc:  # one broken server must not stop the agent
            notes.append(f"mcp:{name}: not loaded ({type(exc).__name__}: {str(exc)[:200]})")
            continue
        for tool in listed:
            try:
                spec = ToolSpec(
                    name=tool.name,
                    # stdio servers run on this machine; without saying so, models assume
                    # every tool is remote and ask the user to upload local files.
                    description=f"{tool.description or ''} {LOCAL_NOTE}".strip(),
                    group=server.group,
                    side_effect=server.side_effect,
                    input_schema=tool.input_schema or {"type": "object"},
                    timeout_s=server.timeout_s,
                    source=f"mcp:{name}",
                )
            except ValidationError as exc:
                notes.append(f"mcp:{name}: skipped tool {tool.name!r} ({exc.errors()[0]['msg']})")
                continue
            tools.append((spec, McpTool(server, tool.name)))
    return tools, notes
