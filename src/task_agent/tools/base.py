"""The one tool contract. Every capability — MCP server tool, Composio action, or wrapped
Python function — is described by a ``ToolSpec`` and run by a ``ToolHandler``. The
side-effect level drives the approval gate.
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from enum import StrEnum
from typing import Any, Protocol, TypeVar

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SideEffect(StrEnum):
    READ = "read"  # no change to the outside world; safe to retry
    WRITE = "write"  # changes something; needs approval or an allowlist rule
    IRREVERSIBLE = "irreversible"  # sends, posts, pays, deletes; always needs approval

    @property
    def needs_approval(self) -> bool:
        return self is not SideEffect.READ


class ToolGroup(StrEnum):
    """Built-in groups. Sub-agents load only the groups they need.

    Third-party tools may declare their own group names; these are just the well-known ones.
    """

    SEARCH = "search"
    BROWSER = "browser"
    CODE = "code"
    FILES = "files"
    COMMS = "comms"
    MEDIA = "media"
    KNOWLEDGE = "knowledge"


_NAME = re.compile(r"^[a-z][a-z0-9_]{1,63}$")


class ToolSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    description: str = Field(
        min_length=20,
        description="One sentence on what it does, one on when NOT to use it.",
    )
    group: str
    side_effect: SideEffect
    input_schema: dict[str, Any] = Field(default_factory=lambda: {"type": "object"})
    timeout_s: float = Field(default=60.0, gt=0)
    version: str = "1"
    source: str = "python"  # "python", "composio", or "mcp:<server>"

    @field_validator("name")
    @classmethod
    def _valid_name(cls, v: str) -> str:
        if not _NAME.match(v):
            raise ValueError("tool names are snake_case, 2-64 chars, starting with a letter")
        return v

    def to_openai_tool(self) -> dict[str, Any]:
        """The function-calling schema models see."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.input_schema,
            },
        }


class ToolOutput(BaseModel):
    """What a handler returns. ``content`` is what the model sees."""

    content: str
    data: Any = None  # structured result for callers and traces; not shown to the model
    files: list[str] = Field(default_factory=list)  # paths produced (e.g. in the sandbox)


R = TypeVar("R")


class ToolContext:
    """Per-task state handed to every tool call.

    Tools that need an expensive resource (a sandbox, a browser) create it lazily with
    ``resource()``; it is reused for the rest of the task and torn down when the task ends,
    so nothing leaks between tasks or users.
    """

    def __init__(self, task_id: str) -> None:
        self.task_id = task_id
        self._resources: dict[str, Any] = {}
        self._stack = AsyncExitStack()

    async def resource(self, key: str, factory: Callable[[AsyncExitStack], Awaitable[R]]) -> R:
        if key not in self._resources:
            self._resources[key] = await factory(self._stack)
        return self._resources[key]  # type: ignore[no-any-return]

    async def aclose(self) -> None:
        self._resources.clear()
        await self._stack.aclose()


class ToolHandler(Protocol):
    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput: ...
