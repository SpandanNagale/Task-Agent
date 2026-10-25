"""The one tool contract. Every capability — MCP server tool or wrapped Python function —
is described by a ``ToolSpec``. The side-effect level drives the approval gate (phase 1).
"""

from __future__ import annotations

import re
from collections.abc import Awaitable, Callable
from enum import StrEnum
from typing import Any

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

ToolHandler = Callable[..., Awaitable[Any]]


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
    version: str = "1"
    source: str = "python"  # "python" or "mcp:<server>"

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
