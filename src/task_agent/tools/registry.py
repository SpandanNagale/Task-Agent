"""In-process tool registry. Tools are registered, never hardcoded into an agent.

Phase 0 ships the contract and lookup by group; phase 1 adds the first tools, MCP loading
and invocation through the approval gate; retrieval by embedding comes once the registry
grows past what fits in a prompt.
"""

from __future__ import annotations

from collections.abc import Iterable

from task_agent.core.errors import ToolRegistryError
from task_agent.tools.base import ToolHandler, ToolSpec


class ToolRegistry:
    def __init__(self) -> None:
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, ToolHandler] = {}

    def register(self, spec: ToolSpec, handler: ToolHandler) -> None:
        if spec.name in self._specs:
            raise ToolRegistryError(f"tool '{spec.name}' is already registered")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = handler

    def get(self, name: str) -> ToolSpec:
        try:
            return self._specs[name]
        except KeyError:
            raise ToolRegistryError(f"unknown tool '{name}'") from None

    def handler(self, name: str) -> ToolHandler:
        self.get(name)
        return self._handlers[name]

    def __contains__(self, name: object) -> bool:
        return name in self._specs

    def __len__(self) -> int:
        return len(self._specs)

    @property
    def groups(self) -> set[str]:
        return {s.group for s in self._specs.values()}

    def list_tools(self, groups: Iterable[str] | None = None) -> list[ToolSpec]:
        """Tools in the given groups (all tools if ``groups`` is None), sorted by name."""
        wanted = None if groups is None else set(groups)
        return sorted(
            (s for s in self._specs.values() if wanted is None or s.group in wanted),
            key=lambda s: s.name,
        )
