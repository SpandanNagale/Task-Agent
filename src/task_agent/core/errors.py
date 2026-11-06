"""Exception hierarchy. Every error the agent raises on purpose derives from TaskAgentError."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from task_agent.core.types import BudgetKind


class TaskAgentError(Exception):
    """Base class for all expected task-agent errors."""


class ConfigError(TaskAgentError):
    """Invalid or incomplete configuration (bad YAML, unknown role, no usable model)."""


class ModelError(TaskAgentError):
    """A model call failed after every fallback in the role's chain was tried."""


class BudgetExceeded(TaskAgentError):
    """A task hit one of its limits. Stops the task with a report, never a silent loop."""

    def __init__(self, kind: BudgetKind, limit: float, used: float) -> None:
        self.kind = kind
        self.limit = limit
        self.used = used
        super().__init__(f"budget exceeded: {kind} used {used:.4g} of {limit:.4g}")


class ToolRegistryError(TaskAgentError):
    """Duplicate tool name, unknown tool, or an invalid tool spec."""


class LoopDetected(TaskAgentError):
    """The agent kept repeating the same action without progress."""


class ToolError(TaskAgentError):
    """A tool failed. The message is shown to the model, so keep it useful and secret-free."""


class TransientToolError(ToolError):
    """A failure worth retrying (timeouts, 429s, 5xx). Only read-only tools are retried."""
