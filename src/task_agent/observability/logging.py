"""Structured logging. JSON in production, readable console output in dev.

Bind ``task_id`` once with ``bind_task`` and every log line in that task carries it, along
with the active trace id, so logs and traces join up.
"""

from __future__ import annotations

import logging
import sys
from collections.abc import MutableMapping
from typing import Any

import structlog

from task_agent.observability.redaction import Redactor


def _add_trace_id(_: Any, __: str, event: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
    from opentelemetry import trace

    ctx = trace.get_current_span().get_span_context()
    if ctx.is_valid:
        event.setdefault("trace_id", format(ctx.trace_id, "032x"))
    return event


def _redactor_processor(redact: Redactor) -> structlog.types.Processor:
    def process(_: Any, __: str, event: MutableMapping[str, Any]) -> MutableMapping[str, Any]:
        for key, value in event.items():
            if isinstance(value, str):
                event[key] = redact(value)
        return event

    return process


class _Stderr:
    """Resolves sys.stderr on every write, so redirected or replaced streams keep working."""

    def write(self, text: str) -> int:
        return sys.stderr.write(text)

    def flush(self) -> None:
        sys.stderr.flush()


def configure_logging(level: str = "INFO", fmt: str = "console") -> None:
    renderer: structlog.types.Processor = (
        structlog.processors.JSONRenderer()
        if fmt == "json"
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso", utc=True),
            _add_trace_id,
            structlog.processors.format_exc_info,
            _redactor_processor(Redactor.from_environment()),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping().get(level.upper(), logging.INFO)
        ),
        logger_factory=structlog.PrintLoggerFactory(file=_Stderr()),  # type: ignore[arg-type]
        cache_logger_on_first_use=True,
    )
    # LiteLLM is chatty at INFO; keep it to warnings unless we are debugging.
    logging.getLogger("LiteLLM").setLevel(logging.DEBUG if level.upper() == "DEBUG" else "WARNING")


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_task(task_id: str) -> None:
    structlog.contextvars.bind_contextvars(task_id=task_id)


def unbind_task() -> None:
    structlog.contextvars.unbind_contextvars("task_id")
