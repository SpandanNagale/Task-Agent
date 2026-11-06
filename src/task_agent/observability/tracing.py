"""Tracing on OpenTelemetry, with Langfuse as the default backend.

Every task is one trace; every model call and (from phase 1) tool call is a child span.
We emit plain OTel spans with GenAI semantic-convention attributes plus the ``langfuse.*``
attributes Langfuse reads, so the same traces also work in Jaeger, Tempo, Honeycomb or any
OTLP collector. Nothing here imports the Langfuse SDK.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from opentelemetry import trace
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import (
    BatchSpanProcessor,
    ConsoleSpanExporter,
    SimpleSpanProcessor,
    SpanExporter,
)
from opentelemetry.trace import Span, Status, StatusCode

from task_agent import __version__
from task_agent.config import Settings
from task_agent.observability.logging import get_logger
from task_agent.observability.redaction import Redactor

_TRACER_NAME = "task_agent"
_MAX_ATTR_CHARS = 32_000

log = get_logger(__name__)

_provider: TracerProvider | None = None
_redact: Redactor = Redactor()
_trace_content = True


def setup_tracing(settings: Settings, *, exporter: SpanExporter | None = None) -> str:
    """Configure the global tracer provider once. Returns the backend name in use.

    ``exporter`` overrides the backend (tests pass an in-memory exporter).
    """
    global _provider, _redact, _trace_content
    if _provider is not None:
        return "already-configured"

    _redact = Redactor.from_environment()
    _trace_content = settings.trace_content
    backend, chosen = _choose_exporter(settings) if exporter is None else ("custom", exporter)
    if chosen is None:
        log.debug("tracing.disabled")
        return backend

    provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": settings.service_name,
                "service.version": __version__,
                "deployment.environment": settings.env,
            }
        )
    )
    processor = SimpleSpanProcessor(chosen) if exporter else BatchSpanProcessor(chosen)
    provider.add_span_processor(processor)
    trace.set_tracer_provider(provider)
    _provider = provider
    log.info("tracing.enabled", backend=backend)
    return backend


def flush_tracing(timeout_ms: int = 10_000) -> None:
    """Export pending spans now. Call before a short-lived process (CLI, eval run) exits;
    the SDK also shuts the provider down at interpreter exit.
    """
    if _provider is not None:
        _provider.force_flush(timeout_ms)


def _choose_exporter(settings: Settings) -> tuple[str, SpanExporter | None]:
    mode = settings.tracing
    has_langfuse = bool(settings.langfuse_public_key and settings.langfuse_secret_key)

    if mode == "off" or (mode == "auto" and not has_langfuse and not settings.otlp_endpoint):
        return "off", None
    if mode == "console":
        return "console", ConsoleSpanExporter()

    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    if mode == "langfuse" or (mode == "auto" and has_langfuse):
        if not has_langfuse:
            log.warning("tracing.langfuse_keys_missing")
            return "off", None
        assert settings.langfuse_public_key
        assert settings.langfuse_secret_key
        creds = (
            f"{settings.langfuse_public_key.get_secret_value()}:"
            f"{settings.langfuse_secret_key.get_secret_value()}"
        )
        token = base64.b64encode(creds.encode()).decode()
        endpoint = settings.langfuse_host.rstrip("/") + "/api/public/otel/v1/traces"
        return "langfuse", OTLPSpanExporter(
            endpoint=endpoint, headers={"Authorization": f"Basic {token}"}
        )

    return "otlp", OTLPSpanExporter(endpoint=settings.otlp_endpoint)


def tracer() -> trace.Tracer:
    return trace.get_tracer(_TRACER_NAME, __version__)


def current_trace_id() -> str | None:
    ctx = trace.get_current_span().get_span_context()
    return format(ctx.trace_id, "032x") if ctx.is_valid else None


def _payload(value: Any) -> str:
    text = value if isinstance(value, str) else json.dumps(value, default=str, ensure_ascii=False)
    text = _redact(text)
    if len(text) > _MAX_ATTR_CHARS:
        text = text[:_MAX_ATTR_CHARS] + "...[truncated]"
    return text


def set_io(span: Span, *, input: Any = None, output: Any = None, trace_level: bool = False) -> None:
    """Attach (redacted, size-capped) input/output to a span, if content tracing is on."""
    if not _trace_content:
        return
    prefix = "langfuse.trace" if trace_level else "langfuse.observation"
    if input is not None:
        span.set_attribute(f"{prefix}.input", _payload(input))
    if output is not None:
        span.set_attribute(f"{prefix}.output", _payload(output))


def record_error(span: Span, exc: BaseException) -> None:
    span.record_exception(exc)
    span.set_status(Status(StatusCode.ERROR, str(exc)))
    span.set_attribute("langfuse.observation.level", "ERROR")
    span.set_attribute("langfuse.observation.status_message", _redact(str(exc))[:1000])


@contextmanager
def task_span(
    task_id: str, prompt: str, *, name: str = "task", metadata: dict[str, Any] | None = None
) -> Iterator[Span]:
    """Root span for one task. Becomes the Langfuse trace."""
    with tracer().start_as_current_span(name, record_exception=False) as span:
        span.set_attribute("task_agent.task_id", task_id)
        span.set_attribute("langfuse.trace.name", name)
        span.set_attribute("langfuse.session.id", task_id)
        span.set_attribute("langfuse.trace.metadata.task_id", task_id)
        for key, value in (metadata or {}).items():
            span.set_attribute(f"langfuse.trace.metadata.{key}", _payload(value))
        set_io(span, input=prompt, trace_level=True)
        yield span


@contextmanager
def generation_span(role: str, messages: list[dict[str, Any]]) -> Iterator[Span]:
    """Child span for one model call through the gateway."""
    with tracer().start_as_current_span(f"llm.{role}", record_exception=False) as span:
        span.set_attribute("langfuse.observation.type", "generation")
        span.set_attribute("gen_ai.operation.name", "chat")
        span.set_attribute("task_agent.model_role", role)
        set_io(span, input=messages)
        yield span


@contextmanager
def tool_span(name: str, side_effect: str, args: Any) -> Iterator[Span]:
    """Child span for one tool call."""
    with tracer().start_as_current_span(f"tool.{name}", record_exception=False) as span:
        span.set_attribute("langfuse.observation.type", "tool")
        span.set_attribute("gen_ai.operation.name", "execute_tool")
        span.set_attribute("gen_ai.tool.name", name)
        span.set_attribute("task_agent.side_effect", side_effect)
        set_io(span, input=args)
        yield span


@contextmanager
def step_span(step: int) -> Iterator[Span]:
    """Groups one think-act cycle: a model call and the tool calls it makes."""
    with tracer().start_as_current_span(f"step.{step}", record_exception=False) as span:
        span.set_attribute("langfuse.observation.type", "span")
        span.set_attribute("task_agent.step", step)
        yield span


def record_generation(
    span: Span,
    *,
    model: str,
    output: Any,
    prompt_tokens: int,
    completion_tokens: int,
    cost_usd: float,
) -> None:
    span.set_attribute("gen_ai.request.model", model)
    span.set_attribute("gen_ai.response.model", model)
    span.set_attribute("langfuse.observation.model.name", model)
    span.set_attribute("gen_ai.usage.input_tokens", prompt_tokens)
    span.set_attribute("gen_ai.usage.output_tokens", completion_tokens)
    span.set_attribute(
        "langfuse.observation.usage_details",
        json.dumps({"input": prompt_tokens, "output": completion_tokens}),
    )
    span.set_attribute("langfuse.observation.cost_details", json.dumps({"total": cost_usd}))
    set_io(span, output=output)
