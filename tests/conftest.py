from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from task_agent.config import Settings
from task_agent.models import ModelGateway, parse_gateway_config
from task_agent.observability.tracing import setup_tracing

ROOT = Path(__file__).resolve().parents[1]
MOCK_MODELS = ROOT / "config" / "models.mock.yaml"
SUITE = ROOT / "evals" / "suite"

# OpenTelemetry allows one global provider per process, so install an in-memory exporter
# once and clear it between tests.
_EXPORTER = InMemorySpanExporter()
setup_tracing(Settings(tracing="off", _env_file=None), exporter=_EXPORTER)


@pytest.fixture
def spans() -> Iterator[InMemorySpanExporter]:
    _EXPORTER.clear()
    yield _EXPORTER
    _EXPORTER.clear()


@pytest.fixture
def mock_gateway() -> ModelGateway:
    return ModelGateway.from_file(MOCK_MODELS)


def make_gateway(answer: str, judge: str = "ok\nPASS") -> ModelGateway:
    """A gateway whose planner always says ``answer`` and whose judge says ``judge``."""
    return ModelGateway(
        parse_gateway_config(
            {
                "defaults": {"timeout_s": 5, "num_retries": 0},
                "roles": {
                    "planner": {
                        "chain": [{"model": "openai/m", "api_key": "x", "mock_response": answer}]
                    },
                    "judge": {
                        "chain": [{"model": "openai/j", "api_key": "x", "mock_response": judge}]
                    },
                },
            }
        )
    )
