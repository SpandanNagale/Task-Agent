"""Phase 0 gate: one task runs end to end and appears as a full trace.

Runs the real CLI in a subprocess with tracing in Langfuse mode, pointed at a local OTLP
receiver that decodes what it gets. This checks the exact wire path Langfuse ingests
(endpoint, auth, span tree, attributes) without Docker or network access.
"""

from __future__ import annotations

import base64
import os
import subprocess
import sys
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any, ClassVar

import pytest
from opentelemetry.proto.collector.trace.v1.trace_service_pb2 import ExportTraceServiceRequest

from tests.conftest import MOCK_MODELS, ROOT


class _Receiver(BaseHTTPRequestHandler):
    requests: ClassVar[list[tuple[str, dict[str, str], bytes]]] = []

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"]))
        self.requests.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        self.send_response(200)
        self.send_header("Content-Type", "application/x-protobuf")
        self.end_headers()

    def log_message(self, *_: Any) -> None:
        pass


@pytest.fixture
def receiver() -> Iterator[tuple[str, list[tuple[str, dict[str, str], bytes]]]]:
    _Receiver.requests = []
    server = HTTPServer(("127.0.0.1", 0), _Receiver)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}", _Receiver.requests
    server.shutdown()


def _attrs(span: Any) -> dict[str, Any]:
    out = {}
    for kv in span.attributes:
        v = kv.value
        out[kv.key] = v.string_value or v.int_value or v.double_value or v.bool_value
    return out


def test_cli_task_exports_full_trace_to_langfuse(receiver) -> None:
    url, requests = receiver
    env = {
        **os.environ,
        "TASK_AGENT_TRACING": "langfuse",
        "LANGFUSE_HOST": url,
        "LANGFUSE_PUBLIC_KEY": "pk-lf-test",
        "LANGFUSE_SECRET_KEY": "sk-lf-test",
    }
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "task_agent.cli",
            "run",
            "What is 6 x 7?",
            "--json",
            "--models",
            str(MOCK_MODELS),
        ],
        cwd=ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert requests, "no spans were exported"

    path, headers, _ = requests[0]
    assert path == "/api/public/otel/v1/traces"
    expected = base64.b64encode(b"pk-lf-test:sk-lf-test").decode()
    assert headers["authorization"] == f"Basic {expected}"

    spans = []
    for _, _, body in requests:
        req = ExportTraceServiceRequest()
        req.ParseFromString(body)
        for rs in req.resource_spans:
            for ss in rs.scope_spans:
                spans.extend(ss.spans)
    by_name = {s.name: s for s in spans}
    assert set(by_name) == {"task", "step.1", "llm.planner"}

    root, step, gen = by_name["task"], by_name["step.1"], by_name["llm.planner"]
    assert gen.trace_id == step.trace_id == root.trace_id
    assert step.parent_span_id == root.span_id
    assert gen.parent_span_id == step.span_id

    root_attrs, gen_attrs = _attrs(root), _attrs(gen)
    assert root_attrs["langfuse.trace.input"] == "What is 6 x 7?"
    assert root_attrs["langfuse.trace.output"] == "42"
    assert root_attrs["task_agent.status"] == "succeeded"
    assert gen_attrs["langfuse.observation.type"] == "generation"
    assert gen_attrs["gen_ai.response.model"] == "openai/mock-planner"
    assert "langfuse.observation.usage_details" in gen_attrs

    # The trace id the CLI reports is the one that was exported.
    assert root.trace_id.hex() in proc.stdout
