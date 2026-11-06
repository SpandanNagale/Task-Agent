import json

import httpx
import pytest

from task_agent.core import ToolError, TransientToolError
from task_agent.tools import http as tool_http
from task_agent.tools.base import ToolContext
from task_agent.tools.builtin.web import ReadUrl, TavilySearch

CTX = ToolContext("t")


@pytest.fixture
def mock_http(monkeypatch: pytest.MonkeyPatch):
    seen: list[httpx.Request] = []

    def install(handler):
        def wrapped(request: httpx.Request) -> httpx.Response:
            seen.append(request)
            return handler(request)

        monkeypatch.setattr(tool_http, "transport", httpx.MockTransport(wrapped))
        return seen

    return install


async def test_tavily_formats_results(mock_http) -> None:
    seen = mock_http(
        lambda r: httpx.Response(
            200,
            json={"results": [{"title": "Python", "url": "https://python.org", "content": "1991"}]},
        )
    )
    out = await TavilySearch("tvly-key")({"query": "python release", "max_results": 50}, CTX)
    assert "1. Python" in out.content and "https://python.org" in out.content
    req = seen[0]
    assert req.headers["Authorization"] == "Bearer tvly-key"
    assert json.loads(req.content)["max_results"] == 10


@pytest.mark.parametrize(
    ("status", "error"),
    [(429, TransientToolError), (503, TransientToolError), (401, ToolError), (400, ToolError)],
)
async def test_http_errors_are_mapped(mock_http, status: int, error: type) -> None:
    mock_http(lambda r: httpx.Response(status, text="nope"))
    with pytest.raises(error):
        await TavilySearch("k")({"query": "x"}, CTX)


async def test_network_failure_is_transient(mock_http) -> None:
    def fail(r: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("down")

    mock_http(fail)
    with pytest.raises(TransientToolError):
        await TavilySearch("k")({"query": "x"}, CTX)


async def test_read_url_firecrawl(mock_http) -> None:
    seen = mock_http(
        lambda r: httpx.Response(
            200,
            json={"success": True, "data": {"markdown": "Body", "metadata": {"title": "T"}}},
        )
    )
    out = await ReadUrl(firecrawl_key="fc-key")({"url": "https://example.com"}, CTX)
    assert out.content == "# T\n\nBody"
    assert seen[0].url.host == "api.firecrawl.dev"


async def test_read_url_falls_back_to_jina(mock_http) -> None:
    seen = mock_http(lambda r: httpx.Response(200, text="Example Domain"))
    out = await ReadUrl()({"url": "https://example.com"}, CTX)
    assert out.content == "Example Domain"
    assert str(seen[0].url) == "https://r.jina.ai/https://example.com"


@pytest.mark.parametrize("url", ["file:///etc/passwd", "example.com", "javascript:alert(1)"])
async def test_read_url_rejects_non_http(url: str) -> None:
    with pytest.raises(ToolError):
        await ReadUrl()({"url": url}, CTX)
