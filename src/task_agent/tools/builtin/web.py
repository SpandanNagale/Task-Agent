"""Web tools: search (Tavily) and page reading (Firecrawl, falling back to Jina Reader)."""

from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from task_agent.core.errors import ToolError
from task_agent.tools.base import SideEffect, ToolContext, ToolGroup, ToolOutput, ToolSpec
from task_agent.tools.http import request

WEB_SEARCH = ToolSpec(
    name="web_search",
    description=(
        "Search the web and get the top results with titles, URLs and short extracts. "
        "Do not use it to read a page you already have the URL for; use read_url instead."
    ),
    group=ToolGroup.SEARCH,
    side_effect=SideEffect.READ,
    timeout_s=30,
    input_schema={
        "type": "object",
        "properties": {
            "query": {"type": "string", "description": "What to search for."},
            "max_results": {"type": "integer", "minimum": 1, "maximum": 10, "default": 5},
        },
        "required": ["query"],
    },
)

READ_URL = ToolSpec(
    name="read_url",
    description=(
        "Fetch a web page or PDF by URL and return its main content as Markdown. "
        "Do not use it to find pages; use web_search first if you lack the URL."
    ),
    group=ToolGroup.SEARCH,
    side_effect=SideEffect.READ,
    timeout_s=60,
    input_schema={
        "type": "object",
        "properties": {"url": {"type": "string", "description": "Absolute http(s) URL."}},
        "required": ["url"],
    },
)


class TavilySearch:
    def __init__(self, api_key: str) -> None:
        self._key = api_key

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        query = str(args["query"]).strip()
        if not query:
            raise ToolError("query is empty")
        max_results = max(1, min(10, int(args.get("max_results", 5))))
        resp = await request(
            "POST",
            "https://api.tavily.com/search",
            service="Tavily",
            headers={"Authorization": f"Bearer {self._key}"},
            json={"query": query, "max_results": max_results, "search_depth": "basic"},
        )
        results = resp.json().get("results", [])
        if not results:
            return ToolOutput(content=f"No results for {query!r}.", data=[])
        lines = [
            f"{i}. {r.get('title', '').strip()}\n"
            f"   {r.get('url', '')}\n"
            f"   {r.get('content', '').strip()}"
            for i, r in enumerate(results, 1)
        ]
        return ToolOutput(content="\n".join(lines), data=results)


def _check_url(url: str) -> str:
    url = url.strip()
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https") or not parsed.netloc:
        raise ToolError(f"not an absolute http(s) URL: {url!r}")
    return url


class ReadUrl:
    """Firecrawl when a key is set (better on JS-heavy pages); Jina Reader otherwise
    (free, no key needed, lower rate limit)."""

    def __init__(self, firecrawl_key: str | None = None, jina_key: str | None = None) -> None:
        self._firecrawl_key = firecrawl_key
        self._jina_key = jina_key

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        url = _check_url(str(args["url"]))
        if self._firecrawl_key:
            resp = await request(
                "POST",
                "https://api.firecrawl.dev/v2/scrape",
                service="Firecrawl",
                timeout_s=60,
                headers={"Authorization": f"Bearer {self._firecrawl_key}"},
                json={"url": url, "formats": ["markdown"], "onlyMainContent": True},
            )
            body = resp.json()
            data = body.get("data") or {}
            markdown = data.get("markdown")
            if not body.get("success", True) or markdown is None:
                raise ToolError(
                    f"Firecrawl could not read {url}: {body.get('error', 'no content')}"
                )
            title = (data.get("metadata") or {}).get("title", "")
            return ToolOutput(content=f"# {title}\n\n{markdown}" if title else markdown)

        headers = {"Accept": "text/plain"}
        if self._jina_key:
            headers["Authorization"] = f"Bearer {self._jina_key}"
        resp = await request(
            "GET", f"https://r.jina.ai/{url}", service="Jina Reader", timeout_s=60, headers=headers
        )
        return ToolOutput(content=resp.text)
