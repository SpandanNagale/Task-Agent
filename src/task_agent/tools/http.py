"""Shared HTTP helper for tools: consistent timeouts and error mapping.

429s, 5xx and network failures become TransientToolError (retried for read tools); auth
failures and other 4xx become ToolError with a message the model can act on.
"""

from __future__ import annotations

from typing import Any

import httpx

from task_agent.core.errors import ToolError, TransientToolError

USER_AGENT = "task-agent/0.1 (+https://github.com/SpandanNagale/Task-Agent)"

# Tests swap in an httpx.MockTransport here; None means real network.
transport: httpx.AsyncBaseTransport | None = None


async def request(
    method: str,
    url: str,
    *,
    service: str,
    timeout_s: float = 30.0,
    headers: dict[str, str] | None = None,
    json: Any = None,
) -> httpx.Response:
    try:
        async with httpx.AsyncClient(
            timeout=timeout_s,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
            transport=transport,
        ) as client:
            resp = await client.request(method, url, headers=headers, json=json)
    except (httpx.TimeoutException, httpx.NetworkError) as exc:
        raise TransientToolError(f"{service} unreachable: {type(exc).__name__}") from exc

    if resp.status_code == 429 or resp.status_code >= 500:
        raise TransientToolError(f"{service} returned HTTP {resp.status_code}; try again later")
    if resp.status_code in (401, 403):
        raise ToolError(f"{service} rejected the API key (HTTP {resp.status_code})")
    if resp.status_code >= 400:
        raise ToolError(f"{service} returned HTTP {resp.status_code}: {resp.text[:300]}")
    return resp
