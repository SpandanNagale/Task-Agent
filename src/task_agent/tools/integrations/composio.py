"""App integrations through Composio (Gmail first; Calendar, Slack, GitHub, Notion later).

Composio holds the OAuth connections, so the agent never sees a password or token. Each
Composio action becomes a ``ToolSpec``. Its side-effect level comes from, in order:

1. our curated table below (reviewed by a human; wins over everything)
2. Composio's MCP-style hints on the tool (``readOnlyHint`` / ``destructiveHint``)
3. ``irreversible``: unknown actions are treated as the most dangerous kind

Connect an account once with ``task-agent connect gmail``.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

from task_agent.core.errors import ToolError, TransientToolError
from task_agent.observability.logging import get_logger
from task_agent.tools.base import SideEffect, ToolContext, ToolGroup, ToolOutput, ToolSpec

log = get_logger(__name__)

CURATED: dict[str, tuple[SideEffect, str]] = {
    "GMAIL_FETCH_EMAILS": (SideEffect.READ, ToolGroup.COMMS),
    "GMAIL_FETCH_MESSAGE_BY_MESSAGE_ID": (SideEffect.READ, ToolGroup.COMMS),
    "GMAIL_FETCH_MESSAGE_BY_THREAD_ID": (SideEffect.READ, ToolGroup.COMMS),
    "GMAIL_LIST_DRAFTS": (SideEffect.READ, ToolGroup.COMMS),
    "GMAIL_CREATE_EMAIL_DRAFT": (SideEffect.WRITE, ToolGroup.COMMS),
    "GMAIL_SEND_EMAIL": (SideEffect.IRREVERSIBLE, ToolGroup.COMMS),
    "GMAIL_REPLY_TO_THREAD": (SideEffect.IRREVERSIBLE, ToolGroup.COMMS),
    "GMAIL_SEND_DRAFT": (SideEffect.IRREVERSIBLE, ToolGroup.COMMS),
    "GMAIL_DELETE_MESSAGE": (SideEffect.IRREVERSIBLE, ToolGroup.COMMS),
    "GMAIL_DELETE_DRAFT": (SideEffect.IRREVERSIBLE, ToolGroup.COMMS),
}


def classify(slug: str, tags: list[str] | None) -> SideEffect:
    if slug in CURATED:
        return CURATED[slug][0]
    tags = tags or []
    if "destructiveHint" in tags:
        return SideEffect.IRREVERSIBLE
    if "readOnlyHint" in tags:
        return SideEffect.READ
    return SideEffect.IRREVERSIBLE


def tool_name(slug: str) -> str:
    return slug.lower()[:64]


def spec_from_composio(tool: Any) -> ToolSpec:
    slug: str = tool.slug
    side_effect = classify(slug, list(getattr(tool, "tags", None) or []))
    group = CURATED.get(slug, (side_effect, ToolGroup.COMMS))[1]
    schema = dict(getattr(tool, "input_parameters", None) or {"type": "object"})
    schema.setdefault("type", "object")
    description = (getattr(tool, "description", "") or getattr(tool, "name", slug)).strip()
    if side_effect is not SideEffect.READ:
        description += " Requires the user's approval before it runs."
    return ToolSpec(
        name=tool_name(slug),
        description=description[:1000].ljust(20),
        group=group,
        side_effect=side_effect,
        input_schema=schema,
        timeout_s=60,
        source="composio",
        version=str(getattr(tool, "version", "1")),
    )


class ComposioAction:
    """Runs one Composio action for the configured user. The SDK is synchronous, so calls
    go to a worker thread to keep the event loop free."""

    def __init__(self, client: Any, slug: str, user_id: str) -> None:
        self._client = client
        self._slug = slug
        self._user_id = user_id

    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        try:
            resp = await asyncio.to_thread(
                self._client.tools.execute,
                self._slug,
                args,
                user_id=self._user_id,
                dangerously_skip_version_check=True,
            )
        except Exception as exc:
            text = str(exc)
            if any(code in text for code in ("429", "502", "503", "504", "timed out")):
                raise TransientToolError(f"Composio unavailable: {text[:200]}") from exc
            if "connected account" in text.lower() or "no connection" in text.lower():
                raise ToolError(
                    "no connected account for this app; ask the user to run "
                    "`task-agent connect <app>`"
                ) from exc
            raise ToolError(f"Composio error: {text[:300]}") from exc

        if not resp.get("successful", False):
            raise ToolError(f"{self._slug} failed: {resp.get('error') or 'unknown error'}")
        data = resp.get("data")
        return ToolOutput(content=json.dumps(data, ensure_ascii=False, default=str), data=data)


def make_client(api_key: str) -> Any:
    from composio import Composio

    return Composio(api_key=api_key, allow_tracking=False)


def load_composio_tools(
    client: Any, slugs: list[str], user_id: str
) -> list[tuple[ToolSpec, ComposioAction]]:
    """Fetch tool schemas once at startup. Unknown slugs are skipped with a warning."""
    tools = client.tools.get_raw_composio_tools(tools=slugs)
    found = {t.slug for t in tools}
    for missing in sorted(set(slugs) - found):
        log.warning("composio.tool_not_found", slug=missing)
    return [(spec_from_composio(t), ComposioAction(client, t.slug, user_id)) for t in tools]
