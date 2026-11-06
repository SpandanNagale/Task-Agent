from types import SimpleNamespace
from typing import Any

import pytest

from task_agent.core import ToolError, TransientToolError
from task_agent.tools import SideEffect
from task_agent.tools.base import ToolContext
from task_agent.tools.integrations.composio import (
    ComposioAction,
    classify,
    load_composio_tools,
    spec_from_composio,
)


@pytest.mark.parametrize(
    ("slug", "tags", "expected"),
    [
        ("GMAIL_SEND_EMAIL", ["readOnlyHint"], SideEffect.IRREVERSIBLE),  # curated wins
        ("GMAIL_FETCH_EMAILS", [], SideEffect.READ),
        ("SLACK_LIST_CHANNELS", ["readOnlyHint"], SideEffect.READ),
        ("NOTION_DELETE_PAGE", ["destructiveHint"], SideEffect.IRREVERSIBLE),
        ("SOMETHING_NEW", [], SideEffect.IRREVERSIBLE),  # unknown -> most dangerous
    ],
)
def test_classify(slug: str, tags: list[str], expected: SideEffect) -> None:
    assert classify(slug, tags) is expected


def _tool(slug: str, **kw: Any) -> SimpleNamespace:
    return SimpleNamespace(
        slug=slug,
        name=slug.title(),
        description=kw.get("description", "Sends an email from the connected account."),
        input_parameters=kw.get("schema", {"properties": {"to": {"type": "string"}}}),
        tags=kw.get("tags", []),
        version="20250101_00",
    )


def test_spec_from_composio() -> None:
    spec = spec_from_composio(_tool("GMAIL_SEND_EMAIL"))
    assert spec.name == "gmail_send_email"
    assert spec.group == "comms" and spec.side_effect is SideEffect.IRREVERSIBLE
    assert spec.input_schema["type"] == "object"
    assert "approval" in spec.description
    assert spec.source == "composio"


class FakeTools:
    def __init__(self, response: Any = None, exc: Exception | None = None) -> None:
        self.response, self.exc = response, exc
        self.calls: list[tuple[Any, ...]] = []

    def execute(self, slug: str, args: dict[str, Any], **kw: Any) -> Any:
        self.calls.append((slug, args, kw))
        if self.exc:
            raise self.exc
        return self.response

    def get_raw_composio_tools(self, tools: list[str]) -> list[Any]:
        return [_tool(s) for s in tools if s.startswith("GMAIL")]


async def test_action_success_and_user() -> None:
    tools = FakeTools({"successful": True, "data": {"id": "m1"}, "error": None})
    out = await ComposioAction(SimpleNamespace(tools=tools), "GMAIL_SEND_EMAIL", "u1")(
        {"to": "a@b.c"}, ToolContext("t")
    )
    assert out.content == '{"id": "m1"}'
    assert tools.calls[0][2]["user_id"] == "u1"


@pytest.mark.parametrize(
    ("tools", "error", "match"),
    [
        (FakeTools({"successful": False, "error": "quota"}), ToolError, "quota"),
        (FakeTools(exc=RuntimeError("HTTP 503")), TransientToolError, "unavailable"),
        (FakeTools(exc=RuntimeError("No connected account found")), ToolError, "connect"),
    ],
)
async def test_action_failures(tools: FakeTools, error: type, match: str) -> None:
    action = ComposioAction(SimpleNamespace(tools=tools), "GMAIL_SEND_EMAIL", "u")
    with pytest.raises(error, match=match):
        await action({}, ToolContext("t"))


def test_load_skips_unknown_slugs() -> None:
    client = SimpleNamespace(tools=FakeTools())
    loaded = load_composio_tools(client, ["GMAIL_FETCH_EMAILS", "NOPE_TOOL"], "u")
    assert [spec.name for spec, _ in loaded] == ["gmail_fetch_emails"]
