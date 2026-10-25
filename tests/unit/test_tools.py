import pytest
from pydantic import ValidationError

from task_agent.core import ToolRegistryError
from task_agent.tools import SideEffect, ToolRegistry, ToolSpec


async def _noop(**_: object) -> str:
    return "ok"


def _spec(name: str, group: str = "search", side: SideEffect = SideEffect.READ) -> ToolSpec:
    return ToolSpec(
        name=name,
        description="Does a thing. Do not use it for other things.",
        group=group,
        side_effect=side,
        input_schema={"type": "object", "properties": {"q": {"type": "string"}}},
    )


def test_register_and_filter() -> None:
    reg = ToolRegistry()
    reg.register(_spec("web_search"), _noop)
    reg.register(_spec("send_email", "comms", SideEffect.IRREVERSIBLE), _noop)
    assert len(reg) == 2
    assert "web_search" in reg
    assert [t.name for t in reg.list_tools(["comms"])] == ["send_email"]
    assert reg.groups == {"search", "comms"}
    assert reg.get("send_email").side_effect.needs_approval
    assert not reg.get("web_search").side_effect.needs_approval
    assert reg.handler("web_search") is _noop


def test_duplicates_and_unknown() -> None:
    reg = ToolRegistry()
    reg.register(_spec("web_search"), _noop)
    with pytest.raises(ToolRegistryError):
        reg.register(_spec("web_search"), _noop)
    with pytest.raises(ToolRegistryError):
        reg.get("missing")


def test_spec_validation_and_openai_schema() -> None:
    with pytest.raises(ValidationError):
        _spec("Bad Name")
    schema = _spec("web_search").to_openai_tool()
    assert schema["function"]["name"] == "web_search"
    assert schema["function"]["parameters"]["properties"]["q"]["type"] == "string"
