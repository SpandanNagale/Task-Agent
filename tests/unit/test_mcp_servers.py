import sys
from pathlib import Path

import pytest

from task_agent.core.errors import ConfigError, ToolError
from task_agent.tools.base import SideEffect, ToolContext
from task_agent.tools.mcp_servers import McpServer, list_mcp_tools, load_mcp_config

SERVER = Path(__file__).parents[1] / "fixtures" / "mcp_echo_server.py"


def test_config_defaults_to_asking_and_missing_file_is_empty(tmp_path) -> None:
    assert load_mcp_config(tmp_path / "nope.yaml") == {}
    path = tmp_path / "mcp.yaml"
    path.write_text("servers:\n  s:\n    command: python\n    group: data\n")
    assert load_mcp_config(path)["s"].side_effect is SideEffect.WRITE
    path.write_text("servers:\n  s:\n    group: data\n")
    with pytest.raises(ConfigError):
        load_mcp_config(path)


async def test_tools_load_and_run_over_stdio() -> None:
    server = McpServer(command=sys.executable, args=[str(SERVER)], group="data", side_effect="read")
    broken = McpServer(command="definitely-not-a-command-xyz", group="data")
    tools, notes = await list_mcp_tools({"echo": server, "broken": broken})

    by_name = {spec.name: (spec, handler) for spec, handler in tools}
    assert set(by_name) == {"add_numbers", "fail_always"}
    assert any("'x'" in n for n in notes)  # name too short: skipped with a note
    assert any("mcp:broken" in n for n in notes)  # a dead server is a note, not a crash

    spec, add = by_name["add_numbers"]
    assert (spec.group, spec.side_effect, spec.source) == ("data", SideEffect.READ, "mcp:echo")
    assert spec.input_schema["required"] == ["a", "b"]
    out = await add({"a": 2, "b": 40}, ToolContext("t"))
    assert out.content == "42"

    with pytest.raises(ToolError, match="fail_always"):
        await by_name["fail_always"][1]({}, ToolContext("t"))
