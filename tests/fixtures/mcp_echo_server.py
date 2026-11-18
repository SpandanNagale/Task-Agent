"""A tiny MCP server over stdio, for tests of the MCP tool loader."""

from mcp.server.mcpserver import MCPServer

server = MCPServer("echo")


@server.tool()
def add_numbers(a: int, b: int) -> str:
    """Add two integers. Do not use for anything but adding two integers."""
    return str(a + b)


@server.tool()
def fail_always() -> str:
    """Always raises, to test error handling. Never use it for real work."""
    raise ValueError("this tool always fails")


@server.tool()
def x() -> str:
    """Bad name and fine description, must be skipped by the loader."""
    return "x"


if __name__ == "__main__":
    server.run()
