"""Drives the real browser tools against a local page (no internet). Skipped when
Chromium is not installed (`uv run playwright install chromium`)."""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

import pytest

from task_agent.core.errors import ToolError
from task_agent.tools.base import ToolContext
from task_agent.tools.builtin.browser import BrowserFactory, browser_tools

FORM = b"""<!doctype html><title>Order</title><h1>Pizza order</h1>
<form method="post" action="/submit">
  <label for="n">Customer name</label><input id="n" name="custname">
  <label for="s">Size</label>
  <select id="s" name="size"><option>Small</option><option>Medium</option></select>
  <label><input type="checkbox" name="topping" value="cheese"> Cheese</label>
  <input type="password" name="pw" aria-label="Password">
  <button type="submit">Submit order</button>
</form>"""


class Handler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(FORM)

    def do_POST(self) -> None:
        body = self.rfile.read(int(self.headers["Content-Length"])).decode()
        fields = {k: v[0] for k, v in parse_qs(body).items()}
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        self.wfile.write(f"<pre>{fields}</pre>".encode())

    def log_message(self, *_a: object) -> None:
        pass


@pytest.fixture
def site():
    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}/"
    server.shutdown()


def _number(snapshot: str, label: str) -> int:
    line = next(ln for ln in snapshot.splitlines() if f'"{label}' in ln and ln.startswith("["))
    return int(line[1 : line.index("]")])


async def test_fill_and_submit_a_form(site) -> None:
    tools = {spec.name: handler for spec, handler in browser_tools(BrowserFactory())}
    ctx = ToolContext("t")
    try:
        try:
            page = (await tools["browser_open"]({"url": site}, ctx)).content
        except Exception as exc:
            pytest.skip(f"browser not available: {exc}")
        assert "Title: Order" in page and "Pizza order" in page

        page = (
            await tools["browser_type"](
                {"element": _number(page, "Customer name"), "text": "Ada"}, ctx
            )
        ).content
        assert 'value="Ada"' in page
        page = (
            await tools["browser_select"](
                {"element": _number(page, "Size"), "option": "Medium"}, ctx
            )
        ).content
        assert "*Medium" in page
        page = (await tools["browser_click"]({"element": _number(page, "Cheese")}, ctx)).content
        assert "(checked)" in page

        with pytest.raises(ToolError, match="never types passwords"):
            await tools["browser_type"]({"element": _number(page, "Password"), "text": "x"}, ctx)
        with pytest.raises(ToolError, match="no element"):
            await tools["browser_click"]({"element": 999}, ctx)

        page = (
            await tools["browser_click"]({"element": _number(page, "Submit order")}, ctx)
        ).content
        assert "'custname': 'Ada'" in page and "'size': 'Medium'" in page
    finally:
        await ctx.aclose()


class FakeVision:
    def __init__(self) -> None:
        self.messages: list = []

    async def complete(self, role, messages, **_kw):
        from tests.fakes import reply

        self.messages = messages
        return reply("A red heading that says Pizza order.")


async def test_browser_look_sends_a_screenshot_to_the_vision_role(site) -> None:
    vision = FakeVision()
    tools = {s.name: h for s, h in browser_tools(BrowserFactory(), vision)}
    ctx = ToolContext("t")
    try:
        try:
            await tools["browser_open"]({"url": site}, ctx)
        except Exception as exc:
            pytest.skip(f"browser not available: {exc}")
        out = await tools["browser_look"]({"question": "What colour is the heading?"}, ctx)
        assert out.content == "A red heading that says Pizza order."
        text, image = vision.messages[0]["content"]
        assert "What colour is the heading?" in text["text"]
        assert image["image_url"]["url"].startswith("data:image/png;base64,iVBOR")  # a PNG
    finally:
        await ctx.aclose()
