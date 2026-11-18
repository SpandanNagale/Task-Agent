from pydantic import SecretStr

from task_agent.config import Settings
from task_agent.tools.loader import build_registry


def _settings(**kw) -> Settings:
    return Settings(_env_file=None, **kw)


def test_no_keys_still_reads_urls(monkeypatch) -> None:
    for key in ("TAVILY_API_KEY", "E2B_API_KEY", "COMPOSIO_API_KEY", "FIRECRAWL_API_KEY"):
        monkeypatch.delenv(key, raising=False)
    registry, notes = build_registry(_settings(browser_enabled=False))
    assert sorted(t.name for t in registry.list_tools()) == ["read_url", "recall", "remember"]
    assert any("TAVILY_API_KEY" in n for n in notes)
    assert any("E2B_API_KEY" in n for n in notes)


def test_keys_register_tools(monkeypatch) -> None:
    registry, _ = build_registry(
        _settings(tavily_api_key=SecretStr("t"), e2b_api_key=SecretStr("e"))
    )
    names = {t.name for t in registry.list_tools()}
    assert {"web_search", "read_url", "run_python", "browser_open", "browser_click"} <= names
    assert registry.groups == {"search", "code", "browser", "knowledge"}


def test_composio_failure_does_not_break_registry(monkeypatch) -> None:
    def boom(*_a, **_k):
        raise RuntimeError("composio down")

    monkeypatch.setattr("task_agent.tools.integrations.composio.make_client", boom)
    registry, notes = build_registry(_settings(composio_api_key=SecretStr("c")))
    assert "read_url" in registry
    assert any("composio" in n for n in notes)


def test_vision_can_be_turned_off(mock_gateway) -> None:
    class WithVision:
        roles = ("vision",)

    on, _ = build_registry(_settings(), WithVision())  # type: ignore[arg-type]
    off, _ = build_registry(_settings(browser_vision=False), WithVision())  # type: ignore[arg-type]
    assert "browser_look" in on and "browser_look" not in off
    assert "browser_open" in off
