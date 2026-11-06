"""Build the tool registry from settings. A tool is registered only when it can work:
missing keys skip the tool with a note instead of failing at call time.
"""

from __future__ import annotations

from task_agent.config import Settings
from task_agent.observability.logging import get_logger
from task_agent.tools.builtin.sandbox import RUN_PYTHON, E2BSandboxFactory, RunPython
from task_agent.tools.builtin.web import READ_URL, WEB_SEARCH, ReadUrl, TavilySearch
from task_agent.tools.registry import ToolRegistry

log = get_logger(__name__)


def _secret(value: object) -> str | None:
    get = getattr(value, "get_secret_value", None)
    return get() if get else None


def build_registry(settings: Settings) -> tuple[ToolRegistry, list[str]]:
    """Returns the registry and human-readable notes on what was skipped and why."""
    registry = ToolRegistry()
    notes: list[str] = []

    tavily = _secret(settings.tavily_api_key)
    if tavily:
        registry.register(WEB_SEARCH, TavilySearch(tavily))
    else:
        notes.append("web_search: set TAVILY_API_KEY (free tier: 1,000 searches/month)")

    registry.register(
        READ_URL,
        ReadUrl(_secret(settings.firecrawl_api_key), _secret(settings.jina_api_key)),
    )
    if not settings.firecrawl_api_key:
        notes.append("read_url: using Jina Reader (free); set FIRECRAWL_API_KEY for JS-heavy pages")

    e2b = _secret(settings.e2b_api_key)
    if e2b:
        factory = E2BSandboxFactory(
            e2b, internet=settings.sandbox_internet, lifetime_s=settings.sandbox_lifetime_s
        )
        registry.register(
            RUN_PYTHON, RunPython(factory, exec_timeout_s=settings.sandbox_exec_timeout_s)
        )
    else:
        notes.append("run_python: set E2B_API_KEY (free Hobby tier: $100 one-time credit)")

    composio = _secret(settings.composio_api_key)
    if composio:
        from task_agent.tools.integrations.composio import load_composio_tools, make_client

        try:
            client = make_client(composio)
            for spec, handler in load_composio_tools(
                client, settings.composio_tools, settings.composio_user_id
            ):
                registry.register(spec, handler)
        except Exception as exc:  # one integration being down must not stop the agent
            log.warning("composio.load_failed", error=str(exc)[:300])
            notes.append(f"composio: tools not loaded ({type(exc).__name__}); see logs")
    else:
        notes.append("gmail tools: set COMPOSIO_API_KEY, then run `task-agent connect gmail`")

    return registry, notes
