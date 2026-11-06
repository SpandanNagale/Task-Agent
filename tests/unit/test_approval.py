import pytest

from task_agent.core import ConfigError
from task_agent.core.approval import Policy, RejectAll, Rule, Verdict
from task_agent.tools import SideEffect, ToolSpec
from tests.conftest import ROOT


def _spec(name: str, side: SideEffect) -> ToolSpec:
    return ToolSpec(name=name, description="x" * 20, group="g", side_effect=side)


def test_defaults_and_tool_overrides() -> None:
    p = Policy(tools={"run_python": Rule.ALLOW, "gmail_send_email": Rule.DENY})
    assert p.decide(_spec("web_search", SideEffect.READ)) is Rule.ALLOW
    assert p.decide(_spec("draft", SideEffect.WRITE)) is Rule.ASK
    assert p.decide(_spec("post", SideEffect.IRREVERSIBLE)) is Rule.ASK
    assert p.decide(_spec("run_python", SideEffect.WRITE)) is Rule.ALLOW
    assert p.decide(_spec("gmail_send_email", SideEffect.IRREVERSIBLE)) is Rule.DENY


def test_partial_defaults_keep_safe_base() -> None:
    p = Policy.model_validate({"defaults": {"write": "deny"}})
    assert p.decide(_spec("rr", SideEffect.READ)) is Rule.ALLOW
    assert p.decide(_spec("ww", SideEffect.WRITE)) is Rule.DENY
    assert p.decide(_spec("ii", SideEffect.IRREVERSIBLE)) is Rule.ASK


def test_irreversible_cannot_default_to_allow() -> None:
    with pytest.raises(ValueError, match="irreversible"):
        Policy.model_validate({"defaults": {"irreversible": "allow"}})


def test_shipped_policy_loads() -> None:
    p = Policy.load(ROOT / "policies" / "default.yaml")
    assert p.decide(_spec("run_python", SideEffect.WRITE)) is Rule.ALLOW
    assert p.decide(_spec("gmail_send_email", SideEffect.IRREVERSIBLE)) is Rule.ASK


def test_bad_policy_file(tmp_path) -> None:
    (tmp_path / "p.yaml").write_text("defaults: {write: sometimes}")
    with pytest.raises(ConfigError):
        Policy.load(tmp_path / "p.yaml")
    with pytest.raises(ConfigError):
        Policy.load(tmp_path / "missing.yaml")


async def test_reject_all() -> None:
    from task_agent.core.approval import ApprovalRequest

    d = await RejectAll("nope").request(
        ApprovalRequest(task_id="t", tool="x", side_effect="write", description="d", args={})
    )
    assert d.verdict is Verdict.REJECT
    assert d.reason == "nope"
