import pytest

from task_agent.core import ConfigError
from task_agent.models import load_gateway_config, parse_gateway_config
from tests.conftest import MOCK_MODELS, ROOT


def test_env_expansion_and_skipping(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("A_KEY", "secret-a")
    monkeypatch.delenv("B_KEY", raising=False)
    monkeypatch.delenv("MODEL_NAME", raising=False)
    cfg = parse_gateway_config(
        {
            "roles": {
                "planner": {
                    "chain": [
                        {"name": "a", "model": "p/${MODEL_NAME:-big}", "api_key": "${A_KEY}"},
                        {"name": "b", "model": "p/x", "api_key": "${B_KEY}"},
                    ]
                }
            }
        }
    )
    chain = cfg.roles["planner"].chain
    assert [e.model for e in chain] == ["p/big"]
    assert chain[0].litellm_params() == {"model": "p/big", "api_key": "secret-a"}
    assert cfg.roles["planner"].skipped == ["b: ${B_KEY} not set"]


def test_alias_inherits_chain_and_merges_params() -> None:
    cfg = parse_gateway_config(
        {
            "roles": {
                "planner": {"params": {"temperature": 0.2}, "chain": [{"model": "p/x"}]},
                "judge": {"alias": "planner", "params": {"temperature": 0}},
            }
        }
    )
    assert cfg.roles["judge"].chain[0].model == "p/x"
    assert cfg.roles["judge"].params == {"temperature": 0}


@pytest.mark.parametrize(
    "roles",
    [
        {"a": {"alias": "b"}, "b": {"alias": "a"}},
        {"a": {"alias": "missing"}},
        {"a": {"alias": "b", "chain": [{"model": "x"}]}, "b": {"chain": [{"model": "y"}]}},
    ],
)
def test_bad_aliases(roles: dict) -> None:
    with pytest.raises(ConfigError):
        parse_gateway_config({"roles": roles})


def test_empty_roles_rejected() -> None:
    with pytest.raises(ConfigError):
        parse_gateway_config({"roles": {}})


def test_shipped_configs_parse() -> None:
    assert load_gateway_config(MOCK_MODELS).usable_roles()
    real = load_gateway_config(ROOT / "config" / "models.yaml")
    assert {"planner", "worker", "router", "judge"} <= set(real.roles)


def test_missing_file(tmp_path) -> None:
    with pytest.raises(ConfigError):
        load_gateway_config(tmp_path / "nope.yaml")
