import json

import pytest

from task_agent.core import Budget, BudgetExceeded, BudgetTracker, ConfigError, ModelError
from task_agent.models import ModelGateway, parse_gateway_config

MSGS = [{"role": "user", "content": "hi"}]


async def test_complete_records_usage_and_span(mock_gateway: ModelGateway, spans) -> None:
    budget = BudgetTracker(Budget())
    c = await mock_gateway.complete("planner", MSGS, budget=budget)
    assert "FINAL ANSWER: 42" in (c.content or "")
    assert c.model == "openai/mock-planner"
    assert c.usage.calls == 1
    assert budget.usage.calls == 1

    (span,) = spans.get_finished_spans()
    assert span.name == "llm.planner"
    attrs = span.attributes
    assert attrs["langfuse.observation.type"] == "generation"
    assert attrs["gen_ai.response.model"] == "openai/mock-planner"
    assert json.loads(attrs["langfuse.observation.input"])[0]["content"] == "hi"


async def test_fallback_to_next_model() -> None:
    gw = ModelGateway(
        parse_gateway_config(
            {
                "defaults": {"timeout_s": 5, "num_retries": 0},
                "roles": {
                    "planner": {
                        "chain": [
                            # nothing listens on port 9: fails fast
                            {
                                "model": "openai/dead",
                                "api_key": "x",
                                "api_base": "http://127.0.0.1:9",
                            },
                            {"model": "openai/alive", "api_key": "x", "mock_response": "fine"},
                        ]
                    }
                },
            }
        )
    )
    c = await gw.complete("planner", MSGS)
    assert c.content == "fine"
    assert c.model == "openai/alive"


async def test_all_models_fail_raises_model_error(spans) -> None:
    gw = ModelGateway(
        parse_gateway_config(
            {
                "defaults": {"timeout_s": 5, "num_retries": 0},
                "roles": {
                    "planner": {
                        "chain": [
                            {
                                "model": "openai/dead",
                                "api_key": "x",
                                "api_base": "http://127.0.0.1:9",
                            }
                        ]
                    }
                },
            }
        )
    )
    with pytest.raises(ModelError):
        await gw.complete("planner", MSGS)
    (span,) = spans.get_finished_spans()
    assert span.attributes["langfuse.observation.level"] == "ERROR"


async def test_unknown_role(mock_gateway: ModelGateway) -> None:
    with pytest.raises(ConfigError, match="no usable models for role 'nope'"):
        await mock_gateway.complete("nope", MSGS)


async def test_exhausted_budget_blocks_call(mock_gateway: ModelGateway, spans) -> None:
    now = [0.0]
    budget = BudgetTracker(Budget(max_wall_s=1), clock=lambda: now[0])
    now[0] = 999.0
    with pytest.raises(BudgetExceeded):
        await mock_gateway.complete("planner", MSGS, budget=budget)
    assert spans.get_finished_spans() == ()


def test_describe(mock_gateway: ModelGateway) -> None:
    info = mock_gateway.describe()
    assert info["planner"]["chain"] == ["mock"]
    assert mock_gateway.roles == ["judge", "planner", "router", "worker"]


async def test_free_tier_costs_zero_but_keeps_list_price() -> None:
    def gw(free: bool) -> ModelGateway:
        entry = {"model": "openai/gpt-4o-mini", "api_key": "x", "mock_response": "hi"}
        if free:
            entry["free_tier"] = True
        return ModelGateway(parse_gateway_config({"roles": {"planner": {"chain": [entry]}}}))

    paid = await gw(False).complete("planner", MSGS)
    free = await gw(True).complete("planner", MSGS)
    assert paid.usage.cost_usd > 0
    assert paid.usage.list_cost_usd == paid.usage.cost_usd
    assert free.usage.cost_usd == 0
    assert free.usage.list_cost_usd == paid.usage.list_cost_usd
