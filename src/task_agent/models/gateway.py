"""The model gateway: every LLM call in the system goes through ``ModelGateway.complete``.

It resolves a role to its fallback chain, applies timeouts clamped to the task's remaining
wall time, records cost and tokens against the task budget, and emits one trace span per
call. Callers never see provider details.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from task_agent.core.budget import BudgetTracker
from task_agent.core.errors import ConfigError, ModelError
from task_agent.core.types import Usage
from task_agent.models.config import GatewayConfig, RoleConfig, load_gateway_config
from task_agent.observability.logging import get_logger
from task_agent.observability.tracing import generation_span, record_error, record_generation

log = get_logger(__name__)

# Use the price map bundled with LiteLLM instead of fetching it on import: no network call at
# startup, works offline, and cost numbers stay fixed for a given LiteLLM version, so eval
# runs stay comparable. Set LITELLM_LOCAL_MODEL_COST_MAP=False to fetch the latest prices.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")

Message = dict[str, Any]


class Completion(BaseModel):
    role: str
    model: str  # the model that actually answered, after any fallback
    content: str | None
    tool_calls: list[dict[str, Any]] = Field(default_factory=list)
    finish_reason: str | None = None
    usage: Usage
    latency_s: float


def _deployment(role: str, index: int) -> str:
    return f"{role}::{index}"


class ModelGateway:
    def __init__(self, config: GatewayConfig) -> None:
        from litellm import Router

        self.config = config
        self._roles = config.usable_roles()
        model_list: list[dict[str, Any]] = []
        fallbacks: list[dict[str, list[str]]] = []
        for role, rc in self._roles.items():
            names = [_deployment(role, i) for i in range(len(rc.chain))]
            for name, entry in zip(names, rc.chain, strict=True):
                model_list.append({"model_name": name, "litellm_params": entry.litellm_params()})
            if len(names) > 1:
                fallbacks.append({names[0]: names[1:]})

        d = config.defaults
        self._router = (
            Router(
                model_list=model_list,
                fallbacks=fallbacks,
                num_retries=d.num_retries,
                timeout=d.timeout_s,
                cooldown_time=d.cooldown_s,
            )
            if model_list
            else None
        )

    @classmethod
    def from_file(cls, path: Path) -> ModelGateway:
        return cls(load_gateway_config(path))

    @property
    def roles(self) -> list[str]:
        return sorted(self._roles)

    def describe(self) -> dict[str, dict[str, Any]]:
        """Resolved chains per role, including entries skipped for missing env vars."""
        return {
            name: {
                "chain": [e.name or e.model for e in rc.chain],
                "skipped": rc.skipped,
                "timeout_s": rc.timeout_s or self.config.defaults.timeout_s,
            }
            for name, rc in sorted(self.config.roles.items())
        }

    def _role(self, role: str) -> RoleConfig:
        rc = self._roles.get(role)
        if rc is None:
            known = ", ".join(self.roles) or "none"
            raise ConfigError(f"no usable models for role '{role}' (usable roles: {known})")
        return rc

    async def complete(
        self,
        role: str,
        messages: list[Message],
        *,
        tools: list[dict[str, Any]] | None = None,
        budget: BudgetTracker | None = None,
        timeout_s: float | None = None,
        **params: Any,
    ) -> Completion:
        rc = self._role(role)
        assert self._router is not None

        timeout = timeout_s or rc.timeout_s or self.config.defaults.timeout_s
        if budget is not None:
            budget.check()
            remaining = budget.remaining_s
            if remaining is not None:
                timeout = max(1.0, min(timeout, remaining))

        call_params = {**rc.params, **params}
        if tools:
            call_params["tools"] = tools

        with generation_span(role, messages) as span:
            started = time.perf_counter()
            try:
                resp = await self._router.acompletion(
                    model=_deployment(role, 0),
                    messages=messages,
                    timeout=timeout,
                    **call_params,
                )
            except Exception as exc:
                record_error(span, exc)
                log.warning("model.call_failed", role=role, error=type(exc).__name__)
                raise ModelError(f"role '{role}': all models failed: {exc}") from exc

            completion = _to_completion(role, resp, time.perf_counter() - started)
            record_generation(
                span,
                model=completion.model,
                output=completion.content
                if not completion.tool_calls
                else {"content": completion.content, "tool_calls": completion.tool_calls},
                prompt_tokens=completion.usage.prompt_tokens,
                completion_tokens=completion.usage.completion_tokens,
                cost_usd=completion.usage.cost_usd,
            )

        log.debug(
            "model.call",
            role=role,
            model=completion.model,
            tokens=completion.usage.total_tokens,
            cost_usd=completion.usage.cost_usd,
            latency_s=round(completion.latency_s, 3),
        )
        if budget is not None:
            budget.charge(completion.usage)
        return completion


def _to_completion(role: str, resp: Any, latency_s: float) -> Completion:
    hidden: dict[str, Any] = getattr(resp, "_hidden_params", None) or {}
    choice = resp.choices[0]
    message = choice.message
    usage = getattr(resp, "usage", None)

    cost = hidden.get("response_cost")
    if cost is None:
        try:
            import litellm

            cost = litellm.completion_cost(completion_response=resp)
        except Exception:  # unknown pricing (local models, new ids): count as free
            cost = 0.0

    tool_calls = [
        tc.model_dump() if hasattr(tc, "model_dump") else dict(tc)
        for tc in (getattr(message, "tool_calls", None) or [])
    ]
    return Completion(
        role=role,
        model=hidden.get("litellm_model_name") or getattr(resp, "model", None) or "unknown",
        content=getattr(message, "content", None),
        tool_calls=tool_calls,
        finish_reason=getattr(choice, "finish_reason", None),
        usage=Usage(
            prompt_tokens=getattr(usage, "prompt_tokens", 0) or 0,
            completion_tokens=getattr(usage, "completion_tokens", 0) or 0,
            cost_usd=float(cost or 0.0),
            calls=1,
        ),
        latency_s=latency_s,
    )
