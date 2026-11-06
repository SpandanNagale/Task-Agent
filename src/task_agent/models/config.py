"""Model-role configuration: which models serve each role, in fallback order.

Code only ever asks for a *role* (planner, worker, router, judge, ...), never a model id, so
swapping models or moving a role from local Ollama to a hosted provider is a YAML change.

Values may reference environment variables as ``${VAR}`` or ``${VAR:-default}``. A chain
entry that references an unset variable with no default is skipped, so a contributor with
only one provider key still gets a working (shorter) chain.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from task_agent.core.errors import ConfigError

_ENV_REF = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class _Missing(Exception):
    def __init__(self, var: str) -> None:
        self.var = var


def _expand(value: Any) -> Any:
    if isinstance(value, str):

        def sub(m: re.Match[str]) -> str:
            var, default = m.group(1), m.group(2)
            resolved = os.environ.get(var)
            if resolved:
                return resolved
            if default is not None:
                return default
            raise _Missing(var)

        return _ENV_REF.sub(sub, value)
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    return value


class ModelEntry(BaseModel):
    """One model in a role's chain. Unknown keys pass straight through to LiteLLM."""

    model_config = ConfigDict(extra="allow")

    model: str = Field(min_length=1)
    name: str | None = None
    # Served on a free tier: actual cost is recorded as $0 (the list price is still tracked
    # as list_cost_usd, so reports can show what the run would cost on a paid plan).
    free_tier: bool = False

    def litellm_params(self) -> dict[str, Any]:
        return self.model_dump(exclude={"name", "free_tier"}, exclude_none=True)


class RoleConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: str = ""
    alias: str | None = None  # reuse another role's chain
    timeout_s: float | None = Field(default=None, gt=0)
    params: dict[str, Any] = Field(default_factory=dict)  # default call params, e.g. temperature
    chain: list[ModelEntry] = Field(default_factory=list)
    skipped: list[str] = Field(default_factory=list, exclude=True)  # why entries were dropped

    @model_validator(mode="after")
    def _alias_or_chain(self) -> RoleConfig:
        if self.alias and self.chain:
            raise ValueError("a role sets either 'alias' or 'chain', not both")
        return self


class GatewayDefaults(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timeout_s: float = Field(default=60.0, gt=0)
    num_retries: int = Field(default=1, ge=0)
    cooldown_s: float = Field(default=30.0, ge=0)


class GatewayConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: GatewayDefaults = Field(default_factory=GatewayDefaults)
    roles: dict[str, RoleConfig]

    @model_validator(mode="after")
    def _resolve_aliases(self) -> GatewayConfig:
        for name, role in self.roles.items():
            seen = {name}
            target = role
            while target.alias:
                if target.alias in seen or target.alias not in self.roles:
                    raise ValueError(f"role '{name}': bad alias chain at '{target.alias}'")
                seen.add(target.alias)
                target = self.roles[target.alias]
            if role.alias:
                role.chain = list(target.chain)
                role.timeout_s = role.timeout_s or target.timeout_s
                role.params = {**target.params, **role.params}
        return self

    def usable_roles(self) -> dict[str, RoleConfig]:
        return {name: role for name, role in self.roles.items() if role.chain}


def parse_gateway_config(raw: dict[str, Any]) -> GatewayConfig:
    roles_raw = raw.get("roles")
    if not isinstance(roles_raw, dict) or not roles_raw:
        raise ConfigError("models config needs a non-empty 'roles' mapping")

    roles: dict[str, Any] = {}
    for role_name, role_raw in roles_raw.items():
        role_raw = dict(role_raw or {})
        chain, skipped = [], []
        for entry in role_raw.pop("chain", []) or []:
            try:
                chain.append(_expand(entry))
            except _Missing as missing:
                label = entry.get("name") or entry.get("model", "?")
                skipped.append(f"{label}: ${{{missing.var}}} not set")
        roles[role_name] = {**_expand(role_raw), "chain": chain, "skipped": skipped}

    try:
        return GatewayConfig.model_validate({"defaults": raw.get("defaults", {}), "roles": roles})
    except ValidationError as exc:
        raise ConfigError(f"invalid models config: {exc}") from exc


def load_gateway_config(path: Path) -> GatewayConfig:
    if not path.is_file():
        raise ConfigError(f"models config not found: {path}")
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"models config is not valid YAML: {path}: {exc}") from exc
    return parse_gateway_config(raw)
