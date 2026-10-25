"""Approval gate and policy engine.

Every tool call is checked against a policy before it runs:

- ``allow``: runs without asking (read tools, sandboxed code)
- ``ask``:   pauses for a human decision: approve, edit the arguments, or reject
- ``deny``:  never runs

Rules come from a YAML policy file. A rule for a specific tool beats the default for its
side-effect level. Irreversible tools can never be set to ``allow`` by default.

The ``Approver`` is an interface so the same gate works in a terminal today and, from
phase 6, as a paused task that resumes when the answer arrives through the API.
"""

from __future__ import annotations

import json
import sys
from enum import StrEnum
from pathlib import Path
from typing import Any, Protocol

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from task_agent.core.errors import ConfigError
from task_agent.tools.base import SideEffect, ToolSpec


class Rule(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


_BASE_DEFAULTS = {
    SideEffect.READ: Rule.ALLOW,
    SideEffect.WRITE: Rule.ASK,
    SideEffect.IRREVERSIBLE: Rule.ASK,
}


class Policy(BaseModel):
    model_config = ConfigDict(extra="forbid")

    defaults: dict[SideEffect, Rule] = Field(default_factory=lambda: dict(_BASE_DEFAULTS))
    tools: dict[str, Rule] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _irreversible_needs_a_human(self) -> Policy:
        if self.defaults.get(SideEffect.IRREVERSIBLE) is Rule.ALLOW:
            raise ValueError(
                "irreversible actions cannot default to 'allow'; allow specific tools instead"
            )
        for level, rule in _BASE_DEFAULTS.items():
            self.defaults.setdefault(level, rule)
        return self

    def decide(self, spec: ToolSpec) -> Rule:
        return self.tools.get(spec.name) or self.defaults[spec.side_effect]

    @classmethod
    def load(cls, path: Path) -> Policy:
        if not path.is_file():
            raise ConfigError(f"policy file not found: {path}")
        try:
            return cls.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        except (yaml.YAMLError, ValidationError) as exc:
            raise ConfigError(f"invalid policy file {path}: {exc}") from exc


class ApprovalRequest(BaseModel):
    task_id: str
    tool: str
    side_effect: SideEffect
    description: str
    args: dict[str, Any]


class Verdict(StrEnum):
    APPROVE = "approve"
    EDIT = "edit"
    REJECT = "reject"


class ApprovalDecision(BaseModel):
    verdict: Verdict
    args: dict[str, Any] | None = None  # replacement arguments when verdict == edit
    reason: str = ""


class Approver(Protocol):
    async def request(self, req: ApprovalRequest) -> ApprovalDecision: ...


class RejectAll:
    """Rejects every request. For evals and non-interactive runs: nothing with a side effect
    happens unless the policy explicitly allows that tool.
    """

    def __init__(self, reason: str = "no human available to approve this action") -> None:
        self.reason = reason

    async def request(self, req: ApprovalRequest) -> ApprovalDecision:
        return ApprovalDecision(verdict=Verdict.REJECT, reason=self.reason)


class ConsoleApprover:
    """Asks on the terminal. Falls back to rejecting when stdin is not interactive."""

    async def request(self, req: ApprovalRequest) -> ApprovalDecision:
        import asyncio

        if not sys.stdin.isatty():
            return ApprovalDecision(verdict=Verdict.REJECT, reason="not an interactive terminal")
        return await asyncio.to_thread(self._ask, req)

    def _ask(self, req: ApprovalRequest) -> ApprovalDecision:
        import typer

        typer.secho(
            f"\n⚠ Approval needed: {req.tool} ({req.side_effect.value})",
            fg=typer.colors.YELLOW,
            bold=True,
            err=True,
        )
        typer.echo(json.dumps(req.args, indent=2, ensure_ascii=False), err=True)
        while True:
            choice = typer.prompt("[a]pprove / [e]dit / [r]eject", default="r", err=True)
            match choice.strip().lower()[:1]:
                case "a":
                    return ApprovalDecision(verdict=Verdict.APPROVE)
                case "r":
                    reason = typer.prompt("reason (optional)", default="", err=True)
                    return ApprovalDecision(verdict=Verdict.REJECT, reason=reason)
                case "e":
                    raw = typer.prompt("new arguments as JSON", err=True)
                    try:
                        args = json.loads(raw)
                    except json.JSONDecodeError as exc:
                        typer.secho(f"not valid JSON: {exc}", fg=typer.colors.RED, err=True)
                        continue
                    if isinstance(args, dict):
                        return ApprovalDecision(verdict=Verdict.EDIT, args=args)
                    typer.secho("arguments must be a JSON object", fg=typer.colors.RED, err=True)
