"""Command-line interface.

task-agent run "..."        run one task (baseline agent with tools, asks before side effects)
task-agent eval             run the eval suite (and GAIA with --gaia 1)
task-agent tools            list registered tools, their side effects and policy
task-agent models           show each model role's fallback chain
task-agent connect gmail    connect an app account through Composio
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer

from task_agent import __version__
from task_agent.config import Settings, get_settings
from task_agent.core.approval import Approver, ConsoleApprover, Policy, RejectAll
from task_agent.core.errors import TaskAgentError
from task_agent.core.runner import DirectRunner, Runner
from task_agent.core.types import Budget, TaskRequest, TaskStatus
from task_agent.models.gateway import ModelGateway
from task_agent.observability.logging import configure_logging
from task_agent.observability.tracing import flush_tracing, setup_tracing
from task_agent.tools.registry import ToolRegistry

app = typer.Typer(
    name="task-agent",
    help="Hand it any task in plain language; get a finished result.",
    no_args_is_help=True,
    add_completion=False,
)


class RunnerKind(StrEnum):
    BASELINE = "baseline"
    DIRECT = "direct"


@dataclass
class Session:
    settings: Settings
    gateway: ModelGateway
    registry: ToolRegistry
    tool_notes: list[str]
    policy: Policy

    def runner(self, kind: RunnerKind, approver: Approver, role: str = "planner") -> Runner:
        if kind is RunnerKind.DIRECT:
            return DirectRunner(self.gateway, role=role)
        from task_agent.core.baseline import BaselineRunner
        from task_agent.core.tool_executor import ToolExecutor

        executor = ToolExecutor(
            self.registry,
            policy=self.policy,
            approver=approver,
            max_output_chars=self.settings.tool_output_max_chars,
        )
        return BaselineRunner(self.gateway, executor, role=role)


@contextmanager
def _session(models_config: Path | None, *, with_tools: bool = True) -> Iterator[Session]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    setup_tracing(settings)
    try:
        from task_agent.tools.loader import build_registry

        gateway = ModelGateway.from_file(models_config or settings.models_config)
        registry, notes = build_registry(settings) if with_tools else (ToolRegistry(), [])
        policy = Policy.load(settings.policy_file) if with_tools else Policy()
        yield Session(settings, gateway, registry, notes, policy)
    except TaskAgentError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    finally:
        flush_tracing()


ModelsOpt = Annotated[
    Path | None,
    typer.Option("--models", help="Models config YAML (default: TASK_AGENT_MODELS_CONFIG)."),
]
RunnerOpt = Annotated[
    RunnerKind,
    typer.Option("--runner", help="baseline: tool-calling agent. direct: one call, no tools."),
]


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def models(models_config: ModelsOpt = None) -> None:
    """Show each role's resolved fallback chain, and entries skipped for missing keys."""
    with _session(models_config, with_tools=False) as s:
        for role, info in s.gateway.describe().items():
            chain = " -> ".join(info["chain"]) or "(no usable models)"
            colour = typer.colors.GREEN if info["chain"] else typer.colors.YELLOW
            typer.secho(f"{role:<10} {chain}", fg=colour)
            for reason in info["skipped"]:
                typer.secho(f"{'':<10} skipped {reason}", fg=typer.colors.BRIGHT_BLACK)


@app.command()
def tools(models_config: ModelsOpt = None) -> None:
    """List registered tools with their group, side effect and approval rule."""
    with _session(models_config) as s:
        for spec in s.registry.list_tools():
            rule = s.policy.decide(spec).value
            typer.echo(f"{spec.name:<36} {spec.group:<8} {spec.side_effect.value:<13} {rule}")
        for note in s.tool_notes:
            typer.secho(f"note: {note}", fg=typer.colors.BRIGHT_BLACK)


@app.command()
def connect(
    app_name: Annotated[str, typer.Argument(metavar="APP", help="Composio toolkit, e.g. gmail.")],
    wait: Annotated[int, typer.Option(help="Seconds to wait for you to finish.")] = 300,
) -> None:
    """Connect an app account through Composio (OAuth in your browser)."""
    settings = get_settings()
    if not settings.composio_api_key:
        typer.secho("set COMPOSIO_API_KEY first (free at composio.dev)", fg=typer.colors.RED)
        raise typer.Exit(2)
    from task_agent.tools.integrations.composio import make_client

    client = make_client(settings.composio_api_key.get_secret_value())
    request = client.toolkits.authorize(user_id=settings.composio_user_id, toolkit=app_name)
    typer.echo(f"Open this link to connect {app_name}:\n\n  {request.redirect_url}\n")
    try:
        request.wait_for_connection(timeout=wait)
    except Exception as exc:
        typer.secho(f"not connected: {exc}", fg=typer.colors.RED)
        raise typer.Exit(1) from exc
    typer.secho(f"{app_name} connected for user '{settings.composio_user_id}'.", fg="green")


@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="The task, in plain language.")],
    runner_kind: RunnerOpt = RunnerKind.BASELINE,
    role: Annotated[str, typer.Option(help="Model role that drives the agent.")] = "planner",
    max_steps: Annotated[int, typer.Option(min=1)] = 20,
    max_cost: Annotated[float, typer.Option(help="USD cap for this task.", min=0)] = 0.10,
    max_wall: Annotated[float, typer.Option(help="Seconds.", min=1)] = 300,
    max_tokens: Annotated[int, typer.Option(min=1)] = 300_000,
    group: Annotated[
        list[str] | None, typer.Option("--group", "-g", help="Limit to these tool groups.")
    ] = None,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full RunResult.")] = False,
    models_config: ModelsOpt = None,
) -> None:
    """Run one task and print its result. Side effects ask for your approval first."""
    with _session(models_config, with_tools=runner_kind is RunnerKind.BASELINE) as s:
        request = TaskRequest(
            prompt=prompt,
            budget=Budget(
                max_steps=max_steps,
                max_cost_usd=max_cost,
                max_wall_s=max_wall,
                max_tokens=max_tokens,
            ),
            allowed_tool_groups=frozenset(group) if group else None,
        )
        result = asyncio.run(s.runner(runner_kind, ConsoleApprover(), role).run(request))

    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    elif result.status is TaskStatus.SUCCEEDED:
        typer.echo(result.output or "")
    else:
        typer.secho(f"{result.status.value}: {result.error}", fg=typer.colors.RED, err=True)
    typer.secho(
        f"[{result.status.value} · {result.steps} step(s) · {result.tools.calls} tool call(s)"
        f" · ${result.usage.cost_usd:.4f} · {result.usage.total_tokens} tokens"
        f" · {result.wall_s:.1f}s · trace {result.trace_id or 'off'}]",
        fg=typer.colors.BRIGHT_BLACK,
        err=True,
    )
    if result.status is not TaskStatus.SUCCEEDED:
        raise typer.Exit(1)


@app.command("eval")
def eval_cmd(
    suite: Annotated[Path, typer.Argument(help="Suite dir or file.")] = Path("evals/suite"),
    runner_kind: RunnerOpt = RunnerKind.BASELINE,
    category: Annotated[list[str] | None, typer.Option("--category", "-c")] = None,
    tag: Annotated[list[str] | None, typer.Option("--tag", "-t")] = None,
    task_id: Annotated[list[str] | None, typer.Option("--id")] = None,
    gaia: Annotated[
        list[int] | None,
        typer.Option("--gaia", help="Run GAIA validation tasks of this level instead."),
    ] = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    concurrency: Annotated[int, typer.Option(min=1, help="Keep low on free tiers.")] = 2,
    groups: Annotated[
        list[str] | None,
        typer.Option("--groups", "-g", help="Tool groups to allow (default: all registered)."),
    ] = None,
    role: Annotated[str, typer.Option(help="Model role that drives the agent.")] = "planner",
    out: Annotated[Path, typer.Option(help="Report directory.")] = Path("evals/reports/runs"),
    label: Annotated[str, typer.Option(help="Free-text label stored with the run.")] = "",
    min_success: Annotated[
        float | None,
        typer.Option(help="Exit 1 if success rate on attempted tasks is below this (0-1)."),
    ] = None,
    models_config: ModelsOpt = None,
) -> None:
    """Run the eval suite and write a JSON + Markdown report.

    Actions needing approval are rejected during evals, so nothing is sent or posted.
    """
    from task_agent.evals import load_suite, run_eval, select_tasks, to_markdown, write_report

    with _session(models_config, with_tools=runner_kind is RunnerKind.BASELINE) as s:
        if gaia:
            from task_agent.evals.gaia import load_gaia

            source = load_gaia(levels=gaia)
        else:
            source = load_suite(suite)
        tasks = select_tasks(
            source, ids=task_id or (), categories=category or (), tags=tag or (), limit=limit
        )
        if not tasks:
            typer.secho("no tasks matched the filters", fg=typer.colors.YELLOW, err=True)
            raise typer.Exit(2)
        available = set(s.registry.groups)
        if groups:
            available &= set(groups)
        approver = RejectAll("evals never perform side effects; describe the action instead")
        report = asyncio.run(
            run_eval(
                tasks,
                s.runner(runner_kind, approver, role),
                s.gateway,
                available_groups=available,
                concurrency=concurrency,
                label=label,
            )
        )

    json_path, md_path = write_report(report, out)
    typer.echo(to_markdown(report))
    typer.secho(f"report: {json_path}  summary: {md_path}", fg=typer.colors.BRIGHT_BLACK, err=True)

    rate = report.overall.success_rate
    if min_success is not None and (rate is None or rate < min_success):
        typer.secho(
            f"success rate {rate if rate is not None else 'n/a'} below {min_success}",
            fg=typer.colors.RED,
            err=True,
        )
        raise typer.Exit(1)


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
