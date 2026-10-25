"""Command-line interface: ``task-agent run``, ``task-agent eval``, ``task-agent models``."""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated

import typer

from task_agent import __version__
from task_agent.config import Settings, get_settings
from task_agent.core.errors import TaskAgentError
from task_agent.core.runner import DirectRunner
from task_agent.core.types import Budget, TaskRequest, TaskStatus
from task_agent.models.gateway import ModelGateway
from task_agent.observability.logging import configure_logging
from task_agent.observability.tracing import flush_tracing, setup_tracing

app = typer.Typer(
    name="task-agent",
    help="Hand it any task in plain language; get a finished result.",
    no_args_is_help=True,
    add_completion=False,
)


@contextmanager
def _session(models_config: Path | None) -> Iterator[tuple[Settings, ModelGateway]]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    setup_tracing(settings)
    try:
        gateway = ModelGateway.from_file(models_config or settings.models_config)
        yield settings, gateway
    except TaskAgentError as exc:
        typer.secho(f"error: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(2) from exc
    finally:
        flush_tracing()


ModelsOpt = Annotated[
    Path | None,
    typer.Option("--models", help="Models config YAML (default: TASK_AGENT_MODELS_CONFIG)."),
]


@app.command()
def version() -> None:
    """Print the version."""
    typer.echo(__version__)


@app.command()
def models(models_config: ModelsOpt = None) -> None:
    """Show each role's resolved fallback chain, and entries skipped for missing keys."""
    with _session(models_config) as (_, gateway):
        for role, info in gateway.describe().items():
            chain = " -> ".join(info["chain"]) or "(no usable models)"
            colour = typer.colors.GREEN if info["chain"] else typer.colors.YELLOW
            typer.secho(f"{role:<10} {chain}", fg=colour)
            for reason in info["skipped"]:
                typer.secho(f"{'':<10} skipped {reason}", fg=typer.colors.BRIGHT_BLACK)


@app.command()
def run(
    prompt: Annotated[str, typer.Argument(help="The task, in plain language.")],
    role: Annotated[str, typer.Option(help="Model role that answers.")] = "planner",
    max_steps: Annotated[int, typer.Option(min=1)] = 25,
    max_cost: Annotated[float, typer.Option(help="USD cap for this task.", min=0)] = 0.50,
    max_wall: Annotated[float, typer.Option(help="Seconds.", min=1)] = 600,
    as_json: Annotated[bool, typer.Option("--json", help="Print the full RunResult.")] = False,
    models_config: ModelsOpt = None,
) -> None:
    """Run one task and print its result."""
    with _session(models_config) as (_, gateway):
        request = TaskRequest(
            prompt=prompt,
            budget=Budget(max_steps=max_steps, max_cost_usd=max_cost, max_wall_s=max_wall),
        )
        result = asyncio.run(DirectRunner(gateway, role=role).run(request))

    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    elif result.status is TaskStatus.SUCCEEDED:
        typer.echo(result.output or "")
    else:
        typer.secho(f"{result.status.value}: {result.error}", fg=typer.colors.RED, err=True)
    typer.secho(
        f"[{result.status.value} · {result.steps} step(s) · ${result.usage.cost_usd:.4f}"
        f" · {result.wall_s:.1f}s · trace {result.trace_id or 'off'}]",
        fg=typer.colors.BRIGHT_BLACK,
        err=True,
    )
    if result.status is not TaskStatus.SUCCEEDED:
        raise typer.Exit(1)


@app.command("eval")
def eval_cmd(
    suite: Annotated[Path, typer.Argument(help="Suite dir or file.")] = Path("evals/suite"),
    category: Annotated[list[str] | None, typer.Option("--category", "-c")] = None,
    tag: Annotated[list[str] | None, typer.Option("--tag", "-t")] = None,
    task_id: Annotated[list[str] | None, typer.Option("--id")] = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    concurrency: Annotated[int, typer.Option(min=1)] = 4,
    groups: Annotated[
        list[str] | None,
        typer.Option("--groups", "-g", help="Tool groups the runner has (phase 0: none)."),
    ] = None,
    role: Annotated[str, typer.Option(help="Model role for the direct runner.")] = "planner",
    out: Annotated[Path, typer.Option(help="Report directory.")] = Path("evals/reports/runs"),
    label: Annotated[str, typer.Option(help="Free-text label stored with the run.")] = "",
    min_success: Annotated[
        float | None,
        typer.Option(help="Exit 1 if success rate on attempted tasks is below this (0-1)."),
    ] = None,
    models_config: ModelsOpt = None,
) -> None:
    """Run the eval suite and write a JSON + Markdown report."""
    from task_agent.evals import load_suite, run_eval, select_tasks, to_markdown, write_report

    with _session(models_config) as (_, gateway):
        tasks = select_tasks(
            load_suite(suite),
            ids=task_id or (),
            categories=category or (),
            tags=tag or (),
            limit=limit,
        )
        if not tasks:
            typer.secho("no tasks matched the filters", fg=typer.colors.YELLOW, err=True)
            raise typer.Exit(2)
        report = asyncio.run(
            run_eval(
                tasks,
                DirectRunner(gateway, role=role),
                gateway,
                available_groups=groups or (),
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
