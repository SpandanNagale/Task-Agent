"""Command-line interface.

task-agent run "..."        run one task (baseline agent with tools, asks before side effects)
task-agent resume TASK_ID   continue an interrupted orchestrator task from its checkpoint
task-agent eval             run the eval suite (GAIA with --gaia 1, WebVoyager with --webvoyager 2)
task-agent ui               web UI: live plan, tool calls, approvals and cost per task
task-agent index PATH       index local text files for the agent's recall tool
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
from task_agent.core.approval import Approver, ConsoleApprover, Policy, RejectAll, Rule
from task_agent.core.errors import TaskAgentError
from task_agent.core.runner import DirectRunner, Runner
from task_agent.core.types import Budget, RunResult, TaskRequest, TaskStatus
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


@app.callback()
def _utf8_console() -> None:
    """Hand it any task in plain language; get a finished result."""
    # Windows consoles default to cp1252, which crashes on non-Latin model output.
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")


class RunnerKind(StrEnum):
    BASELINE = "baseline"
    DIRECT = "direct"
    ORCHESTRATOR = "orchestrator"


@dataclass
class Session:
    settings: Settings
    gateway: ModelGateway
    registry: ToolRegistry
    tool_notes: list[str]
    policy: Policy

    def runner(
        self,
        kind: RunnerKind,
        approver: Approver,
        role: str = "planner",
        worker_role: str = "worker",
        *,
        checkpoint: bool = True,
        skills: bool = False,
    ) -> Runner:
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
        if kind is RunnerKind.ORCHESTRATOR:
            from task_agent.core.orchestrator import OrchestratorRunner

            return OrchestratorRunner(
                self.gateway,
                executor,
                planner_role=role,
                worker_role=worker_role,
                checkpoint_db=self.settings.checkpoint_db if checkpoint else None,
            )
        from task_agent.memory import MemoryStore

        store = MemoryStore(self.settings.memory_db) if skills else None
        return BaselineRunner(self.gateway, executor, role=role, skills=store)


@contextmanager
def _session(models_config: Path | None, *, with_tools: bool = True) -> Iterator[Session]:
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_format)
    setup_tracing(settings)
    try:
        from task_agent.tools.loader import build_registry

        gateway = ModelGateway.from_file(models_config or settings.models_config)
        registry, notes = build_registry(settings, gateway) if with_tools else (ToolRegistry(), [])
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
    typer.Option(
        "--runner",
        help="baseline: one tool-calling agent. orchestrator: planner + sub-agents. "
        "direct: one call, no tools.",
    ),
]
WorkerRoleOpt = Annotated[
    str, typer.Option("--worker-role", help="Model role for orchestrator sub-agents.")
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
    worker_role: WorkerRoleOpt = "worker",
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
    with _session(models_config, with_tools=runner_kind is not RunnerKind.DIRECT) as s:
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
        runner = s.runner(
            runner_kind, ConsoleApprover(), role, worker_role, skills=s.settings.skills
        )
        result = asyncio.run(runner.run(request))
    _print_result(result, as_json)


@app.command()
def ui(
    port: Annotated[int, typer.Option(help="Port on 127.0.0.1.")] = 8765,
    role: Annotated[str, typer.Option(help="Model role that drives the agent.")] = "planner",
    worker_role: WorkerRoleOpt = "worker",
    models_config: ModelsOpt = None,
) -> None:
    """Web UI on this machine: start tasks, watch them live, answer approvals."""
    from task_agent.ui import serve

    with _session(models_config) as s:

        def make_runner(kind: str, approver: Approver) -> Runner:
            return s.runner(RunnerKind(kind), approver, role, worker_role, skills=s.settings.skills)

        typer.echo(f"Task Agent UI: http://127.0.0.1:{port}  (Ctrl+C to stop)")
        asyncio.run(serve(make_runner, port))


@app.command()
def serve(
    host: Annotated[str, typer.Option(help="Interface to bind; 0.0.0.0 behind a proxy.")] = (
        "127.0.0.1"
    ),
    port: Annotated[int, typer.Option()] = 8000,
    allowed_host: Annotated[
        list[str] | None,
        typer.Option("--allowed-host", help="Host header to accept (repeat). Default: local."),
    ] = None,
    role: Annotated[str, typer.Option(help="Model role that drives the agent.")] = "planner",
    worker_role: WorkerRoleOpt = "worker",
    models_config: ModelsOpt = None,
) -> None:
    """The /v1 HTTP API for programs (API keys: `task-agent keys create`). Docs at /docs."""
    import uvicorn

    from task_agent.api import KeyStore, create_api

    with _session(models_config) as s:

        def make_runner(kind: str, approver: Approver) -> Runner:
            return s.runner(RunnerKind(kind), approver, role, worker_role, skills=s.settings.skills)

        api = create_api(make_runner, KeyStore(s.settings.api_db), allowed_hosts=allowed_host)
        typer.echo(f"Task Agent API: http://{host}:{port}/docs  (Ctrl+C to stop)")
        uvicorn.run(api, host=host, port=port, log_level="warning", proxy_headers=True)


keys_app = typer.Typer(help="Manage /v1 API keys.", no_args_is_help=True)
app.add_typer(keys_app, name="keys")


@keys_app.command("create")
def keys_create(
    name: Annotated[str, typer.Argument(help="Who or what the key is for.")],
    scope: Annotated[
        list[str] | None,
        typer.Option(help="tasks:read, tasks:write, approvals:write (repeat)."),
    ] = None,
    group: Annotated[
        list[str] | None, typer.Option(help="Tool groups this key may use (repeat). Default: all.")
    ] = None,
    rate: Annotated[int, typer.Option(help="Requests per minute.")] = 30,
    running: Annotated[int, typer.Option(help="Tasks running at once.")] = 3,
    daily_tasks: Annotated[int, typer.Option()] = 200,
    daily_cost: Annotated[float, typer.Option(help="USD per UTC day.")] = 1.0,
) -> None:
    """Create a key. The secret is printed once; only its hash is stored."""
    from task_agent.api import KeyStore

    try:
        key_id, secret = KeyStore(get_settings().api_db).create(
            name,
            set(scope or ["tasks:read", "tasks:write"]),
            groups=set(group) if group else None,
            rate_per_min=rate,
            max_running=running,
            daily_tasks=daily_tasks,
            daily_cost_usd=daily_cost,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(f"key id {key_id}\n{secret}\nStore it now: it cannot be shown again.")


@keys_app.command("list")
def keys_list() -> None:
    """Show keys (never the secrets)."""
    from task_agent.api import KeyStore

    for k in KeyStore(get_settings().api_db).list():
        state = "revoked" if k["revoked"] else "active"
        typer.echo(
            f"{k['id']}  {state:7}  {k['name']}  scopes={k['scopes']}"
            f"  groups={k['groups'] or 'all'}"
            f"  {k['rate_per_min']}/min  running<={k['max_running']}"
            f"  {k['daily_tasks']} tasks/day  ${k['daily_cost_usd']}/day"
        )


@keys_app.command("revoke")
def keys_revoke(key_id: Annotated[str, typer.Argument()]) -> None:
    from task_agent.api import KeyStore

    if not KeyStore(get_settings().api_db).revoke(key_id):
        typer.secho(f"no key {key_id}", fg="red", err=True)
        raise typer.Exit(1)
    typer.echo(f"revoked {key_id}")


@app.command()
def index(
    paths: Annotated[
        list[Path] | None, typer.Argument(help="Files or folders of text files.")
    ] = None,
) -> None:
    """Index local text files so the agent's recall tool can search them.

    Re-indexing a file replaces its old chunks. With no paths, shows what is stored.
    """
    from task_agent.memory import TEXT_SUFFIXES, MemoryStore

    store = MemoryStore(get_settings().memory_db)
    for path in paths or []:
        files = sorted(p for p in path.rglob("*") if p.is_file()) if path.is_dir() else [path]
        for file in files:
            if file.suffix.lower() not in TEXT_SUFFIXES:
                continue
            typer.echo(f"{file}: {store.index_file(file)} chunk(s)")
    counts = store.count()
    typer.echo(
        f"stored: {counts.get('skill', 0)} skill(s), {counts.get('memory', 0)} memory note(s), "
        f"{counts.get('knowledge', 0)} knowledge chunk(s)"
    )


@app.command()
def resume(
    task_id: Annotated[str, typer.Argument(help="Task id printed by an orchestrator run.")],
    role: Annotated[str, typer.Option(help="Model role for the planner.")] = "planner",
    worker_role: WorkerRoleOpt = "worker",
    as_json: Annotated[bool, typer.Option("--json", help="Print the full RunResult.")] = False,
    models_config: ModelsOpt = None,
) -> None:
    """Continue an interrupted orchestrator task from its last checkpoint.

    The task's budget starts fresh; finished steps are not run again.
    """
    from task_agent.core.orchestrator import load_request

    with _session(models_config) as s:
        request = asyncio.run(load_request(s.settings.checkpoint_db, task_id))
        if request is None:
            typer.secho(f"no checkpoint for {task_id}", fg=typer.colors.RED, err=True)
            raise typer.Exit(2)
        runner = s.runner(RunnerKind.ORCHESTRATOR, ConsoleApprover(), role, worker_role)
        result = asyncio.run(runner.run(request))
    _print_result(result, as_json)


def _print_result(result: RunResult, as_json: bool) -> None:
    if as_json:
        typer.echo(result.model_dump_json(indent=2))
    elif result.status is TaskStatus.SUCCEEDED:
        typer.echo(result.output or "")
    else:
        typer.secho(f"{result.status.value}: {result.error}", fg=typer.colors.RED, err=True)
    typer.secho(
        f"[{result.status.value} · {result.steps} step(s) · {result.tools.calls} tool call(s)"
        f" · ${result.usage.cost_usd:.4f} · {result.usage.total_tokens} tokens"
        f" · {result.wall_s:.1f}s · {result.task_id} · trace {result.trace_id or 'off'}]",
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
    webvoyager: Annotated[
        int | None,
        typer.Option(
            "--webvoyager",
            min=1,
            help="Run WebVoyager golden-answer tasks instead: this many per website.",
        ),
    ] = None,
    wv_site: Annotated[
        list[str] | None, typer.Option("--wv-site", help="Limit WebVoyager to these sites.")
    ] = None,
    limit: Annotated[int | None, typer.Option(min=1)] = None,
    concurrency: Annotated[int, typer.Option(min=1, help="Keep low on free tiers.")] = 2,
    groups: Annotated[
        list[str] | None,
        typer.Option("--groups", "-g", help="Tool groups to allow (default: all registered)."),
    ] = None,
    role: Annotated[str, typer.Option(help="Model role that drives the agent.")] = "planner",
    worker_role: WorkerRoleOpt = "worker",
    allow: Annotated[
        list[str] | None,
        typer.Option(
            "--allow",
            help="Let this tool run without approval for this eval run only (repeatable). "
            "Everything else needing approval is still rejected.",
        ),
    ] = None,
    skills: Annotated[
        bool,
        typer.Option(
            "--skills",
            help="Reuse and save skills during this run (off by default so runs stay "
            "independent; use it to measure skill reuse).",
        ),
    ] = False,
    out: Annotated[Path, typer.Option(help="Report directory.")] = Path("evals/reports/runs"),
    label: Annotated[str, typer.Option(help="Free-text label stored with the run.")] = "",
    min_success: Annotated[
        float | None,
        typer.Option(help="Exit 1 if success rate on attempted tasks is below this (0-1)."),
    ] = None,
    models_config: ModelsOpt = None,
) -> None:
    """Run the eval suite and write a JSON + Markdown report.

    Actions needing approval are rejected during evals, so nothing is sent or posted,
    except tools you name with --allow.
    """
    from task_agent.evals import load_suite, run_eval, select_tasks, to_markdown, write_report

    with _session(models_config, with_tools=runner_kind is not RunnerKind.DIRECT) as s:
        if gaia:
            from task_agent.evals.gaia import load_gaia

            source = load_gaia(levels=gaia)
        elif webvoyager:
            from task_agent.evals.webvoyager import load_webvoyager

            source = load_webvoyager(per_site=webvoyager, sites=wv_site or ())
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
        for name in allow or ():
            s.policy.tools[name] = Rule.ALLOW
        roles = {role, "judge"} | (
            {worker_role} if runner_kind is RunnerKind.ORCHESTRATOR else set()
        )
        dead = asyncio.run(_preflight(s.gateway, roles))
        if dead:
            for line in dead:
                typer.secho(f"preflight: {line}", fg=typer.colors.RED, err=True)
            typer.secho("not running the eval: fix the models above first", fg="red", err=True)
            raise typer.Exit(2)
        approver = RejectAll("evals never perform side effects; describe the action instead")
        report = asyncio.run(
            run_eval(
                tasks,
                s.runner(runner_kind, approver, role, worker_role, checkpoint=False, skills=skills),
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


async def _preflight(gateway: ModelGateway, roles: set[str]) -> list[str]:
    """One tiny call per role, so a revoked key fails in seconds instead of failing every
    task of a long eval run (the router reports a 401 as a cooldown, which hides it)."""
    problems = []
    for name in sorted(roles):
        if name not in gateway.roles:
            continue
        try:
            await gateway.complete(
                name, [{"role": "user", "content": "Reply with OK."}], timeout_s=60
            )
        except TaskAgentError as exc:
            problems.append(f"role '{name}' failed: {str(exc)[:300]}")
    return problems


def main() -> None:  # pragma: no cover
    try:
        app()
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":  # pragma: no cover
    main()
