# Task Agent

**Any task in, finished result out.** Hand it a task in plain language and it plans the work,
picks or spawns the right workers and tools, and carries the task through to a verified
result, or comes back to you with a clear question.

The goal is to show how much of a standard agent benchmark **local models plus good
orchestration** can solve on a 12 GB consumer GPU, and at what cost per task. Success rates
and cost per task will be published here as the phases land.

> **Status: Phase 0 (foundations).** The model gateway, budgets, tracing and eval harness
> work end to end. The agent doesn't use tools yet; those arrive in Phase 1. See the
> [roadmap](docs/ROADMAP.md).

## Design principles

1. **Earn every agent.** Start with one agent and its tools. Add the orchestrator and
   sub-agents only where they beat that baseline on the eval suite.
2. **Plan, act, observe, replan.** The agent runs a loop, never a fixed pipeline.
3. **Every capability is a tool** behind one interface (MCP), chosen by its description.
4. **Code is the universal fallback.** When no tool fits, the agent writes one and runs it
   in a sandbox.
5. **A human approves every side effect.** Sending, posting, paying and deleting all wait
   for you.
6. **Observable and measurable.** Every step is traced and every capability is benchmarked.
7. **Local-first where it's cheap.** A small local model handles routing and simple steps;
   the strongest model you can afford does the planning.

## Quick start

Requires Python 3.11+ and [uv](https://docs.astral.sh/uv/).

```bash
git clone https://github.com/SpandanNagale/Task-Agent && cd Task-Agent
uv sync

# No keys needed: mock models, fully offline
uv run task-agent run "What is 6 x 7?" --models config/models.mock.yaml
uv run task-agent eval --models config/models.mock.yaml
```

With real models:

```bash
cp .env.example .env          # add a key for at least one provider
uv run task-agent models      # shows which models each role resolved to
uv run task-agent run "How many days are there from 2024-01-15 to 2026-03-01?"
uv run task-agent eval --tag smoke
```

### Tracing with Langfuse (self-hosted)

```bash
docker compose -f deploy/compose.dev.yml --env-file .env up -d --wait
uv run task-agent run "Draft a two-line thank-you note to my team."
# Open http://localhost:3000 (login: dev@task-agent.local / taskagent-dev). The task shows
# up as a trace, with every model call, its tokens and its cost.
```

Tracing is plain OpenTelemetry, so you can set `TASK_AGENT_TRACING=otlp` and
`TASK_AGENT_OTLP_ENDPOINT` to send traces to Jaeger, Tempo, Honeycomb or any OTLP collector.

## How it fits together

```
 you ─► CLI / API ─► Runner ─► ModelGateway (role → fallback chain, via LiteLLM)
                       │              │
                       │              └─► budgets: steps · cost · tokens · wall time
                       ├─► ToolRegistry (MCP + Python tools, side-effect levels)   [phase 1]
                       └─► OpenTelemetry trace per task ─► Langfuse / any OTLP backend
```

- **Model roles, not model IDs.** Code asks for `planner`, `worker`, `router` or `judge`.
  [`config/models.yaml`](config/models.yaml) maps each role to a fallback chain (Ollama
  Cloud → OpenRouter → Gemini → Groq for planning; local Ollama first for workers). Moving a
  role from local to hosted is a config change.
- **Budgets on every task.** Each task has limits on steps, cost, tokens and wall time.
  Hitting one stops the task with a report instead of looping silently.
- **One `Runner` protocol.** The CLI, the eval harness and (later) the queue workers depend
  only on it. Each phase swaps the agent behind it, and the baseline stays runnable for
  comparisons.

More detail is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Evaluation

The eval suite has 20 internal tasks in five categories: research, browsing, data & code,
communication, and multi-step. Each task has a deterministic checker or an LLM-judge rubric.
Tasks that need tool groups the current agent doesn't have are reported as *skipped*, not
failed, so one suite serves every phase. See [evals/README.md](evals/README.md).

```bash
uv run task-agent eval --category data_code --concurrency 8 --label "baseline qwen3-14b"
uv run task-agent eval --tag smoke --min-success 0.6   # non-zero exit below the bar
```

## Repository layout

```
src/task_agent/
  core/            contracts (TaskRequest, RunResult, Budget), budgets, runners
  models/          role-based model gateway on LiteLLM
  tools/           the ToolSpec contract and registry
  observability/   OpenTelemetry tracing, structured logs, secret redaction
  evals/           suite schema, checkers, runner, reports
  cli.py           `task-agent` command
config/            model role chains (real + offline mock)
evals/suite/       internal eval tasks (YAML)
deploy/            docker-compose stack and Dockerfile
docs/              architecture and roadmap
```

## Contributing

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) first. The short version:
`uv sync`, `uv run pytest`, `uv run ruff check .`, `uv run mypy`, and any new capability
comes with eval tasks that cover it.

Security issues: see [SECURITY.md](SECURITY.md). Please don't open public issues for them.

## License

[Apache 2.0](LICENSE)
