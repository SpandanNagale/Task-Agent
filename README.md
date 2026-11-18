# Task Agent

**Any task in, finished result out.** Hand it a task in plain language and it plans the work,
picks or spawns the right workers and tools, and carries the task through to a verified
result, or comes back to you with a clear question.

The goal is to show how much of a standard agent benchmark **local models plus good
orchestration** can solve on a 12 GB consumer GPU, and at what cost per task. Success rates
and cost per task will be published here as the phases land.

> **Status: Phase 4 done (skills, memory, local knowledge, web UI); Phase 5 (local benchmarks) next.** A planner splits the task into steps and runs them as
> sub-agents in parallel (`--runner orchestrator`), next to the Phase 1 single-agent
> baseline. Tools: web search, page reading, a cloud code sandbox and Gmail, behind an
> approval gate. Runs on free tiers or fully on a local GPU. See the
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

Fully local on a GPU with [Ollama](https://ollama.com) (`ollama pull qwen3:14b`, then set
`OLLAMA_BASE_URL=http://localhost:11434` in `.env`):

```bash
uv run task-agent run "..." --role worker                       # baseline agent
uv run task-agent run "..." --runner orchestrator --role worker  # planner + sub-agents
uv run task-agent resume TASK_ID                                 # continue an interrupted run
uv run task-agent ui                 # web UI at http://127.0.0.1:8765: live plan, approvals, cost
uv run task-agent index ~/notes      # index text files for the agent's recall tool
```

With real models and tools. Every service below has a free tier, and nothing runs on
your machine except the CLI:

```bash
cp .env.example .env    # add CEREBRAS_API_KEY (or Groq / OpenRouter / Gemini),
                        # TAVILY_API_KEY, E2B_API_KEY, and optionally COMPOSIO_API_KEY
uv run task-agent models          # which models each role resolved to
uv run task-agent tools           # which tools loaded, their side effects and approval rule
uv run task-agent run "What is the sum of all primes below 1000?"
uv run task-agent run "Find the arXiv id of 'Attention Is All You Need'"

uv run task-agent connect gmail   # one-time OAuth through Composio
uv run task-agent run "Draft an email to me summarising today's unread mail"
```

### HTTP API

`/v1` for programs: API keys with scopes and per-key limits, live progress as server-sent
events, approvals over HTTP, RFC 9457 errors and idempotent task creation. OpenAPI docs at
`/docs`.

```bash
uv run task-agent keys create my-app --scope tasks:read --scope tasks:write --group search
uv run task-agent serve --port 8000

curl -X POST localhost:8000/v1/tasks -H "Authorization: Bearer $KEY" \
     -H "Idempotency-Key: 1" -H "Content-Type: application/json" \
     -d '{"prompt": "What is 17 * 23 + 4?"}'               # 202 {"task_id": ...}
curl -N localhost:8000/v1/tasks/$ID/events -H "Authorization: Bearer $KEY"  # SSE until done
```

Anything that writes to the outside world (sending, posting, deleting) stops and shows
you the exact action. You can approve it, edit the arguments, or reject it. The rules are
in [`policies/default.yaml`](policies/default.yaml).

### Free-tier defaults

| Piece | Default | Free tier |
|---|---|---|
| Planner model | `gpt-oss-120b` on Cerebras → Groq → OpenRouter → Ollama Cloud, then Gemini Flash | all free; one key is enough |
| Web search | Tavily | 1,000 searches/month |
| Page reading | Jina Reader (Firecrawl if you add a key) | free |
| Code sandbox | E2B | $100 one-time credit |
| Apps | Composio (Gmail first) | free tier |

Per-task budget defaults: 20 steps, 300k tokens, 5 minutes, and $0.10. Free-tier calls are
recorded as $0, and reports also show the paid list-price equivalent.

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
                       ├─► ToolExecutor ─► policy ─► approval gate ─► ToolRegistry
                       │                     (search · read · sandbox · Gmail)
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
uv run task-agent eval --label "baseline gpt-oss-120b"
uv run task-agent eval --runner direct --label "no tools floor"
uv run task-agent eval --gaia 1 --limit 20             # GAIA level 1 (needs HF_TOKEN)
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
  memory.py        skills, remembered facts, local knowledge (SQLite FTS5)
  ui.py, ui.html   local web UI (Starlette + server-sent events)
  api.py           /v1 HTTP API: keys, scopes, limits, idempotency (FastAPI)
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
