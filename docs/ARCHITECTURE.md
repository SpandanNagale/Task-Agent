# Architecture

This document describes what exists today and the seams later phases plug into. For what gets
built when, see [ROADMAP.md](ROADMAP.md).

## Target system

```
                 ┌──────────── you: CLI / web UI / API ◄──── approval gate ◄─┐
                 │                                                           │
   memory ◄────► orchestrator (plans, dispatches, checks, replans)           │
                 │  briefs down, results up                                  │
                 ▼                                                           │
   sub-agents: researcher · browser operator · coder/analyst ·               │
               communicator · writer · runtime-defined roles                 │
                 │  tool calls                                               │
                 ▼                                                           │
   tool registry (MCP) ── search · browser · sandbox · apps · files · media ─┘
                                                          write actions pause here
   ──────────────────────────────────────────────────────────────────────────
   model gateway (planner · worker · router · judge via LiteLLM)   tracing (OTel → Langfuse)
```

## Layers and contracts

### `core/types.py`: data contracts

`TaskRequest`, `RunResult`, `Budget`, `Usage` and `TaskStatus` are plain, JSON-serialisable
pydantic models. They are the payloads of the future `/v1` API, the job queue and the event
stream, so they hold no live handles. Changing a field here is an API change.

`RunResult.versions` records the agent version, the runner, the prompt version and the model
that actually served each role, so any result can be reproduced.

### `core/runner.py`: the Runner protocol

```python
class Runner(Protocol):
    name: str

    async def run(self, request: TaskRequest) -> RunResult: ...
```

Every caller (CLI, eval harness, later the queue workers) depends only on this protocol.
Phases replace the implementation:

| Phase | Runner | What it does |
|---|---|---|
| 0 | `DirectRunner` | one model call, no tools; proves the plumbing and sets a "no tools" floor |
| 1 | `BaselineRunner` | single tool-calling agent loop with approval gate |
| 2 | `OrchestratorRunner` | LangGraph planner/replanner with sub-agents, kept only where it beats the baseline |

Runners are stateless across tasks: everything a run needs comes in the request or from
injected services. That keeps any worker able to run any task.

### `core/budget.py`: budgets

`BudgetTracker` enforces `max_steps`, `max_cost_usd`, `max_tokens` and `max_wall_s`. The
gateway charges it after every call and clamps each call's timeout to the remaining wall
time. A hit raises `BudgetExceeded`, and the runner turns that into a
`budget_exceeded` result with a report. It never loops silently.

### `models/`: model gateway

- Code asks for a **role**, never a model ID. `config/models.yaml` maps each role to an
  ordered fallback chain. Each entry is passed straight to LiteLLM, so any provider LiteLLM
  supports works.
- `${VAR}` / `${VAR:-default}` read the environment. Entries that need an unset variable are
  skipped, so contributors with one key still get a working chain. `task-agent models` shows
  what resolved.
- `alias:` lets a role reuse another role's chain (router → worker, judge → planner) with its
  own parameters.
- Under the hood this is a LiteLLM `Router`: per-model retries, cooldown of failing
  deployments, and fallbacks down the chain.

### `tools/`: tool contract

`ToolSpec` is the one schema for every capability: name, description, group, input JSON
schema, and side-effect level (`read`, `write`, `irreversible`). The side-effect level drives
the approval gate. Phase 0 ships the contract and an in-memory registry. Phase 1 adds the
first tools, MCP loading and invocation through the gate.

Write the description as one sentence on what the tool does and one on when *not* to use it.
The model picks tools by reading these descriptions.

### `observability/`: traces, logs, redaction

- **Tracing** is OpenTelemetry. One trace per task, one child span per model call (and,
  later, per tool call and step). Spans carry GenAI semantic-convention attributes plus the
  `langfuse.*` attributes Langfuse reads, so Langfuse is the default backend but any OTLP
  collector works. Inputs and outputs are redacted and size-capped, and you can turn them off
  with `TASK_AGENT_TRACE_CONTENT=false`.
- **Logs** are structlog: JSON in containers, readable console output in dev. Every line in
  a task carries `task_id` and `trace_id`, so logs and traces join up.
- **Redaction** scrubs known secret values from the environment, plus common key formats,
  out of logs and traces.

### `evals/`: harness

Suite YAML → `EvalTask` → run through any `Runner` → graded by a checker (exact, one-of,
contains, regex, numeric, or LLM judge) → `EvalReport` (JSON plus a Markdown summary).
Tasks that need tool groups the runner lacks are skipped, not failed. Concurrency is bounded,
one crashing task doesn't sink the run, and judge cost is reported separately from agent cost.

## Scaling seams (built in now, used later)

| Concern | Seam today | Used in |
|---|---|---|
| Stateless API replicas | contracts are serialisable; runners hold no cross-task state | phase 6 |
| Queue + worker pool | `Runner.run(TaskRequest) -> RunResult` is the whole worker body | phase 6 |
| Resume after crash | LangGraph checkpointer (Postgres) behind the runner | phase 2 |
| Provider outages | role fallback chains, cooldowns, per-call timeouts | now |
| Cost control | per-task budgets now; per-key monthly caps later | now / phase 6 |
| Reproducibility | `RunResult.versions` (agent, runner, prompt, served models) | now |
| Secrets hygiene | env-only secrets, redaction in logs and traces | now |
| Swappable observability | vendor-neutral OTel spans | now |

## Local services (`deploy/compose.dev.yml`)

Postgres (app and Langfuse databases), Redis, Qdrant, and Langfuse (web, worker,
ClickHouse, MinIO). Ollama is optional (`--profile local-models`). Ports bind to localhost
only. Langfuse is provisioned headlessly with fixed dev API keys, so tracing works on first
start.
