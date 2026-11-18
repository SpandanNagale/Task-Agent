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
| 2 | `OrchestratorRunner` | LangGraph planner/replanner with sub-agents, kept only where it beats the baseline (current) |

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
- `free_tier: true` on an entry records actual cost as $0 and keeps the paid list price in
  `list_cost_usd`, so budgets reflect real spend and reports can still show the
  paid-plan equivalent.
- LiteLLM's bundled price map is used instead of fetching one at import, so there's no
  network call at startup and cost numbers stay fixed per LiteLLM version.

### `core/baseline.py`: the Phase 1 agent

A plain loop with no framework: the model sees the task and the tools for its allowed
groups. It calls tools, reads their results, and stops when it answers without a tool call.
Guards:

- On the last allowed step, tools are withdrawn so the model must answer.
- An identical call (same tool, same arguments) is warned about on its 3rd repeat. The 4th
  repeat stops the task with `LoopDetected`.
- An empty reply gets one nudge, then the task fails.
- Per-task resources (the sandbox) are always torn down in `finally`.

### `core/orchestrator.py`: the Phase 2 agent

A LangGraph graph: `plan -> execute -> review -> execute ... -> END`.

- **plan**: the `planner` role answers directly, or returns JSON steps, each with a role,
  a self-contained goal, a success check and `depends_on`. One corrective retry on bad JSON.
- **execute**: every step whose dependencies are done runs now, in parallel (capped), as a
  sub-agent: the same `tool_loop` as the baseline, on the `worker` role, with a role
  prompt, an 8-step budget and only its role's tool groups. A sub-agent sees its brief and
  its dependencies' results (wrapped as `<step_result>` data), never the conversation.
- **review**: when a wave has a failure or nothing is left to run, the planner finishes or
  appends new steps. After `max_replans` it must answer.
- Preset roles: researcher (search), coder (code, files), communicator (comms), writer (no
  tools). Any other role name is a runtime role: the planner writes its instructions and
  tool groups.
- Budgets stay task-wide: every planner and sub-agent call counts as a step.
- State is checkpointed per task id in `.task-agent/checkpoints.db` (SQLite).
  `task-agent resume TASK_ID` continues an interrupted task without re-running finished
  steps (its budget starts fresh). Postgres replaces SQLite with the phase 6 workers.

### `tools/builtin/browser.py`: the browser (Phase 3)

Playwright Chromium exposed as ordinary tools in the `browser` group, so browsing runs in
the same loop, approval gate, budgets and traces as every other tool (no separate browsing
agent with its own model calls):

- `browser_open`, `browser_snapshot` (read): every call returns a snapshot of URL, title,
  numbered interactive elements (`[7] textbox "Customer name" value=""`, dropdown options,
  checked state) and the page text. Numbers are re-assigned on every snapshot.
- `browser_click`, `browser_type`, `browser_select` (write, ask by default): each returns
  the page afterwards, so the model verifies every action by reading its result.
- `browser_look` (read): screenshot of the page, answered by the `vision` model role
  (qwen2.5vl locally, the model JARVIS uses). The fallback for canvas, charts and layout.
- One browser context per task, closed when the task ends, or one persistent profile
  (`TASK_AGENT_BROWSER_PROFILE_DIR`) for sites you log into by hand. The agent never types
  into password fields, and a CAPTCHA or bot check in a snapshot tells it to stop and hand
  over. `TASK_AGENT_BROWSER_HEADLESS=false` shows the window.

Small local models have small context windows, so `tool_loop` keeps only the newest two
tool results whole and cuts older large ones to their start (`compact_tool_outputs`), and
local Ollama models get an explicit `num_ctx` (Ollama's default silently drops the start of
long prompts).

### `memory.py`: skills, memory and knowledge (Phase 4)

One SQLite FTS5 table (`.task-agent/memory.db`), BM25 keyword search with stemming, three
kinds of note:

- **skill**: after a successful baseline task, the tool calls that worked (failed and
  rejected calls dropped) are saved under the task's prompt. Before a task, a skill whose
  prompt shares at least 60% of its content words is added to the context as a hint to make
  the same calls and skip exploring. A longer rerun never replaces a shorter recipe. Gave-up
  answers are not saved. `run` and the UI use skills; evals only with `--skills`.
- **memory**: facts the user asked to keep, written by the `remember` tool (write side
  effect, so it asks: an injected page must not plant lasting memories).
- **knowledge**: chunks of local text files from `task-agent index PATH`.

`recall` (read) searches memory and knowledge. Keyword search misses paraphrases; Mem0 or
sqlite-vec can replace `MemoryStore.search` when that matters.

### `core/events.py` and `ui.py`: live events and the web UI (Phase 4)

Runners emit events (`model` with cumulative usage, `tool`, `plan`, `step`, `skill`,
`answer`) into a sink held in a context variable, so nothing is wired through call
signatures and parallel sub-agents report into their own task. The CLI binds no sink.

`task-agent ui` serves one static page and a small JSON API on 127.0.0.1: start a task,
stream its events (server-sent events, replayed from the start for late viewers), and
answer approvals (`WebApprover` parks the tool call on a future until the click). Only JSON
POSTs and localhost Host headers are accepted (no cross-site form posts, no DNS rebinding);
agent output is rendered as text only. Tasks live in memory until the server stops.

### `api.py`: the `/v1` HTTP API (Phase 6)

`task-agent serve` is the same task lifecycle (`TaskRecord.start`/`follow`, shared with the
UI) behind API keys. A key is `ta_<id>_<secret>`; only its SHA-256 is stored, with its
scopes (`tasks:read`, `tasks:write`, `approvals:write`), an optional tool-group allow list
(a request can narrow it, never widen it) and its limits: requests per minute, tasks running
at once, tasks and USD per UTC day. Per-task budgets are clamped to server ceilings and to
the key's remaining daily spend. A key sees only its own tasks: anything else is a 404.

Keys without `approvals:write` get `RejectAll`; with it, approvals are answered over HTTP
and rejected after `approval_timeout_s` so an abandoned task cannot hold a worker.
`Idempotency-Key` makes `POST /v1/tasks` safe to retry (same key and body: the first task;
different body: 422). Errors are RFC 9457 `application/problem+json`, with `Retry-After` on
429s. Keys, spend and idempotency records are SQLite; tasks and rate windows are in process
memory until the Redis queue lands.

### `core/tool_executor.py` and `core/approval.py`: the only path to a side effect

Every tool call from a model goes through `ToolExecutor`:

1. **Validate.** Unknown tool, disallowed group, bad JSON or missing required fields come
   back to the model as an `ERROR:` message it can fix, never as a crash.
2. **Policy.** `policies/default.yaml` maps each tool, or each side-effect level, to
   `allow`, `ask` or `deny`. `irreversible` can never default to `allow`.
3. **Approval.** `ask` goes to an `Approver`: `ConsoleApprover` in the CLI, and `RejectAll`
   in evals and non-interactive runs. An edit replaces the arguments.
4. **Run.** Each call runs under the tool's timeout. Read-only tools retry transient
   failures (429, 5xx, network) with exponential backoff and jitter. Tools with side
   effects never retry, so nothing is sent twice.
5. **Return.** Output is size-capped and wrapped in
   `<tool_output trust="untrusted">…</tool_output>`, with closing tags escaped so content
   can't break out. The system prompt tells the model never to follow instructions found
   inside.

`ToolStats` on every `RunResult` counts calls, errors, approvals requested, edited and
rejected, and policy blocks. These feed the tool-error-rate and approval metrics.

### `tools/`: tool contract

`ToolSpec` is the one schema for every capability: name, description, group, input JSON
schema, timeout, and side-effect level (`read`, `write`, `irreversible`). Handlers are
`async (args, ctx) -> ToolOutput`. `ToolContext` gives each task lazily created resources
(the sandbox) that are torn down when the task ends.

`tools/loader.py` registers a tool only if its key is present:

| Tool | Group | Side effect | Backend |
|---|---|---|---|
| `web_search` | search | read | Tavily |
| `read_url` | search | read | Firecrawl, falling back to Jina Reader (no key needed) |
| `run_python` | code | write (sandbox only) | E2B, one sandbox per task, internet off by default |
| `gmail_*` | comms | read / write / irreversible | Composio, with side effects from a curated table, then Composio's hints, then `irreversible` |

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
