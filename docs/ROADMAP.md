# Roadmap

Seven phases take the project from an empty repo to a production API. Phases 0 to 5 are
built and tested locally. Phase 6 deploys only after the release checklist passes. Each phase
ends at a **gate** that must pass before the next one starts. A phase that misses its gate
gets fixed before any new features start.

## Phase 0: Foundations (done)

- [x] Monorepo on uv, src layout, ruff, mypy (strict), pytest, pre-commit, CI
- [x] Model gateway on LiteLLM with `planner`, `worker`, `router` and `judge` roles and fallback chains
- [x] Per-task budgets: steps, cost, tokens, wall time
- [x] OpenTelemetry tracing with Langfuse as the default backend; redaction of secrets
- [x] docker-compose dev stack: Postgres, Redis, Qdrant, Langfuse (Ollama optional)
- [x] Eval runner with 20 internal tasks across five categories; JSON + Markdown reports
- [x] Tool contract (`ToolSpec`, side-effect levels) and registry

**Gate:** one task runs end to end and appears as a full trace.

## Phase 1: Single-agent baseline (done)

- [x] One tool-calling agent loop (act, observe, repeat) with step, token, time and cost budgets
- [x] Tools: Tavily (search), Jina Reader / Firecrawl (read), E2B cloud sandbox (code), Gmail via Composio
- [x] Approval gate: write/irreversible tools pause for approve / edit / reject; policy file `policies/default.yaml`
- [x] Tool outputs wrapped as untrusted data; repeated-action detection; read-only retries, no write retries
- [x] GAIA loader and official scorer; tool-error and approval metrics in eval reports
- [x] Local-only baseline on the internal suite (qwen3:14b on an RTX 5070, read_url only)
- [ ] Hybrid baseline and the GAIA level-1 subset (needs TAVILY / E2B keys and HF_TOKEN)

**Gate:** baseline success rate, cost and steps per task recorded, local-only and hybrid.

Fixed while re-checking phase 1 on a real machine: `.env` keys never reached
`models.yaml` (only the process environment did); the CLI crashed on Windows consoles on
non-Latin model output; the LLM judge marked "... PASS." as a fail when the verdict shared
a line with its reasoning; eval reports always showed 0 tool calls.

## Phase 2: Orchestrator and existing agents (done)

- [x] LangGraph planner and replanner (`--runner orchestrator`), preset and runtime sub-agent roles, parallel fan-out
- [x] Checkpointer: resume an interrupted task (`task-agent resume TASK_ID`). SQLite locally;
      Postgres moves in with the phase 6 workers (psycopg async does not run on Windows' default loop)
- [x] MCP client: servers in `config/mcp.yaml` run in their own venvs; their tools join the registry behind the approval gate
- [x] DataSleuth as an MCP server (`data` group, `analyze_dataset`) with 3 eval tasks on seeded traps
- [x] The research pipeline (`research_topic`) and ML-Agent (`build_ml_pipeline`) as MCP servers, 3 eval tasks each.
      ML-Agent runs code on this machine, so it asks for approval; evals opt in with `--allow build_ml_pipeline`

**Gate:** same suite re-run; multi-agent kept only where it beats the baseline.

Results, qwen3:14b on an RTX 5070 for every role (2026-11-07):

| suite | baseline | orchestrator |
|---|---|---|
| internal, search + code (15 tasks; browsing and Gmail skipped) | 15/15, 1.7 steps, 12s | 15/15, 3.2 steps, 28s |
| DataSleuth tasks (3) | 3/3 | 2/3 |
| research-pipeline tasks (3) | 3/3 | 3/3 |
| ML-Agent tasks (3) | 1/3 | 1/3 |

The orchestrator does not beat the baseline yet: same success, about 2x the steps and wall
time. Per "earn every agent", the baseline stays the default runner; the orchestrator stays
for tasks the suite does not cover yet (long multi-part tasks) and for the phase 5 ablation.
The leakage task (dsleuth-001) is noisy: it depends on DataSleuth's critic catching the leak.

What the wrapped-agent tasks showed:

- research-pipeline routes its own LLM calls to hosted free tiers (Ollama Cloud, Groq,
  OpenRouter, Gemini), not the local GPU; one of its runs failed on a Groq rate limit. The
  orchestrator answered these tasks with plain web search through its researcher role.
- ML-Agent's leakage auditor only warns on `exit_interview_completed`, so it trains a
  "100% accurate" model on the leak (mlagent-001), and its UCI dataset search returns 404s,
  so the iris task trains on the wrong data (mlagent-003). Both are ML-Agent fixes.

MCP servers on Windows: each server keeps the protocol on private copies of stdin/stdout and
points fds 0/1 at devnull/stderr, so prints and child processes cannot corrupt or block it.

Fixed while wiring DataSleuth in: its sandbox inherited stdin, which hangs every script
under an MCP stdio server on Windows; its profiler read True/False columns as empty numeric
stats; the planner only saw group names, so it never routed to MCP tools; a runtime role
with no tools listed got none, and invented results.

## Phase 3: Browser and documents (done, Docling deferred)

- [x] Browser: Playwright tools in the `browser` group (open, snapshot, click, type, select), numbered
      DOM elements, a fresh snapshot after every action; click/type/select ask for approval
- [x] Persistent profile option (`TASK_AGENT_BROWSER_PROFILE_DIR`), headed mode, no typing into
      password fields, CAPTCHA/bot checks hand over to the user
- [x] Screen-vision fallback `browser_look`: page screenshot read by a new `vision` role (qwen2.5vl, as in JARVIS)
- [x] DocIntel as an MCP server (`extract_document`, group `files`) with 3 eval tasks
- [x] WebVoyager loader (`eval --webvoyager N`): golden-answer tasks only, judged against the reference
- [ ] Docling for PDFs and office documents (blocked on a slow PyPI mirror during this phase)

**Gate:** browsing and form tasks pass; WebVoyager baseline recorded.

Results (2026-11-07), baseline runner:

| suite | local (qwen3:14b) | hybrid (gpt-oss-120b planner) |
|---|---|---|
| browsing (4, incl. a form) | 4/4 | |
| DocIntel documents (3) | 3/3 | |
| WebVoyager golden, 1 per site (14) | 3/14, 3.1 steps | 6/14, 11.7 steps (4 errors: 2 provider outages, 1 loop, 1 step budget) |

The local model reads pages correctly but gives up after one or two actions; the hosted
planner keeps navigating. Worth trying next: a longer step budget for browsing tasks, and a
second planner provider so one outage does not fail the task.

Fixed along the way: Ollama's default context (~2k tokens) silently dropped the start of long
prompts, so the model never saw its task on large pages. Local models now set `num_ctx`, and
the agent loop shortens older tool outputs. Earlier local results were run without this and
should be re-run. Also in ML-Agent: its leakage checks crashed on pandas 3 text columns (two of
five never ran), it picked a leaky boolean as the target for user CSVs, and its UCI dataset
search used a retired API; mlagent-001 and the iris task now pass.

## Phase 4: Memory and UI (done)

- [x] Skills store: tool calls that worked are saved per task and replayed as a hint
- [x] Long-term memory (`remember`, asks first) and a local knowledge base (`task-agent index`),
  searched by `recall`. SQLite FTS5 keyword search instead of Mem0: no server or embedding
  model; swap in Mem0/sqlite-vec if paraphrased lookups matter
- [x] Loop detection (since Phase 1: repeated identical calls warn, then stop the task)
- [x] Web UI (`task-agent ui`): live plan, steps, tool calls, approvals (approve, edit,
  reject), tokens and cost per task. One static page on Starlette + SSE instead of Next.js;
  Next.js with the phase 6 API

**Gate:** a repeated task reuses a saved skill and takes fewer steps. **Passed** on
WebVoyager (8 golden tasks, 1 per site, hosted gpt-oss-120b, run twice with `--skills`):

| Run | Passed | Errors | Mean steps | Tool calls | Mean time |
|---|---|---|---|---|---|
| 1 (saves skills) | 4/8 | 2 | 10.1 | 73 | 36 s |
| 2 (reuses 6 skills) | 4/8 | 2 | **8.0** | 56 | **23 s** |

Biggest cuts: Amazon 20 to 11 steps, Allrecipes 20 to 11, BBC News 10 to 7. Coursera went
10 to 14 (both runs ended in loop detection); the ArXiv errors were Ollama Cloud 500s.

On the internal suite (16 tool tasks) skills barely matter: 13 tasks already finish in the
minimum steps (one tool call and the answer). Three runs: 2.31, 2.50, 2.19 mean steps, all
15/16. The first try made things worse (verification calls, and longer reruns overwrote
lean recipes); fixed by telling the model not to add calls and keeping the shorter recipe.

## Phase 5: Local benchmarks

- Full suite, GAIA and WebVoyager subsets
- Ablations: single vs multi-agent, frontier-only vs hybrid, with vs without skills, DOM-only vs hybrid vision
- README with architecture, results and demo video

**Gate:** release checklist passes locally: scores, injection set, load test, worker-kill test.

## Phase 6: Production API and deployment

- [x] Async `/v1` API (FastAPI + SSE): API keys with scopes, rate limits, cost caps, RFC 9457 errors, idempotency keys (`task-agent serve`, `task-agent keys`)
- Redis queue and worker pool, per-task sandbox and browser containers
- CI/CD, monitoring, live on a cloud VM

## Later

Keel as the Spec Writer sub-agent, a managed container platform, media tools, scheduled tasks.

## Decisions

- [x] Planner: `gpt-oss-120b` on free tiers (Cerebras, Groq, OpenRouter, Ollama Cloud), Gemini Flash as last resort. Budget: $0/month; per-task limits of 20 steps, 300k tokens, 5 min
- [x] Sandbox: E2B for v1 (free Hobby credit); a local Docker backend later behind the same interface
- [x] Integrations: Composio first (Gmail); individual MCP servers later
- [x] Local worker/router model: Qwen3 14B on the RTX 5070 via Ollama

## Open decisions

- [ ] Project name (the package is `task_agent` for now)
- [ ] Cloud provider and hosting budget for phase 6
- [ ] Who can use the hosted API, and which tool groups the public demo allows
