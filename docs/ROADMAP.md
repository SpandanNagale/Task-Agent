# Roadmap

Seven phases take the project from an empty repo to a production API. Phases 0 to 5 are
built and tested locally. Phase 6 deploys only after the release checklist passes. Each phase
ends at a **gate** that must pass before the next one starts. A phase that misses its gate
gets fixed before any new features start.

## Phase 0: Foundations (current)

- [x] Monorepo on uv, src layout, ruff, mypy (strict), pytest, pre-commit, CI
- [x] Model gateway on LiteLLM with `planner`, `worker`, `router` and `judge` roles and fallback chains
- [x] Per-task budgets: steps, cost, tokens, wall time
- [x] OpenTelemetry tracing with Langfuse as the default backend; redaction of secrets
- [x] docker-compose dev stack: Postgres, Redis, Qdrant, Langfuse (Ollama optional)
- [x] Eval runner with 20 internal tasks across five categories; JSON + Markdown reports
- [x] Tool contract (`ToolSpec`, side-effect levels) and registry

**Gate:** one task runs end to end and appears as a full trace.

## Phase 1: Single-agent baseline

- One tool-calling agent loop (plan, act, observe, replan) with step and cost budgets
- Tools: Tavily (search), Firecrawl (read), Docker sandbox (code), Gmail (via Composio)
- Approval gate: write/irreversible tools pause; policy file `policies/default.yaml`
- Tool outputs tagged as untrusted data; repeated-action detection
- Baseline runs on the internal suite and a GAIA level-1 subset

**Gate:** baseline success rate, cost and steps per task recorded, local-only and hybrid.

## Phase 2: Orchestrator and existing agents

- LangGraph planner and replanner, runtime sub-agent roles, parallel fan-out
- Postgres checkpointer: resume a task after a crash, replay any step
- DataSleuth, the research pipeline and ML-Agent wrapped as MCP sub-agents

**Gate:** same suite re-run; multi-agent kept only where it beats the baseline.

## Phase 3: Browser

- browser-use on Playwright: persistent profiles, DOM view, screen-vision fallback, verify each action
- Docling and DocIntel for documents; WebVoyager subset added to eval runs

**Gate:** browsing and form tasks pass; WebVoyager baseline recorded.

## Phase 4: Memory and UI

- Mem0 long-term memory, skills store, RAG knowledge tool, loop detection
- Web UI (Next.js): live plan, steps, approvals, cost per task

**Gate:** a repeated task reuses a saved skill and takes fewer steps.

## Phase 5: Local benchmarks

- Full suite, GAIA and WebVoyager subsets
- Ablations: single vs multi-agent, frontier-only vs hybrid, with vs without skills, DOM-only vs hybrid vision
- README with architecture, results and demo video

**Gate:** release checklist passes locally: scores, injection set, load test, worker-kill test.

## Phase 6: Production API and deployment

- Async `/v1` API (FastAPI + SSE): API keys with scopes, rate limits, cost caps, RFC 9457 errors, idempotency keys
- Redis queue and worker pool, per-task sandbox and browser containers
- CI/CD, monitoring, live on a cloud VM

## Later

Keel as the Spec Writer sub-agent, a managed container platform, media tools, scheduled tasks.

## Open decisions

- [ ] Project name (the package is `task_agent` for now)
- [ ] Planner model and monthly API budget (sets per-task cost limits)
- [ ] Local worker/router model (first candidate: Qwen3 14B on the RTX 5070)
- [ ] Local Docker sandbox or E2B for v1
- [ ] Composio or individual MCP servers for app integrations
- [ ] Cloud provider and hosting budget for phase 6
- [ ] Who can use the hosted API, and which tool groups the public demo allows
