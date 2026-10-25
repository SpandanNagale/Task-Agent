# Evals

The project is judged at three levels: public benchmarks (GAIA, WebVoyager, added from
phase 1), this **internal task suite**, and unit evals per component.

## Running

```bash
uv run task-agent eval                                  # whole suite
uv run task-agent eval --tag smoke                      # tasks that need no tools
uv run task-agent eval -c research -c data_code --limit 5
uv run task-agent eval --groups search --groups code    # declare tool groups the runner has
uv run task-agent eval --label "qwen3-14b worker" --min-success 0.5
```

Each run writes `evals/reports/runs/<run_id>.json` (every task, answer, cost, trace id) and
`<run_id>.md` (the summary table and failures). Runs you want to cite go in
`evals/reports/published/`.

A task is **skipped** when it `requires` tool groups the runner doesn't have. Skipped tasks
don't count against the success rate, so one suite serves every phase.

## Writing a task

Tasks live in `evals/suite/<category>.yaml`:

```yaml
tasks:
  - id: data-002                # unique, lowercase, stable: results are tracked by id
    category: data_code         # research | browsing | data_code | communication | multi_step
    prompt: ...                 # exactly what the agent sees
    checker: { type: numeric, expected: 62.5, abs_tol: 0.001 }
    requires: [code]            # tool groups it cannot be solved without
    tags: [smoke]
    budget: { max_steps: 10 }   # optional per-task override
    notes: why this task exists # never shown to the agent
```

| checker | passes when |
|---|---|
| `exact` | the answer equals `expected` after normalisation (case, whitespace, quotes, trailing punctuation) |
| `one_of` | the answer equals any of `options` |
| `contains` | it has every string in `all_of`, at least one in `any_of`, and none in `none_of` |
| `regex` | `pattern` matches somewhere in the answer |
| `numeric` | the first number in the answer is within `abs_tol` / `rel_tol` of `expected` |
| `llm_judge` | the `judge` role grades the answer against `rubric` and returns PASS |

Guidelines:

- **Prefer deterministic checkers.** Use `llm_judge` only for open-ended output such as
  drafts and summaries, and write rubrics as concrete PASS conditions.
- **Stable answers only.** Avoid facts that change: prices, "latest version", today's news.
- **One capability per task** where possible. Use `multi_step` for deliberate combinations.
- **New capability, new tasks.** A tool or sub-agent enters the project with 3 to 5 tasks that
  cover it.
- Verify the expected answer yourself (with code where you can) before committing.
