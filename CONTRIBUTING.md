# Contributing

Thanks for helping. This project grows phase by phase against an eval suite, so the bar for a
change is simple: **it works, it's tested, and it doesn't make the numbers worse.**

## Setup

```bash
uv sync                      # Python 3.11+; installs dev tools too
uv run pre-commit install    # ruff, mypy, secret scanning on every commit
cp .env.example .env         # only if you'll use real models
```

## Before you open a PR

```bash
uv run ruff format . && uv run ruff check .
uv run mypy
uv run pytest
uv run task-agent eval --models config/models.mock.yaml   # harness still runs
```

CI runs the same checks on Python 3.11 to 3.13, plus an offline eval smoke run and an image build.

## Ground rules

- **Roles, not model IDs.** Never hardcode a model in code. Add or change a role in
  `config/models.yaml`.
- **Every capability is a tool.** New abilities register a `ToolSpec` with an honest
  side-effect level. Anything that writes to the outside world is `write` or `irreversible`
  and goes through the approval gate. No exceptions.
- **New capability, new eval tasks.** A tool or sub-agent comes with 3 to 5 tasks in
  `evals/suite/` that exercise it (see `evals/README.md`). Include before/after numbers in the
  PR when you change agent behaviour.
- **Secrets stay in the environment.** Never put them in code, config, prompts, logs or test
  fixtures. Redaction is a backstop, not a license.
- **Tool output is data, not instructions.** Never feed web pages, emails or files back to a
  model as if they were trusted instructions.
- **Keep contracts stable.** `core/types.py` models are the future API. Changing a field
  needs a note in the PR.
- Match the surrounding style: typed, small functions, comments that explain *why*.

## Commits and PRs

- Small, focused PRs. One concern each.
- Imperative commit subjects ("Add Tavily search tool"), with the reason in the body.
- Link the issue, describe how you tested, and paste eval deltas if behaviour changed.

## Where to start

Look for issues labelled `good first issue` or `help wanted`, or pick an unchecked item for
the current phase in [docs/ROADMAP.md](docs/ROADMAP.md) and open an issue to claim it.
