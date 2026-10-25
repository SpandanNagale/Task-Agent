.PHONY: install check fmt lint type test eval-mock up down

install:  ## install deps and git hooks
	uv sync
	uv run pre-commit install

check: lint type test  ## everything CI runs (minus docker)

fmt:
	uv run ruff format .
	uv run ruff check --fix .

lint:
	uv run ruff format --check .
	uv run ruff check .

type:
	uv run mypy

test:
	uv run pytest

eval-mock:  ## offline eval smoke run
	uv run task-agent eval --models config/models.mock.yaml --label local-smoke

up:  ## start Postgres, Redis, Qdrant, Langfuse
	docker compose -f deploy/compose.dev.yml --env-file .env up -d --wait

down:
	docker compose -f deploy/compose.dev.yml down
