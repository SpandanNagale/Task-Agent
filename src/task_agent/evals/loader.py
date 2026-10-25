"""Load and filter eval suites."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import yaml
from pydantic import ValidationError

from task_agent.core.errors import ConfigError
from task_agent.evals.schema import EvalSuiteFile, EvalTask


def load_suite(path: Path) -> list[EvalTask]:
    """Load every ``*.yaml`` file under ``path`` (or one file). Task ids must be unique."""
    files = sorted(path.rglob("*.yaml")) if path.is_dir() else [path]
    if not files or not all(f.is_file() for f in files):
        raise ConfigError(f"no eval suite files at {path}")

    tasks: dict[str, EvalTask] = {}
    for file in files:
        try:
            raw = yaml.safe_load(file.read_text(encoding="utf-8")) or {}
            suite = EvalSuiteFile.model_validate(raw)
        except (yaml.YAMLError, ValidationError) as exc:
            raise ConfigError(f"invalid eval file {file}: {exc}") from exc
        for task in suite.tasks:
            if task.id in tasks:
                raise ConfigError(f"duplicate eval task id '{task.id}' in {file}")
            tasks[task.id] = task
    return list(tasks.values())


def select_tasks(
    tasks: Iterable[EvalTask],
    *,
    ids: Iterable[str] = (),
    categories: Iterable[str] = (),
    tags: Iterable[str] = (),
    limit: int | None = None,
) -> list[EvalTask]:
    """Filter by id, category and tag (each filter is an OR within itself, AND across)."""
    ids, categories, tags = set(ids), set(categories), set(tags)
    selected = [
        t
        for t in tasks
        if (not ids or t.id in ids)
        and (not categories or t.category.value in categories)
        and (not tags or t.tags & tags)
    ]
    return selected[:limit] if limit is not None else selected
