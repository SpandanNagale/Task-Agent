"""Load and filter eval suites."""

from __future__ import annotations

import functools
import threading
from collections.abc import Iterable
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import yaml
from pydantic import ValidationError

from task_agent.core.errors import ConfigError
from task_agent.evals.schema import EvalSuiteFile, EvalTask


def load_suite(path: Path) -> list[EvalTask]:
    """Load every ``*.yaml`` file under ``path`` (or one file). Task ids must be unique.

    ``{files}`` in a suite file becomes the absolute path of the ``files/`` folder next to
    it, so tasks can point tools that run elsewhere (MCP servers) at their input files.
    ``{files_url}`` becomes the http:// address of that folder on a local static server
    (started on first use), for browser tasks on pages we control (the injection set).
    """
    files = sorted(path.rglob("*.yaml")) if path.is_dir() else [path]
    if not files or not all(f.is_file() for f in files):
        raise ConfigError(f"no eval suite files at {path}")

    tasks: dict[str, EvalTask] = {}
    for file in files:
        try:
            text = file.read_text(encoding="utf-8")
            folder = (file.parent / "files").resolve()
            text = text.replace("{files}", folder.as_posix())
            if "{files_url}" in text:
                text = text.replace("{files_url}", serve_folder(folder))
            raw = yaml.safe_load(text) or {}
            suite = EvalSuiteFile.model_validate(raw)
        except (yaml.YAMLError, ValidationError) as exc:
            raise ConfigError(f"invalid eval file {file}: {exc}") from exc
        for task in suite.tasks:
            if task.id in tasks:
                raise ConfigError(f"duplicate eval task id '{task.id}' in {file}")
            tasks[task.id] = task
    return list(tasks.values())


@functools.cache
def serve_folder(folder: Path) -> str:
    """Serve ``folder`` read-only on 127.0.0.1 (a free port) for the rest of the process."""

    class Quiet(SimpleHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), partial(Quiet, directory=str(folder)))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return f"http://127.0.0.1:{server.server_address[1]}"


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
