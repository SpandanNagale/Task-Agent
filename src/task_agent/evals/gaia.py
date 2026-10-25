"""GAIA benchmark loader (https://huggingface.co/datasets/gaia-benchmark/GAIA).

GAIA is gated: accept its terms on Hugging Face and set HF_TOKEN. Its authors ask that the
answers not be republished in crawlable form, so the data is downloaded into a local cache
at run time and never committed to this repository.

Install the extra first: ``uv sync --extra gaia``.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from task_agent.core.errors import ConfigError
from task_agent.evals.schema import Category, EvalTask, GaiaChecker

REPO_ID = "gaia-benchmark/GAIA"


def task_from_row(row: Mapping[str, Any]) -> EvalTask | None:
    """Convert one GAIA metadata row. Rows without a public answer (the test split) are
    dropped. Tasks with an attached file need the 'files' tool group until file inputs land.
    """
    answer = row.get("Final answer")
    if answer in (None, "", "?"):
        return None
    level = int(row.get("Level", 0))
    has_file = bool(row.get("file_name"))
    return EvalTask(
        id=f"gaia-{row['task_id']}".lower()[:64],
        category=Category.GAIA,
        prompt=str(row["Question"]),
        checker=GaiaChecker(expected=str(answer)),
        requires=frozenset({"search", "files"} if has_file else {"search"}),
        tags=frozenset({"gaia", f"gaia-level{level}"}),
    )


def _read_rows(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as fh:
            return [json.loads(line) for line in fh if line.strip()]
    try:
        import pyarrow.parquet as pq
    except ImportError as exc:  # pragma: no cover
        raise ConfigError("reading GAIA parquet needs pyarrow: uv sync --extra gaia") from exc
    return pq.read_table(path).to_pylist()  # type: ignore[no-any-return]


def load_gaia(
    levels: Iterable[int] = (1,),
    *,
    split: str = "validation",
    cache_dir: Path | None = None,
    local_dir: Path | None = None,
) -> list[EvalTask]:
    """Load GAIA tasks of the given levels. ``local_dir`` skips the download."""
    if local_dir is None:
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise ConfigError("GAIA needs huggingface_hub: uv sync --extra gaia") from exc
        try:
            local_dir = Path(
                snapshot_download(
                    REPO_ID,
                    repo_type="dataset",
                    allow_patterns=[f"2023/{split}/*"],
                    cache_dir=cache_dir,
                    token=os.environ.get("HF_TOKEN"),
                )
            )
        except Exception as exc:
            raise ConfigError(
                f"could not download GAIA ({type(exc).__name__}). Accept the dataset terms at "
                f"https://huggingface.co/datasets/{REPO_ID} and set HF_TOKEN."
            ) from exc

    files = sorted((local_dir / "2023" / split).glob("metadata*.jsonl")) or sorted(
        (local_dir / "2023" / split).glob("metadata*.parquet")
    )
    if not files:
        raise ConfigError(f"no GAIA metadata found under {local_dir / '2023' / split}")

    wanted = set(levels)
    tasks: dict[str, EvalTask] = {}
    for file in files:
        for row in _read_rows(file):
            if int(row.get("Level", 0)) not in wanted:
                continue
            task = task_from_row(row)
            if task is not None:
                tasks[task.id] = task
    return list(tasks.values())
