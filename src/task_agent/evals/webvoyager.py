"""WebVoyager benchmark loader (https://github.com/MinorJerry/WebVoyager).

643 tasks on 15 live websites. Each has a reference answer typed "golden" (stable: a fact
that should not change) or "possible" (one valid answer when the benchmark was made; live
sites change). Only golden tasks are loaded, so scores stay comparable between runs. The
original benchmark grades screenshots with GPT-4V; here a judge model compares the agent's
answer with the golden reference.

The data is downloaded into a local cache at run time and never committed.
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from task_agent.core.errors import ConfigError
from task_agent.evals.schema import Category, EvalTask, LLMJudgeChecker

BASE_URL = "https://raw.githubusercontent.com/MinorJerry/WebVoyager/main/data"
FILES = ("WebVoyager_data.jsonl", "reference_answer.json")


def _download(cache_dir: Path) -> None:
    import httpx

    cache_dir.mkdir(parents=True, exist_ok=True)
    for name in FILES:
        target = cache_dir / name
        if target.is_file() and target.stat().st_size > 0:
            continue
        try:
            resp = httpx.get(f"{BASE_URL}/{name}", timeout=120, follow_redirects=True)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise ConfigError(f"could not download WebVoyager {name}: {exc}") from exc
        target.write_bytes(resp.content)


def tasks_from_data(
    rows: Iterable[dict[str, Any]],
    references: dict[str, Any],
    *,
    per_site: int | None = 2,
    sites: Iterable[str] = (),
) -> list[EvalTask]:
    """Golden-answer tasks, the first ``per_site`` of each site (None: all of them)."""
    golden: dict[tuple[str, int], str] = {}
    for site, block in references.items():
        for ans in block.get("answers", []):
            if ans.get("type") == "golden":
                golden[(site, int(ans["id"]))] = str(ans["ans"])

    wanted = {s.lower() for s in sites}
    taken: dict[str, int] = defaultdict(int)
    tasks = []
    for row in rows:
        site = row["web_name"]
        number = int(str(row["id"]).rsplit("--", 1)[1])
        reference = golden.get((site, number))
        if reference is None or (wanted and site.lower() not in wanted):
            continue
        if per_site is not None and taken[site] >= per_site:
            continue
        taken[site] += 1
        slug = site.lower().replace(" ", "-")
        tasks.append(
            EvalTask(
                id=f"wv-{slug}-{number}",
                category=Category.WEBVOYAGER,
                prompt=f"{row['ques']} Start from {row['web']}",
                checker=LLMJudgeChecker(
                    rubric=(
                        f"A reference answer for this task is: {reference}\n"
                        "PASS if the answer gives the same key facts as the reference (names, "
                        "numbers, items), allowing different wording, formatting or extra detail. "
                        "FAIL if a key fact differs or is missing, or the answer says the task "
                        "could not be done."
                    )
                ),
                requires=frozenset({"browser"}),
                tags=frozenset({"webvoyager", f"webvoyager-{slug}"}),
            )
        )
    return tasks


def load_webvoyager(
    *, per_site: int | None = 2, sites: Iterable[str] = (), cache_dir: Path | None = None
) -> list[EvalTask]:
    cache_dir = cache_dir or Path(".task-agent/webvoyager")
    _download(cache_dir)
    with (cache_dir / FILES[0]).open(encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    references = json.loads((cache_dir / FILES[1]).read_text(encoding="utf-8"))
    return tasks_from_data(rows, references, per_site=per_site, sites=sites)
