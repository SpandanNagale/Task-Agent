"""Re-grade an eval report's 'checker error' tasks (e.g. after a judge outage) without
re-running the agent. Rewrites the report's JSON and Markdown in place.

    uv run python -m scripts.regrade evals/reports/runs/.../eval_X.json
"""

from __future__ import annotations

import asyncio
import sys
from collections import defaultdict
from pathlib import Path

from task_agent.config import get_settings
from task_agent.evals import load_suite, to_markdown
from task_agent.evals.checkers import check
from task_agent.evals.runner import EvalReport, Outcome, summarise
from task_agent.models.gateway import ModelGateway


async def regrade(path: Path) -> int:
    report = EvalReport.model_validate_json(path.read_text(encoding="utf-8"))
    tasks = {t.id: t for t in load_suite(Path("evals/suite"))}
    gateway = ModelGateway.from_file(get_settings().models_config)
    fixed = 0
    for o in report.results:
        if o.outcome is not Outcome.ERROR or not o.detail.startswith("checker error"):
            continue
        graded = await check(tasks[o.task_id], o.answer, gateway)
        o.outcome = Outcome.PASSED if graded.passed else Outcome.FAILED
        o.detail, o.check = graded.detail, graded
        fixed += 1
    by_cat = defaultdict(list)
    for o in report.results:
        by_cat[o.category].append(o)
    report.overall = summarise(report.results)
    report.by_category = {c: summarise(v) for c, v in sorted(by_cat.items())}
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    path.with_suffix(".md").write_text(to_markdown(report), encoding="utf-8")
    print(f"re-graded {fixed} task(s): {report.overall.passed}/{report.overall.total} passed")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(regrade(Path(sys.argv[1]))))
