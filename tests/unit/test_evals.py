import pytest

from task_agent.core import ConfigError
from task_agent.core.runner import DirectRunner
from task_agent.evals import Category, Outcome, load_suite, run_eval, select_tasks, to_markdown
from task_agent.evals.report import write_report
from tests.conftest import SUITE, make_gateway


def test_shipped_suite_is_valid() -> None:
    tasks = load_suite(SUITE)
    assert len(tasks) >= 20
    by_cat = {c: [t for t in tasks if t.category is c] for c in Category if c is not Category.GAIA}
    assert all(by_cat.values()), "every category needs tasks"
    # Phase 0 gate needs tasks that run with no tools at all.
    assert any(not t.requires for t in tasks)


def test_duplicate_ids_rejected(tmp_path) -> None:
    body = (
        "tasks:\n  - id: dup-001\n    category: research\n    prompt: a valid prompt here\n"
        "    checker: {type: exact, expected: x}\n"
    )
    (tmp_path / "a.yaml").write_text(body)
    (tmp_path / "b.yaml").write_text(body)
    with pytest.raises(ConfigError, match="duplicate"):
        load_suite(tmp_path)


def test_select_tasks() -> None:
    tasks = load_suite(SUITE)
    smoke = select_tasks(tasks, tags=["smoke"])
    assert smoke
    assert all("smoke" in t.tags for t in smoke)
    assert [t.id for t in select_tasks(tasks, ids=["research-001"])] == ["research-001"]
    assert len(select_tasks(tasks, categories=["browsing"], limit=2)) == 2


async def test_run_eval_grades_and_skips(tmp_path) -> None:
    tasks = select_tasks(load_suite(SUITE), ids=["research-001", "data-002", "browse-001"])
    gw = make_gateway("thinking...\nFINAL ANSWER: 1991")
    report = await run_eval(tasks, DirectRunner(gw), gw, concurrency=2)

    outcome = {r.task_id: r.outcome for r in report.results}
    assert outcome == {
        "research-001": Outcome.PASSED,
        "data-002": Outcome.FAILED,
        "browse-001": Outcome.SKIPPED,
    }
    assert report.overall.success_rate == 0.5
    assert report.overall.skipped == 1

    md = to_markdown(report)
    assert "data-002" in md
    assert "**overall**" in md
    json_path, md_path = write_report(report, tmp_path)
    assert json_path.exists()
    assert md_path.exists()


async def test_groups_unlock_tasks() -> None:
    tasks = select_tasks(load_suite(SUITE), ids=["data-001"])
    gw = make_gateway("FINAL ANSWER: 76127")
    report = await run_eval(tasks, DirectRunner(gw), gw, available_groups=["code"])
    assert report.results[0].outcome is Outcome.PASSED
