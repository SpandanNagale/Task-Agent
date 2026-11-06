import json

import pytest

from task_agent.evals.checkers import check_static, gaia_score
from task_agent.evals.gaia import load_gaia, task_from_row
from task_agent.evals.schema import Category


@pytest.mark.parametrize(
    ("answer", "expected", "ok"),
    [
        ("17", "17", True),
        ("$1,000", "1000", True),
        ("17.0", "17", True),
        ("18", "17", False),
        ("St. Petersburg", "Saint Petersburg", False),
        ("saint   petersburg!", "Saint Petersburg", True),
        ("a, b, 3", "a,b,3", True),
        ("a; b", "a, b, c", False),
        ("b, a", "a, b", False),
    ],
)
def test_gaia_scorer(answer: str, expected: str, ok: bool) -> None:
    assert gaia_score(answer, expected) is ok


ROWS = [
    {
        "task_id": "AAA-1",
        "Question": "How many legs does a spider have?",
        "Level": 1,
        "Final answer": "8",
        "file_name": "",
    },
    {
        "task_id": "BBB-2",
        "Question": "Read the attached sheet and sum column B.",
        "Level": 1,
        "Final answer": "12",
        "file_name": "sheet.xlsx",
    },
    {
        "task_id": "CCC-3",
        "Question": "A harder multi-hop question here.",
        "Level": 2,
        "Final answer": "x",
        "file_name": "",
    },
    {
        "task_id": "DDD-4",
        "Question": "A test split question with no answer.",
        "Level": 1,
        "Final answer": "?",
        "file_name": "",
    },
]


def test_task_from_row() -> None:
    task = task_from_row(ROWS[0])
    assert task is not None
    assert task.id == "gaia-aaa-1" and task.category is Category.GAIA
    assert task.requires == {"search"}
    assert check_static(task.checker, "8").passed
    assert task_from_row(ROWS[1]).requires == {"search", "files"}
    assert task_from_row(ROWS[3]) is None


def test_load_gaia_from_local_dir(tmp_path) -> None:
    split = tmp_path / "2023" / "validation"
    split.mkdir(parents=True)
    (split / "metadata.jsonl").write_text("\n".join(json.dumps(r) for r in ROWS))
    tasks = load_gaia([1], local_dir=tmp_path)
    assert sorted(t.id for t in tasks) == ["gaia-aaa-1", "gaia-bbb-2"]
    assert len(load_gaia([1, 2], local_dir=tmp_path)) == 3
