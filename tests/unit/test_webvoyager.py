from task_agent.evals.schema import Category, LLMJudgeChecker
from task_agent.evals.webvoyager import tasks_from_data

ROWS = [
    {"web_name": "ArXiv", "id": "ArXiv--0", "ques": "Find paper A.", "web": "https://arxiv.org/"},
    {"web_name": "ArXiv", "id": "ArXiv--1", "ques": "Find paper B.", "web": "https://arxiv.org/"},
    {"web_name": "ArXiv", "id": "ArXiv--2", "ques": "Find paper C.", "web": "https://arxiv.org/"},
    {
        "web_name": "Google Map",
        "id": "Google Map--4",
        "ques": "Find X.",
        "web": "https://maps.google.com/",
    },
]
REFS = {
    "ArXiv": {
        "answers": [
            {"id": 0, "type": "possible", "ans": "live"},
            {"id": 1, "type": "golden", "ans": "2106.01234"},
            {"id": 2, "type": "golden", "ans": "2201.00001"},
        ]
    },
    "Google Map": {"answers": [{"id": 4, "type": "golden", "ans": "X Street"}]},
}


def test_only_golden_tasks_capped_per_site() -> None:
    tasks = tasks_from_data(ROWS, REFS, per_site=1)
    assert [t.id for t in tasks] == ["wv-arxiv-1", "wv-google-map-4"]
    t = tasks[0]
    assert t.category is Category.WEBVOYAGER and t.requires == {"browser"}
    assert t.prompt == "Find paper B. Start from https://arxiv.org/"
    assert isinstance(t.checker, LLMJudgeChecker) and "2106.01234" in t.checker.rubric


def test_site_filter_and_no_cap() -> None:
    tasks = tasks_from_data(ROWS, REFS, per_site=None, sites=["arxiv"])
    assert [t.id for t in tasks] == ["wv-arxiv-1", "wv-arxiv-2"]
