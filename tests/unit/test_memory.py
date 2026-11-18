from __future__ import annotations

from typing import Any

from task_agent.core.baseline import BaselineRunner
from task_agent.core.events import listen
from task_agent.core.tool_executor import ToolExecutor
from task_agent.core.types import TaskRequest, TaskStatus
from task_agent.memory import MemoryStore, Recall, Remember, chunk_text, find_skill, skill_recipe
from task_agent.tools.base import SideEffect, ToolContext, ToolOutput, ToolSpec
from task_agent.tools.registry import ToolRegistry
from tests.fakes import ScriptedModel, reply, tool_call


class Calc:
    async def __call__(self, args: dict[str, Any], ctx: ToolContext) -> ToolOutput:
        return ToolOutput(content=str(eval(args["expr"], {"__builtins__": {}})))


def make_runner(model: ScriptedModel, store: MemoryStore | None) -> BaselineRunner:
    reg = ToolRegistry()
    reg.register(
        ToolSpec(
            name="calc",
            description="Evaluate arithmetic. Do not use for anything but plain arithmetic.",
            group="code",
            side_effect=SideEffect.READ,
            input_schema={"type": "object", "required": ["expr"]},
        ),
        Calc(),
    )
    return BaselineRunner(model, ToolExecutor(reg), skills=store)


def test_search_ranks_and_filters_by_kind(tmp_path) -> None:
    store = MemoryStore(tmp_path / "m.db")
    store.add("memory", "timezone", "The user's timezone is IST (UTC+5:30).")
    store.add("knowledge", "notes.md (part 1)", "Invoices are paid within 30 days.")
    store.add("skill", "convert currency", "1. web_search {}")
    hits = store.search("what timezone is the user in?", {"memory", "knowledge"})
    assert [h.title for h in hits] == ["timezone"]
    assert store.search('"; DROP TABLE notes; --', {"memory"}) == []  # user text is quoted
    assert store.count() == {"memory": 1, "knowledge": 1, "skill": 1}


def test_chunk_text_respects_size() -> None:
    text = "\n\n".join(["a" * 600, "b" * 600, "c" * 600, "d" * 4000])
    chunks = chunk_text(text, size=1500)
    assert all(len(c) <= 1500 for c in chunks)
    assert "".join(chunks).replace("\n", "") == text.replace("\n", "")


def test_index_file_replaces_old_chunks(tmp_path) -> None:
    store = MemoryStore(tmp_path / "m.db")
    doc = tmp_path / "policy.md"
    doc.write_text("Refunds take 14 days.", encoding="utf-8")
    assert store.index_file(doc) == 1
    doc.write_text("Refunds take 7 days.\n\nShipping is free.", encoding="utf-8")
    assert store.index_file(doc) == 1
    (hit,) = store.search("refunds", {"knowledge"})
    assert "7 days" in hit.body


async def test_recall_and_remember_tools(tmp_path) -> None:
    store = MemoryStore(tmp_path / "m.db")
    ctx = ToolContext("t1")
    assert (
        await Remember(store)({"text": "Prefers answers in metric units."}, ctx)
    ).content == "Saved."
    out = await Recall(store)({"query": "units preference metric"}, ctx)
    assert "metric units" in out.content
    out = await Recall(store)({"query": "zebra"}, ctx)
    assert out.content.startswith("No saved notes")


def test_skill_recipe_keeps_only_calls_that_worked() -> None:
    messages = [
        {"role": "assistant", "tool_calls": [tool_call("calc", {"expr": "1/0"}, "a")]},
        {"role": "tool", "tool_call_id": "a", "content": "ERROR: division by zero"},
        {"role": "assistant", "tool_calls": [tool_call("calc", {"expr": "6*7"}, "b")]},
        {"role": "tool", "tool_call_id": "b", "content": "<tool_output>42</tool_output>"},
    ]
    recipe = skill_recipe(messages, "42")
    assert recipe is not None and "6*7" in recipe and "1/0" not in recipe
    assert skill_recipe(messages, "I could not find it") is None
    assert skill_recipe(messages[:2], "42") is None


async def test_repeated_task_reuses_the_saved_skill(tmp_path) -> None:
    store = MemoryStore(tmp_path / "m.db")
    prompt = "What is 6 times 7 plus 100 divided by 4?"
    first = ScriptedModel(
        reply("try", [tool_call("calc", {"expr": "6*7+100/"}, "a")]),  # a failed attempt
        reply("again", [tool_call("calc", {"expr": "6*7+100/4"}, "b")]),
        reply("FINAL ANSWER: 67"),
    )
    r1 = await make_runner(first, store).run(TaskRequest(prompt=prompt))
    assert r1.status is TaskStatus.SUCCEEDED and r1.steps == 3
    assert find_skill(store, "what is 6 times 7 plus 100 divided by 4") is not None
    assert find_skill(store, "summarise today's news about chess") is None

    events: list[dict[str, Any]] = []
    second = ScriptedModel(
        reply("reuse", [tool_call("calc", {"expr": "6*7+100/4"}, "c")]),
        reply("FINAL ANSWER: 67"),
    )
    with listen(events.append):
        r2 = await make_runner(second, store).run(TaskRequest(prompt=prompt))
    assert r2.status is TaskStatus.SUCCEEDED and r2.steps == 2
    skill_msg = second.calls[0]["messages"][1]
    assert skill_msg["role"] == "system" and "6*7+100/4" in skill_msg["content"]
    assert '6*7+100/"' not in skill_msg["content"]  # the failed call is not in the recipe
    kinds = [e["type"] for e in events]
    assert kinds[0] == "skill" and "tool" in kinds and "model" in kinds


async def test_no_store_means_no_skills(tmp_path) -> None:
    model = ScriptedModel(reply("FINAL ANSWER: 1"))
    await make_runner(model, None).run(TaskRequest(prompt="anything"))
    assert len(model.calls[0]["messages"]) == 2


async def test_a_longer_rerun_does_not_replace_a_lean_skill(tmp_path) -> None:
    store = MemoryStore(tmp_path / "m.db")
    prompt = "What is 6 times 7 plus 100 divided by 4?"
    lean = ScriptedModel(
        reply("go", [tool_call("calc", {"expr": "6*7+100/4"})]), reply("FINAL ANSWER: 67")
    )
    await make_runner(lean, store).run(TaskRequest(prompt=prompt))
    wander = ScriptedModel(
        reply("check", [tool_call("calc", {"expr": "6*7"}, "a")]),
        reply("again", [tool_call("calc", {"expr": "42+25"}, "b")]),
        reply("FINAL ANSWER: 67"),
    )
    await make_runner(wander, store).run(TaskRequest(prompt=prompt))
    skill = find_skill(store, prompt)
    assert skill is not None and "6*7+100/4" in skill.body and "42+25" not in skill.body
