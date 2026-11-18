"""Release check: a worker killed mid-task (no cleanup, no signal handlers) loses no work.

The task resumes from its last checkpoint in a new process: finished steps are not re-run.
"""

from __future__ import annotations

import json
import subprocess
import sys

from task_agent.core.orchestrator import OrchestratorRunner, load_request
from task_agent.core.tool_executor import ToolExecutor
from task_agent.core.types import Budget, TaskStatus
from task_agent.tools.registry import ToolRegistry
from tests.conftest import ROOT
from tests.fakes import ScriptedModel, reply


async def test_killed_worker_resumes_without_redoing_steps(tmp_path) -> None:
    db = tmp_path / "cp.db"
    proc = subprocess.Popen(  # noqa: ASYNC220 - the test is about a real process
        [sys.executable, "-m", "tests.fixtures.killable_orchestrator", str(db), "task_kill"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        text=True,
    )
    try:
        assert proc.stdout is not None
        for line in proc.stdout:  # log lines first; blocks until step b is in flight
            if line.strip() == "KILL-ME":
                break
        else:
            raise AssertionError("worker exited before reaching step b")
    finally:
        proc.kill()  # TerminateProcess / SIGKILL: no chance to clean up
        proc.wait(timeout=30)

    request = await load_request(db, "task_kill")
    assert request is not None and request.prompt == "two steps"
    model = ScriptedModel(reply("FINAL ANSWER: B"), reply(json.dumps({"final_answer": "A B"})))
    runner = OrchestratorRunner(model, ToolExecutor(ToolRegistry()), checkpoint_db=db)
    result = await runner.run(request.model_copy(update={"budget": Budget(max_steps=5)}))

    assert result.status is TaskStatus.SUCCEEDED, result.error
    assert result.output == "A B"
    assert len(model.calls) == 2  # only step b and the review: no re-plan, no step a
    assert '<step_result id="a"' in model.calls[0]["messages"][1]["content"]
