"""Runs an orchestrator task that hangs in its second step, for the worker-kill test.

python -m tests.fixtures.killable_orchestrator DB TASK_ID
Prints KILL-ME once step a is checkpointed and step b's model call is in flight.
"""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path
from typing import Any

from tests.fakes import ScriptedModel, reply

from task_agent.core.orchestrator import OrchestratorRunner
from task_agent.core.tool_executor import ToolExecutor
from task_agent.core.types import Budget, TaskRequest
from task_agent.models.gateway import Completion
from task_agent.tools.registry import ToolRegistry


class HangsWhenEmpty(ScriptedModel):
    async def complete(self, role: str, messages: list[dict[str, Any]], **kw: Any) -> Completion:
        if not self.script:
            print("KILL-ME", flush=True)
            await asyncio.sleep(3600)
        return await super().complete(role, messages, **kw)


async def main(db: Path, task_id: str) -> None:
    steps = [
        {"id": "a", "role": "writer", "goal": "first"},
        {"id": "b", "role": "writer", "goal": "second", "depends_on": ["a"]},
    ]
    model = HangsWhenEmpty(reply(json.dumps({"steps": steps})), reply("FINAL ANSWER: A"))
    runner = OrchestratorRunner(model, ToolExecutor(ToolRegistry()), checkpoint_db=db)
    await runner.run(TaskRequest(task_id=task_id, prompt="two steps", budget=Budget(max_steps=10)))


if __name__ == "__main__":
    asyncio.run(main(Path(sys.argv[1]), sys.argv[2]))
