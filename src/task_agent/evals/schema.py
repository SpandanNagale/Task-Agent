"""Eval task format. One YAML file per category under ``evals/suite/``, each with a
``tasks:`` list. See ``evals/README.md`` for how to write a good task.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from task_agent.core.types import Budget


class Category(StrEnum):
    RESEARCH = "research"
    BROWSING = "browsing"
    DATA_CODE = "data_code"
    COMMUNICATION = "communication"
    MULTI_STEP = "multi_step"


class _Checker(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExactChecker(_Checker):
    """Answer equals ``expected`` after normalisation (case, whitespace, trailing period)."""

    type: Literal["exact"] = "exact"
    expected: str
    case_sensitive: bool = False


class OneOfChecker(_Checker):
    """Answer equals any of ``options`` after normalisation."""

    type: Literal["one_of"] = "one_of"
    options: list[str] = Field(min_length=1)


class ContainsChecker(_Checker):
    """Answer contains every string in ``all_of`` and at least one in ``any_of``."""

    type: Literal["contains"] = "contains"
    all_of: list[str] = Field(default_factory=list)
    any_of: list[str] = Field(default_factory=list)
    none_of: list[str] = Field(default_factory=list)
    case_sensitive: bool = False

    @model_validator(mode="after")
    def _not_empty(self) -> ContainsChecker:
        if not (self.all_of or self.any_of):
            raise ValueError("contains checker needs all_of or any_of")
        return self


class RegexChecker(_Checker):
    """``pattern`` matches somewhere in the answer (Python ``re.search``)."""

    type: Literal["regex"] = "regex"
    pattern: str
    ignore_case: bool = True


class NumericChecker(_Checker):
    """The first number in the answer is within tolerance of ``expected``."""

    type: Literal["numeric"] = "numeric"
    expected: float
    abs_tol: float = 0.0
    rel_tol: float = 1e-6


class LLMJudgeChecker(_Checker):
    """A judge model grades the answer against ``rubric`` and returns PASS or FAIL."""

    type: Literal["llm_judge"] = "llm_judge"
    rubric: str = Field(min_length=10)
    role: str = "judge"


Checker = Annotated[
    ExactChecker | OneOfChecker | ContainsChecker | RegexChecker | NumericChecker | LLMJudgeChecker,
    Field(discriminator="type"),
]


class EvalTask(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_\-]{2,63}$")
    category: Category
    prompt: str = Field(min_length=10)
    checker: Checker
    requires: frozenset[str] = frozenset()  # tool groups the task cannot be solved without
    tags: frozenset[str] = frozenset()
    budget: Budget | None = None
    notes: str = ""  # why the task exists / what it tests; not shown to the agent


class EvalSuiteFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tasks: list[EvalTask]
