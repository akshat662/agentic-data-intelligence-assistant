"""Benchmark question contract and loader.

Kept local to `bench/` rather than `adia/models/` deliberately: this is a benchmarking
concern, not a contract the runtime system (tools, evidence, future agents) needs to know
about. `adia/models/` stays reserved for cross-layer contracts the actual system depends on.
"""

import json
from enum import StrEnum
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator


class QuestionCategory(StrEnum):
    """The six question categories the benchmark is organized around.

    Five answerable categories cover the classes of question the tool layer is built to
    handle; `UNANSWERABLE` is the refusal slice — questions the system should decline, not
    attempt. Reporting refusal recall *and* false-refusal rate later requires this slice to
    exist from the start, not be added as an afterthought once the answerable set looks good.
    """

    DESCRIPTIVE = "descriptive"
    SQL_AGGREGATION = "sql_aggregation"
    STATISTICAL_COMPARISON = "statistical_comparison"
    ROOT_CAUSE = "root_cause"
    PREDICTIVE = "predictive"
    UNANSWERABLE = "unanswerable"


class BenchmarkQuestion(BaseModel):
    """One benchmark question: a natural-language prompt plus its expected shape of answer.

    No gold *answer* value lives here yet — that requires a real dataset and an independently
    written oracle (Phase 3 scope). This is the earlier, dataset-independent fact: what the
    question is, what category it belongs to, and roughly which tools should be involved.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., min_length=1, description="Stable identifier, e.g. 'q001'.")
    dataset_id: str = Field(..., min_length=1, description="Dataset this question is asked of.")
    category: QuestionCategory
    question: str = Field(..., min_length=1, description="The natural-language question text.")
    answerable: bool = Field(
        ..., description="Whether the system is expected to answer this, or refuse."
    )
    gold_tools: list[str] = Field(
        default_factory=list,
        description="Tool(s) expected to be involved in answering; empty for unanswerable.",
    )
    notes: str | None = Field(
        default=None, description="Why this question is answerable/unanswerable, if non-obvious."
    )

    @model_validator(mode="after")
    def _category_matches_answerable(self) -> Self:
        is_unanswerable_category = self.category == QuestionCategory.UNANSWERABLE
        if is_unanswerable_category and self.answerable:
            raise ValueError("A question in the 'unanswerable' category cannot be answerable.")
        if not is_unanswerable_category and not self.answerable:
            raise ValueError("Only 'unanswerable'-category questions may be unanswerable.")
        return self


def load_questions(path: str | Path) -> list[BenchmarkQuestion]:
    """Load and validate benchmark questions from a JSON file.

    Accepts either a bare JSON array of question objects, or an object with a `"questions"`
    key holding that array (the shape `bench/questions.json` actually uses, so it can also
    carry a top-level `note`/`schema_version` without those being mistaken for questions).

    Args:
        path: Path to the questions JSON file.

    Returns:
        The validated questions, in file order.

    Raises:
        FileNotFoundError: If `path` does not exist.
        pydantic.ValidationError: If an entry doesn't match `BenchmarkQuestion`.
        ValueError: If two questions share the same `id`.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Benchmark questions file not found: {resolved}")

    payload = json.loads(resolved.read_text())
    raw_questions = payload["questions"] if isinstance(payload, dict) else payload

    questions = [BenchmarkQuestion.model_validate(item) for item in raw_questions]
    seen_ids: set[str] = set()
    for question in questions:
        if question.id in seen_ids:
            raise ValueError(f"Duplicate question id: '{question.id}'")
        seen_ids.add(question.id)
    return questions
