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


class EvaluationTier(StrEnum):
    """A cross-cutting grouping of `QuestionCategory` by how many steps answering typically
    takes, used to report evaluation metrics without inventing a second question taxonomy.

    Not stored on `BenchmarkQuestion` -- derived from `category` via `_TIER_BY_CATEGORY`, so
    the mapping lives in exactly one place and existing question JSON never needs to carry it.
    """

    DIRECT = "direct"
    INVESTIGATION = "investigation"
    REFUSAL = "refusal"


#: The single source of truth for which categories belong to which evaluation tier.
_TIER_BY_CATEGORY: dict[QuestionCategory, EvaluationTier] = {
    QuestionCategory.DESCRIPTIVE: EvaluationTier.DIRECT,
    QuestionCategory.SQL_AGGREGATION: EvaluationTier.DIRECT,
    QuestionCategory.STATISTICAL_COMPARISON: EvaluationTier.DIRECT,
    QuestionCategory.PREDICTIVE: EvaluationTier.DIRECT,
    QuestionCategory.ROOT_CAUSE: EvaluationTier.INVESTIGATION,
    QuestionCategory.UNANSWERABLE: EvaluationTier.REFUSAL,
}


class InvestigationExpectation(BaseModel):
    """Investigation-specific evaluation rubric, set only on `root_cause` questions.

    This is a rubric for a human reviewer (and for the deterministic checks
    `bench/evaluation_report.py` can actually run), not a gold answer -- `expected_observation`
    and `acceptable_conclusion_style` require an independent oracle to grade automatically,
    which does not exist yet (see `bench/README.md`'s "Implemented vs. Planned"). Only
    `forbidden_causal_phrases` is deterministically checkable today: a case-insensitive scan
    of the final answer text, run by `bench/evaluation_report.py`.
    """

    model_config = ConfigDict(frozen=True)

    expected_observation: str = Field(
        ..., min_length=1, description="The anchor fact a correct investigation should find."
    )
    expected_analysis_dimensions: list[str] = Field(
        ...,
        min_length=1,
        description="Distinct candidate explanations a good plan should test (not tool names).",
    )
    acceptable_conclusion_style: str = Field(
        ...,
        min_length=1,
        description="Rubric prose describing what a properly-hedged conclusion sounds like.",
    )
    forbidden_causal_phrases: list[str] = Field(
        default_factory=list,
        description="Phrases that must not appear in the final answer, checked verbatim "
        "(case-insensitive) by bench/evaluation_report.py.",
    )


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
    investigation: InvestigationExpectation | None = Field(
        default=None,
        description="Investigation rubric; only ever set on 'root_cause' questions.",
    )

    @property
    def evaluation_tier(self) -> EvaluationTier:
        """This question's cross-cutting evaluation tier, derived from `category`."""
        return _TIER_BY_CATEGORY[self.category]

    @model_validator(mode="after")
    def _category_matches_answerable(self) -> Self:
        is_unanswerable_category = self.category == QuestionCategory.UNANSWERABLE
        if is_unanswerable_category and self.answerable:
            raise ValueError("A question in the 'unanswerable' category cannot be answerable.")
        if not is_unanswerable_category and not self.answerable:
            raise ValueError("Only 'unanswerable'-category questions may be unanswerable.")
        return self

    @model_validator(mode="after")
    def _investigation_only_on_root_cause(self) -> Self:
        if self.investigation is not None and self.category != QuestionCategory.ROOT_CAUSE:
            raise ValueError("`investigation` may only be set on 'root_cause' questions.")
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
