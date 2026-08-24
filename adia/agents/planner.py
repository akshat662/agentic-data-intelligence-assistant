"""The planner agent: proposes a plan; Python decides whether to trust it.

`create_plan` asks an LLM for a short sequence of steps — which tool family each step uses,
why, and what (if anything) it depends on. It never executes anything, never generates a
final answer, and never touches evidence directly; its only output is a list of
`PlanStep` (`adia.models.plan`), the same contract the rest of the system already uses.

Every proposed step is validated in Python before it can enter `AgentState`: an unsupported
tool family, a malformed step, a dependency on a step that doesn't exist in the same plan, or
any failure to reach the LLM at all collapses the *whole* plan to empty rather than admitting
a partially-trusted one. `expected_output`/`success_criteria` — required by `PlanStep` but not
really a judgment call — are filled in deterministically per tool family in Python, not asked
of the LLM, keeping its proposal surface to exactly what needs reasoning: which tool, why,
and in what order.
"""

from collections.abc import Callable
from typing import Self

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field, model_validator

from adia.agents.llm_config import load_llm_settings
from adia.models.catalog import DatasetCatalog
from adia.models.plan import PlanStep
from adia.models.state import FeasibilityResult, FeasibilityVerdict

#: The only tool families a plan step may name. Anything else is rejected, not guessed at.
_SUPPORTED_TOOL_FAMILIES = frozenset(
    {"profile_dataset", "run_sql", "compare_groups", "compute_correlation", "train_model"}
)

#: Deterministic, per-tool-family fill-in for the two `PlanStep` fields the LLM is not asked
#: to produce — these describe the tool's own contract, not a judgment call about this plan.
_EXPECTED_OUTPUT_BY_TOOL_FAMILY: dict[str, str] = {
    "profile_dataset": "Dataset-level and column-level summary statistics.",
    "run_sql": "Rows returned by a read-only SQL query.",
    "compare_groups": "Per-group count/mean/median/std and pairwise mean differences.",
    "compute_correlation": "A correlation matrix and ranked column-pair correlations.",
    "train_model": (
        "A trained model's evaluation metric against a naive baseline, plus feature "
        "importance."
    ),
}
_SUCCESS_CRITERIA_BY_TOOL_FAMILY: dict[str, str] = {
    family: f"{family} returns ok=True." for family in _SUPPORTED_TOOL_FAMILIES
}

_MAX_STEPS = 6

_SYSTEM_PROMPT = (
    "You are the planning component of a data analysis system. Given a user's question, a "
    "dataset's catalog, and a confirmation that the question is answerable, propose a short "
    "plan of steps to answer it. You do NOT execute anything and do NOT choose tool "
    "arguments, SQL text, or specific columns to use -- only which tool each step should use "
    "and why.\n\n"
    "Available tool families (choose only from this list):\n"
    "- profile_dataset: dataset-level and column-level summary statistics.\n"
    "- run_sql: read-only SQL aggregation/filtering over the dataset.\n"
    "- compare_groups: compares a numeric metric across the groups of a categorical column.\n"
    "- compute_correlation: pairwise correlation between numeric columns (not a causal "
    "claim).\n"
    "- train_model: fits a small classification/regression model and reports it against a "
    "naive baseline.\n\n"
    "Rules:\n"
    f"- Produce at most {_MAX_STEPS} steps.\n"
    "- Prefer profile_dataset first for any non-trivial question, unless the question is "
    "already narrow enough that profiling adds nothing.\n"
    "- Each step needs a step_id you choose and is unique within this plan (e.g. 'step_1'), "
    "a tool_family from the list above, a one-sentence purpose, and depends_on listing any "
    "OTHER step_id in this same plan that must run first (usually empty).\n"
    "- Do not propose tool arguments, SQL text, or column selections -- only the plan shape.\n\n"
    "For a 'why'/'how' investigation question (asking for a reason or explanation, not just a "
    "number), shape the plan as an investigation rather than a single lookup:\n"
    "- One observation step first, with empty depends_on, that establishes the fact being "
    "explained (e.g. which category/group has the extreme value).\n"
    "- Several supporting-analysis steps after it, each with depends_on listing the "
    "observation step's step_id, each testing one distinct possible explanation (e.g. order "
    "volume, average order value, a regional/segment split, a sub-category breakdown) rather "
    "than all reproducing the same comparison.\n"
    "- Do not propose a step whose only purpose is to state a causal conclusion -- that is the "
    "Synthesizer's job, from the evidence these steps produce, not something any tool computes."
)


class _PlanStepLLMOutput(BaseModel):
    """Raw, unverified shape for one proposed step."""

    model_config = ConfigDict(frozen=True)

    step_id: str = Field(..., min_length=1)
    tool_family: str = Field(..., min_length=1)
    purpose: str = Field(..., min_length=1, description="What this step should accomplish.")
    depends_on: list[str] = Field(default_factory=list)


class _PlannerLLMOutput(BaseModel):
    """Raw, unverified shape for the full proposed plan."""

    model_config = ConfigDict(frozen=True)

    steps: list[_PlanStepLLMOutput] = Field(default_factory=list, max_length=_MAX_STEPS)

    @model_validator(mode="after")
    def _step_ids_unique(self) -> Self:
        ids = [step.step_id for step in self.steps]
        if len(ids) != len(set(ids)):
            raise ValueError("Proposed plan has duplicate step_id values.")
        return self


#: Signature every `llm_call` — real or a test's fake — must satisfy.
LLMCall = Callable[[str, DatasetCatalog, FeasibilityResult], _PlannerLLMOutput]


def create_plan(
    question: str,
    catalog: DatasetCatalog,
    feasibility: FeasibilityResult,
    *,
    llm_call: LLMCall | None = None,
) -> list[PlanStep]:
    """Propose and validate a plan for answering `question` against `catalog`.

    Args:
        question: The user's question, verbatim.
        catalog: The dataset's catalog — the only thing the LLM is shown about the data.
        feasibility: The feasibility agent's result for this question. If its verdict is
            not `FEASIBLE`, no LLM call is made at all — there is nothing to plan for a
            question already ruled out.
        llm_call: Override for the LLM call, e.g. a fake for tests. Defaults to a real
            OpenAI call built from `adia.agents.llm_config.load_llm_settings`.

    Returns:
        A validated list of `PlanStep`. Empty if the question wasn't feasible, if the LLM
        could not be reached, or if its proposal failed validation for any reason (unknown
        tool family, malformed step, duplicate or unresolvable step dependency) — a
        partially-trusted plan is never returned.
    """
    if feasibility.verdict != FeasibilityVerdict.FEASIBLE:
        return []

    resolved_call = llm_call or _call_openai
    try:
        raw = resolved_call(question, catalog, feasibility)
        return _build_plan(raw)
    except Exception:  # the LLM is never trusted to be reachable or well-behaved
        return []


def _build_plan(raw: _PlannerLLMOutput) -> list[PlanStep]:
    """Validate a raw LLM plan and convert it into trusted `PlanStep` instances.

    Raises:
        ValueError: If any step names an unsupported `tool_family`, or depends on a step_id
            that doesn't resolve to another step in the same plan.
        pydantic.ValidationError: If a step is otherwise malformed for `PlanStep` (e.g. a
            self-dependency, caught by `PlanStep`'s own validator).
    """
    if not raw.steps:
        return []

    known_ids = {step.step_id for step in raw.steps}
    for step in raw.steps:
        if step.tool_family not in _SUPPORTED_TOOL_FAMILIES:
            raise ValueError(
                f"Step '{step.step_id}' names unsupported tool_family '{step.tool_family}'."
            )
        unknown_deps = sorted(d for d in step.depends_on if d not in known_ids)
        if unknown_deps:
            raise ValueError(f"Step '{step.step_id}' depends on unknown step(s): {unknown_deps}.")

    return [
        PlanStep(
            id=step.step_id,
            intent=step.purpose,
            tool_family=step.tool_family,
            depends_on=step.depends_on,
            expected_output=_EXPECTED_OUTPUT_BY_TOOL_FAMILY[step.tool_family],
            success_criteria=_SUCCESS_CRITERIA_BY_TOOL_FAMILY[step.tool_family],
        )
        for step in raw.steps
    ]


def _call_openai(
    question: str, catalog: DatasetCatalog, feasibility: FeasibilityResult
) -> _PlannerLLMOutput:
    """The real LLM call: build a `ChatOpenAI` client from the environment and invoke it.

    Raises whatever `load_llm_settings` or the API call itself raises — `create_plan` is
    responsible for catching that, not this function.
    """
    settings = load_llm_settings()
    model = ChatOpenAI(model=settings.model, api_key=settings.api_key, temperature=0)
    structured_model = model.with_structured_output(_PlannerLLMOutput)
    messages = _build_messages(question, catalog, feasibility)
    return structured_model.invoke(messages)


def _build_messages(
    question: str, catalog: DatasetCatalog, feasibility: FeasibilityResult
) -> list[BaseMessage]:
    """Build the prompt: the fixed system rules plus the question, catalog, and feasibility."""
    column_lines = "\n".join(f"- {col.name} ({col.semantic_type.value})" for col in catalog.columns)
    human_content = (
        f"Dataset: {catalog.dataset_id} ({catalog.row_count} rows)\n"
        f"Columns:\n{column_lines}\n\n"
        f"Feasibility assessment: {feasibility.reason}\n\n"
        f"Question: {question}"
    )
    return [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_content)]
