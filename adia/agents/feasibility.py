"""The feasibility agent: the first and only place an LLM's judgment is trusted — partially.

`assess_feasibility` asks an LLM whether a question can be answered from a dataset's catalog,
which columns it believes are relevant, and why. It never trusts that answer as-is: every
column name the LLM names is cross-checked in Python against the catalog's real columns
before a `FeasibilityResult` is built, and a hallucinated column forces the verdict to
`INFEASIBLE` regardless of what the LLM claimed. The LLM reasons; Python verifies. Any
failure to reach the LLM at all (no API key, a network error, a malformed response) also
degrades to a safe `INFEASIBLE` result rather than raising — this agent never crashes a run.
"""

from collections.abc import Callable
from typing import Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from adia.agents.llm_config import load_llm_settings
from adia.models.catalog import DatasetCatalog
from adia.models.state import FeasibilityResult, FeasibilityVerdict

_SYSTEM_PROMPT = (
    "You are the feasibility-assessment component of a data analysis system. You are given "
    "a user's question and the exact catalog of a dataset (its columns, types, and "
    "cardinality) — nothing else. Decide whether the question is answerable from this data "
    "alone.\n\n"
    "Rules:\n"
    "- Only ever name columns that appear verbatim in the catalog below. Never invent, "
    "guess, or assume a column exists.\n"
    "- If the question needs information not present in the catalog (e.g. future/forecast "
    "data, external data, free-text fields that don't exist), the verdict is 'infeasible' "
    "and you must say what's missing in missing_capabilities.\n"
    "- If the question is ambiguous about which column or entity it refers to, use "
    "'needs_clarification' rather than guessing.\n"
    "- Otherwise, if the question can plausibly be answered using only the columns listed, "
    "the verdict is 'feasible'.\n"
    "- relevant_columns must be a subset of the catalog's column names.\n"
    "- reasoning must briefly justify the verdict in one or two sentences."
)


class _FeasibilityLLMOutput(BaseModel):
    """Raw, unverified shape the LLM is asked to produce.

    Never returned to a caller directly — `assess_feasibility` cross-checks and reshapes
    this into the real, trusted `FeasibilityResult` contract.
    """

    model_config = ConfigDict(frozen=True)

    verdict: Literal["feasible", "infeasible", "needs_clarification"]
    relevant_columns: list[str] = Field(
        default_factory=list, description="Columns the LLM believes the question refers to."
    )
    missing_capabilities: list[str] = Field(
        default_factory=list,
        description="Analytical capabilities the LLM believes are needed but unavailable.",
    )
    reasoning: str = Field(..., min_length=1, description="The LLM's explanation for its verdict.")


#: Signature every `llm_call` — real or a test's fake — must satisfy.
LLMCall = Callable[[str, DatasetCatalog], _FeasibilityLLMOutput]


def assess_feasibility(
    question: str,
    catalog: DatasetCatalog,
    *,
    llm_call: LLMCall | None = None,
) -> FeasibilityResult:
    """Assess whether `question` is answerable from `catalog`, using an LLM plus verification.

    Args:
        question: The user's question, verbatim.
        catalog: The dataset's catalog — the only thing the LLM is shown about the data.
        llm_call: Override for the LLM call, e.g. a fake for tests. Defaults to a real
            OpenAI call built from `adia.agents.llm_config.load_llm_settings`.

    Returns:
        A `FeasibilityResult`. `verdict` is forced to `INFEASIBLE` if the LLM named any
        column not present in `catalog`, regardless of what it claimed; the same happens,
        with a explanatory `reason`, if the LLM could not be reached at all.
    """
    resolved_call = llm_call or _call_openai

    try:
        raw = resolved_call(question, catalog)
    except Exception as exc:  # the LLM is never trusted to be reachable, only to reason
        return FeasibilityResult(
            verdict=FeasibilityVerdict.INFEASIBLE,
            reason=f"Feasibility could not be assessed: {exc}",
        )

    known_columns = set(catalog.column_names())
    hallucinated = sorted(c for c in raw.relevant_columns if c not in known_columns)

    if hallucinated:
        return FeasibilityResult(
            verdict=FeasibilityVerdict.INFEASIBLE,
            reason=(
                f"The model referenced column(s) {hallucinated} that do not exist in the "
                f"'{catalog.dataset_id}' dataset. Original reasoning: {raw.reasoning}"
            ),
            missing_columns=hallucinated,
            missing_capabilities=raw.missing_capabilities,
        )

    return FeasibilityResult(
        verdict=FeasibilityVerdict(raw.verdict),
        reason=raw.reasoning,
        missing_columns=[],
        missing_capabilities=raw.missing_capabilities,
    )


def _call_openai(question: str, catalog: DatasetCatalog) -> _FeasibilityLLMOutput:
    """The real LLM call: build a `ChatOpenAI` client from the environment and invoke it.

    Raises whatever `load_llm_settings` or the API call itself raises — `assess_feasibility`
    is responsible for catching that, not this function.
    """
    settings = load_llm_settings()
    model = ChatOpenAI(model=settings.model, api_key=settings.api_key, temperature=0)
    structured_model = model.with_structured_output(_FeasibilityLLMOutput)
    messages = _build_messages(question, catalog)
    return structured_model.invoke(messages)


def _build_messages(question: str, catalog: DatasetCatalog) -> list[BaseMessage]:
    """Build the prompt: the fixed system rules plus the question and the exact catalog."""
    column_lines = "\n".join(
        f"- {col.name} ({col.semantic_type.value}, {col.unique_count} distinct values, "
        f"{col.null_rate:.1%} missing)"
        for col in catalog.columns
    )
    human_content = (
        f"Dataset: {catalog.dataset_id} ({catalog.row_count} rows)\n"
        f"Columns:\n{column_lines}\n\n"
        f"Question: {question}"
    )
    return [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_content)]
