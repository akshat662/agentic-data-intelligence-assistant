"""Tool argument generation: converts a validated PlanStep into validated `run_sql` arguments.

`generate_tool_arguments` asks an LLM to propose a SQL query for a `run_sql` PlanStep, given
the step's intent and the dataset's catalog. It never lets that proposal reach DuckDB as-is:
the raw response is first checked against a private, minimal schema (`_RunSqlLLMOutput`), then
explicitly checked for a non-blank `query`, and finally passed through the existing
`adia.tools.sql_guard.check_sql` guard -- the same guard `run_sql` itself applies -- before
anything is returned. A missing query, a hallucinated table/column, a destructive statement, or
any other guard rejection collapses the whole result to `None` rather than returning a
partially-trusted query. Any failure to reach the LLM at all also degrades to `None`, the same
"never crash a run" contract every other agent in this system follows.

Scope: only `run_sql` is supported. A `PlanStep` naming any other `tool_family` is rejected
immediately, without an LLM call -- generating arguments for `compare_groups`,
`compute_correlation`, or `train_model` is out of scope for this phase.
"""

from collections.abc import Callable

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from adia.agents.llm_config import load_llm_settings
from adia.models.catalog import DatasetCatalog
from adia.models.plan import PlanStep
from adia.tools.run_sql import RunSqlArgs
from adia.tools.sql_guard import check_sql

#: The only `tool_family` this module knows how to generate arguments for.
_SUPPORTED_TOOL_FAMILY = "run_sql"

_SYSTEM_PROMPT = (
    "You are the SQL argument-generation component of a data analysis system. Given one plan "
    "step's intent and a dataset's exact catalog, propose a single read-only SQL SELECT query "
    "that fulfills the step's intent.\n\n"
    "Rules:\n"
    "- Reference exactly one table, named after the dataset_id given below.\n"
    "- Only reference columns that appear verbatim in the catalog. Never invent, guess, or "
    "assume a column exists.\n"
    "- Propose only a single SELECT statement (CTEs are fine). Never propose INSERT, UPDATE, "
    "DELETE, DROP, ALTER, CREATE, or any statement that isn't a read-only SELECT.\n"
    "- Do not include a trailing semicolon or multiple statements."
)


class _RunSqlLLMOutput(BaseModel):
    """Raw, unverified shape the LLM is asked to produce.

    Never returned to a caller directly -- `generate_tool_arguments` checks `query` is
    non-blank and then re-validates it in full through `sql_guard.check_sql` before trusting it.
    """

    model_config = ConfigDict(frozen=True)

    query: str = Field(default="", description="A single read-only SELECT statement.")


#: Signature every `llm_call` -- real or a test's fake -- must satisfy.
LLMCall = Callable[[PlanStep, DatasetCatalog], _RunSqlLLMOutput]


def generate_tool_arguments(
    step: PlanStep,
    catalog: DatasetCatalog,
    dataset_id: str,
    *,
    llm_call: LLMCall | None = None,
) -> RunSqlArgs | None:
    """Propose and validate `run_sql` arguments for one plan step.

    Args:
        step: The plan step to generate arguments for. Only `tool_family == 'run_sql'` is
            supported; anything else returns `None` without an LLM call.
        catalog: The dataset's catalog -- the only thing the LLM is shown about the data, and
            what its proposed query is validated against.
        dataset_id: The dataset (and therefore table) this query must be scoped to.
        llm_call: Override for the LLM call, e.g. a fake for tests. Defaults to a real OpenAI
            call built from `adia.agents.llm_config.load_llm_settings`.

    Returns:
        Validated `RunSqlArgs` wrapping a guarded, safe-to-execute query, or `None` if the step
        isn't a `run_sql` step, the LLM couldn't be reached, its proposal had no query, or the
        proposed query failed `sql_guard` (unknown table, unknown column, destructive
        statement, multi-statement, or any other guard rejection).
    """
    if step.tool_family != _SUPPORTED_TOOL_FAMILY:
        return None

    resolved_call = llm_call or _call_openai
    try:
        raw = resolved_call(step, catalog)
        return _build_args(raw, catalog=catalog, dataset_id=dataset_id)
    except Exception:  # the LLM is never trusted to be reachable or well-behaved
        return None


def _build_args(raw: _RunSqlLLMOutput, *, catalog: DatasetCatalog, dataset_id: str) -> RunSqlArgs:
    """Validate a raw LLM proposal and convert it into trusted, guarded `RunSqlArgs`.

    Raises:
        ValueError: If the LLM proposed no query (blank or whitespace-only).
        SqlGuardError: If the proposed query fails `sql_guard.check_sql` (unknown table,
            unknown column, non-SELECT statement, or multiple statements).
    """
    query = raw.query.strip()
    if not query:
        raise ValueError("LLM did not propose a SQL query.")

    guarded = check_sql(query, catalog=catalog, table_name=dataset_id)
    return RunSqlArgs(query=guarded.sql)


def _call_openai(step: PlanStep, catalog: DatasetCatalog) -> _RunSqlLLMOutput:
    """The real LLM call: build a `ChatOpenAI` client from the environment and invoke it.

    Raises whatever `load_llm_settings` or the API call itself raises --
    `generate_tool_arguments` is responsible for catching that, not this function.
    """
    settings = load_llm_settings()
    model = ChatOpenAI(model=settings.model, api_key=settings.api_key, temperature=0)
    structured_model = model.with_structured_output(_RunSqlLLMOutput)
    messages = _build_messages(step, catalog)
    return structured_model.invoke(messages)


def _build_messages(step: PlanStep, catalog: DatasetCatalog) -> list[BaseMessage]:
    """Build the prompt: the fixed system rules plus the step's intent and the exact catalog."""
    column_lines = "\n".join(f"- {col.name} ({col.semantic_type.value})" for col in catalog.columns)
    human_content = (
        f"Dataset: {catalog.dataset_id} ({catalog.row_count} rows)\n"
        f"Columns:\n{column_lines}\n\n"
        f"Plan step intent: {step.intent}"
    )
    return [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_content)]
