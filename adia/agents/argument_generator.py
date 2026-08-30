from collections.abc import Callable
from typing import Any, Literal

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from adia.agents.llm_config import load_llm_settings
from adia.models.catalog import DatasetCatalog
from adia.models.plan import PlanStep
from adia.tools.compare_groups import CompareGroupsArgs
from adia.tools.correlation import ComputeCorrelationArgs
from adia.tools.ml_model import TrainModelArgs
from adia.tools.run_sql import RunSqlArgs
from adia.tools.segment_contribution import SegmentContributionArgs
from adia.tools.sql_guard import check_sql

#: The tool families this module knows how to generate arguments for.
_SUPPORTED_TOOL_FAMILIES = frozenset(
    {"run_sql", "compare_groups", "compute_correlation", "train_model", "segment_contribution"}
)

_SYSTEM_PROMPT = (
    "You are the argument-generation component of a data analysis system. Given one plan "
    "step's intent and a dataset's exact catalog, propose the precise arguments needed to "
    "run the tool assigned to this step.\n\n"
    "Rules:\n"
    "- Only reference columns that appear verbatim in the catalog. Never invent, guess, or "
    "assume a column exists.\n"
    "- For run_sql: Propose a single read-only SELECT statement. Reference exactly one "
    "table, named after the dataset_id given below. Do not include a trailing semicolon "
    "or multiple statements.\n"
    "- For compare_groups: Propose a categorical group_column and a numeric metric_column.\n"
    "- For compute_correlation: Propose a list of numeric columns to correlate, or null "
    "to correlate all numeric columns.\n"
    "- For train_model: Propose a target_column, a list of feature_columns, the "
    "task_type (classification or regression), and the model_type.\n"
    "  - classification models: 'logistic_regression', 'random_forest_classifier'\n"
    "  - regression models: 'linear_regression', 'random_forest_regressor'\n"
    "- For segment_contribution: Propose an entity_column (categorical -- what to break the "
    "total down by, e.g. Sub-Category) and a metric_column (numeric -- what to decompose, "
    "e.g. Sales). Optionally propose parent_column and parent_value TOGETHER to scope the "
    "decomposition to one value (e.g. parent_column='Category', parent_value='Technology') -- "
    "ground parent_value in the findings from steps this one depends on, if any are shown "
    "below. Otherwise leave both null.\n"
)


class _RunSqlLLMOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    query: str = Field(default="", description="A single read-only SELECT statement.")

class _CompareGroupsLLMOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    group_column: str = Field(description="Categorical column to group by.")
    metric_column: str = Field(description="Numeric column to compare across groups.")

class _ComputeCorrelationLLMOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    columns: list[str] | None = Field(
        default=None, description="Numeric columns to correlate, or null for all numeric."
    )

class _TrainModelLLMOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    target_column: str = Field(description="Column to predict.")
    feature_columns: list[str] = Field(description="Numeric columns to use as features.")
    task_type: Literal["classification", "regression"] = Field(
        description="The type of machine learning task."
    )
    model_type: str = Field(
        description=(
            "The model type (e.g. logistic_regression, random_forest_classifier, "
            "linear_regression, random_forest_regressor)."
        )
    )

class _SegmentContributionLLMOutput(BaseModel):
    model_config = ConfigDict(frozen=True)
    entity_column: str = Field(description="Categorical column to break the total down by.")
    metric_column: str = Field(description="Numeric column to decompose.")
    parent_column: str | None = Field(
        default=None, description="Optional column to scope the decomposition to one value of."
    )
    parent_value: str | None = Field(
        default=None, description="The value of parent_column to scope to, if given."
    )


OutputArgs = (
    RunSqlArgs | CompareGroupsArgs | ComputeCorrelationArgs | TrainModelArgs
    | SegmentContributionArgs
)

#: Signature every `llm_call` -- real or a test's fake -- must satisfy.
LLMCall = Callable[[PlanStep, DatasetCatalog], Any]


def generate_tool_arguments(
    step: PlanStep,
    catalog: DatasetCatalog,
    dataset_id: str,
    *,
    dependency_context: str = "",
    llm_call: LLMCall | None = None,
) -> OutputArgs | None:
    """Propose and validate arguments for one plan step.

    Args:
        step: The plan step to generate arguments for.
        catalog: The dataset's catalog.
        dataset_id: The dataset this query must be scoped to.
        dependency_context: Rendered evidence (e.g. from
            `adia.evidence.renderer.render_evidence_context`) produced by this step's own
            `depends_on` steps, if any -- shown to the LLM alongside the catalog so it can
            ground its proposal in a prior step's finding (e.g. a category identified
            upstream) instead of only the dataset schema. Defaults to `""`, which produces
            the exact same prompt as before this parameter existed -- every dependency-free
            step (the only kind that existed previously) is unaffected. Only reaches the real
            LLM call path; a caller-supplied `llm_call` still receives just `(step, catalog)`,
            unchanged, so it can be threaded straight to a rendering helper without knowing
            about this parameter.
        llm_call: Override for the LLM call, e.g. a fake for tests.

    Returns:
        Validated arguments wrapping a safe-to-execute query/config, or `None` if the step
        isn't supported, the LLM couldn't be reached, or the proposed arguments failed validation.
    """
    if step.tool_family not in _SUPPORTED_TOOL_FAMILIES:
        return None

    try:
        raw = (
            llm_call(step, catalog)
            if llm_call is not None
            else _call_openai(step, catalog, dependency_context)
        )
        return _build_args(raw, step.tool_family, catalog=catalog, dataset_id=dataset_id)
    except Exception:  # the LLM is never trusted to be reachable or well-behaved
        return None


def _build_args(
    raw: Any, tool_family: str, *, catalog: DatasetCatalog, dataset_id: str
) -> OutputArgs:
    """Validate a raw LLM proposal and convert it into trusted, guarded arguments."""
    if tool_family == "run_sql":
        query = raw.query.strip()
        if not query:
            raise ValueError("LLM did not propose a SQL query.")
        guarded = check_sql(query, catalog=catalog, table_name=dataset_id)
        return RunSqlArgs(query=guarded.sql)

    elif tool_family == "compare_groups":
        if raw.group_column not in catalog.column_names():
            raise ValueError(f"Unknown group_column: {raw.group_column}")
        if raw.metric_column not in catalog.column_names():
            raise ValueError(f"Unknown metric_column: {raw.metric_column}")
        return CompareGroupsArgs(
            dataset_id=dataset_id,
            source_path=catalog.source_path,
            group_column=raw.group_column,
            metric_column=raw.metric_column
        )

    elif tool_family == "compute_correlation":
        if raw.columns is not None:
            for col in raw.columns:
                if col not in catalog.column_names():
                    raise ValueError(f"Unknown column: {col}")
        return ComputeCorrelationArgs(
            dataset_id=dataset_id,
            source_path=catalog.source_path,
            columns=raw.columns
        )

    elif tool_family == "train_model":
        if raw.target_column not in catalog.column_names():
            raise ValueError(f"Unknown target_column: {raw.target_column}")
        for col in raw.feature_columns:
            if col not in catalog.column_names():
                raise ValueError(f"Unknown feature column: {col}")
        return TrainModelArgs(
            dataset_id=dataset_id,
            source_path=catalog.source_path,
            target_column=raw.target_column,
            feature_columns=raw.feature_columns,
            task_type=raw.task_type,
            model_type=raw.model_type
        )

    elif tool_family == "segment_contribution":
        if raw.entity_column not in catalog.column_names():
            raise ValueError(f"Unknown entity_column: {raw.entity_column}")
        if raw.metric_column not in catalog.column_names():
            raise ValueError(f"Unknown metric_column: {raw.metric_column}")
        if raw.parent_column is not None and raw.parent_column not in catalog.column_names():
            raise ValueError(f"Unknown parent_column: {raw.parent_column}")
        return SegmentContributionArgs(
            dataset_id=dataset_id,
            source_path=catalog.source_path,
            entity_column=raw.entity_column,
            metric_column=raw.metric_column,
            parent_column=raw.parent_column,
            parent_value=raw.parent_value,
        )
    raise ValueError(f"Unsupported tool family: {tool_family}")


def _call_openai(step: PlanStep, catalog: DatasetCatalog, dependency_context: str = "") -> Any:
    """The real LLM call: build a `ChatOpenAI` client from the environment and invoke it."""
    settings = load_llm_settings()
    model = ChatOpenAI(model=settings.model, api_key=settings.api_key, temperature=0)

    schema_map = {
        "run_sql": _RunSqlLLMOutput,
        "compare_groups": _CompareGroupsLLMOutput,
        "compute_correlation": _ComputeCorrelationLLMOutput,
        "train_model": _TrainModelLLMOutput,
        "segment_contribution": _SegmentContributionLLMOutput,
    }
    structured_model = model.with_structured_output(schema_map[step.tool_family])
    messages = _build_messages(step, catalog, dependency_context)
    return structured_model.invoke(messages)


def _build_messages(
    step: PlanStep, catalog: DatasetCatalog, dependency_context: str = ""
) -> list[BaseMessage]:
    """Build the prompt: the fixed system rules plus the step's intent and the exact catalog.

    Args:
        step: The plan step to generate arguments for.
        catalog: The dataset's catalog.
        dependency_context: Rendered evidence from this step's own dependency steps, if any.
            When blank (the default, and the only case that existed before this parameter),
            the prompt is byte-identical to before -- no extra section is added.
    """
    column_lines = "\n".join(f"- {col.name} ({col.semantic_type.value})" for col in catalog.columns)
    human_content = (
        f"Dataset: {catalog.dataset_id} ({catalog.row_count} rows)\n"
        f"Columns:\n{column_lines}\n\n"
        f"Plan step intent: {step.intent}"
    )
    if dependency_context:
        human_content += (
            "\n\nFindings from steps this one depends on (ground your proposal in these "
            f"where relevant):\n{dependency_context}"
        )
    return [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_content)]
