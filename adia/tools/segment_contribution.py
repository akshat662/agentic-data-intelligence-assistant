"""Deterministic segment-contribution (decomposition) tool.

`segment_contribution` answers "which sub-groups drive this total, and by how much" --
e.g. which Sub-Categories make up most of Technology's Sales. For each distinct value of
`entity_column` (optionally scoped to rows where `parent_column == parent_value`, e.g.
`Category == "Technology"`), it reports row count, summed `metric_column`, mean
`metric_column`, and that entity's share of the in-scope total, ranked by total descending.

This is pure pandas group-by/sum arithmetic -- no LLM call, no reasoning, no judgment. Like
`adia.tools.compare_groups` and `adia.tools.correlation`, a share ranking says nothing about
*why* one segment is larger than another: `causal_claim_allowed` is always `False`, so
`adia.validate.static.validate_answer` rejects causal language in any answer that cites it.
"""

import time
from typing import Any, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError, model_validator

from adia.data.loader import load_dataset
from adia.evidence.ids import compute_args_hash, generate_evidence_id
from adia.evidence.store import EvidenceStore
from adia.models.errors import ToolError, ToolErrorKind
from adia.models.evidence import Evidence
from adia.models.provenance import Provenance
from adia.models.tool_result import ToolResult

_TOOL_NAME = "segment_contribution"


class SegmentContributionArgs(BaseModel):
    """Validated input contract for `segment_contribution`."""

    dataset_id: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    entity_column: str = Field(
        ..., min_length=1, description="Categorical column to break the total down by."
    )
    metric_column: str = Field(..., min_length=1, description="Numeric column to decompose.")
    parent_column: str | None = Field(
        default=None, description="Optional column to scope the decomposition to one value of."
    )
    parent_value: str | None = Field(
        default=None, description="The value of parent_column to scope to, e.g. 'Technology'."
    )
    top_k: int = Field(default=10, gt=0, description="Maximum number of ranked entities to report.")

    @model_validator(mode="after")
    def _parent_column_and_value_together(self) -> Self:
        if (self.parent_column is None) != (self.parent_value is None):
            raise ValueError(
                "parent_column and parent_value must both be provided together, or both omitted."
            )
        return self


def segment_contribution(
    dataset_id: str,
    source_path: str,
    entity_column: str,
    metric_column: str,
    evidence_store: EvidenceStore,
    *,
    parent_column: str | None = None,
    parent_value: str | None = None,
    top_k: int = 10,
    plan_step_id: str | None = None,
) -> ToolResult:
    """Decompose a numeric metric's total across the entities of a categorical column.

    Rows with a missing `entity_column` value are dropped before grouping, the same
    convention `adia.tools.compare_groups` uses for `group_column`. When `parent_column`/
    `parent_value` are given, the decomposition is computed only over rows where
    `parent_column == parent_value` (e.g. break Sales down by Sub-Category, but only within
    Category == "Technology") -- both must be given together, or neither.

    Args:
        dataset_id: Stable identifier for the dataset.
        source_path: Path to a `.parquet` or `.csv` file.
        entity_column: Categorical column whose distinct values are ranked by contribution.
        metric_column: Numeric column to decompose.
        evidence_store: Store to record the resulting evidence in.
        parent_column: Optional column to restrict the decomposition to one value of.
        parent_value: The value of `parent_column` to restrict to. Required if
            `parent_column` is given, and only meaningful together with it.
        top_k: Maximum number of ranked entities to report (highest total first).
        plan_step_id: ID of the plan step this call belongs to, if any.

    Returns:
        A `ToolResult`. On success, `data` holds `entity_column`, `metric_column`,
        `parent_column`/`parent_value` (echoed, possibly `None`), `entity_count` (distinct
        entities in scope, before `top_k` truncation), `overall_total`/`overall_count`/
        `overall_mean` (over all rows in scope, not just the reported entities), `entities`
        (up to `top_k` records -- `entity`, `rank` (1 = largest total), `count`, `total`,
        `mean`, `share_of_total` -- ordered by `total` descending), and
        `causal_claim_allowed: False`. On failure, `error` describes exactly what was
        rejected or what went wrong.
    """
    started = time.perf_counter()
    args = {
        "dataset_id": dataset_id,
        "source_path": source_path,
        "entity_column": entity_column,
        "metric_column": metric_column,
        "parent_column": parent_column,
        "parent_value": parent_value,
        "top_k": top_k,
    }

    try:
        SegmentContributionArgs(**args)
    except ValidationError as exc:
        return _error_result(args, ToolErrorKind.VALIDATION, str(exc), started)

    try:
        df = load_dataset(source_path)
    except FileNotFoundError as exc:
        return _error_result(args, ToolErrorKind.NOT_FOUND, str(exc), started)
    except ValueError as exc:
        return _error_result(args, ToolErrorKind.EXECUTION, str(exc), started)

    columns_to_check = [entity_column, metric_column]
    if parent_column is not None:
        columns_to_check.append(parent_column)
    for column in columns_to_check:
        if column not in df.columns:
            return _error_result(
                args,
                ToolErrorKind.VALIDATION,
                f"Unknown column '{column}'.",
                started,
                details={"column": column},
            )

    if not pd.api.types.is_numeric_dtype(df[metric_column]):
        return _error_result(
            args,
            ToolErrorKind.VALIDATION,
            f"Column '{metric_column}' is not numeric; cannot decompose it.",
            started,
            details={"column": metric_column, "dtype": str(df[metric_column].dtype)},
        )

    working = df
    if parent_column is not None:
        working = working[working[parent_column] == parent_value]
    working = working[[entity_column, metric_column]].dropna(subset=[entity_column])

    if working.empty:
        scope = f" where '{parent_column}' == {parent_value!r}" if parent_column else ""
        return _error_result(
            args,
            ToolErrorKind.INSUFFICIENT_DATA,
            f"No valid rows to decompose{scope} after dropping missing "
            f"'{entity_column}' values.",
            started,
        )

    # No `summary.empty` check needed here: `working` is already confirmed non-empty above,
    # and grouping a non-empty frame always yields at least one group.
    summary = working.groupby(entity_column, observed=True)[metric_column].agg(
        ["count", "sum", "mean"]
    )

    overall_total = _json_safe(working[metric_column].sum())
    overall_count = int(working[metric_column].count())
    overall_mean = _json_safe(working[metric_column].mean()) if overall_count else None

    ranked = summary.reset_index().sort_values(
        by=["sum", entity_column], ascending=[False, True]
    )
    entity_count = len(ranked)

    entities = []
    # .iterrows(), not .itertuples(): entity_column is a runtime value (e.g. "Sub-Category",
    # not a valid Python identifier), and itertuples silently renames such columns to
    # positional fields (_0, _1, ...) instead of raising, which would look fine here and
    # break at runtime -- bracket access on a row Series always works regardless of the name.
    for rank, (_, row) in enumerate(ranked.head(top_k).iterrows(), start=1):
        total = _json_safe(row["sum"])
        share = None if not overall_total or total is None else total / overall_total
        entities.append(
            {
                "entity": _json_safe(row[entity_column]),
                "rank": rank,
                "count": int(row["count"]),
                "total": total,
                "mean": _json_safe(row["mean"]),
                "share_of_total": share,
            }
        )

    data = {
        "entity_column": entity_column,
        "metric_column": metric_column,
        "parent_column": parent_column,
        "parent_value": parent_value,
        "entity_count": entity_count,
        "overall_total": overall_total,
        "overall_count": overall_count,
        "overall_mean": overall_mean,
        "entities": entities,
        "causal_claim_allowed": False,
    }

    evidence_id = generate_evidence_id(_TOOL_NAME, args)
    provenance = Provenance(
        tool_name=_TOOL_NAME,
        args=args,
        args_hash=compute_args_hash(args),
        row_count=len(working),
        library_versions={"pandas": pd.__version__},
    )
    evidence = evidence_store.add(
        Evidence(
            id=evidence_id,
            tool=_TOOL_NAME,
            data=data,
            provenance=provenance,
            plan_step_id=plan_step_id,
        )
    )

    return ToolResult(
        ok=True,
        tool=_TOOL_NAME,
        evidence_id=evidence.id,
        data=evidence.data,
        provenance=evidence.provenance,
        duration_ms=_elapsed_ms(started),
    )


def _json_safe(value: Any) -> Any:
    """Convert a pandas/numpy scalar into a plain JSON-safe, full-precision Python value."""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if pd.isna(value) else float(value)
    if isinstance(value, float):
        return None if pd.isna(value) else value
    if pd.isna(value):
        return None
    return value


def _error_result(
    args: dict[str, Any],
    kind: ToolErrorKind,
    message: str,
    started: float,
    *,
    details: dict[str, Any] | None = None,
) -> ToolResult:
    """Build a typed `ToolResult` failure. The tool never lets an exception reach its caller."""
    return ToolResult(
        ok=False,
        tool=_TOOL_NAME,
        error=ToolError(kind=kind, message=message, details=details or {}),
        duration_ms=_elapsed_ms(started),
    )


def _elapsed_ms(started: float) -> float:
    """Milliseconds elapsed since a `time.perf_counter()` reading."""
    return (time.perf_counter() - started) * 1000
