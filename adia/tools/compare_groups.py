"""Deterministic group comparison tool.

`compare_groups` computes descriptive statistics for a numeric metric split by a categorical
group column — count, mean, median, and standard deviation per group, plus pairwise mean
differences between groups. It performs no hypothesis testing (no p-values, no test
selection, no effect sizes): that is out of scope for this phase and, if added later,
belongs in its own typed contract rather than folded silently into this tool's output.
"""

import itertools
import time
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from adia.data.loader import load_dataset
from adia.evidence.ids import compute_args_hash, generate_evidence_id
from adia.evidence.store import EvidenceStore
from adia.models.errors import ToolError, ToolErrorKind
from adia.models.evidence import Evidence
from adia.models.provenance import Provenance
from adia.models.tool_result import ToolResult

_TOOL_NAME = "compare_groups"


class CompareGroupsArgs(BaseModel):
    """Validated input contract for `compare_groups`."""

    dataset_id: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    group_column: str = Field(..., min_length=1, description="Categorical column to group by.")
    metric_column: str = Field(
        ..., min_length=1, description="Numeric column to compare across groups."
    )


def compare_groups(
    dataset_id: str,
    source_path: str,
    group_column: str,
    metric_column: str,
    evidence_store: EvidenceStore,
    *,
    plan_step_id: str | None = None,
) -> ToolResult:
    """Compare a numeric metric across the groups defined by a categorical column.

    Rows with a missing `group_column` value are dropped before grouping. Within a group,
    `count`/`mean`/`median`/`std` are computed over non-missing `metric_column` values only
    (pandas' default `NaN`-skipping behavior); a group with fewer than 2 valid values reports
    `std: None` rather than a defined-but-meaningless sample standard deviation.

    Args:
        dataset_id: Stable identifier for the dataset.
        source_path: Path to a `.parquet` or `.csv` file.
        group_column: Column whose distinct values define the groups.
        metric_column: Numeric column to summarize within each group.
        evidence_store: Store to record the resulting evidence in.
        plan_step_id: ID of the plan step this call belongs to, if any.

    Returns:
        A `ToolResult`. On success, `data` holds `group_column`, `metric_column`,
        `group_count`, a `groups` list (one entry per group — `group`, `count`, `mean`,
        `median`, `std` — ordered by group value ascending), and `pairwise_differences`
        (one entry per unordered group pair — `group_a`, `group_b`, `mean_difference` —
        computed as `mean(group_b) - mean(group_a)`). On failure, `error` describes exactly
        what was rejected or what went wrong.
    """
    started = time.perf_counter()
    args = {
        "dataset_id": dataset_id,
        "source_path": source_path,
        "group_column": group_column,
        "metric_column": metric_column,
    }

    try:
        CompareGroupsArgs(
            dataset_id=dataset_id,
            source_path=source_path,
            group_column=group_column,
            metric_column=metric_column,
        )
    except ValidationError as exc:
        return _error_result(args, ToolErrorKind.VALIDATION, str(exc), started)

    try:
        df = load_dataset(source_path)
    except FileNotFoundError as exc:
        return _error_result(args, ToolErrorKind.NOT_FOUND, str(exc), started)
    except ValueError as exc:
        return _error_result(args, ToolErrorKind.EXECUTION, str(exc), started)

    for column in (group_column, metric_column):
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
            f"Column '{metric_column}' is not numeric; cannot compute mean/median/std.",
            started,
            details={"column": metric_column, "dtype": str(df[metric_column].dtype)},
        )

    working = df[[group_column, metric_column]].dropna(subset=[group_column])
    summary = working.groupby(group_column, observed=True)[metric_column].agg(
        ["count", "mean", "median", "std"]
    )

    if summary.empty:
        return _error_result(
            args,
            ToolErrorKind.INSUFFICIENT_DATA,
            f"No valid rows to compare after dropping missing '{group_column}' values.",
            started,
        )

    summary = summary.sort_index()
    groups = [
        {
            "group": _json_safe(group_value),
            "count": int(row["count"]),
            "mean": _json_safe(row["mean"]),
            "median": _json_safe(row["median"]),
            "std": _json_safe(row["std"]),
        }
        for group_value, row in summary.iterrows()
    ]

    group_values = [g["group"] for g in groups]
    means_by_group = {g["group"]: g["mean"] for g in groups}
    pairwise_differences = [
        {
            "group_a": a,
            "group_b": b,
            "mean_difference": (
                None
                if means_by_group[a] is None or means_by_group[b] is None
                else means_by_group[b] - means_by_group[a]
            ),
        }
        for a, b in itertools.combinations(group_values, 2)
    ]

    data = {
        "group_column": group_column,
        "metric_column": metric_column,
        "group_count": len(groups),
        "groups": groups,
        "pairwise_differences": pairwise_differences,
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
