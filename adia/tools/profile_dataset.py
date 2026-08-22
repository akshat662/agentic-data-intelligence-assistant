"""Deterministic dataset profiling tool.

`profile_dataset` is the tool a planner is expected to call first on any non-trivial
question: before computing anything, know what the data actually looks like. It builds on
the same catalog machinery from Phase 1A (`adia.data.catalog.build_catalog`) and enriches it
with statistics a profiling *tool* call needs beyond the lightweight catalog used for column
cross-checking — categorical/boolean value frequencies and dataset-level memory usage —
without changing the `DatasetCatalog`/`ColumnProfile` contracts themselves.
"""

import time
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field, ValidationError

from adia.data.catalog import build_catalog
from adia.data.loader import load_dataset
from adia.evidence.ids import compute_args_hash, generate_evidence_id
from adia.evidence.store import EvidenceStore
from adia.models.catalog import ColumnProfile, DatasetCatalog, SemanticType
from adia.models.errors import ToolError, ToolErrorKind
from adia.models.evidence import Evidence
from adia.models.provenance import Provenance
from adia.models.tool_result import ToolResult

_TOOL_NAME = "profile_dataset"
_CATEGORICAL_LIKE_TYPES = {SemanticType.CATEGORICAL, SemanticType.BOOLEAN}


class ProfileDatasetArgs(BaseModel):
    """Validated input contract for `profile_dataset`."""

    dataset_id: str = Field(..., min_length=1, description="Stable identifier for the dataset.")
    source_path: str = Field(
        ..., min_length=1, description="Path to the dataset's Parquet/CSV file."
    )


def profile_dataset(
    dataset_id: str,
    source_path: str,
    evidence_store: EvidenceStore,
    *,
    plan_step_id: str | None = None,
    top_k: int = 5,
) -> ToolResult:
    """Profile a dataset: dataset-level shape plus per-column statistics.

    Loads the dataset, builds its `DatasetCatalog` via the existing data layer, and enriches
    it with dataset-level memory usage and per-column categorical/boolean value frequencies.
    The result is deterministic: profiling the same dataset with the same `top_k` twice
    produces identical output and an identical evidence ID.

    Args:
        dataset_id: Stable identifier for the dataset (used as its catalog ID).
        source_path: Path to a `.parquet` or `.csv` file.
        evidence_store: Store to record the resulting evidence in.
        plan_step_id: ID of the plan step this call belongs to, if any.
        top_k: Number of most frequent values to report for categorical/boolean columns.

    Returns:
        A `ToolResult`. On success, `data` holds dataset-level fields (`row_count`,
        `column_count`, `column_names`, `memory_bytes`) plus a `columns` list with one entry
        per column: name, dtype, semantic type, missing count/percentage, unique count,
        numeric min/max, datetime min/max, and — for categorical/boolean columns —
        `top_values`. On failure, `error` describes exactly what went wrong.
    """
    started = time.perf_counter()
    args = {"dataset_id": dataset_id, "source_path": source_path, "top_k": top_k}

    try:
        ProfileDatasetArgs(dataset_id=dataset_id, source_path=source_path)
    except ValidationError as exc:
        return _error_result(args, ToolErrorKind.VALIDATION, str(exc), started)

    try:
        df = load_dataset(source_path)
    except FileNotFoundError as exc:
        return _error_result(args, ToolErrorKind.NOT_FOUND, str(exc), started)
    except ValueError as exc:
        return _error_result(args, ToolErrorKind.EXECUTION, str(exc), started)

    catalog = build_catalog(df, dataset_id=dataset_id, source_path=source_path)
    profile_data = _build_profile_data(df, catalog, top_k=top_k)

    evidence_id = generate_evidence_id(_TOOL_NAME, args)
    provenance = Provenance(
        tool_name=_TOOL_NAME,
        args=args,
        args_hash=compute_args_hash(args),
        row_count=catalog.row_count,
        library_versions={"pandas": pd.__version__},
    )
    evidence = evidence_store.add(
        Evidence(
            id=evidence_id,
            tool=_TOOL_NAME,
            data=profile_data,
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


def _build_profile_data(df: pd.DataFrame, catalog: DatasetCatalog, *, top_k: int) -> dict[str, Any]:
    """Assemble the full dataset+column profile dict from a DataFrame and its catalog."""
    columns = [
        _column_profile_data(df[column.name], column, top_k=top_k) for column in catalog.columns
    ]
    return {
        "dataset_id": catalog.dataset_id,
        "row_count": catalog.row_count,
        "column_count": len(catalog.columns),
        "column_names": catalog.column_names(),
        "memory_bytes": int(df.memory_usage(deep=True).sum()),
        "columns": columns,
    }


def _column_profile_data(
    series: pd.Series, column: ColumnProfile, *, top_k: int
) -> dict[str, Any]:
    """Combine a `ColumnProfile` with categorical/boolean top-value statistics."""
    return {
        "name": column.name,
        "dtype": column.dtype,
        "semantic_type": column.semantic_type.value,
        "missing_count": column.null_count,
        "missing_percentage": column.null_rate * 100,
        "unique_count": column.unique_count,
        "min_value": column.min_value,
        "max_value": column.max_value,
        "min_date": column.min_date,
        "max_date": column.max_date,
        "top_values": _top_values(series, column.semantic_type, top_k=top_k),
    }


def _top_values(
    series: pd.Series, semantic_type: SemanticType, *, top_k: int
) -> list[dict[str, Any]] | None:
    """Most frequent values for a categorical/boolean column, or `None` for other types.

    Ties are broken by value (ascending) after count (descending), so the result is
    deterministic regardless of pandas' internal grouping order.
    """
    if semantic_type not in _CATEGORICAL_LIKE_TYPES:
        return None
    counts = series.dropna().value_counts()
    if counts.empty:
        return []
    ordered = counts.reset_index()
    ordered.columns = ["value", "count"]
    ordered = ordered.sort_values(by=["count", "value"], ascending=[False, True], kind="stable")
    return [
        {"value": _json_safe(row.value), "count": int(row.count)}
        for row in ordered.head(top_k).itertuples(index=False)
    ]


def _json_safe(value: Any) -> Any:
    """Convert a pandas/numpy scalar into a plain JSON-safe Python value.

    Checked before the native-type passthrough because `numpy.bool_`/`numpy.integer` are not
    subclasses of Python's `bool`/`int` — a plain `isinstance(value, bool | int | float)`
    check would silently miss them and fall through to `str()`, corrupting e.g. a boolean
    `True` into the string `"True"`.
    """
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, bool | str | int | float) or value is None:
        return value
    return str(value)


def _error_result(
    args: dict[str, Any], kind: ToolErrorKind, message: str, started: float
) -> ToolResult:
    """Build a typed `ToolResult` failure. The tool never lets an exception reach its caller."""
    return ToolResult(
        ok=False,
        tool=_TOOL_NAME,
        error=ToolError(kind=kind, message=message),
        duration_ms=_elapsed_ms(started),
    )


def _elapsed_ms(started: float) -> float:
    """Milliseconds elapsed since a `time.perf_counter()` reading."""
    return (time.perf_counter() - started) * 1000
