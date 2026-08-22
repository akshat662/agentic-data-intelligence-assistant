"""Deterministic correlation analysis tool.

`compute_correlation` computes pairwise Pearson correlation between numeric columns of a
dataset and returns both the full correlation matrix and a deterministically ordered list of
per-pair statistics. The output is descriptive only: `causal_claim_allowed` is always
`False`, a flag a future validation layer can read to reject any causal wording built on top
of a bare correlation.
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

_TOOL_NAME = "compute_correlation"


class ComputeCorrelationArgs(BaseModel):
    """Validated input contract for `compute_correlation`."""

    dataset_id: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    columns: list[str] | None = Field(
        default=None,
        description="Numeric columns to correlate; defaults to every numeric column.",
    )


def compute_correlation(
    dataset_id: str,
    source_path: str,
    evidence_store: EvidenceStore,
    *,
    columns: list[str] | None = None,
    plan_step_id: str | None = None,
) -> ToolResult:
    """Compute pairwise Pearson correlation between numeric columns of a dataset.

    Args:
        dataset_id: Stable identifier for the dataset.
        source_path: Path to a `.parquet` or `.csv` file.
        evidence_store: Store to record the resulting evidence in.
        columns: Numeric columns to correlate. If omitted, every numeric column in the
            dataset is used, in ascending name order.
        plan_step_id: ID of the plan step this call belongs to, if any.

    Returns:
        A `ToolResult`. On success, `data` holds `columns` (the numeric columns used, in
        matrix order), `matrix` (a `columns`-by-`columns` correlation grid), `pairs` (one
        entry per unordered column pair — `column_a`, `column_b`, `correlation`, `n` — the
        number of rows with both values present), and `causal_claim_allowed: False`. On
        failure, `error` describes exactly what was rejected or what went wrong.
    """
    started = time.perf_counter()
    args = {"dataset_id": dataset_id, "source_path": source_path, "columns": columns}

    try:
        ComputeCorrelationArgs(dataset_id=dataset_id, source_path=source_path, columns=columns)
    except ValidationError as exc:
        return _error_result(args, ToolErrorKind.VALIDATION, str(exc), started)

    try:
        df = load_dataset(source_path)
    except FileNotFoundError as exc:
        return _error_result(args, ToolErrorKind.NOT_FOUND, str(exc), started)
    except ValueError as exc:
        return _error_result(args, ToolErrorKind.EXECUTION, str(exc), started)

    if columns is not None:
        unknown = [c for c in columns if c not in df.columns]
        if unknown:
            return _error_result(
                args,
                ToolErrorKind.VALIDATION,
                f"Unknown column(s): {', '.join(unknown)}.",
                started,
                details={"columns": unknown},
            )
        non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
        if non_numeric:
            return _error_result(
                args,
                ToolErrorKind.VALIDATION,
                f"Column(s) not numeric: {', '.join(non_numeric)}.",
                started,
                details={"columns": non_numeric},
            )
        selected = sorted(columns)
    else:
        selected = sorted(c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]))

    if len(selected) < 2:
        return _error_result(
            args,
            ToolErrorKind.INSUFFICIENT_DATA,
            f"At least 2 numeric columns are required to compute correlation; "
            f"found {len(selected)}.",
            started,
            details={"numeric_columns": selected},
        )

    numeric_df = df[selected]
    corr = numeric_df.corr(method="pearson")

    matrix = [[_json_safe(corr.loc[a, b]) for b in selected] for a in selected]
    pairs = [
        {
            "column_a": a,
            "column_b": b,
            "correlation": _json_safe(corr.loc[a, b]),
            "n": int(numeric_df[[a, b]].dropna().shape[0]),
        }
        for a, b in itertools.combinations(selected, 2)
    ]

    data = {
        "columns": selected,
        "matrix": matrix,
        "pairs": pairs,
        "causal_claim_allowed": False,
    }

    evidence_id = generate_evidence_id(_TOOL_NAME, args)
    provenance = Provenance(
        tool_name=_TOOL_NAME,
        args=args,
        args_hash=compute_args_hash(args),
        row_count=len(df),
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
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if pd.isna(value) else float(value)
    if isinstance(value, float):
        return None if pd.isna(value) else value
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
