"""Deterministic, guarded SQL execution tool.

`run_sql` is the only way user-authored SQL reaches data in this system. Every query passes
through `adia.tools.sql_guard.check_sql` first; only a validated, safe-to-execute query is
ever handed to DuckDB. The tool never raises into its caller — every outcome, success or
failure, comes back as a `ToolResult` — and every successful call writes a full-precision
`Evidence` record through the evidence layer built in Phase 1B-1.
"""

import time
from typing import Any

import duckdb
from pydantic import BaseModel, Field, ValidationError

from adia.data.loader import load_dataset
from adia.evidence.ids import compute_args_hash, generate_evidence_id
from adia.evidence.store import EvidenceStore
from adia.models.catalog import DatasetCatalog
from adia.models.errors import ToolError, ToolErrorKind
from adia.models.evidence import Evidence
from adia.models.provenance import Provenance
from adia.models.tool_result import ToolResult
from adia.tools.duckdb_client import build_connection, execute_query
from adia.tools.sql_guard import SqlGuardError, check_sql

_TOOL_NAME = "run_sql"


class RunSqlArgs(BaseModel):
    """Validated input contract for `run_sql`."""

    query: str = Field(..., min_length=1, description="A single read-only SELECT statement.")


def run_sql(
    query: str,
    catalog: DatasetCatalog,
    evidence_store: EvidenceStore,
    *,
    plan_step_id: str | None = None,
    default_limit: int = 10_000,
) -> ToolResult:
    """Validate, guard, and execute a read-only SQL query against one dataset.

    The dataset is loaded from `catalog.source_path` and registered as a single in-memory
    table named `catalog.dataset_id`; the query may reference only that table (and its own
    CTEs) and only columns present in `catalog`. On success, a full-precision `Evidence`
    record is written to `evidence_store`, keyed by a hash of `(tool, query, dataset_id)` so
    an identical call is a cache hit rather than a repeated execution.

    Args:
        query: Raw SQL text, expected to be a single `SELECT` statement.
        catalog: Catalog of the dataset to query.
        evidence_store: Store to record the resulting evidence in.
        plan_step_id: ID of the plan step this call belongs to, if any.
        default_limit: Row limit injected by the guard when the query has none.

    Returns:
        A `ToolResult`. On success, `data` holds `{"rows", "row_count", "columns"}` and
        `evidence_id` names the written `Evidence` record. On failure, `error` describes
        exactly what was rejected or what went wrong, and no evidence is written.
    """
    started = time.perf_counter()
    args = {"query": query, "dataset_id": catalog.dataset_id}

    try:
        RunSqlArgs(query=query)
    except ValidationError as exc:
        return _error_result(args, ToolErrorKind.VALIDATION, str(exc), started)

    try:
        guarded = check_sql(
            query, catalog=catalog, table_name=catalog.dataset_id, default_limit=default_limit
        )
    except SqlGuardError as exc:
        return _error_result(
            args, ToolErrorKind.GUARD_REJECTED, str(exc), started, details=exc.details
        )

    try:
        df = load_dataset(catalog.source_path)
    except FileNotFoundError as exc:
        return _error_result(args, ToolErrorKind.NOT_FOUND, str(exc), started)
    except ValueError as exc:
        return _error_result(args, ToolErrorKind.EXECUTION, str(exc), started)

    conn = build_connection(catalog.dataset_id, df)
    try:
        rows = execute_query(conn, guarded.sql)
    except duckdb.Error as exc:
        return _error_result(args, ToolErrorKind.EXECUTION, str(exc), started)
    finally:
        conn.close()

    evidence_id = generate_evidence_id(_TOOL_NAME, args)
    provenance = Provenance(
        tool_name=_TOOL_NAME,
        args=args,
        args_hash=compute_args_hash(args),
        source_query=guarded.sql,
        row_count=len(rows),
        library_versions={"duckdb": duckdb.__version__},
    )
    data = {
        "rows": rows,
        "row_count": len(rows),
        "columns": list(rows[0].keys()) if rows else [],
    }
    evidence = evidence_store.add(
        Evidence(
            id=evidence_id,
            tool=_TOOL_NAME,
            data=data,
            provenance=provenance,
            plan_step_id=plan_step_id,
        )
    )

    warnings = []
    if guarded.limit_injected:
        warnings.append(f"No LIMIT was specified; a limit of {default_limit} was applied.")

    return ToolResult(
        ok=True,
        tool=_TOOL_NAME,
        evidence_id=evidence.id,
        data=evidence.data,
        provenance=evidence.provenance,
        warnings=warnings,
        duration_ms=_elapsed_ms(started),
    )


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
