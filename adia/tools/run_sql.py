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

#: Row cap for `_format_rows_preview` -- generous enough to answer a typical aggregation/
#: lookup question, bounded so a wide result set can't blow up the Synthesizer's prompt.
_PREVIEW_ROW_LIMIT = 50


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
        A `ToolResult`. On success, `data` holds `{"rows", "row_count", "columns", "rows_preview"}`
        -- `rows_preview` is a Markdown-table rendering of `rows` (see `_format_rows_preview`),
        capped at `_PREVIEW_ROW_LIMIT` rows, kept alongside the full `rows` so the Synthesizer
        can actually see fetched values even when there are too many rows for the evidence
        renderer's generic list-summarization to expand inline -- and `evidence_id` names the
        written `Evidence` record. On failure, `error` describes exactly what was rejected or
        what went wrong, and no evidence is written.
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
        "rows_preview": _format_rows_preview(rows),
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


def _format_rows_preview(rows: list[dict[str, Any]], *, limit: int = _PREVIEW_ROW_LIMIT) -> str:
    """Render fetched rows as a Markdown table, capped at `limit` rows.

    `data["rows"]` alone is not enough for the Synthesizer to see a query's actual result:
    `adia.evidence.renderer`'s generic list-summarization deliberately collapses any list over
    10 items down to a bare count (by design, so one tool's huge result doesn't blow up every
    other evidence record's rendered size) -- which means a `run_sql` call returning more than
    10 rows (e.g. one row per country, one per category) would otherwise leave the Synthesizer
    with only a row count and no actual fetched values to cite, no matter how small the result
    genuinely is. A plain string is a leaf value to that same renderer regardless of how many
    rows went into building it, so this preview survives the summarization untouched and always
    reaches the Synthesizer's prompt. `data["rows"]` itself is left as the full, unlimited,
    machine-readable result -- this is a human/LLM-readable rendering of it, not a replacement,
    and the grounding validator (`adia.validate.static.validate_answer`) checks claimed numbers
    against `data["rows"]` directly, not against this string.

    Args:
        rows: Query result rows, each a column-name-keyed dict (as returned by
            `adia.tools.duckdb_client.execute_query`).
        limit: Maximum number of rows to render before truncating.

    Returns:
        A Markdown table (header, separator, one line per row), with a trailing note if `rows`
        was truncated to `limit`. `"(no rows returned)"` if `rows` is empty.
    """
    if not rows:
        return "(no rows returned)"

    columns = list(rows[0].keys())
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(str(row.get(col, "")) for col in columns) + " |"
        for row in rows[:limit]
    ]
    table = "\n".join([header, separator, *body])

    if len(rows) > limit:
        table += f"\n... ({len(rows) - limit} more row(s) truncated)"
    return table


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
