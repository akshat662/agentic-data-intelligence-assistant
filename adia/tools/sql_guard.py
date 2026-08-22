"""SQL safety guard: parses, validates, and rewrites a query before execution.

Only a single, read-only `SELECT` (optionally with CTEs) may pass through this guard. Every
other statement type — `INSERT`, `UPDATE`, `DELETE`, `DROP`, `ALTER`, `CREATE`, and anything
else sqlglot recognizes as non-`SELECT` (`ATTACH`, `COPY`, `PRAGMA`, multi-statement batches,
set operations like `UNION`) — is rejected before it ever reaches DuckDB. Table references are
checked against a single allowed table plus the query's own CTEs; column references are
checked against the dataset catalog.

Known limitation: column validation accepts any name that is either a real catalog column or
an alias defined by a `SELECT` projection anywhere in the query (so `ORDER BY total_price`
after `SUM(price) AS total_price` is accepted, as is a CTE's own output alias). This means a
column name that happens to collide with an unrelated alias elsewhere in the same query would
not be caught — an accepted tradeoff for this phase's scope, not a soundness guarantee.
"""

from dataclasses import dataclass
from typing import Any

import sqlglot
from sqlglot import exp

from adia.models.catalog import DatasetCatalog

_DISALLOWED_STATEMENT_NAMES: dict[type[exp.Expression], str] = {
    exp.Insert: "INSERT",
    exp.Update: "UPDATE",
    exp.Delete: "DELETE",
    exp.Drop: "DROP",
    exp.Alter: "ALTER",
    exp.Create: "CREATE",
}


class SqlGuardError(Exception):
    """Raised when a query fails the SQL safety guard.

    `details` carries structured context (e.g. the offending table or column name) so a
    caller can build a typed `ToolError` without re-parsing the message string.
    """

    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


@dataclass(frozen=True)
class GuardedQuery:
    """A query that has passed the guard, rewritten to a safe, executable form."""

    sql: str
    limit_injected: bool


def check_sql(
    query: str,
    *,
    catalog: DatasetCatalog,
    table_name: str,
    default_limit: int = 10_000,
) -> GuardedQuery:
    """Validate and rewrite a user-supplied SQL query before execution.

    Args:
        query: The raw SQL text to validate.
        catalog: The dataset catalog whose columns define the allowed column set.
        table_name: The single table name this query is permitted to reference.
        default_limit: Row limit injected when the query has none.

    Returns:
        A `GuardedQuery` wrapping the safe, dialect-normalized SQL to execute.

    Raises:
        SqlGuardError: If the query is not a single read-only `SELECT`, references a table
            other than `table_name` (or one of its own CTEs), or references a column not
            present in `catalog` and not defined as an alias within the query itself.
    """
    statements = _parse(query)
    if len(statements) != 1:
        raise SqlGuardError(
            "Only a single SQL statement is permitted per query.",
            details={"statement_count": len(statements)},
        )
    statement = statements[0]
    if statement is None:
        raise SqlGuardError("Query could not be parsed.")

    _check_is_select(statement)
    _check_tables(statement, table_name=table_name)
    _check_columns(statement, catalog=catalog)

    limit_injected = statement.args.get("limit") is None
    if limit_injected:
        statement = statement.limit(default_limit)

    return GuardedQuery(sql=statement.sql(dialect="duckdb"), limit_injected=limit_injected)


def _parse(query: str) -> list[exp.Expression | None]:
    """Parse `query` into statements, converting parser failures into `SqlGuardError`."""
    try:
        return sqlglot.parse(query, read="duckdb")
    except sqlglot.errors.SqlglotError as exc:
        raise SqlGuardError(f"Query could not be parsed: {exc}") from exc


def _check_is_select(statement: exp.Expression) -> None:
    """Reject anything that is not a plain `SELECT` (with optional CTEs)."""
    if isinstance(statement, exp.Select):
        return
    kind = _DISALLOWED_STATEMENT_NAMES.get(type(statement), type(statement).__name__.upper())
    raise SqlGuardError(
        f"{kind} statements are not permitted; only read-only SELECT queries are allowed.",
        details={"statement_type": kind},
    )


def _check_tables(statement: exp.Expression, *, table_name: str) -> None:
    """Reject any table reference other than `table_name` or the query's own CTEs."""
    cte_names = {cte.alias_or_name.lower() for cte in statement.find_all(exp.CTE)}
    allowed = {table_name.lower()} | cte_names
    for table in statement.find_all(exp.Table):
        if table.name.lower() not in allowed:
            raise SqlGuardError(
                f"Unknown table '{table.name}'. Only '{table_name}' is queryable.",
                details={"table": table.name},
            )


def _check_columns(statement: exp.Expression, *, catalog: DatasetCatalog) -> None:
    """Reject any column reference not in the catalog and not a query-defined alias."""
    known = {c.lower() for c in catalog.column_names()}
    for select in statement.find_all(exp.Select):
        for projection in select.selects:
            if projection.alias:
                known.add(projection.alias.lower())

    for column in statement.find_all(exp.Column):
        if column.name.lower() not in known:
            raise SqlGuardError(
                f"Unknown column '{column.name}'.",
                details={"column": column.name},
            )
