"""DuckDB connection helpers for querying an in-memory, pre-registered DataFrame.

Deliberately narrow: a query never references a filesystem path — the caller loads the
dataset once and registers it under a fixed table name, so the only thing user-supplied SQL
can name is that one in-memory relation. Combined with the SQL guard's statement-type and
table checks, this means arbitrary file/network access from within a query is not just
disallowed but structurally unreachable.
"""

from typing import Any

import duckdb
import pandas as pd


def build_connection(table_name: str, df: pd.DataFrame) -> duckdb.DuckDBPyConnection:
    """Create an in-memory DuckDB connection with `df` registered as `table_name`.

    Args:
        table_name: Name the dataset is queryable as.
        df: The dataset to register.

    Returns:
        A connection with exactly one relation registered.
    """
    conn = duckdb.connect(database=":memory:")
    conn.register(table_name, df)
    return conn


def execute_query(conn: duckdb.DuckDBPyConnection, sql: str) -> list[dict[str, Any]]:
    """Execute a query and return rows as JSON-serializable records.

    Args:
        conn: An open connection, as returned by `build_connection`.
        sql: The (already guarded) SQL to execute.

    Returns:
        One dict per result row, with `pandas.Timestamp` values converted to ISO 8601
        strings and any missing value (`NaN`, `NaT`, `None`, `pandas.NA`) converted to `None`
        so the result is safe to persist as `Evidence.data` and round-trip through JSON.
    """
    result_df = conn.execute(sql).fetchdf()
    return [
        {column: _sanitize_value(value) for column, value in row.items()}
        for row in result_df.to_dict(orient="records")
    ]


def _sanitize_value(value: Any) -> Any:
    """Convert a single cell value into a JSON-safe, full-precision Python value.

    Checked in this order because `pandas.isna` on a real `Timestamp` is always `False`
    (so that branch never fires for a real value), while `pandas.NaT` is *not* an instance
    of `Timestamp` and must fall through to the `isna` check to be caught as missing.
    """
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if pd.isna(value):
        return None
    return value
