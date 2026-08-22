"""Deterministic analytical tools: zero-LLM, typed I/O, evidence-producing.

No dependency on `adia.graph` or `adia.agents` — every tool here is a plain Python function
callable and testable on its own, per ADIA's core thesis that the LLM never computes a number
itself.
"""

from adia.tools.duckdb_client import build_connection, execute_query
from adia.tools.run_sql import RunSqlArgs, run_sql
from adia.tools.sql_guard import GuardedQuery, SqlGuardError, check_sql

__all__ = [
    "GuardedQuery",
    "RunSqlArgs",
    "SqlGuardError",
    "build_connection",
    "check_sql",
    "execute_query",
    "run_sql",
]
