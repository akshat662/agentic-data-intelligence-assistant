"""Tests for adia.tools.duckdb_client (value sanitization for evidence storage)."""

import pandas as pd

from adia.tools.duckdb_client import build_connection, execute_query


def test_timestamp_columns_are_isoformat_strings():
    df = pd.DataFrame({"order_date": pd.to_datetime(["2024-01-01", "2024-02-15"])})
    conn = build_connection("orders", df)
    rows = execute_query(conn, "SELECT order_date FROM orders")
    conn.close()
    assert rows == [{"order_date": "2024-01-01T00:00:00"}, {"order_date": "2024-02-15T00:00:00"}]


def test_null_numeric_values_become_none_not_nan():
    df = pd.DataFrame({"price": [10.0, None, 30.0]})
    conn = build_connection("orders", df)
    rows = execute_query(conn, "SELECT price FROM orders ORDER BY price NULLS FIRST")
    conn.close()
    assert rows[0]["price"] is None
    assert rows[1]["price"] == 10.0


def test_result_is_json_round_trippable():
    import json

    df = pd.DataFrame({"price": [10.0, None], "order_date": pd.to_datetime(["2024-01-01", None])})
    conn = build_connection("orders", df)
    rows = execute_query(conn, "SELECT price, order_date FROM orders")
    conn.close()
    assert json.loads(json.dumps(rows)) == rows
