"""Tests for adia.tools.run_sql."""

import pandas as pd
import pytest

from adia.data.catalog import build_catalog
from adia.evidence.renderer import render_evidence
from adia.evidence.store import EvidenceStore
from adia.models.errors import ToolErrorKind
from adia.tools.run_sql import run_sql


@pytest.fixture
def orders_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
            "region": ["north", "south", "north", "east", "south"],
            "order_date": pd.to_datetime(
                ["2024-01-01", "2024-02-15", "2024-01-20", "2024-03-01", "2024-03-15"]
            ),
        }
    )


@pytest.fixture
def catalog(tmp_path, orders_df):
    path = tmp_path / "orders.parquet"
    orders_df.to_parquet(path)
    return build_catalog(orders_df, dataset_id="orders", source_path=str(path))


@pytest.fixture
def wide_orders_df() -> pd.DataFrame:
    # 55 rows -- past the evidence renderer's small-list expansion threshold (10) and past
    # run_sql's own preview row limit (50), to exercise both boundaries in one fixture.
    return pd.DataFrame(
        {
            "region": [f"region_{i}" for i in range(55)],
            "price": [float(i) for i in range(55)],
        }
    )


@pytest.fixture
def wide_catalog(tmp_path, wide_orders_df):
    path = tmp_path / "wide_orders.parquet"
    wide_orders_df.to_parquet(path)
    return build_catalog(wide_orders_df, dataset_id="orders", source_path=str(path))


@pytest.fixture
def store() -> EvidenceStore:
    return EvidenceStore()


class TestValidQuery:
    def test_select_returns_ok_result_with_correct_rows(self, catalog, store):
        result = run_sql(
            "SELECT region, price FROM orders WHERE region = 'north'", catalog, store
        )
        assert result.ok is True
        assert result.data["row_count"] == 2
        assert {r["region"] for r in result.data["rows"]} == {"north"}

    def test_aggregate_query_returns_correct_values(self, catalog, store):
        result = run_sql(
            "SELECT region, SUM(price) AS total_price FROM orders GROUP BY region "
            "ORDER BY total_price DESC",
            catalog,
            store,
        )
        assert result.ok is True
        totals = {row["region"]: row["total_price"] for row in result.data["rows"]}
        assert totals == {"north": 40.0, "south": 70.0, "east": 40.0}

    def test_evidence_id_and_evidence_store_are_populated(self, catalog, store):
        result = run_sql("SELECT price FROM orders", catalog, store)
        assert result.evidence_id is not None
        stored = store.get(result.evidence_id)
        assert stored is not None
        assert stored.tool == "run_sql"

    def test_no_limit_query_gets_limit_warning(self, catalog, store):
        result = run_sql("SELECT price FROM orders", catalog, store)
        assert any("LIMIT" in w for w in result.warnings)

    def test_query_with_explicit_limit_has_no_warning(self, catalog, store):
        result = run_sql("SELECT price FROM orders LIMIT 2", catalog, store)
        assert result.warnings == []


class TestInputAndDataErrors:
    def test_empty_query_returns_validation_error(self, catalog, store):
        result = run_sql("", catalog, store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION

    def test_missing_dataset_file_returns_not_found_error(self, catalog, store):
        missing = catalog.model_copy(update={"source_path": "/no/such/file.parquet"})
        result = run_sql("SELECT price FROM orders", missing, store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.NOT_FOUND

    def test_unsupported_dataset_extension_returns_execution_error(self, catalog, store, tmp_path):
        bad_path = tmp_path / "orders.txt"
        bad_path.write_text("not a real dataset")
        misconfigured = catalog.model_copy(update={"source_path": str(bad_path)})
        result = run_sql("SELECT price FROM orders", misconfigured, store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.EXECUTION

    def test_duckdb_execution_failure_is_reported_as_execution_error(self, catalog, store):
        result = run_sql("SELECT CAST(region AS INTEGER) FROM orders", catalog, store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.EXECUTION


class TestBlockedQueries:
    def test_drop_query_returns_guard_rejected_error(self, catalog, store):
        result = run_sql("DROP TABLE orders", catalog, store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.GUARD_REJECTED

    def test_no_evidence_written_on_failure(self, catalog, store):
        run_sql("DROP TABLE orders", catalog, store)
        assert len(store) == 0


class TestInvalidColumns:
    def test_unknown_column_returns_guard_rejected_error(self, catalog, store):
        result = run_sql("SELECT nonexistent FROM orders", catalog, store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.GUARD_REJECTED
        assert result.error.details.get("column") == "nonexistent"


class TestRowsPreview:
    """`rows_preview` -- a Markdown-table string -- must survive into the evidence's rendered
    summary even when there are more rows than the evidence renderer's generic list-
    summarization (`adia.evidence.renderer`) would otherwise expand inline (>10 items), since
    that's the only way the Synthesizer actually sees fetched values for such a result.
    """

    def test_small_result_preview_contains_header_and_values(self, catalog, store):
        result = run_sql(
            "SELECT region, price FROM orders WHERE region = 'north'", catalog, store
        )
        preview = result.data["rows_preview"]
        assert "region" in preview and "price" in preview
        assert "north" in preview

    def test_preview_present_in_rendered_evidence_summary_beyond_ten_rows(
        self, wide_catalog, store
    ):
        # 55 rows -- well past the renderer's small-list expansion threshold (10), the exact
        # shape that silently dropped every fetched value before this fix.
        result = run_sql("SELECT region, price FROM orders", wide_catalog, store)
        assert result.data["row_count"] == 55
        stored = store.get(result.evidence_id)
        rendered = render_evidence(stored)
        # The generic renderer still only reports a count for the raw `rows` list...
        assert rendered.key_values["rows_count"] == 55
        assert "rows[0].region" not in rendered.key_values
        # ...but the preview string is a leaf value, so it survives untouched and carries the
        # actual fetched values into what the Synthesizer is shown.
        assert rendered.key_values["rows_preview"] == result.data["rows_preview"]
        assert "region_0" in rendered.summary

    def test_preview_truncated_and_notes_remaining_rows(self, wide_catalog, store):
        result = run_sql("SELECT region, price FROM orders", wide_catalog, store)
        assert result.data["row_count"] == 55
        assert "5 more row" in result.data["rows_preview"]
        # header + sep + 50 rows joined by "\n" (51 newlines) + 1 for the truncation note line
        assert result.data["rows_preview"].count("\n") == 52
        assert "region_49" in result.data["rows_preview"]
        assert "region_50" not in result.data["rows_preview"]

    def test_empty_result_preview_says_no_rows(self, catalog, store):
        result = run_sql("SELECT price FROM orders WHERE region = 'nowhere'", catalog, store)
        assert result.data["rows_preview"] == "(no rows returned)"


class TestDeterminism:
    def test_repeated_call_produces_identical_evidence_id(self, catalog, store):
        result1 = run_sql("SELECT price FROM orders WHERE region = 'north'", catalog, store)
        result2 = run_sql("SELECT price FROM orders WHERE region = 'north'", catalog, store)
        assert result1.evidence_id == result2.evidence_id
        assert len(store) == 1

    def test_repeated_call_produces_identical_data(self, catalog, store):
        result1 = run_sql("SELECT price FROM orders", catalog, store)
        result2 = run_sql("SELECT price FROM orders", catalog, EvidenceStore())
        assert result1.data == result2.data

    def test_different_queries_produce_different_evidence_ids(self, catalog, store):
        result1 = run_sql("SELECT price FROM orders", catalog, store)
        result2 = run_sql("SELECT region FROM orders", catalog, store)
        assert result1.evidence_id != result2.evidence_id
