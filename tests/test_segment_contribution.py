"""Tests for adia.tools.segment_contribution."""

import pandas as pd
import pytest

from adia.evidence.store import EvidenceStore
from adia.models.errors import ToolErrorKind
from adia.tools.segment_contribution import segment_contribution


@pytest.fixture
def orders_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Category": [
                "Technology",
                "Technology",
                "Technology",
                "Furniture",
                "Furniture",
                "Technology",
            ],
            "Sub-Category": ["Phones", "Chairs", "Phones", "Chairs", "Tables", None],
            "Sales": [100.0, 50.0, 150.0, 80.0, 20.0, 999.0],
        }
    )


@pytest.fixture
def dataset_path(tmp_path, orders_df) -> str:
    path = tmp_path / "orders.parquet"
    orders_df.to_parquet(path)
    return str(path)


@pytest.fixture
def store() -> EvidenceStore:
    return EvidenceStore()


def _entity(result, name: str) -> dict:
    return next(e for e in result.data["entities"] if e["entity"] == name)


class TestParentScopedDecomposition:
    def test_scoped_totals_and_ranking(self, dataset_path, store):
        result = segment_contribution(
            "orders",
            dataset_path,
            "Sub-Category",
            "Sales",
            store,
            parent_column="Category",
            parent_value="Technology",
        )
        assert result.ok is True
        # Technology rows: Phones=100, Chairs=50, Phones=150 (the null Sub-Category row is
        # dropped) -- Phones totals 250, Chairs totals 50.
        assert result.data["overall_total"] == 300.0
        assert result.data["overall_count"] == 3
        assert result.data["entity_count"] == 2
        phones = _entity(result, "Phones")
        assert phones["rank"] == 1
        assert phones["count"] == 2
        assert phones["total"] == 250.0
        assert phones["mean"] == 125.0
        assert phones["share_of_total"] == pytest.approx(250.0 / 300.0)
        chairs = _entity(result, "Chairs")
        assert chairs["rank"] == 2
        assert chairs["share_of_total"] == pytest.approx(50.0 / 300.0)

    def test_furniture_scope_is_independent_of_technology(self, dataset_path, store):
        result = segment_contribution(
            "orders",
            dataset_path,
            "Sub-Category",
            "Sales",
            store,
            parent_column="Category",
            parent_value="Furniture",
        )
        assert result.data["overall_total"] == 100.0
        assert {e["entity"] for e in result.data["entities"]} == {"Chairs", "Tables"}

    def test_rows_with_missing_entity_are_excluded(self, dataset_path, store):
        result = segment_contribution(
            "orders",
            dataset_path,
            "Sub-Category",
            "Sales",
            store,
            parent_column="Category",
            parent_value="Technology",
        )
        # 999.0 belongs to the row with a null Sub-Category -- must not be counted anywhere.
        total_counted = sum(e["count"] for e in result.data["entities"])
        assert total_counted == 3

    def test_causal_claim_not_allowed(self, dataset_path, store):
        result = segment_contribution("orders", dataset_path, "Category", "Sales", store)
        assert result.data["causal_claim_allowed"] is False


class TestUnscopedDecomposition:
    def test_no_parent_ranks_across_whole_dataset(self, dataset_path, store):
        result = segment_contribution("orders", dataset_path, "Category", "Sales", store)
        assert result.ok is True
        tech = _entity(result, "Technology")
        furniture = _entity(result, "Furniture")
        assert tech["rank"] == 1
        assert furniture["rank"] == 2
        assert tech["total"] == pytest.approx(100.0 + 50.0 + 150.0 + 999.0)


class TestRanking:
    def test_ranks_are_contiguous_starting_at_one(self, dataset_path, store):
        result = segment_contribution(
            "orders",
            dataset_path,
            "Sub-Category",
            "Sales",
            store,
            parent_column="Category",
            parent_value="Technology",
        )
        ranks = sorted(e["rank"] for e in result.data["entities"])
        assert ranks == list(range(1, len(ranks) + 1))

    def test_entities_ordered_by_total_descending(self, dataset_path, store):
        result = segment_contribution("orders", dataset_path, "Category", "Sales", store)
        totals = [e["total"] for e in result.data["entities"]]
        assert totals == sorted(totals, reverse=True)

    def test_top_k_truncates_but_entity_count_reflects_full_scope(self, tmp_path, store):
        df = pd.DataFrame(
            {
                "group": ["a", "b", "c", "d"],
                "value": [40.0, 30.0, 20.0, 10.0],
            }
        )
        path = tmp_path / "many.parquet"
        df.to_parquet(path)
        result = segment_contribution("orders", str(path), "group", "value", store, top_k=2)
        assert result.data["entity_count"] == 4
        assert len(result.data["entities"]) == 2
        assert [e["entity"] for e in result.data["entities"]] == ["a", "b"]


class TestParentColumnValuePairing:
    def test_parent_column_without_parent_value_is_rejected(self, dataset_path, store):
        result = segment_contribution(
            "orders", dataset_path, "Sub-Category", "Sales", store, parent_column="Category"
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION

    def test_parent_value_without_parent_column_is_rejected(self, dataset_path, store):
        result = segment_contribution(
            "orders", dataset_path, "Sub-Category", "Sales", store, parent_value="Technology"
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION


class TestDatasetLoadErrors:
    def test_missing_dataset_file_returns_not_found_error(self, store):
        result = segment_contribution(
            "orders", "/no/such/file.parquet", "Category", "Sales", store
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.NOT_FOUND


class TestMissingColumns:
    def test_unknown_entity_column(self, dataset_path, store):
        result = segment_contribution("orders", dataset_path, "nonexistent", "Sales", store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION
        assert result.error.details["column"] == "nonexistent"

    def test_unknown_metric_column(self, dataset_path, store):
        result = segment_contribution("orders", dataset_path, "Category", "nonexistent", store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION

    def test_unknown_parent_column(self, dataset_path, store):
        result = segment_contribution(
            "orders",
            dataset_path,
            "Sub-Category",
            "Sales",
            store,
            parent_column="nonexistent",
            parent_value="x",
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION
        assert result.error.details["column"] == "nonexistent"


class TestInvalidDataTypes:
    def test_non_numeric_metric_column_rejected(self, dataset_path, store):
        result = segment_contribution("orders", dataset_path, "Sales", "Category", store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION
        assert result.error.details["column"] == "Category"


class TestInsufficientData:
    def test_parent_value_matching_no_rows_returns_insufficient_data(self, dataset_path, store):
        result = segment_contribution(
            "orders",
            dataset_path,
            "Sub-Category",
            "Sales",
            store,
            parent_column="Category",
            parent_value="Office Supplies",
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.INSUFFICIENT_DATA

    def test_all_missing_entity_values_returns_insufficient_data(self, tmp_path, store):
        df = pd.DataFrame({"group": [None, None], "value": [1.0, 2.0]})
        path = tmp_path / "empty.parquet"
        df.to_parquet(path)
        result = segment_contribution("orders", str(path), "group", "value", store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.INSUFFICIENT_DATA


class TestZeroTotal:
    def test_zero_overall_total_reports_none_shares_not_a_crash(self, tmp_path, store):
        df = pd.DataFrame({"group": ["a", "a", "b"], "value": [10.0, -10.0, 0.0]})
        path = tmp_path / "zero_total.parquet"
        df.to_parquet(path)
        result = segment_contribution("orders", str(path), "group", "value", store)
        assert result.ok is True
        assert result.data["overall_total"] == 0.0
        assert all(e["share_of_total"] is None for e in result.data["entities"])


class TestEvidenceGeneration:
    def test_evidence_id_and_store_populated(self, dataset_path, store):
        result = segment_contribution("orders", dataset_path, "Category", "Sales", store)
        assert result.evidence_id is not None
        stored = store.get(result.evidence_id)
        assert stored is not None
        assert stored.tool == "segment_contribution"
        assert stored.data == result.data

    def test_no_evidence_written_on_failure(self, dataset_path, store):
        segment_contribution("orders", dataset_path, "nonexistent", "Sales", store)
        assert len(store) == 0


class TestDeterminism:
    def test_repeated_call_produces_identical_evidence_id(self, dataset_path, store):
        result1 = segment_contribution("orders", dataset_path, "Category", "Sales", store)
        result2 = segment_contribution("orders", dataset_path, "Category", "Sales", store)
        assert result1.evidence_id == result2.evidence_id
        assert len(store) == 1

    def test_repeated_call_produces_identical_data(self, dataset_path, store):
        result1 = segment_contribution("orders", dataset_path, "Category", "Sales", store)
        result2 = segment_contribution(
            "orders", dataset_path, "Category", "Sales", EvidenceStore()
        )
        assert result1.data == result2.data
