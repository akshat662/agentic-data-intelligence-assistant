"""Tests for adia.tools.profile_dataset."""

import pandas as pd
import pytest

from adia.evidence.store import EvidenceStore
from adia.models.errors import ToolErrorKind
from adia.tools.profile_dataset import profile_dataset


@pytest.fixture
def orders_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [10.0, 20.0, None, 40.0, 50.0],
            "region": ["north", "south", "north", "east", "south"],
            "order_date": pd.to_datetime(
                ["2024-01-01", "2024-02-15", "2024-01-20", "2024-03-01", None]
            ),
            "is_returned": pd.array([True, False, True, False, None], dtype="boolean"),
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


def _column(result, name: str) -> dict:
    return next(c for c in result.data["columns"] if c["name"] == name)


class TestDatasetLevel:
    def test_row_and_column_counts(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        assert result.ok is True
        assert result.data["row_count"] == 5
        assert result.data["column_count"] == 4
        assert result.data["column_names"] == ["price", "region", "order_date", "is_returned"]

    def test_memory_bytes_is_positive(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        assert result.data["memory_bytes"] > 0


class TestNumericColumns:
    def test_min_max_and_type(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        price = _column(result, "price")
        assert price["semantic_type"] == "numeric"
        assert price["min_value"] == 10.0
        assert price["max_value"] == 50.0

    def test_numeric_column_has_no_top_values(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        assert _column(result, "price")["top_values"] is None


class TestCategoricalColumns:
    def test_top_values_counts_and_order(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        region = _column(result, "region")
        assert region["semantic_type"] == "categorical"
        assert region["top_values"] == [
            {"value": "north", "count": 2},
            {"value": "south", "count": 2},
            {"value": "east", "count": 1},
        ]

    def test_top_k_limits_results(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store, top_k=1)
        assert len(_column(result, "region")["top_values"]) == 1

    def test_all_null_categorical_column_returns_empty_top_values(self, tmp_path, store):
        df = pd.DataFrame({"tag": pd.Series([None, None, None], dtype="object")})
        path = tmp_path / "tags.parquet"
        df.to_parquet(path)
        result = profile_dataset("tags", str(path), store)
        assert _column(result, "tag")["top_values"] == []


class TestMissingValues:
    def test_missing_count_and_percentage(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        price = _column(result, "price")
        assert price["missing_count"] == 1
        assert price["missing_percentage"] == pytest.approx(20.0)

    def test_no_missing_values_reports_zero(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        region = _column(result, "region")
        assert region["missing_count"] == 0
        assert region["missing_percentage"] == 0.0


class TestDatetimeColumns:
    def test_date_range_detected(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        order_date = _column(result, "order_date")
        assert order_date["semantic_type"] == "datetime"
        assert order_date["min_date"] is not None
        assert order_date["max_date"] is not None
        assert order_date["min_date"] < order_date["max_date"]

    def test_datetime_column_has_no_top_values(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        assert _column(result, "order_date")["top_values"] is None


class TestBooleanColumns:
    def test_semantic_type_and_top_values(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        is_returned = _column(result, "is_returned")
        assert is_returned["semantic_type"] == "boolean"
        assert is_returned["top_values"] == [
            {"value": False, "count": 2},
            {"value": True, "count": 2},
        ]

    def test_boolean_values_are_real_booleans_not_strings(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        values = [entry["value"] for entry in _column(result, "is_returned")["top_values"]]
        assert all(isinstance(v, bool) for v in values)

    def test_missing_boolean_counted(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        assert _column(result, "is_returned")["missing_count"] == 1


class TestEvidenceGeneration:
    def test_evidence_id_and_store_populated(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        assert result.evidence_id is not None
        stored = store.get(result.evidence_id)
        assert stored is not None
        assert stored.tool == "profile_dataset"
        assert stored.data == result.data

    def test_provenance_records_args(self, dataset_path, store):
        result = profile_dataset("orders", dataset_path, store)
        assert result.provenance.tool_name == "profile_dataset"
        assert result.provenance.args["dataset_id"] == "orders"


class TestDeterminism:
    def test_repeated_call_produces_identical_evidence_id(self, dataset_path, store):
        result1 = profile_dataset("orders", dataset_path, store)
        result2 = profile_dataset("orders", dataset_path, store)
        assert result1.evidence_id == result2.evidence_id
        assert len(store) == 1

    def test_repeated_call_produces_identical_data(self, dataset_path, store):
        result1 = profile_dataset("orders", dataset_path, store)
        result2 = profile_dataset("orders", dataset_path, EvidenceStore())
        assert result1.data == result2.data

    def test_different_top_k_produces_different_evidence_id(self, dataset_path, store):
        result1 = profile_dataset("orders", dataset_path, store, top_k=1)
        result2 = profile_dataset("orders", dataset_path, store, top_k=5)
        assert result1.evidence_id != result2.evidence_id
        assert len(store) == 2


class TestErrorHandling:
    def test_empty_dataset_id_returns_validation_error(self, dataset_path, store):
        result = profile_dataset("", dataset_path, store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION

    def test_missing_file_returns_not_found_error(self, store):
        result = profile_dataset("orders", "/no/such/file.parquet", store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.NOT_FOUND

    def test_unsupported_extension_returns_execution_error(self, store, tmp_path):
        bad_path = tmp_path / "orders.txt"
        bad_path.write_text("not a real dataset")
        result = profile_dataset("orders", str(bad_path), store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.EXECUTION

    def test_no_evidence_written_on_failure(self, store):
        profile_dataset("orders", "/no/such/file.parquet", store)
        assert len(store) == 0
