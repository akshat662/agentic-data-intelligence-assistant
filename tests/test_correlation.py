"""Tests for adia.tools.correlation."""

import pandas as pd
import pytest

from adia.evidence.store import EvidenceStore
from adia.models.errors import ToolErrorKind
from adia.tools.correlation import compute_correlation


@pytest.fixture
def orders_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [10.0, 20.0, 30.0, 40.0, 50.0],
            "quantity": [1, 2, 3, 4, 5],
            "discount": [5.0, 4.0, 3.0, 2.0, 1.0],
            "region": ["north", "south", "east", "west", "north"],
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


def _pair(result, a: str, b: str) -> dict:
    return next(
        p
        for p in result.data["pairs"]
        if {p["column_a"], p["column_b"]} == {a, b}
    )


class TestCorrelationOutput:
    def test_defaults_to_all_numeric_columns_sorted(self, dataset_path, store):
        result = compute_correlation("orders", dataset_path, store)
        assert result.ok is True
        assert result.data["columns"] == ["discount", "price", "quantity"]

    def test_perfect_positive_correlation(self, dataset_path, store):
        result = compute_correlation("orders", dataset_path, store)
        pair = _pair(result, "price", "quantity")
        assert pair["correlation"] == pytest.approx(1.0)
        assert pair["n"] == 5

    def test_perfect_negative_correlation(self, dataset_path, store):
        result = compute_correlation("orders", dataset_path, store)
        pair = _pair(result, "price", "discount")
        assert pair["correlation"] == pytest.approx(-1.0)

    def test_causal_claim_allowed_is_always_false(self, dataset_path, store):
        result = compute_correlation("orders", dataset_path, store)
        assert result.data["causal_claim_allowed"] is False

    def test_matrix_shape_matches_columns(self, dataset_path, store):
        result = compute_correlation("orders", dataset_path, store)
        matrix = result.data["matrix"]
        assert len(matrix) == len(result.data["columns"])
        assert all(len(row) == len(result.data["columns"]) for row in matrix)

    def test_matrix_diagonal_is_self_correlation(self, dataset_path, store):
        result = compute_correlation("orders", dataset_path, store)
        columns = result.data["columns"]
        matrix = result.data["matrix"]
        for i, col in enumerate(columns):
            assert matrix[i][i] == pytest.approx(1.0), col

    def test_explicit_columns_are_used(self, dataset_path, store):
        result = compute_correlation(
            "orders", dataset_path, store, columns=["price", "discount"]
        )
        assert result.data["columns"] == ["discount", "price"]

    def test_constant_column_correlation_is_none(self, tmp_path, store):
        df = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [5.0, 5.0, 5.0]})
        path = tmp_path / "constant.parquet"
        df.to_parquet(path)
        result = compute_correlation("orders", str(path), store)
        assert _pair(result, "a", "b")["correlation"] is None


class TestDatasetLoadErrors:
    def test_missing_dataset_file_returns_not_found_error(self, store):
        result = compute_correlation("orders", "/no/such/file.parquet", store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.NOT_FOUND

    def test_unsupported_dataset_extension_returns_execution_error(self, store, tmp_path):
        bad_path = tmp_path / "orders.txt"
        bad_path.write_text("not a real dataset")
        result = compute_correlation("orders", str(bad_path), store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.EXECUTION


class TestMissingAndInvalidColumns:
    def test_empty_dataset_id_returns_validation_error(self, dataset_path, store):
        result = compute_correlation("", dataset_path, store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION

    def test_unknown_column_rejected(self, dataset_path, store):
        result = compute_correlation("orders", dataset_path, store, columns=["price", "bogus"])
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION
        assert "bogus" in result.error.details["columns"]

    def test_non_numeric_column_rejected(self, dataset_path, store):
        result = compute_correlation("orders", dataset_path, store, columns=["price", "region"])
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION
        assert "region" in result.error.details["columns"]

    def test_fewer_than_two_numeric_columns_is_insufficient_data(self, tmp_path, store):
        df = pd.DataFrame({"price": [1.0, 2.0, 3.0], "region": ["a", "b", "c"]})
        path = tmp_path / "one_numeric.parquet"
        df.to_parquet(path)
        result = compute_correlation("orders", str(path), store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.INSUFFICIENT_DATA


class TestEvidenceGeneration:
    def test_evidence_id_and_store_populated(self, dataset_path, store):
        result = compute_correlation("orders", dataset_path, store)
        assert result.evidence_id is not None
        stored = store.get(result.evidence_id)
        assert stored is not None
        assert stored.tool == "compute_correlation"
        assert stored.data == result.data

    def test_no_evidence_written_on_failure(self, dataset_path, store):
        compute_correlation("orders", dataset_path, store, columns=["price", "bogus"])
        assert len(store) == 0


class TestDeterminism:
    def test_repeated_call_produces_identical_evidence_id(self, dataset_path, store):
        result1 = compute_correlation("orders", dataset_path, store)
        result2 = compute_correlation("orders", dataset_path, store)
        assert result1.evidence_id == result2.evidence_id
        assert len(store) == 1

    def test_repeated_call_produces_identical_data(self, dataset_path, store):
        result1 = compute_correlation("orders", dataset_path, store)
        result2 = compute_correlation("orders", dataset_path, EvidenceStore())
        assert result1.data == result2.data

    def test_different_columns_produce_different_evidence_ids(self, dataset_path, store):
        result1 = compute_correlation("orders", dataset_path, store, columns=["price", "quantity"])
        result2 = compute_correlation("orders", dataset_path, store, columns=["price", "discount"])
        assert result1.evidence_id != result2.evidence_id
