"""Tests for adia.tools.compare_groups."""

import pandas as pd
import pytest

from adia.evidence.store import EvidenceStore
from adia.models.errors import ToolErrorKind
from adia.tools.compare_groups import compare_groups


@pytest.fixture
def orders_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "region": ["north", "south", "north", "east", "south", "north", None],
            "price": [10.0, 20.0, 30.0, None, 50.0, 15.0, 99.0],
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


def _group(result, name: str) -> dict:
    return next(g for g in result.data["groups"] if g["group"] == name)


class TestValidComparison:
    def test_group_stats_are_correct(self, dataset_path, store):
        result = compare_groups("orders", dataset_path, "region", "price", store)
        assert result.ok is True
        north = _group(result, "north")
        assert north["count"] == 3
        assert north["mean"] == pytest.approx(18.333333, rel=1e-4)
        assert north["median"] == 15.0
        assert north["std"] == pytest.approx(10.408330, rel=1e-4)

    def test_groups_ordered_ascending(self, dataset_path, store):
        result = compare_groups("orders", dataset_path, "region", "price", store)
        assert [g["group"] for g in result.data["groups"]] == ["east", "north", "south"]

    def test_group_with_no_valid_metric_reports_none_stats(self, dataset_path, store):
        result = compare_groups("orders", dataset_path, "region", "price", store)
        east = _group(result, "east")
        assert east["count"] == 0
        assert east["mean"] is None
        assert east["std"] is None

    def test_group_count(self, dataset_path, store):
        result = compare_groups("orders", dataset_path, "region", "price", store)
        assert result.data["group_count"] == 3

    def test_rows_with_missing_group_are_excluded(self, dataset_path, store):
        result = compare_groups("orders", dataset_path, "region", "price", store)
        total = sum(g["count"] for g in result.data["groups"])
        assert total == 5  # 6 non-null-region rows, one of which has a null price

    def test_single_group_has_no_pairwise_differences(self, tmp_path, store):
        df = pd.DataFrame({"region": ["north", "north"], "price": [10.0, 20.0]})
        path = tmp_path / "single.parquet"
        df.to_parquet(path)
        result = compare_groups("orders", str(path), "region", "price", store)
        assert result.data["pairwise_differences"] == []


class TestPairwiseDifferences:
    def test_mean_difference_direction(self, dataset_path, store):
        result = compare_groups("orders", dataset_path, "region", "price", store)
        north_south = next(
            d
            for d in result.data["pairwise_differences"]
            if d["group_a"] == "north" and d["group_b"] == "south"
        )
        # south mean (35.0) - north mean (18.333...)
        assert north_south["mean_difference"] == pytest.approx(16.666666, rel=1e-4)

    def test_pair_count_matches_combinations(self, dataset_path, store):
        result = compare_groups("orders", dataset_path, "region", "price", store)
        assert len(result.data["pairwise_differences"]) == 3  # C(3, 2)

    def test_pair_involving_empty_group_has_none_difference(self, dataset_path, store):
        result = compare_groups("orders", dataset_path, "region", "price", store)
        pair = next(
            d
            for d in result.data["pairwise_differences"]
            if "east" in (d["group_a"], d["group_b"])
        )
        assert pair["mean_difference"] is None


class TestGroupColumnTypes:
    def test_boolean_group_column(self, tmp_path, store):
        df = pd.DataFrame(
            {
                "is_returned": pd.array([True, True, False, False, False], dtype="boolean"),
                "price": [10.0, 20.0, 30.0, 40.0, 50.0],
            }
        )
        path = tmp_path / "bool_group.parquet"
        df.to_parquet(path)
        result = compare_groups("orders", str(path), "is_returned", "price", store)
        assert result.ok is True
        assert [g["group"] for g in result.data["groups"]] == [False, True]

    def test_numeric_group_column(self, tmp_path, store):
        df = pd.DataFrame({"tier": [3, 1, 1, 2], "price": [10.0, 20.0, 30.0, 40.0]})
        path = tmp_path / "numeric_group.parquet"
        df.to_parquet(path)
        result = compare_groups("orders", str(path), "tier", "price", store)
        assert result.ok is True
        assert [g["group"] for g in result.data["groups"]] == [1, 2, 3]

    def test_datetime_group_column(self, tmp_path, store):
        df = pd.DataFrame(
            {
                "signup_date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-02-01"]),
                "price": [10.0, 20.0, 30.0],
            }
        )
        path = tmp_path / "datetime_group.parquet"
        df.to_parquet(path)
        result = compare_groups("orders", str(path), "signup_date", "price", store)
        assert result.ok is True
        assert result.data["groups"][0]["group"] == "2024-01-01T00:00:00"


class TestDatasetLoadErrors:
    def test_missing_dataset_file_returns_not_found_error(self, store):
        result = compare_groups("orders", "/no/such/file.parquet", "region", "price", store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.NOT_FOUND

    def test_unsupported_dataset_extension_returns_execution_error(self, store, tmp_path):
        bad_path = tmp_path / "orders.txt"
        bad_path.write_text("not a real dataset")
        result = compare_groups("orders", str(bad_path), "region", "price", store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.EXECUTION


class TestMissingColumns:
    def test_empty_group_column_returns_validation_error(self, dataset_path, store):
        result = compare_groups("orders", dataset_path, "", "price", store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION

    def test_unknown_group_column(self, dataset_path, store):
        result = compare_groups("orders", dataset_path, "nonexistent", "price", store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION
        assert result.error.details["column"] == "nonexistent"

    def test_unknown_metric_column(self, dataset_path, store):
        result = compare_groups("orders", dataset_path, "region", "nonexistent", store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION


class TestInvalidDataTypes:
    def test_non_numeric_metric_column_rejected(self, dataset_path, store):
        result = compare_groups("orders", dataset_path, "price", "region", store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION
        assert result.error.details["column"] == "region"


class TestInsufficientData:
    def test_all_missing_group_values_returns_insufficient_data(self, tmp_path, store):
        df = pd.DataFrame({"region": [None, None], "price": [1.0, 2.0]})
        path = tmp_path / "empty.parquet"
        df.to_parquet(path)
        result = compare_groups("orders", str(path), "region", "price", store)
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.INSUFFICIENT_DATA


class TestEvidenceGeneration:
    def test_evidence_id_and_store_populated(self, dataset_path, store):
        result = compare_groups("orders", dataset_path, "region", "price", store)
        assert result.evidence_id is not None
        stored = store.get(result.evidence_id)
        assert stored is not None
        assert stored.tool == "compare_groups"
        assert stored.data == result.data

    def test_no_evidence_written_on_failure(self, dataset_path, store):
        compare_groups("orders", dataset_path, "nonexistent", "price", store)
        assert len(store) == 0


class TestDeterminism:
    def test_repeated_call_produces_identical_evidence_id(self, dataset_path, store):
        result1 = compare_groups("orders", dataset_path, "region", "price", store)
        result2 = compare_groups("orders", dataset_path, "region", "price", store)
        assert result1.evidence_id == result2.evidence_id
        assert len(store) == 1

    def test_repeated_call_produces_identical_data(self, dataset_path, store):
        result1 = compare_groups("orders", dataset_path, "region", "price", store)
        result2 = compare_groups("orders", dataset_path, "region", "price", EvidenceStore())
        assert result1.data == result2.data
