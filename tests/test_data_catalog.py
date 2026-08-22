"""Tests for adia.data.catalog (column profiling and catalog generation)."""

import pandas as pd
import pytest

from adia.data.catalog import build_catalog, load_catalog, profile_column, save_catalog
from adia.models.catalog import SemanticType


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price": [10.0, 20.0, None, 40.0],
            "region": ["north", "south", "north", "east"],
            "order_date": pd.to_datetime(["2024-01-01", "2024-02-15", "2024-01-20", "2024-03-01"]),
            "is_returned": [True, False, False, False],
        }
    )


class TestProfileColumn:
    def test_numeric_column(self, sample_df):
        profile = profile_column(sample_df["price"])
        assert profile.semantic_type == SemanticType.NUMERIC
        assert profile.non_null_count == 3
        assert profile.null_count == 1
        assert profile.null_rate == pytest.approx(0.25)
        assert profile.min_value == 10.0
        assert profile.max_value == 40.0

    def test_categorical_column(self, sample_df):
        profile = profile_column(sample_df["region"])
        assert profile.semantic_type == SemanticType.CATEGORICAL
        assert profile.unique_count == 3
        assert profile.min_value is None
        assert profile.max_value is None

    def test_datetime_column(self, sample_df):
        profile = profile_column(sample_df["order_date"])
        assert profile.semantic_type == SemanticType.DATETIME
        assert profile.min_date is not None
        assert profile.max_date is not None
        assert profile.min_date < profile.max_date

    def test_boolean_column(self, sample_df):
        profile = profile_column(sample_df["is_returned"])
        assert profile.semantic_type == SemanticType.BOOLEAN
        assert profile.unique_count == 2

    def test_all_null_column_has_no_range(self):
        series = pd.Series([None, None, None], name="empty", dtype="float64")
        profile = profile_column(series)
        assert profile.non_null_count == 0
        assert profile.min_value is None
        assert profile.max_value is None


class TestBuildCatalog:
    def test_builds_one_profile_per_column(self, sample_df):
        catalog = build_catalog(sample_df, dataset_id="orders", source_path="orders.parquet")
        assert catalog.dataset_id == "orders"
        assert catalog.row_count == 4
        assert catalog.column_names() == ["price", "region", "order_date", "is_returned"]
        assert len(catalog.columns) == 4

    def test_save_and_load_round_trip(self, sample_df, tmp_path):
        catalog = build_catalog(sample_df, dataset_id="orders", source_path="orders.parquet")
        path = tmp_path / "orders_catalog.json"
        save_catalog(catalog, path)
        restored = load_catalog(path)
        assert restored == catalog
