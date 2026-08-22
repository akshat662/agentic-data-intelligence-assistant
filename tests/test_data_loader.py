"""Tests for adia.data.loader."""

import pandas as pd
import pytest

from adia.data.loader import load_dataset


@pytest.fixture
def sample_df() -> pd.DataFrame:
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


def test_load_parquet(tmp_path, sample_df):
    path = tmp_path / "sample.parquet"
    sample_df.to_parquet(path)
    loaded = load_dataset(path)
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_load_csv(tmp_path, sample_df):
    path = tmp_path / "sample.csv"
    sample_df.to_csv(path, index=False)
    loaded = load_dataset(path)
    pd.testing.assert_frame_equal(loaded, sample_df)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_dataset(tmp_path / "does_not_exist.parquet")


def test_unsupported_extension_raises(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("not a dataset")
    with pytest.raises(ValueError, match="Unsupported dataset file extension"):
        load_dataset(path)
