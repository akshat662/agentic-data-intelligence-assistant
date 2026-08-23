"""Tests for adia.data.registry and adia.models.dataset."""

import json

import pytest
from pydantic import ValidationError

from adia.data.registry import get_dataset_config, load_registry, save_registry
from adia.models.dataset import DatasetConfig


def _config(dataset_id: str = "orders", **overrides) -> DatasetConfig:
    defaults = dict(
        dataset_id=dataset_id,
        file_path=f"data/processed/{dataset_id}.parquet",
        description="Test dataset.",
    )
    defaults.update(overrides)
    return DatasetConfig(**defaults)


class TestDatasetConfig:
    def test_target_columns_optional(self):
        config = _config()
        assert config.target_columns is None

    def test_target_columns_can_be_set(self):
        config = _config(target_columns=["is_returned"])
        assert config.target_columns == ["is_returned"]

    def test_missing_description_rejected(self):
        with pytest.raises(ValidationError):
            DatasetConfig(dataset_id="orders", file_path="orders.parquet")

    def test_does_not_require_file_to_exist(self):
        # Registration is purely declarative -- an unregistered/nonexistent path is valid.
        config = _config(file_path="/no/such/file.parquet")
        assert config.file_path == "/no/such/file.parquet"


class TestLoadRegistry:
    def test_round_trip(self, tmp_path):
        configs = [_config("orders"), _config("churn", description="Telco churn dataset.")]
        path = tmp_path / "registry.json"
        save_registry(configs, path)

        registry = load_registry(path)
        assert set(registry) == {"orders", "churn"}
        assert registry["orders"] == configs[0]
        assert registry["churn"] == configs[1]

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_registry(tmp_path / "does_not_exist.json")

    def test_duplicate_dataset_id_rejected(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(
            json.dumps(
                [
                    {"dataset_id": "orders", "file_path": "a.parquet", "description": "A"},
                    {"dataset_id": "orders", "file_path": "b.parquet", "description": "B"},
                ]
            )
        )
        with pytest.raises(ValueError, match="Duplicate dataset_id"):
            load_registry(path)

    def test_malformed_entry_raises_validation_error(self, tmp_path):
        path = tmp_path / "registry.json"
        path.write_text(json.dumps([{"dataset_id": "orders"}]))  # missing required fields
        with pytest.raises(ValidationError):
            load_registry(path)

    def test_empty_registry_is_valid(self, tmp_path):
        path = tmp_path / "registry.json"
        save_registry([], path)
        assert load_registry(path) == {}


class TestGetDatasetConfig:
    def test_returns_matching_config(self):
        config = _config("orders")
        registry = {"orders": config}
        assert get_dataset_config(registry, "orders") == config

    def test_unregistered_dataset_raises_keyerror(self):
        with pytest.raises(KeyError, match="not registered"):
            get_dataset_config({}, "nonexistent")
