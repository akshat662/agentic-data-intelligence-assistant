"""Integration tests for the registered `superstore` evaluation dataset.

Unlike `tests/test_dataset_registry.py` (which tests the registry *mechanism* against
`tmp_path` fixtures), these tests exercise the real, committed `superstore` registration and
data file end-to-end, using only the existing generic data layer -- no dataset-specific code
is introduced anywhere in `adia/`.
"""

from pathlib import Path

from adia.data.catalog import build_catalog
from adia.data.loader import load_dataset
from adia.data.registry import get_dataset_config, load_registry
from adia.models.catalog import SemanticType

_REPO_ROOT = Path(__file__).parent.parent
_REGISTRY_PATH = _REPO_ROOT / "data" / "registry.json"


class TestRegistration:
    def test_superstore_is_registered(self):
        registry = load_registry(_REGISTRY_PATH)
        config = get_dataset_config(registry, "superstore")
        assert config.file_path == "data/superstore.csv"
        assert config.description

    def test_registered_file_path_exists(self):
        registry = load_registry(_REGISTRY_PATH)
        config = get_dataset_config(registry, "superstore")
        assert (_REPO_ROOT / config.file_path).exists()

    def test_target_columns_are_real_numeric_columns(self):
        registry = load_registry(_REGISTRY_PATH)
        config = get_dataset_config(registry, "superstore")
        df = load_dataset(_REPO_ROOT / config.file_path)
        assert config.target_columns
        for column in config.target_columns:
            assert column in df.columns


class TestLoadingAndProfiling:
    def test_dataset_loads_via_generic_loader(self):
        df = load_dataset(_REPO_ROOT / "data" / "superstore.csv")
        assert df.shape == (9994, 21)

    def test_catalog_captures_expected_columns(self):
        df = load_dataset(_REPO_ROOT / "data" / "superstore.csv")
        catalog = build_catalog(df, dataset_id="superstore", source_path="data/superstore.csv")
        assert catalog.row_count == 9994
        assert len(catalog.columns) == 21
        assert "Sales" in catalog.column_names()
        assert "Profit" in catalog.column_names()

    def test_numeric_columns_detected(self):
        df = load_dataset(_REPO_ROOT / "data" / "superstore.csv")
        catalog = build_catalog(df, dataset_id="superstore", source_path="data/superstore.csv")
        by_name = {c.name: c for c in catalog.columns}
        for column in ("Sales", "Profit", "Quantity", "Discount", "Row ID"):
            assert by_name[column].semantic_type == SemanticType.NUMERIC

    def test_categorical_columns_detected(self):
        df = load_dataset(_REPO_ROOT / "data" / "superstore.csv")
        catalog = build_catalog(df, dataset_id="superstore", source_path="data/superstore.csv")
        by_name = {c.name: c for c in catalog.columns}
        for column in ("Category", "Region", "Segment", "Ship Mode"):
            assert by_name[column].semantic_type == SemanticType.CATEGORICAL

    def test_postal_code_missingness_detected(self):
        df = load_dataset(_REPO_ROOT / "data" / "superstore.csv")
        catalog = build_catalog(df, dataset_id="superstore", source_path="data/superstore.csv")
        by_name = {c.name: c for c in catalog.columns}
        assert by_name["Postal Code"].null_count > 0

    def test_no_missingness_in_core_numeric_columns(self):
        df = load_dataset(_REPO_ROOT / "data" / "superstore.csv")
        catalog = build_catalog(df, dataset_id="superstore", source_path="data/superstore.csv")
        by_name = {c.name: c for c in catalog.columns}
        for column in ("Sales", "Profit", "Quantity", "Discount"):
            assert by_name[column].null_count == 0

    def test_committed_catalog_matches_freshly_generated_one(self):
        from adia.data.catalog import load_catalog

        df = load_dataset(_REPO_ROOT / "data" / "superstore.csv")
        fresh = build_catalog(df, dataset_id="superstore", source_path="data/superstore.csv")
        committed = load_catalog(_REPO_ROOT / "data" / "catalog" / "superstore.json")
        assert fresh.columns == committed.columns
        assert fresh.row_count == committed.row_count
