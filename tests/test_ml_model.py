"""Tests for adia.tools.ml_model."""

import pandas as pd
import pytest

from adia.evidence.store import EvidenceStore
from adia.models.errors import ToolErrorKind
from adia.tools.ml_model import train_model


@pytest.fixture
def classification_df() -> pd.DataFrame:
    rows = list(range(-30, 30))
    return pd.DataFrame(
        {
            "f1": [float(i) for i in rows],
            "f2": [float(i) * 0.3 for i in rows],
            "region": ["north" if i % 2 == 0 else "south" for i in rows],
            "label": ["yes" if i >= 0 else "no" for i in rows],
        }
    )


@pytest.fixture
def classification_path(tmp_path, classification_df) -> str:
    path = tmp_path / "classification.parquet"
    classification_df.to_parquet(path)
    return str(path)


@pytest.fixture
def regression_df() -> pd.DataFrame:
    rows = list(range(-30, 30))
    return pd.DataFrame(
        {
            "f1": [float(i) for i in rows],
            "f2": [float(i) * 0.3 for i in rows],
            "target": [float(i) * 2 - float(i) * 0.3 for i in rows],
        }
    )


@pytest.fixture
def regression_path(tmp_path, regression_df) -> str:
    path = tmp_path / "regression.parquet"
    regression_df.to_parquet(path)
    return str(path)


@pytest.fixture
def store() -> EvidenceStore:
    return EvidenceStore()


class TestClassificationTraining:
    @pytest.mark.parametrize("model_type", ["logistic_regression", "random_forest_classifier"])
    def test_trains_and_reports_metric(self, classification_path, store, model_type):
        result = train_model(
            "orders",
            classification_path,
            "label",
            ["f1", "f2"],
            "classification",
            model_type,
            store,
        )
        assert result.ok is True
        assert result.data["model_type"] == model_type
        assert result.data["task_type"] == "classification"
        assert result.data["metric_name"] == "accuracy"
        assert 0.0 <= result.data["metric_value"] <= 1.0

    def test_beats_baseline_on_separable_data(self, classification_path, store):
        result = train_model(
            "orders",
            classification_path,
            "label",
            ["f1", "f2"],
            "classification",
            "logistic_regression",
            store,
        )
        assert result.data["metric_value"] > result.data["baseline_metric_value"]

    def test_feature_importance_present_and_ranked(self, classification_path, store):
        result = train_model(
            "orders",
            classification_path,
            "label",
            ["f1", "f2"],
            "classification",
            "random_forest_classifier",
            store,
        )
        importance = result.data["feature_importance"]
        assert {entry["feature"] for entry in importance} == {"f1", "f2"}
        assert importance == sorted(importance, key=lambda entry: -entry["importance"])

    def test_class_labels_reported(self, classification_path, store):
        result = train_model(
            "orders",
            classification_path,
            "label",
            ["f1", "f2"],
            "classification",
            "logistic_regression",
            store,
        )
        assert result.data["metadata"]["class_labels"] == ["no", "yes"]

    def test_split_sizes_sum_to_rows_used(self, classification_path, store):
        result = train_model(
            "orders",
            classification_path,
            "label",
            ["f1", "f2"],
            "classification",
            "logistic_regression",
            store,
        )
        meta = result.data["metadata"]
        assert meta["n_train"] + meta["n_test"] == meta["n_rows_used"]


class TestRegressionTraining:
    @pytest.mark.parametrize("model_type", ["linear_regression", "random_forest_regressor"])
    def test_trains_and_reports_metric(self, regression_path, store, model_type):
        result = train_model(
            "orders", regression_path, "target", ["f1", "f2"], "regression", model_type, store
        )
        assert result.ok is True
        assert result.data["metric_name"] == "r2"

    def test_linear_regression_recovers_near_perfect_fit(self, regression_path, store):
        result = train_model(
            "orders",
            regression_path,
            "target",
            ["f1", "f2"],
            "regression",
            "linear_regression",
            store,
        )
        assert result.data["metric_value"] > 0.99

    def test_feature_importance_present_for_linear_model(self, regression_path, store):
        result = train_model(
            "orders",
            regression_path,
            "target",
            ["f1", "f2"],
            "regression",
            "linear_regression",
            store,
        )
        importance = result.data["feature_importance"]
        assert {entry["feature"] for entry in importance} == {"f1", "f2"}

    def test_class_labels_none_for_regression(self, regression_path, store):
        result = train_model(
            "orders",
            regression_path,
            "target",
            ["f1", "f2"],
            "regression",
            "linear_regression",
            store,
        )
        assert result.data["metadata"]["class_labels"] is None


class TestInvalidTargetColumn:
    def test_unknown_target_column(self, classification_path, store):
        result = train_model(
            "orders",
            classification_path,
            "nonexistent",
            ["f1", "f2"],
            "classification",
            "logistic_regression",
            store,
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION

    def test_target_column_as_feature_rejected(self, classification_path, store):
        result = train_model(
            "orders",
            classification_path,
            "label",
            ["f1", "label"],
            "classification",
            "logistic_regression",
            store,
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION

    def test_non_numeric_target_for_regression_rejected(self, classification_path, store):
        result = train_model(
            "orders",
            classification_path,
            "label",
            ["f1", "f2"],
            "regression",
            "linear_regression",
            store,
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION


class TestInvalidFeatureTypes:
    def test_non_numeric_feature_column_rejected(self, classification_path, store):
        result = train_model(
            "orders",
            classification_path,
            "label",
            ["f1", "region"],
            "classification",
            "logistic_regression",
            store,
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION
        assert result.error.details["columns"] == ["region"]


class TestTestSizeValidation:
    def test_test_size_out_of_range_rejected(self, classification_path, store):
        result = train_model(
            "orders",
            classification_path,
            "label",
            ["f1", "f2"],
            "classification",
            "logistic_regression",
            store,
            test_size=1.5,
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION

    def test_test_size_too_small_for_class_count_is_execution_error(self, tmp_path, store):
        # 10 "a" rows + 2 "b" rows passes the _MIN_ROWS and per-class-count checks, but a
        # 5% test split can't allocate a test-set member to every class -- sklearn itself
        # raises, and the tool must convert that into a typed EXECUTION error, not crash.
        df = pd.DataFrame(
            {
                "f1": list(range(12)),
                "f2": list(range(12)),
                "label": ["a"] * 10 + ["b"] * 2,
            }
        )
        path = tmp_path / "sparse_split.parquet"
        df.to_parquet(path)
        result = train_model(
            "orders",
            str(path),
            "label",
            ["f1", "f2"],
            "classification",
            "logistic_regression",
            store,
            test_size=0.05,
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.EXECUTION


class TestUnknownModelOrTaskType:
    def test_unknown_model_type_for_task_rejected(self, classification_path, store):
        result = train_model(
            "orders",
            classification_path,
            "label",
            ["f1", "f2"],
            "classification",
            "linear_regression",
            store,
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION

    def test_invalid_task_type_rejected(self, classification_path, store):
        result = train_model(
            "orders",
            classification_path,
            "label",
            ["f1", "f2"],
            "clustering",
            "logistic_regression",
            store,
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.VALIDATION


class TestDatasetLoadErrors:
    def test_missing_dataset_file(self, store):
        result = train_model(
            "orders",
            "/no/such/file.parquet",
            "label",
            ["f1", "f2"],
            "classification",
            "logistic_regression",
            store,
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.NOT_FOUND

    def test_unsupported_extension(self, store, tmp_path):
        bad_path = tmp_path / "data.txt"
        bad_path.write_text("not a dataset")
        result = train_model(
            "orders",
            str(bad_path),
            "label",
            ["f1", "f2"],
            "classification",
            "logistic_regression",
            store,
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.EXECUTION


class TestInsufficientData:
    def test_too_few_rows(self, tmp_path, store):
        df = pd.DataFrame({"f1": [1.0, 2.0], "f2": [3.0, 4.0], "label": ["a", "b"]})
        path = tmp_path / "tiny.parquet"
        df.to_parquet(path)
        result = train_model(
            "orders", str(path), "label", ["f1", "f2"], "classification",
            "logistic_regression", store,
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.INSUFFICIENT_DATA

    def test_single_class_rejected(self, tmp_path, store):
        df = pd.DataFrame({"f1": list(range(20)), "f2": list(range(20)), "label": ["a"] * 20})
        path = tmp_path / "single_class.parquet"
        df.to_parquet(path)
        result = train_model(
            "orders", str(path), "label", ["f1", "f2"], "classification",
            "logistic_regression", store,
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.INSUFFICIENT_DATA

    def test_sparse_class_rejected(self, tmp_path, store):
        df = pd.DataFrame(
            {"f1": list(range(20)), "f2": list(range(20)), "label": ["a"] * 19 + ["b"]}
        )
        path = tmp_path / "sparse_class.parquet"
        df.to_parquet(path)
        result = train_model(
            "orders", str(path), "label", ["f1", "f2"], "classification",
            "logistic_regression", store,
        )
        assert result.ok is False
        assert result.error.kind == ToolErrorKind.INSUFFICIENT_DATA


class TestEvidenceGeneration:
    def test_evidence_id_and_store_populated(self, classification_path, store):
        result = train_model(
            "orders",
            classification_path,
            "label",
            ["f1", "f2"],
            "classification",
            "logistic_regression",
            store,
        )
        assert result.evidence_id is not None
        stored = store.get(result.evidence_id)
        assert stored is not None
        assert stored.tool == "train_model"
        assert stored.data == result.data

    def test_provenance_seed_recorded(self, classification_path, store):
        result = train_model(
            "orders",
            classification_path,
            "label",
            ["f1", "f2"],
            "classification",
            "logistic_regression",
            store,
        )
        assert result.provenance.seed == 42

    def test_no_evidence_written_on_failure(self, classification_path, store):
        train_model(
            "orders",
            classification_path,
            "nonexistent",
            ["f1", "f2"],
            "classification",
            "logistic_regression",
            store,
        )
        assert len(store) == 0


class TestDeterminism:
    def test_repeated_call_produces_identical_evidence_id(self, classification_path, store):
        result1 = train_model(
            "orders", classification_path, "label", ["f1", "f2"], "classification",
            "random_forest_classifier", store,
        )
        result2 = train_model(
            "orders", classification_path, "label", ["f1", "f2"], "classification",
            "random_forest_classifier", store,
        )
        assert result1.evidence_id == result2.evidence_id
        assert len(store) == 1

    def test_repeated_call_produces_identical_data(self, classification_path, store):
        result1 = train_model(
            "orders", classification_path, "label", ["f1", "f2"], "classification",
            "random_forest_classifier", store,
        )
        result2 = train_model(
            "orders", classification_path, "label", ["f1", "f2"], "classification",
            "random_forest_classifier", EvidenceStore(),
        )
        assert result1.data == result2.data

    def test_different_model_type_produces_different_evidence_id(
        self, classification_path, store
    ):
        result1 = train_model(
            "orders", classification_path, "label", ["f1", "f2"], "classification",
            "logistic_regression", store,
        )
        result2 = train_model(
            "orders", classification_path, "label", ["f1", "f2"], "classification",
            "random_forest_classifier", store,
        )
        assert result1.evidence_id != result2.evidence_id
