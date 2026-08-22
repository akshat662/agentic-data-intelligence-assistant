"""Deterministic, explainable machine learning tool.

`train_model` fits one of a small, fixed set of standard scikit-learn models — no AutoML, no
hyperparameter search — on a held-out split and reports its evaluation metric against an
honest baseline (majority-class for classification, mean-predictor for regression), plus
feature importance where the model type supports it. The tool never chooses a model on the
caller's behalf and never tunes one; `model_type` is a required argument, and every
hyperparameter left unspecified uses scikit-learn's own default. A model score with no
baseline next to it is not evidence, so a baseline is always computed and reported alongside.
"""

import time
from collections.abc import Callable
from typing import Any, Literal

import numpy as np
import pandas as pd
import sklearn
from pydantic import BaseModel, Field, ValidationError
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from adia.data.loader import load_dataset
from adia.evidence.ids import compute_args_hash, generate_evidence_id
from adia.evidence.store import EvidenceStore
from adia.models.errors import ToolError, ToolErrorKind
from adia.models.evidence import Evidence
from adia.models.provenance import Provenance
from adia.models.tool_result import ToolResult

_TOOL_NAME = "train_model"
_RANDOM_STATE = 42
_MIN_ROWS = 10
_MIN_CLASS_COUNT = 2

_MODEL_REGISTRY: dict[str, dict[str, Callable[[], Any]]] = {
    "classification": {
        "logistic_regression": lambda: LogisticRegression(max_iter=1000),
        "random_forest_classifier": lambda: RandomForestClassifier(random_state=_RANDOM_STATE),
    },
    "regression": {
        "linear_regression": LinearRegression,
        "random_forest_regressor": lambda: RandomForestRegressor(random_state=_RANDOM_STATE),
    },
}
"""XGBoost variants were marked optional in the task and are intentionally omitted here to
avoid an extra heavy dependency; the registry shape makes adding them later a pure addition,
not a redesign."""


class TrainModelArgs(BaseModel):
    """Validated input contract for `train_model`."""

    dataset_id: str = Field(..., min_length=1)
    source_path: str = Field(..., min_length=1)
    target_column: str = Field(..., min_length=1)
    feature_columns: list[str] = Field(..., min_length=1)
    task_type: Literal["classification", "regression"]
    model_type: str = Field(..., min_length=1)


def train_model(
    dataset_id: str,
    source_path: str,
    target_column: str,
    feature_columns: list[str],
    task_type: str,
    model_type: str,
    evidence_store: EvidenceStore,
    *,
    plan_step_id: str | None = None,
    test_size: float = 0.2,
) -> ToolResult:
    """Train one fixed-hyperparameter model and evaluate it on a held-out split.

    Rows with a missing value in `target_column` or any `feature_columns` are dropped before
    splitting. The split is a single deterministic `train_test_split` (stratified by target
    for classification), seeded by a fixed random state — not k-fold cross-validation, and
    not repeated: this tool reports one honest evaluation, not a tuned best-of-many.

    Args:
        dataset_id: Stable identifier for the dataset.
        source_path: Path to a `.parquet` or `.csv` file.
        target_column: Column to predict.
        feature_columns: Numeric columns to train on. Must not include `target_column`.
        task_type: `"classification"` or `"regression"`.
        model_type: One of the models registered for `task_type` — for classification,
            `"logistic_regression"` or `"random_forest_classifier"`; for regression,
            `"linear_regression"` or `"random_forest_regressor"`.
        evidence_store: Store to record the resulting evidence in.
        plan_step_id: ID of the plan step this call belongs to, if any.
        test_size: Fraction of rows held out for evaluation.

    Returns:
        A `ToolResult`. On success, `data` holds `model_type`, `task_type`, `metric_name` /
        `metric_value` (accuracy for classification, R² for regression),
        `baseline_metric_name` / `baseline_metric_value`, `feature_importance` (a list of
        `{feature, importance}` ranked descending, or `None` if the model type has no
        importance/coefficient attribute), and `metadata` (row/split counts, feature and
        target columns, seed, and — for classification — the sorted class labels). On
        failure, `error` describes exactly what was rejected or what went wrong.
    """
    started = time.perf_counter()
    args = {
        "dataset_id": dataset_id,
        "source_path": source_path,
        "target_column": target_column,
        "feature_columns": feature_columns,
        "task_type": task_type,
        "model_type": model_type,
        "test_size": test_size,
    }

    try:
        TrainModelArgs(
            dataset_id=dataset_id,
            source_path=source_path,
            target_column=target_column,
            feature_columns=feature_columns,
            task_type=task_type,
            model_type=model_type,
        )
    except ValidationError as exc:
        return _error_result(args, ToolErrorKind.VALIDATION, str(exc), started)

    if not 0.0 < test_size < 1.0:
        return _error_result(
            args,
            ToolErrorKind.VALIDATION,
            f"test_size must be between 0 and 1 (exclusive); got {test_size}.",
            started,
        )

    valid_models = _MODEL_REGISTRY[task_type]
    if model_type not in valid_models:
        return _error_result(
            args,
            ToolErrorKind.VALIDATION,
            f"Unknown model_type '{model_type}' for task_type '{task_type}'.",
            started,
            details={"model_type": model_type, "valid_options": sorted(valid_models)},
        )

    try:
        df = load_dataset(source_path)
    except FileNotFoundError as exc:
        return _error_result(args, ToolErrorKind.NOT_FOUND, str(exc), started)
    except ValueError as exc:
        return _error_result(args, ToolErrorKind.EXECUTION, str(exc), started)

    for column in (target_column, *feature_columns):
        if column not in df.columns:
            return _error_result(
                args,
                ToolErrorKind.VALIDATION,
                f"Unknown column '{column}'.",
                started,
                details={"column": column},
            )

    if target_column in feature_columns:
        return _error_result(
            args,
            ToolErrorKind.VALIDATION,
            f"target_column '{target_column}' cannot also be a feature column.",
            started,
            details={"column": target_column},
        )

    non_numeric_features = [
        c for c in feature_columns if not pd.api.types.is_numeric_dtype(df[c])
    ]
    if non_numeric_features:
        return _error_result(
            args,
            ToolErrorKind.VALIDATION,
            f"Feature column(s) not numeric: {', '.join(non_numeric_features)}. "
            "This tool does not encode categorical features.",
            started,
            details={"columns": non_numeric_features},
        )

    if task_type == "regression" and not pd.api.types.is_numeric_dtype(df[target_column]):
        return _error_result(
            args,
            ToolErrorKind.VALIDATION,
            f"target_column '{target_column}' is not numeric; required for regression.",
            started,
            details={"column": target_column, "dtype": str(df[target_column].dtype)},
        )

    working = df[[target_column, *feature_columns]].dropna()
    if len(working) < _MIN_ROWS:
        return _error_result(
            args,
            ToolErrorKind.INSUFFICIENT_DATA,
            f"At least {_MIN_ROWS} complete rows are required to train a model; "
            f"found {len(working)}.",
            started,
        )

    x = working[feature_columns]
    y = working[target_column]

    class_counts = None
    if task_type == "classification":
        class_counts = y.value_counts()
        if class_counts.shape[0] < 2:
            return _error_result(
                args,
                ToolErrorKind.INSUFFICIENT_DATA,
                "At least 2 distinct classes are required for classification.",
                started,
            )
        sparse_classes = class_counts[class_counts < _MIN_CLASS_COUNT].index.tolist()
        if sparse_classes:
            return _error_result(
                args,
                ToolErrorKind.INSUFFICIENT_DATA,
                f"Class(es) with fewer than {_MIN_CLASS_COUNT} examples: {sparse_classes}.",
                started,
                details={"sparse_classes": [str(c) for c in sparse_classes]},
            )

    try:
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=_RANDOM_STATE,
            stratify=y if task_type == "classification" else None,
        )
        model = valid_models[model_type]()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)

        baseline = _make_baseline(task_type)
        baseline.fit(x_train, y_train)
        baseline_predictions = baseline.predict(x_test)

        if task_type == "classification":
            metric_name = "accuracy"
            metric_value = accuracy_score(y_test, predictions)
            baseline_value = accuracy_score(y_test, baseline_predictions)
        else:
            metric_name = "r2"
            metric_value = r2_score(y_test, predictions)
            baseline_value = r2_score(y_test, baseline_predictions)
    except ValueError as exc:
        return _error_result(args, ToolErrorKind.EXECUTION, str(exc), started)

    metadata = {
        "n_rows_used": len(working),
        "n_train": len(x_train),
        "n_test": len(x_test),
        "n_features": len(feature_columns),
        "feature_columns": feature_columns,
        "target_column": target_column,
        "test_size": test_size,
        "random_state": _RANDOM_STATE,
        "class_labels": (
            sorted(_json_safe(c) for c in class_counts.index)
            if class_counts is not None
            else None
        ),
    }

    data = {
        "model_type": model_type,
        "task_type": task_type,
        "metric_name": metric_name,
        "metric_value": _json_safe(metric_value),
        "baseline_metric_name": metric_name,
        "baseline_metric_value": _json_safe(baseline_value),
        "feature_importance": _feature_importance(model, feature_columns),
        "metadata": metadata,
    }

    evidence_id = generate_evidence_id(_TOOL_NAME, args)
    provenance = Provenance(
        tool_name=_TOOL_NAME,
        args=args,
        args_hash=compute_args_hash(args),
        seed=_RANDOM_STATE,
        row_count=len(working),
        library_versions={"scikit-learn": sklearn.__version__},
    )
    evidence = evidence_store.add(
        Evidence(
            id=evidence_id,
            tool=_TOOL_NAME,
            data=data,
            provenance=provenance,
            plan_step_id=plan_step_id,
        )
    )

    return ToolResult(
        ok=True,
        tool=_TOOL_NAME,
        evidence_id=evidence.id,
        data=evidence.data,
        provenance=evidence.provenance,
        duration_ms=_elapsed_ms(started),
    )


def _make_baseline(task_type: str) -> DummyClassifier | DummyRegressor:
    """A naive baseline for the given task: majority-class, or mean-of-target."""
    if task_type == "classification":
        return DummyClassifier(strategy="most_frequent", random_state=_RANDOM_STATE)
    return DummyRegressor(strategy="mean")


def _feature_importance(model: Any, feature_columns: list[str]) -> list[dict[str, Any]] | None:
    """Rank features by the model's own importance/coefficient attribute, if it has one.

    Random forests expose `feature_importances_` directly; linear/logistic models expose
    `coef_`, averaged across output classes via `abs()` so binary and multiclass logistic
    regression are handled by the same code path. Ties are broken by feature name ascending
    for deterministic output.
    """
    if hasattr(model, "feature_importances_"):
        raw = model.feature_importances_
    elif hasattr(model, "coef_"):
        raw = np.mean(np.abs(np.atleast_2d(model.coef_)), axis=0)
    else:
        return None

    ranked = sorted(
        zip(feature_columns, raw, strict=True),
        key=lambda item: (-float(item[1]), item[0]),
    )
    return [{"feature": name, "importance": _json_safe(value)} for name, value in ranked]


def _json_safe(value: Any) -> Any:
    """Convert a pandas/numpy scalar into a plain JSON-safe, full-precision Python value."""
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return None if pd.isna(value) else float(value)
    if isinstance(value, float):
        return None if pd.isna(value) else value
    return value


def _error_result(
    args: dict[str, Any],
    kind: ToolErrorKind,
    message: str,
    started: float,
    *,
    details: dict[str, Any] | None = None,
) -> ToolResult:
    """Build a typed `ToolResult` failure. The tool never lets an exception reach its caller."""
    return ToolResult(
        ok=False,
        tool=_TOOL_NAME,
        error=ToolError(kind=kind, message=message, details=details or {}),
        duration_ms=_elapsed_ms(started),
    )


def _elapsed_ms(started: float) -> float:
    """Milliseconds elapsed since a `time.perf_counter()` reading."""
    return (time.perf_counter() - started) * 1000
