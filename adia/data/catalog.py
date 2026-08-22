"""Dataset catalog generation: profiling a DataFrame into a `DatasetCatalog`."""

from pathlib import Path

import pandas as pd

from adia.models.catalog import ColumnProfile, DatasetCatalog, SemanticType


def _classify_semantic_type(series: pd.Series) -> SemanticType:
    """Classify a column's semantic type from its pandas dtype."""
    if pd.api.types.is_bool_dtype(series):
        return SemanticType.BOOLEAN
    if pd.api.types.is_datetime64_any_dtype(series):
        return SemanticType.DATETIME
    if pd.api.types.is_numeric_dtype(series):
        return SemanticType.NUMERIC
    if (
        pd.api.types.is_object_dtype(series)
        or pd.api.types.is_string_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
    ):
        return SemanticType.CATEGORICAL
    return SemanticType.OTHER


def profile_column(series: pd.Series) -> ColumnProfile:
    """Build a `ColumnProfile` for a single column.

    Args:
        series: The column to profile.

    Returns:
        A `ColumnProfile` capturing dtype, missingness, cardinality, and (where applicable)
        numeric range or date range.
    """
    semantic_type = _classify_semantic_type(series)
    non_null_count = int(series.notna().sum())
    null_count = int(series.isna().sum())
    total = len(series)
    null_rate = (null_count / total) if total else 0.0
    unique_count = int(series.nunique(dropna=True))

    min_value: float | None = None
    max_value: float | None = None
    min_date: str | None = None
    max_date: str | None = None

    if semantic_type == SemanticType.NUMERIC and non_null_count:
        min_value = float(series.min())
        max_value = float(series.max())
    elif semantic_type == SemanticType.DATETIME and non_null_count:
        min_date = pd.Timestamp(series.min()).isoformat()
        max_date = pd.Timestamp(series.max()).isoformat()

    return ColumnProfile(
        name=str(series.name),
        dtype=str(series.dtype),
        semantic_type=semantic_type,
        non_null_count=non_null_count,
        null_count=null_count,
        null_rate=null_rate,
        unique_count=unique_count,
        min_value=min_value,
        max_value=max_value,
        min_date=min_date,
        max_date=max_date,
    )


def build_catalog(df: pd.DataFrame, dataset_id: str, source_path: str) -> DatasetCatalog:
    """Build a `DatasetCatalog` by profiling every column of `df`.

    Args:
        df: The dataset to profile.
        dataset_id: Stable identifier for this dataset.
        source_path: Path the dataset was loaded from, recorded for provenance.

    Returns:
        A `DatasetCatalog` with one `ColumnProfile` per column.
    """
    columns = [profile_column(df[col]) for col in df.columns]
    return DatasetCatalog(
        dataset_id=dataset_id,
        source_path=str(source_path),
        row_count=len(df),
        columns=columns,
    )


def save_catalog(catalog: DatasetCatalog, path: str | Path) -> None:
    """Serialize a `DatasetCatalog` to a JSON file."""
    Path(path).write_text(catalog.model_dump_json(indent=2))


def load_catalog(path: str | Path) -> DatasetCatalog:
    """Load a `DatasetCatalog` previously written by `save_catalog`."""
    return DatasetCatalog.model_validate_json(Path(path).read_text())
