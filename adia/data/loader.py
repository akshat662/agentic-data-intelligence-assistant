"""Dataset loading utilities.

Reads a dataset file (Parquet or CSV) into a pandas DataFrame. Deliberately format-agnostic
and free of any dataset-specific cleaning logic — dataset-specific ingestion pipelines are a
separate concern from this generic loader.
"""

from pathlib import Path

import pandas as pd

_SUPPORTED_SUFFIXES = {".parquet", ".csv"}


def load_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a Parquet or CSV file into a DataFrame.

    Args:
        path: Path to a `.parquet` or `.csv` file.

    Returns:
        The loaded DataFrame, unmodified.

    Raises:
        FileNotFoundError: If `path` does not exist.
        ValueError: If the file extension is not supported.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Dataset file not found: {resolved}")
    if resolved.suffix not in _SUPPORTED_SUFFIXES:
        raise ValueError(
            f"Unsupported dataset file extension '{resolved.suffix}'. "
            f"Supported: {sorted(_SUPPORTED_SUFFIXES)}"
        )
    if resolved.suffix == ".parquet":
        return pd.read_parquet(resolved)
    return pd.read_csv(resolved)
