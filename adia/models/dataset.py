"""Dataset registration contract: metadata for a dataset before it is ever loaded or profiled.

Distinct from `DatasetCatalog` (`adia.models.catalog`), which describes an *already-loaded*
dataset's schema and profile. `DatasetConfig` is the earlier, smaller fact — "this dataset
exists, here is where to find it" — the thing a registry maps dataset IDs to.
"""

from pydantic import BaseModel, ConfigDict, Field


class DatasetConfig(BaseModel):
    """Registration metadata for one dataset.

    Recorded once when a dataset is added to the system; consumed by the data layer to
    locate the file, and by future agents/benchmarks to know what's available without
    reading the file itself. Registering a dataset does not require its file to exist yet —
    that is only checked when something actually tries to load it.
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str = Field(..., min_length=1, description="Stable identifier for the dataset.")
    file_path: str = Field(
        ..., min_length=1, description="Path to the dataset's Parquet/CSV file."
    )
    description: str = Field(
        ..., min_length=1, description="Human-readable summary of what this dataset contains."
    )
    target_columns: list[str] | None = Field(
        default=None,
        description="Columns expected to serve as ML prediction targets, if any.",
    )
