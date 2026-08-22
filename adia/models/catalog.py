"""Dataset catalog contracts: the frozen schema/profile snapshot agents reason over.

Not one of the six contracts called out by name in the Phase 1A brief, but required by it
indirectly: the data layer must produce a "clear schema" for catalogs, and `AgentState.catalog`
needs a real type rather than a bare `dict`.
"""

from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class SemanticType(StrEnum):
    """Coarse semantic classification of a column, used to decide which profiling stats apply."""

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    BOOLEAN = "boolean"
    OTHER = "other"


class ColumnProfile(BaseModel):
    """Profile of a single column, as recorded in a dataset's catalog."""

    model_config = ConfigDict(frozen=True)

    name: str
    dtype: str = Field(..., description="Underlying pandas dtype, as a string (e.g. 'int64').")
    semantic_type: SemanticType
    non_null_count: int = Field(..., ge=0)
    null_count: int = Field(..., ge=0)
    null_rate: float = Field(..., ge=0, le=1)
    unique_count: int = Field(
        ..., ge=0, description="Column cardinality — the relevant figure for categorical columns."
    )
    min_value: float | None = Field(default=None, description="Minimum value, numeric only.")
    max_value: float | None = Field(default=None, description="Maximum value, numeric only.")
    min_date: str | None = Field(
        default=None, description="Earliest date (ISO 8601), datetime only."
    )
    max_date: str | None = Field(default=None, description="Latest date (ISO 8601), datetime only.")


class DatasetCatalog(BaseModel):
    """Frozen schema/profile snapshot of one dataset.

    Agents reason about a dataset only through its catalog, never by reading raw rows directly
    — this keeps column-reference hallucinations checkable in Python (cross-check against
    `columns`) instead of trusted on the model's word.
    """

    model_config = ConfigDict(frozen=True)

    dataset_id: str
    source_path: str
    row_count: int = Field(..., ge=0)
    columns: list[ColumnProfile]
    generated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    def column_names(self) -> list[str]:
        """Convenience accessor for cross-checking LLM-referenced column names."""
        return [c.name for c in self.columns]
