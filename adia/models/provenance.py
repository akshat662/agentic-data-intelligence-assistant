"""Provenance metadata attached to every tool result and evidence record."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Provenance(BaseModel):
    """Records exactly how a piece of evidence was produced.

    Every tool call attaches a `Provenance` record — the exact arguments used, a stable hash
    of those arguments, relevant library versions, and an optional seed — so that any number
    appearing in a final answer can be traced back to precisely what produced it and, in
    principle, reproduced.
    """

    model_config = ConfigDict(frozen=True)

    tool_name: str = Field(
        ..., description="Registered name of the tool that produced this result."
    )
    args: dict[str, Any] = Field(
        default_factory=dict, description="Exact arguments the tool was called with."
    )
    args_hash: str = Field(
        ..., description="Stable hash of `args`, used for caching and determinism checks."
    )
    seed: int | None = Field(default=None, description="Random seed used, if applicable.")
    library_versions: dict[str, str] = Field(
        default_factory=dict,
        description="Versions of libraries relevant to reproducing this result.",
    )
    source_query: str | None = Field(
        default=None, description="Raw SQL or query string executed, if applicable."
    )
    row_count: int | None = Field(
        default=None, description="Number of rows read or processed, if applicable."
    )
    generated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="UTC timestamp when this result was produced.",
    )
