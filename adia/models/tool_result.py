"""The universal envelope every deterministic tool returns."""

from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from adia.models.errors import ToolError
from adia.models.provenance import Provenance


class ToolResult(BaseModel):
    """Universal return type for every tool call.

    Exactly one of (`data` + `evidence_id` + `provenance`) or `error` is populated, selected
    by `ok`. This uniformity lets the deterministic router branch on tool outcomes without
    knowing anything about a specific tool's internals.
    """

    model_config = ConfigDict(frozen=True)

    ok: bool
    tool: str = Field(..., description="Registered name of the tool that produced this result.")
    evidence_id: str | None = Field(
        default=None, description="ID of the Evidence record written for this result, if ok."
    )
    data: Any = Field(
        default=None, description="Full-precision payload. Rounding happens only at render time."
    )
    provenance: Provenance | None = Field(default=None)
    warnings: list[str] = Field(default_factory=list)
    error: ToolError | None = Field(default=None)
    duration_ms: float = Field(
        ..., ge=0, description="Wall-clock time the call took, in milliseconds."
    )

    @model_validator(mode="after")
    def _check_ok_shape(self) -> Self:
        """Enforce the ok/error mutual exclusivity that makes this envelope trustworthy."""
        if self.ok:
            if self.error is not None:
                raise ValueError("ToolResult.error must be None when ok=True.")
            if self.evidence_id is None or self.data is None or self.provenance is None:
                raise ValueError(
                    "ToolResult must set evidence_id, data, and provenance when ok=True."
                )
        elif self.error is None:
            raise ValueError("ToolResult.error must be set when ok=False.")
        return self
