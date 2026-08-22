"""A single fact in the evidence store: one tool result, addressable by ID."""

from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from adia.models.provenance import Provenance


class Evidence(BaseModel):
    """An immutable, addressable record of one computed fact.

    The Synthesizer never writes numbers directly — it cites an evidence ID, and the
    deterministic renderer substitutes the actual value from here. This indirection is what
    makes grounding enforceable in code rather than by prompt instruction.
    """

    model_config = ConfigDict(frozen=True)

    id: str = Field(..., description="Stable evidence identifier, e.g. 'ev_03'.")
    tool: str = Field(..., description="Name of the tool that produced this evidence.")
    data: Any = Field(..., description="Full-precision value(s) this evidence record holds.")
    provenance: Provenance
    plan_step_id: str | None = Field(
        default=None, description="ID of the plan step this evidence was produced for, if any."
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
