"""Request/response contracts for the API layer. No logic lives here — validation constraints
only. Response shapes are deliberately close to what `bench/runner.py::run_question` already
derives from a finished `AgentState`, so the API surfaces the same facts the benchmark records.
"""

from typing import Any, Literal

from pydantic import BaseModel, Field

from adia.evidence.renderer import RenderedEvidence

#: Also doubles as path-safety validation: a `dataset_id` matching this pattern can never
#: contain `/`, `..`, or other characters that would let it escape the upload directory when
#: used to build a filename.
DATASET_ID_PATTERN = r"^[a-zA-Z0-9_-]+$"


class ChatRequest(BaseModel):
    """Body of `POST /chat`."""

    dataset_id: str = Field(..., min_length=1, pattern=DATASET_ID_PATTERN)
    question: str = Field(..., min_length=1)


class ChatResponse(BaseModel):
    """Response of `POST /chat` — the facts a client needs from a finished `AgentState`."""

    run_id: str
    dataset_id: str
    question: str
    answer: str | None
    validation_passed: bool | None
    evidence_ids: list[str]
    tools_used: list[str]
    feasibility_verdict: str | None
    refused: bool
    duration_ms: float


class DatasetUploadResponse(BaseModel):
    """Response of `POST /datasets`."""

    dataset_id: str
    description: str
    file_path: str
    row_count: int
    column_count: int


class HealthResponse(BaseModel):
    """Response of `GET /health`."""

    status: str = "ok"


class ErrorResponse(BaseModel):
    """Response body for any error the API returns, including unhandled ones."""

    detail: str


# --- POST /chat/stream events -------------------------------------------------------------
#
# One JSON object per Server-Sent Event, emitted as the graph runs. Evidence is always carried
# as `RenderedEvidence` (`adia.evidence.renderer`) -- the same bounded, tool-agnostic summary
# the Synthesizer itself is shown -- never the raw, unbounded `Evidence.data` a tool produced.


class StreamPhaseEvent(BaseModel):
    """One graph node has completed. `data` is a small, curated, node-specific summary --
    never a raw dump of the node's full state update.
    """

    type: Literal["phase"] = "phase"
    node: str = Field(..., description="Name of the node that just completed, e.g. 'planner'.")
    data: dict[str, Any] = Field(default_factory=dict)


class StreamEvidenceEvent(BaseModel):
    """One new evidence record was produced by a tool."""

    type: Literal["evidence"] = "evidence"
    evidence: RenderedEvidence


class StreamFinalEvent(BaseModel):
    """The run has finished: the validated final answer and its supporting evidence."""

    type: Literal["final"] = "final"
    run_id: str
    dataset_id: str
    question: str
    answer: str | None
    validation_passed: bool | None
    evidence: list[RenderedEvidence]
    tools_used: list[str]
    feasibility_verdict: str | None
    refused: bool
    duration_ms: float


class StreamErrorEvent(BaseModel):
    """The run failed unexpectedly. Never carries internals -- same principle as the API's
    generic 500 handler (`adia.api.app`).
    """

    type: Literal["error"] = "error"
    detail: str = "Internal server error."


StreamEvent = StreamPhaseEvent | StreamEvidenceEvent | StreamFinalEvent | StreamErrorEvent
