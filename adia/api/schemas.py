"""Request/response contracts for the API layer. No logic lives here — validation constraints
only. Response shapes are deliberately close to what `bench/runner.py::run_question` already
derives from a finished `AgentState`, so the API surfaces the same facts the benchmark records.
"""

from pydantic import BaseModel, Field

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
