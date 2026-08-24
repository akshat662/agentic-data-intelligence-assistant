"""API routes. Each handler only validates input, calls into `adia.api.service`, and translates
service-level exceptions into HTTP responses -- no business logic lives here.
"""

from collections.abc import Iterator
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from .schemas import (
    DATASET_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    DatasetUploadResponse,
    HealthResponse,
    StreamEvent,
)
from .service import (
    DatasetAlreadyRegisteredError,
    GraphRunner,
    InvalidCsvError,
    StreamGraphRunner,
    get_graph_runner,
    get_registry_path,
    get_stream_graph_runner,
    get_upload_dir,
    register_dataset,
    run_chat,
    stream_chat,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """Liveness check."""
    return HealthResponse()


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    runner: GraphRunner = Depends(get_graph_runner),
) -> ChatResponse:
    """Run a question through the ADIA graph and return a structured result."""
    return run_chat(request.dataset_id, request.question, run_graph_fn=runner)


@router.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    runner: StreamGraphRunner = Depends(get_stream_graph_runner),
) -> StreamingResponse:
    """Run a question through the ADIA graph, streaming progress as Server-Sent Events.

    Emits one `data: <json>\\n\\n` frame per event from `adia.api.service.stream_chat`
    (`StreamPhaseEvent`, `StreamEvidenceEvent`, ..., ending in exactly one `StreamFinalEvent`
    or `StreamErrorEvent`) -- see `adia/api/schemas.py` for the event shapes.
    """
    events = stream_chat(request.dataset_id, request.question, stream_graph_fn=runner)
    return StreamingResponse(_as_sse(events), media_type="text/event-stream")


def _as_sse(events: Iterator[StreamEvent]) -> Iterator[str]:
    """Format typed stream events as `text/event-stream` frames -- wire format only, no logic."""
    for event in events:
        yield f"data: {event.model_dump_json()}\n\n"


@router.post("/datasets", response_model=DatasetUploadResponse, status_code=201)
async def upload_dataset(
    dataset_id: str = Form(..., min_length=1, pattern=DATASET_ID_PATTERN),
    description: str = Form(..., min_length=1),
    file: UploadFile = File(...),
    registry_path: Path = Depends(get_registry_path),
    upload_dir: Path = Depends(get_upload_dir),
) -> DatasetUploadResponse:
    """Upload a CSV dataset and register it for future `/chat` requests."""
    contents = await file.read()
    try:
        return register_dataset(
            dataset_id,
            description,
            file.filename,
            contents,
            registry_path=registry_path,
            upload_dir=upload_dir,
        )
    except DatasetAlreadyRegisteredError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except InvalidCsvError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
