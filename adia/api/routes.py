"""API routes. Each handler only validates input, calls into `adia.api.service`, and translates
service-level exceptions into HTTP responses -- no business logic lives here.
"""

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from .schemas import (
    DATASET_ID_PATTERN,
    ChatRequest,
    ChatResponse,
    DatasetUploadResponse,
    HealthResponse,
)
from .service import (
    DatasetAlreadyRegisteredError,
    GraphRunner,
    InvalidCsvError,
    get_graph_runner,
    get_registry_path,
    get_upload_dir,
    register_dataset,
    run_chat,
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
