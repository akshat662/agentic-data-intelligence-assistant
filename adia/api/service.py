"""Orchestration behind the API routes. Every function here only calls existing, already-tested
modules (`adia.graph.state`, `adia.graph.workflow`, `adia.data.registry`, `adia.data.loader`) —
no agent, planner, tool, or validation logic is reimplemented.
"""

import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

from adia.data.loader import load_dataset
from adia.data.registry import load_registry, save_registry
from adia.evidence.renderer import render_evidence
from adia.graph.state import create_initial_state
from adia.graph.workflow import run_graph, stream_graph
from adia.models.dataset import DatasetConfig
from adia.models.state import AgentState

from .schemas import (
    ChatResponse,
    DatasetUploadResponse,
    StreamErrorEvent,
    StreamEvent,
    StreamEvidenceEvent,
    StreamFinalEvent,
    StreamPhaseEvent,
)

#: `adia/api/service.py` -> `adia/api` -> `adia` -> repo root -> `data/`.
_REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = _REPO_ROOT / "data" / "registry.json"
UPLOAD_DIR = _REPO_ROOT / "data" / "uploads"

#: Signature every graph runner -- the real one or a test's fake -- must satisfy. Mirrors
#: `adia.cli.GraphRunner`.
GraphRunner = Callable[[AgentState], AgentState]

#: Signature every streaming graph runner -- the real `stream_graph` or a test's fake -- must
#: satisfy.
StreamGraphRunner = Callable[[AgentState], Iterator[tuple[str, dict[str, Any], AgentState]]]


class DatasetAlreadyRegisteredError(ValueError):
    """Raised when a dataset upload names a `dataset_id` that is already registered."""


class InvalidCsvError(ValueError):
    """Raised when an uploaded file is missing, not a `.csv`, empty, or fails to parse."""


def get_graph_runner() -> GraphRunner:
    """Default graph runner dependency. Overridden in tests to avoid real LLM calls."""
    return run_graph


def get_stream_graph_runner() -> StreamGraphRunner:
    """Default streaming graph runner dependency. Overridden in tests to avoid real LLM calls."""
    return stream_graph


def get_registry_path() -> Path:
    """Default registry-path dependency. Overridden in tests to avoid touching the real repo."""
    return REGISTRY_PATH


def get_upload_dir() -> Path:
    """Default upload-dir dependency. Overridden in tests to avoid touching the real repo."""
    return UPLOAD_DIR


def run_chat(
    dataset_id: str,
    question: str,
    *,
    run_graph_fn: GraphRunner = run_graph,
) -> ChatResponse:
    """Run one question through the ADIA graph and shape the result for the API.

    Args:
        dataset_id: Dataset to answer against. Resolved against the dataset registry by
            `feasibility_node`, not here -- an unregistered ID surfaces as a grounded refusal
            from the graph itself, not an API-level error.
        question: The user's question, verbatim.
        run_graph_fn: Override for the graph runner, e.g. a fake for tests.

    Returns:
        A `ChatResponse` built from the finished `AgentState`.
    """
    started = time.perf_counter()
    initial_state = create_initial_state(question, dataset_id)
    final_state = run_graph_fn(initial_state)
    duration_ms = (time.perf_counter() - started) * 1000

    return ChatResponse(
        run_id=final_state.run_id,
        dataset_id=dataset_id,
        question=question,
        answer=final_state.final_answer,
        validation_passed=(
            final_state.validation.passed if final_state.validation else None
        ),
        evidence_ids=sorted(final_state.evidence),
        tools_used=sorted({e.tool for e in final_state.evidence.values()}),
        feasibility_verdict=(
            final_state.feasibility.verdict.value if final_state.feasibility else None
        ),
        refused=final_state.refusal is not None,
        duration_ms=duration_ms,
    )


def stream_chat(
    dataset_id: str,
    question: str,
    *,
    stream_graph_fn: StreamGraphRunner = stream_graph,
) -> Iterator[StreamEvent]:
    """Run one question through the ADIA graph, yielding progress as it goes.

    Mirrors `run_chat`'s field derivation for the final event, but is otherwise a distinct code
    path -- `POST /chat` is untouched by this function. Never raises: any failure inside
    `stream_graph_fn` (including one from a real, unreachable LLM) is caught and turned into a
    `StreamErrorEvent` with a fixed, internals-free message, the same principle
    `adia.api.app`'s generic exception handler applies to the rest of the API.

    Args:
        dataset_id: Dataset to answer against. Resolved against the dataset registry by
            `feasibility_node`, not here.
        question: The user's question, verbatim.
        stream_graph_fn: Override for the streaming graph runner, e.g. a fake for tests.

    Yields:
        One `StreamPhaseEvent` per completed graph node, one `StreamEvidenceEvent` per new
        evidence record produced by `execute_tools`, and exactly one `StreamFinalEvent` last --
        unless the run fails, in which case a `StreamErrorEvent` is yielded instead of the
        final event.
    """
    started = time.perf_counter()
    initial_state = create_initial_state(question, dataset_id)
    final_state: AgentState | None = None
    try:
        for node_name, partial, state in stream_graph_fn(initial_state):
            final_state = state
            yield StreamPhaseEvent(node=node_name, data=_phase_data(node_name, partial, state))
            if node_name == "execute_tools":
                for evidence in partial.get("evidence", {}).values():
                    yield StreamEvidenceEvent(evidence=render_evidence(evidence))

        if final_state is None:
            raise RuntimeError("Graph produced no state.")  # pragma: no cover -- defensive

        duration_ms = (time.perf_counter() - started) * 1000
        yield StreamFinalEvent(
            run_id=final_state.run_id,
            dataset_id=dataset_id,
            question=question,
            answer=final_state.final_answer,
            validation_passed=(
                final_state.validation.passed if final_state.validation else None
            ),
            evidence=[
                render_evidence(e)
                for e in sorted(final_state.evidence.values(), key=lambda e: e.id)
            ],
            tools_used=sorted({e.tool for e in final_state.evidence.values()}),
            feasibility_verdict=(
                final_state.feasibility.verdict.value if final_state.feasibility else None
            ),
            refused=final_state.refusal is not None,
            duration_ms=duration_ms,
        )
    except Exception:
        yield StreamErrorEvent()


def _phase_data(node_name: str, partial: dict[str, Any], state: AgentState) -> dict[str, Any]:
    """Small, curated, node-specific summary for a `StreamPhaseEvent` -- never the raw partial
    state update, which may carry full model instances (a plan, an evidence dict) with no
    business being sent to a client as an opaque "phase" payload.
    """
    if node_name == "feasibility":
        if state.feasibility is None:
            return {}
        return {"verdict": state.feasibility.verdict.value, "reason": state.feasibility.reason}
    if node_name == "planner":
        return {
            "step_count": len(state.plan),
            "tools": [step.tool_family for step in state.plan],
        }
    if node_name == "execute_tools":
        return {
            "evidence_count": len(partial.get("evidence", {})),
            "error_count": len(partial.get("errors", [])),
        }
    if node_name == "validation":
        return {"passed": state.validation.passed if state.validation else None}
    if node_name == "refusal":
        return {"reason": state.refusal.reason if state.refusal else None}
    return {}


def register_dataset(
    dataset_id: str,
    description: str,
    filename: str | None,
    contents: bytes,
    *,
    registry_path: Path = REGISTRY_PATH,
    upload_dir: Path = UPLOAD_DIR,
) -> DatasetUploadResponse:
    """Save an uploaded CSV and register it so future chat requests can use it.

    The client-supplied `filename` is used only to check the `.csv` extension -- it is never
    used to build the on-disk path. The stored file is always named `<dataset_id>.csv` under
    `upload_dir`, and `dataset_id` is validated (by `ChatRequest`/`schemas.DATASET_ID_PATTERN`
    at the route layer) to contain only `[a-zA-Z0-9_-]`, which rules out `/`, `..`, and any
    other path-traversal character before it ever reaches this function.

    Args:
        dataset_id: Path-safe identifier to register the dataset under.
        description: Human-readable description, stored in the registry entry.
        filename: The client-supplied filename, checked for a `.csv` extension only.
        contents: Raw file bytes.
        registry_path: Registry file to read/append. Overridable so tests use a `tmp_path`
            instead of the real repo registry.
        upload_dir: Directory uploaded files are saved into. Overridable for the same reason.

    Returns:
        A `DatasetUploadResponse` describing the registered dataset.

    Raises:
        DatasetAlreadyRegisteredError: If `dataset_id` is already registered.
        InvalidCsvError: If `filename` isn't a `.csv`, `contents` is empty, or the file fails
            to parse as CSV.
    """
    if not filename or not filename.lower().endswith(".csv"):
        raise InvalidCsvError(f"Only .csv files are supported, got '{filename}'.")
    if not contents:
        raise InvalidCsvError("Uploaded file is empty.")

    registry = load_registry(registry_path) if Path(registry_path).exists() else {}
    if dataset_id in registry:
        raise DatasetAlreadyRegisteredError(f"Dataset '{dataset_id}' is already registered.")

    upload_dir = Path(upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / f"{dataset_id}.csv"
    dest.write_bytes(contents)

    try:
        df = load_dataset(dest)
    except Exception as exc:
        dest.unlink(missing_ok=True)
        raise InvalidCsvError(f"Uploaded file could not be parsed as CSV: {exc}") from exc

    config = DatasetConfig(dataset_id=dataset_id, file_path=str(dest), description=description)
    save_registry([*registry.values(), config], registry_path)

    return DatasetUploadResponse(
        dataset_id=dataset_id,
        description=description,
        file_path=str(dest),
        row_count=len(df),
        column_count=len(df.columns),
    )
