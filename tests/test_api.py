"""Tests for the FastAPI demo layer. No real graph run (and therefore no real LLM call) is ever
made here -- `/chat` tests override `get_graph_runner` with a fake, exactly like
`tests/test_cli.py` overrides `run_graph_fn`.
"""

import json

import pytest
from fastapi.testclient import TestClient

from adia.api.app import app
from adia.api.service import (
    get_graph_runner,
    get_registry_path,
    get_stream_graph_runner,
    get_upload_dir,
)
from adia.evidence.ids import compute_args_hash, generate_evidence_id
from adia.models.evidence import Evidence
from adia.models.provenance import Provenance
from adia.models.state import AgentState, FeasibilityResult, FeasibilityVerdict, ValidationResult


def _make_evidence(tool: str, args: dict) -> Evidence:
    return Evidence(
        id=generate_evidence_id(tool, args),
        tool=tool,
        data={"row_count": 9994},
        provenance=Provenance(tool_name=tool, args=args, args_hash=compute_args_hash(args)),
    )


def _fake_run_graph(final_answer, *, passed=True, evidence=None, feasible=True, refused=False):
    evidence = evidence or {}

    def _run(initial_state: AgentState) -> AgentState:
        return initial_state.model_copy(
            update={
                "final_answer": final_answer,
                "validation": ValidationResult(passed=passed),
                "evidence": evidence,
                "feasibility": FeasibilityResult(
                    verdict=(
                        FeasibilityVerdict.FEASIBLE if feasible else FeasibilityVerdict.INFEASIBLE
                    ),
                    reason="ok" if feasible else "not answerable",
                ),
                "refusal": (
                    None
                    if not refused
                    else FeasibilityResult(verdict=FeasibilityVerdict.INFEASIBLE, reason="nope")
                ),
            }
        )

    return _run


def _fake_stream_graph(steps):
    """Build a `stream_graph_fn`-shaped fake from `[(node_name, partial_update), ...]`.

    Mirrors `adia.graph.workflow.stream_graph`'s own contract: each partial is folded onto a
    running copy of the initial state, and `(node_name, partial, state_so_far)` is yielded.
    """

    def _stream(initial_state: AgentState):
        state = initial_state
        for node_name, partial in steps:
            state = state.model_copy(update=partial)
            yield node_name, partial, state

    return _stream


def _fake_stream_graph_raising(exc: Exception):
    """A `stream_graph_fn`-shaped fake that raises partway through -- never reaches the graph
    or an LLM.
    """

    def _stream(initial_state: AgentState):
        raise exc
        yield  # pragma: no cover -- unreachable, makes this a generator function

    return _stream


def _parse_sse(text: str) -> list[dict]:
    """Parse a `text/event-stream` body of `data: <json>\\n\\n` frames into a list of dicts."""
    return [json.loads(line[len("data: ") :]) for line in text.split("\n\n") if line.strip()]


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


class TestHealth:
    def test_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestChat:
    def test_happy_path_returns_structured_result(self, client):
        ev = _make_evidence("run_sql", {"query": "select 1"})
        app.dependency_overrides[get_graph_runner] = lambda: _fake_run_graph(
            f"There are 9994 rows [[{ev.id}]].", evidence={ev.id: ev}
        )

        response = client.post(
            "/chat", json={"dataset_id": "superstore", "question": "How many rows?"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["dataset_id"] == "superstore"
        assert body["question"] == "How many rows?"
        assert "9994" in body["answer"]
        assert body["validation_passed"] is True
        assert body["evidence_ids"] == [ev.id]
        assert body["tools_used"] == ["run_sql"]
        assert body["feasibility_verdict"] == "feasible"
        assert body["refused"] is False
        assert body["duration_ms"] >= 0

    def test_refused_question_is_reported(self, client):
        app.dependency_overrides[get_graph_runner] = lambda: _fake_run_graph(
            "This question cannot be answered.", feasible=False, refused=True
        )

        response = client.post(
            "/chat", json={"dataset_id": "superstore", "question": "What's the weather?"}
        )

        assert response.status_code == 200
        body = response.json()
        assert body["refused"] is True
        assert body["feasibility_verdict"] == "infeasible"

    def test_never_calls_the_real_graph_runner(self):
        def _fail_if_called(*args, **kwargs):
            raise AssertionError("the real graph runner must never be invoked in tests")

        app.dependency_overrides[get_graph_runner] = lambda: _fail_if_called
        with TestClient(app, raise_server_exceptions=False) as unsafe_client:
            response = unsafe_client.post(
                "/chat", json={"dataset_id": "superstore", "question": "How many rows?"}
            )
        app.dependency_overrides.clear()
        assert response.status_code == 500  # our fake raised, not the real graph

    def test_blank_question_is_rejected(self, client):
        response = client.post("/chat", json={"dataset_id": "superstore", "question": ""})
        assert response.status_code == 422

    def test_invalid_dataset_id_is_rejected(self, client):
        response = client.post(
            "/chat", json={"dataset_id": "../etc/passwd", "question": "How many rows?"}
        )
        assert response.status_code == 422

    def test_missing_fields_is_rejected(self, client):
        response = client.post("/chat", json={"dataset_id": "superstore"})
        assert response.status_code == 422

    def test_unexpected_failure_returns_generic_500(self):
        def _boom(*args, **kwargs):
            raise RuntimeError("db password is hunter2")

        app.dependency_overrides[get_graph_runner] = lambda: _boom
        with TestClient(app, raise_server_exceptions=False) as unsafe_client:
            response = unsafe_client.post(
                "/chat", json={"dataset_id": "superstore", "question": "How many rows?"}
            )
        app.dependency_overrides.clear()

        assert response.status_code == 500
        assert response.json() == {"detail": "Internal server error."}
        assert "hunter2" not in response.text


class TestChatStream:
    def test_feasible_run_streams_phases_evidence_then_final(self, client):
        ev = _make_evidence("profile_dataset", {"dataset_id": "superstore"})
        steps = [
            (
                "feasibility",
                {
                    "catalog": None,
                    "feasibility": FeasibilityResult(
                        verdict=FeasibilityVerdict.FEASIBLE, reason="ok"
                    ),
                },
            ),
            ("planner", {"plan": []}),
            ("execute_tools", {"evidence": {ev.id: ev}, "errors": []}),
            ("synthesizer", {"draft_answer": "draft", "rendered_answer": "draft"}),
            (
                "validation",
                {
                    "validation": ValidationResult(passed=True),
                    "final_answer": f"There are 9994 rows [[{ev.id}]].",
                },
            ),
        ]
        app.dependency_overrides[get_stream_graph_runner] = lambda: _fake_stream_graph(steps)

        response = client.post(
            "/chat/stream", json={"dataset_id": "superstore", "question": "How many rows?"}
        )

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/event-stream")
        events = _parse_sse(response.text)

        types = [e["type"] for e in events]
        assert types == ["phase", "phase", "phase", "evidence", "phase", "phase", "final"]

        assert events[0] == {
            "type": "phase",
            "node": "feasibility",
            "data": {"verdict": "feasible", "reason": "ok"},
        }
        assert events[2]["node"] == "execute_tools"
        assert events[2]["data"] == {"evidence_count": 1, "error_count": 0}
        assert events[3]["evidence"]["evidence_id"] == ev.id
        assert events[3]["evidence"]["tool"] == "profile_dataset"
        assert events[5]["data"] == {"passed": True}

        final = events[-1]
        assert final["dataset_id"] == "superstore"
        assert final["question"] == "How many rows?"
        assert "9994" in final["answer"]
        assert final["validation_passed"] is True
        assert final["tools_used"] == ["profile_dataset"]
        assert final["refused"] is False
        assert final["evidence"][0]["evidence_id"] == ev.id

    def test_refusal_run_streams_refusal_phase_then_final(self, client):
        steps = [
            (
                "feasibility",
                {
                    "catalog": None,
                    "feasibility": FeasibilityResult(
                        verdict=FeasibilityVerdict.INFEASIBLE, reason="no such column"
                    ),
                },
            ),
            (
                "refusal",
                {
                    "draft_answer": "refused",
                    "rendered_answer": "refused",
                    "refusal": FeasibilityResult(
                        verdict=FeasibilityVerdict.INFEASIBLE, reason="no such column"
                    ),
                },
            ),
            (
                "validation",
                {
                    "validation": ValidationResult(passed=True),
                    "final_answer": "This question cannot be answered.",
                },
            ),
        ]
        app.dependency_overrides[get_stream_graph_runner] = lambda: _fake_stream_graph(steps)

        response = client.post(
            "/chat/stream", json={"dataset_id": "superstore", "question": "What's the weather?"}
        )
        events = _parse_sse(response.text)

        assert [e["type"] for e in events] == ["phase", "phase", "phase", "final"]
        assert events[1]["node"] == "refusal"
        assert events[1]["data"] == {"reason": "no such column"}
        assert events[-1]["refused"] is True
        assert events[-1]["feasibility_verdict"] == "infeasible"

    def test_never_calls_the_real_graph_runner(self, client):
        def _fail_if_called(initial_state):
            raise AssertionError("the real streaming graph runner must never run in tests")
            yield  # pragma: no cover -- unreachable, makes this a generator function

        app.dependency_overrides[get_stream_graph_runner] = lambda: _fail_if_called
        response = client.post(
            "/chat/stream", json={"dataset_id": "superstore", "question": "How many rows?"}
        )
        events = _parse_sse(response.text)
        assert events == [{"type": "error", "detail": "Internal server error."}]

    def test_unexpected_failure_mid_stream_yields_error_event_not_a_broken_response(self, client):
        app.dependency_overrides[get_stream_graph_runner] = lambda: _fake_stream_graph_raising(
            RuntimeError("db password is hunter2")
        )

        response = client.post(
            "/chat/stream", json={"dataset_id": "superstore", "question": "How many rows?"}
        )

        assert response.status_code == 200  # the stream itself starts fine; the error is a frame
        events = _parse_sse(response.text)
        assert events == [{"type": "error", "detail": "Internal server error."}]
        assert "hunter2" not in response.text

    def test_blank_question_is_rejected(self, client):
        response = client.post("/chat/stream", json={"dataset_id": "superstore", "question": ""})
        assert response.status_code == 422

    def test_invalid_dataset_id_is_rejected(self, client):
        response = client.post(
            "/chat/stream", json={"dataset_id": "../etc/passwd", "question": "How many rows?"}
        )
        assert response.status_code == 422


class TestDatasetUpload:
    @pytest.fixture(autouse=True)
    def _isolated_paths(self, tmp_path):
        registry_path = tmp_path / "registry.json"
        upload_dir = tmp_path / "uploads"
        app.dependency_overrides[get_registry_path] = lambda: registry_path
        app.dependency_overrides[get_upload_dir] = lambda: upload_dir
        self.registry_path = registry_path
        self.upload_dir = upload_dir
        yield

    def test_valid_csv_is_registered(self, client):
        csv_bytes = b"a,b,c\n1,2,3\n4,5,6\n"
        response = client.post(
            "/datasets",
            data={"dataset_id": "mydata", "description": "test dataset"},
            files={"file": ("mydata.csv", csv_bytes, "text/csv")},
        )

        assert response.status_code == 201
        body = response.json()
        assert body["dataset_id"] == "mydata"
        assert body["row_count"] == 2
        assert body["column_count"] == 3

        assert (self.upload_dir / "mydata.csv").exists()
        assert self.registry_path.exists()
        assert '"dataset_id": "mydata"' in self.registry_path.read_text()

    def test_client_supplied_filename_never_used_for_the_path(self, client):
        csv_bytes = b"a,b\n1,2\n"
        response = client.post(
            "/datasets",
            data={"dataset_id": "safe_id", "description": "test"},
            files={"file": ("../../evil.csv", csv_bytes, "text/csv")},
        )

        assert response.status_code == 201
        assert (self.upload_dir / "safe_id.csv").exists()
        assert not (self.upload_dir / "evil.csv").exists()

    def test_non_csv_extension_is_rejected(self, client):
        response = client.post(
            "/datasets",
            data={"dataset_id": "mydata", "description": "test dataset"},
            files={"file": ("mydata.txt", b"not a csv", "text/plain")},
        )
        assert response.status_code == 400

    def test_duplicate_dataset_id_is_rejected(self, client):
        csv_bytes = b"a,b\n1,2\n"
        first = client.post(
            "/datasets",
            data={"dataset_id": "dup", "description": "first"},
            files={"file": ("first.csv", csv_bytes, "text/csv")},
        )
        assert first.status_code == 201

        second = client.post(
            "/datasets",
            data={"dataset_id": "dup", "description": "second"},
            files={"file": ("second.csv", csv_bytes, "text/csv")},
        )
        assert second.status_code == 409

    def test_malformed_csv_is_rejected_and_cleaned_up(self, client):
        malformed = b"\x00\x01\x02\xff\xfe\xfd\x00\x01\x02\xff\xfe\xfd"  # not valid UTF-8 text
        response = client.post(
            "/datasets",
            data={"dataset_id": "broken", "description": "test"},
            files={"file": ("broken.csv", malformed, "text/csv")},
        )
        assert response.status_code == 400
        assert not (self.upload_dir / "broken.csv").exists()

    def test_invalid_dataset_id_is_rejected(self, client):
        response = client.post(
            "/datasets",
            data={"dataset_id": "../escape", "description": "test"},
            files={"file": ("data.csv", b"a,b\n1,2\n", "text/csv")},
        )
        assert response.status_code == 422

    def test_empty_file_is_rejected(self, client):
        response = client.post(
            "/datasets",
            data={"dataset_id": "empty", "description": "test"},
            files={"file": ("empty.csv", b"", "text/csv")},
        )
        assert response.status_code == 400
