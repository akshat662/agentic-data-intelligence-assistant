"""Tests for the FastAPI demo layer. No real graph run (and therefore no real LLM call) is ever
made here -- `/chat` tests override `get_graph_runner` with a fake, exactly like
`tests/test_cli.py` overrides `run_graph_fn`.
"""

import pytest
from fastapi.testclient import TestClient

from adia.api.app import app
from adia.api.service import get_graph_runner, get_registry_path, get_upload_dir
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
