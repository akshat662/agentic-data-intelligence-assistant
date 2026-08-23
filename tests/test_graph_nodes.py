"""Tests for adia.graph.nodes.

Uses the real, registered `superstore` dataset (Phase 2B.5) rather than synthetic fixtures
where possible: these placeholder nodes are already wired to real Phase 1/2 infrastructure,
so testing them against a fake in-memory dataset would exercise less of the actual system.
"""

import pytest

from adia.data.catalog import build_catalog
from adia.data.loader import load_dataset
from adia.evidence.store import EvidenceStore
from adia.graph.nodes import (
    execute_tools_node,
    feasibility_node,
    planner_node,
    synthesizer_node,
    validation_node,
)
from adia.graph.state import create_initial_state
from adia.models.errors import ToolErrorKind
from adia.models.evidence import Evidence
from adia.models.plan import PlanStep
from adia.models.provenance import Provenance
from adia.models.state import FeasibilityResult, FeasibilityVerdict


@pytest.fixture
def superstore_catalog():
    df = load_dataset("data/superstore.csv")
    return build_catalog(df, dataset_id="superstore", source_path="data/superstore.csv")


def _mock_feasibility(verdict: FeasibilityVerdict, **kwargs):
    """A stand-in for `adia.agents.feasibility.assess_feasibility` — no LLM call, no API key."""

    def _assess(question, catalog):
        return FeasibilityResult(verdict=verdict, reason="mocked for test", **kwargs)

    return _assess


class TestFeasibilityNode:
    def test_registered_dataset_calls_feasibility_agent(self, monkeypatch):
        monkeypatch.setattr(
            "adia.graph.nodes.assess_feasibility",
            _mock_feasibility(FeasibilityVerdict.FEASIBLE),
        )
        state = create_initial_state("How many rows?", "superstore")
        update = feasibility_node(state)
        assert update["feasibility"].verdict == FeasibilityVerdict.FEASIBLE
        assert update["catalog"] is not None
        assert update["catalog"].dataset_id == "superstore"

    def test_agent_infeasible_verdict_is_passed_through(self, monkeypatch):
        monkeypatch.setattr(
            "adia.graph.nodes.assess_feasibility",
            _mock_feasibility(FeasibilityVerdict.INFEASIBLE, missing_columns=["Employee Name"]),
        )
        state = create_initial_state("Sales by employee?", "superstore")
        update = feasibility_node(state)
        assert update["feasibility"].verdict == FeasibilityVerdict.INFEASIBLE
        assert update["feasibility"].missing_columns == ["Employee Name"]
        # The dataset itself still loaded fine -- catalog is populated regardless of verdict.
        assert update["catalog"] is not None

    def test_unregistered_dataset_is_infeasible_without_calling_agent(self, monkeypatch):
        def _fail_if_called(question, catalog):
            raise AssertionError("assess_feasibility should not be called: dataset unregistered")

        monkeypatch.setattr("adia.graph.nodes.assess_feasibility", _fail_if_called)
        state = create_initial_state("How many rows?", "nonexistent_dataset")
        update = feasibility_node(state)
        assert update["feasibility"].verdict == FeasibilityVerdict.INFEASIBLE
        assert "catalog" not in update

    def test_registered_but_unloadable_dataset_is_infeasible_without_calling_agent(
        self, monkeypatch
    ):
        def _broken_loader(_path):
            raise FileNotFoundError("simulated missing file")

        def _fail_if_called(question, catalog):
            raise AssertionError("assess_feasibility should not be called: dataset unloadable")

        monkeypatch.setattr("adia.graph.nodes.load_dataset", _broken_loader)
        monkeypatch.setattr("adia.graph.nodes.assess_feasibility", _fail_if_called)
        state = create_initial_state("How many rows?", "superstore")
        update = feasibility_node(state)
        assert update["feasibility"].verdict == FeasibilityVerdict.INFEASIBLE
        assert "failed to load" in update["feasibility"].reason
        assert "catalog" not in update


class TestPlannerNode:
    def test_calls_planner_agent_when_feasible(self, superstore_catalog, monkeypatch):
        expected_plan = [
            PlanStep(
                id="step_1",
                intent="Profile it.",
                tool_family="profile_dataset",
                expected_output="stats",
                success_criteria="ok",
            )
        ]
        monkeypatch.setattr("adia.graph.nodes.create_plan", lambda *a, **k: expected_plan)
        state = create_initial_state("How many rows?", "superstore")
        state = state.model_copy(
            update={
                "catalog": superstore_catalog,
                "feasibility": FeasibilityResult(
                    verdict=FeasibilityVerdict.FEASIBLE, reason="mocked"
                ),
            }
        )
        update = planner_node(state)
        assert update["plan"] == expected_plan

    def test_produces_no_plan_without_catalog(self):
        state = create_initial_state("How many rows?", "superstore")
        assert planner_node(state) == {}

    def test_produces_no_plan_without_feasibility_result(self, superstore_catalog):
        state = create_initial_state("How many rows?", "superstore")
        state = state.model_copy(update={"catalog": superstore_catalog})
        assert planner_node(state) == {}


class TestExecuteToolsNode:
    def test_runs_profile_dataset_step(self, superstore_catalog):
        state = create_initial_state("How many rows?", "superstore")
        step = PlanStep(
            id="step_1",
            intent="Profile it.",
            tool_family="profile_dataset",
            expected_output="stats",
            success_criteria="ok",
        )
        state = state.model_copy(update={"catalog": superstore_catalog, "plan": [step]})
        update = execute_tools_node(state)
        assert len(update["evidence"]) == 1
        assert update["errors"] == []
        evidence = next(iter(update["evidence"].values()))
        assert evidence.tool == "profile_dataset"

    def test_unsupported_tool_family_produces_typed_error(self, superstore_catalog):
        state = create_initial_state("Sales by segment?", "superstore")
        step = PlanStep(
            id="step_1",
            intent="Compare.",
            tool_family="compare_groups",
            expected_output="group stats",
            success_criteria="ok",
        )
        state = state.model_copy(update={"catalog": superstore_catalog, "plan": [step]})
        update = execute_tools_node(state)
        assert update["evidence"] == {}
        assert len(update["errors"]) == 1
        assert update["errors"][0].kind == ToolErrorKind.UNKNOWN

    def test_runs_run_sql_step_with_mocked_argument_generator(
        self, superstore_catalog, monkeypatch
    ):
        from adia.tools.run_sql import RunSqlArgs

        monkeypatch.setattr(
            "adia.graph.nodes.generate_tool_arguments",
            lambda *a, **k: RunSqlArgs(query="SELECT Sales FROM superstore LIMIT 5"),
        )
        state = create_initial_state("Total sales by region?", "superstore")
        step = PlanStep(
            id="step_1",
            intent="Aggregate.",
            tool_family="run_sql",
            expected_output="rows",
            success_criteria="ok",
        )
        state = state.model_copy(update={"catalog": superstore_catalog, "plan": [step]})
        update = execute_tools_node(state)
        assert update["errors"] == []
        assert len(update["evidence"]) == 1
        evidence = next(iter(update["evidence"].values()))
        assert evidence.tool == "run_sql"

    def test_run_sql_step_with_rejected_arguments_produces_typed_error(
        self, superstore_catalog, monkeypatch
    ):
        monkeypatch.setattr("adia.graph.nodes.generate_tool_arguments", lambda *a, **k: None)
        state = create_initial_state("Total sales by region?", "superstore")
        step = PlanStep(
            id="step_1",
            intent="Aggregate.",
            tool_family="run_sql",
            expected_output="rows",
            success_criteria="ok",
        )
        state = state.model_copy(update={"catalog": superstore_catalog, "plan": [step]})
        update = execute_tools_node(state)
        assert update["evidence"] == {}
        assert len(update["errors"]) == 1
        assert update["errors"][0].kind == ToolErrorKind.VALIDATION

    def test_no_plan_is_a_no_op(self, superstore_catalog):
        state = create_initial_state("How many rows?", "superstore")
        state = state.model_copy(update={"catalog": superstore_catalog})
        assert execute_tools_node(state) == {}

    def test_tool_failure_is_recorded_as_a_typed_error(self, superstore_catalog):
        broken_catalog = superstore_catalog.model_copy(
            update={"source_path": "/no/such/file.csv"}
        )
        step = PlanStep(
            id="step_1",
            intent="Profile it.",
            tool_family="profile_dataset",
            expected_output="stats",
            success_criteria="ok",
        )
        state = create_initial_state("How many rows?", "superstore")
        state = state.model_copy(update={"catalog": broken_catalog, "plan": [step]})
        update = execute_tools_node(state)
        assert update["evidence"] == {}
        assert len(update["errors"]) == 1
        assert update["errors"][0].kind == ToolErrorKind.NOT_FOUND

    def test_seeds_from_existing_evidence(self, superstore_catalog):
        args = {"dataset_id": "superstore", "source_path": "data/superstore.csv", "top_k": 5}
        from adia.evidence.ids import compute_args_hash, generate_evidence_id

        existing = Evidence(
            id=generate_evidence_id("profile_dataset", args),
            tool="profile_dataset",
            data={"row_count": 9994},
            provenance=Provenance(
                tool_name="profile_dataset", args=args, args_hash=compute_args_hash(args)
            ),
        )
        step = PlanStep(
            id="step_1",
            intent="Profile it.",
            tool_family="profile_dataset",
            expected_output="stats",
            success_criteria="ok",
        )
        state = create_initial_state("How many rows?", "superstore")
        state = state.model_copy(
            update={
                "catalog": superstore_catalog,
                "plan": [step],
                "evidence": {existing.id: existing},
            }
        )
        update = execute_tools_node(state)
        # The real profile_dataset call resolves to the same content-addressed ID and is a
        # cache hit: the pre-seeded record (row_count=9994) is what comes back, not a fresh
        # recomputation, proving the seed actually took effect rather than being ignored.
        assert existing.id in update["evidence"]
        assert update["evidence"][existing.id].data["row_count"] == 9994


class TestSynthesizerNode:
    def test_no_evidence_produces_honest_placeholder(self):
        state = create_initial_state("How many rows?", "superstore")
        update = synthesizer_node(state)
        assert "No evidence was collected" in update["draft_answer"]
        assert update["draft_answer"] == update["rendered_answer"]

    def test_evidence_produces_citation_bearing_draft(self, superstore_catalog):
        state = create_initial_state("How many rows?", "superstore")
        state = state.model_copy(update={"catalog": superstore_catalog})
        store = EvidenceStore()
        from adia.tools.profile_dataset import profile_dataset

        result = profile_dataset("superstore", "data/superstore.csv", store)
        evidence = {result.evidence_id: store.get(result.evidence_id)}
        state = state.model_copy(update={"evidence": evidence})
        update = synthesizer_node(state)
        assert f"[[{result.evidence_id}]]" in update["draft_answer"]

    def test_evidence_with_no_scalar_values_is_reported_honestly(self):
        args = {"dataset_id": "superstore", "source_path": "data/superstore.csv"}
        from adia.evidence.ids import compute_args_hash, generate_evidence_id

        empty_evidence = Evidence(
            id=generate_evidence_id("profile_dataset", args),
            tool="profile_dataset",
            data={},
            provenance=Provenance(
                tool_name="profile_dataset", args=args, args_hash=compute_args_hash(args)
            ),
        )
        state = create_initial_state("How many rows?", "superstore")
        state = state.model_copy(update={"evidence": {empty_evidence.id: empty_evidence}})
        update = synthesizer_node(state)
        assert "reported no scalar values" in update["draft_answer"]


class TestValidationNode:
    def test_grounded_answer_passes_and_sets_final_answer(self):
        state = create_initial_state("How many rows?", "superstore")
        state = state.model_copy(update={"rendered_answer": "No numeric claims here."})
        update = validation_node(state)
        assert update["validation"].passed is True
        assert update["final_answer"] == "No numeric claims here."

    def test_ungrounded_answer_fails_and_clears_final_answer(self):
        state = create_initial_state("How many rows?", "superstore")
        state = state.model_copy(update={"rendered_answer": "Revenue was 999999.99."})
        update = validation_node(state)
        assert update["validation"].passed is False
        assert update["final_answer"] is None

    def test_handles_missing_rendered_answer(self):
        state = create_initial_state("How many rows?", "superstore")
        update = validation_node(state)
        assert update["validation"].passed is True
