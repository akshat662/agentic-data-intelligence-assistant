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
    VALIDATION_FALLBACK_ANSWER,
    _topological_order,
    execute_tools_node,
    feasibility_node,
    planner_node,
    refusal_node,
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


class TestTopologicalOrder:
    def _step(self, step_id: str, depends_on: list[str] | None = None) -> PlanStep:
        return PlanStep(
            id=step_id,
            intent="do something",
            tool_family="run_sql",
            depends_on=depends_on or [],
            expected_output="rows",
            success_criteria="ok",
        )

    def test_single_step_no_dependencies(self):
        a = self._step("a")
        ordered, errors = _topological_order([a])
        assert ordered == [a]
        assert errors == []

    def test_orders_dependent_step_after_its_dependency_even_when_listed_first(self):
        a = self._step("a")
        b = self._step("b", depends_on=["a"])
        ordered, errors = _topological_order([b, a])  # deliberately out of order
        assert [step.id for step in ordered] == ["a", "b"]
        assert errors == []

    def test_independent_steps_preserve_original_relative_order(self):
        a, b, c = self._step("a"), self._step("b"), self._step("c")
        ordered, errors = _topological_order([a, b, c])
        assert [step.id for step in ordered] == ["a", "b", "c"]
        assert errors == []

    def test_diamond_dependencies_ordered_correctly(self):
        a = self._step("a")
        b = self._step("b", depends_on=["a"])
        c = self._step("c", depends_on=["a"])
        d = self._step("d", depends_on=["b", "c"])
        ordered, errors = _topological_order([d, c, b, a])
        ids = [step.id for step in ordered]
        assert ids[0] == "a"
        assert ids[-1] == "d"
        assert set(ids[1:3]) == {"b", "c"}
        assert errors == []

    def test_two_step_cycle_produces_typed_error_for_both_steps(self):
        a = self._step("a", depends_on=["b"])
        b = self._step("b", depends_on=["a"])
        ordered, errors = _topological_order([a, b])
        assert ordered == []
        assert len(errors) == 2
        assert all(e.kind == ToolErrorKind.VALIDATION for e in errors)

    def test_unknown_dependency_produces_typed_error(self):
        a = self._step("a", depends_on=["ghost"])
        ordered, errors = _topological_order([a])
        assert ordered == []
        assert len(errors) == 1
        assert errors[0].kind == ToolErrorKind.VALIDATION
        assert "ghost" in errors[0].message

    def test_partial_cycle_still_orders_the_valid_portion(self):
        a = self._step("a")
        b = self._step("b", depends_on=["c"])
        c = self._step("c", depends_on=["b"])
        ordered, errors = _topological_order([a, b, c])
        assert [step.id for step in ordered] == ["a"]
        assert len(errors) == 2
        assert {e.kind for e in errors} == {ToolErrorKind.VALIDATION}


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
            tool_family="magic_tool",
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

    def test_dependent_step_receives_dependency_evidence_even_when_listed_first(
        self, superstore_catalog, monkeypatch
    ):
        from adia.tools.run_sql import RunSqlArgs

        captured_contexts: dict[str, str] = {}

        def _fake_generate_tool_arguments(
            step, catalog, dataset_id, *, dependency_context="", **kwargs
        ):
            captured_contexts[step.id] = dependency_context
            # Distinct queries per step -- identical queries would resolve to the same
            # content-addressed evidence ID and collapse into a single cached record.
            limit = 1 if step.id == "anchor" else 2
            return RunSqlArgs(query=f"SELECT Sales FROM superstore LIMIT {limit}")

        monkeypatch.setattr(
            "adia.graph.nodes.generate_tool_arguments", _fake_generate_tool_arguments
        )

        anchor = PlanStep(
            id="anchor",
            intent="Find the category with the lowest total Sales.",
            tool_family="run_sql",
            expected_output="rows",
            success_criteria="ok",
        )
        dependent = PlanStep(
            id="dependent",
            intent="Check order volume for that category.",
            tool_family="run_sql",
            depends_on=["anchor"],
            expected_output="rows",
            success_criteria="ok",
        )
        # Listed dependent-first, deliberately out of dependency order.
        state = create_initial_state("Why is that category lowest?", "superstore")
        state = state.model_copy(
            update={"catalog": superstore_catalog, "plan": [dependent, anchor]}
        )
        update = execute_tools_node(state)

        assert update["errors"] == []
        assert len(update["evidence"]) == 2
        assert captured_contexts["anchor"] == ""
        assert captured_contexts["dependent"] != ""

    def test_step_with_no_dependencies_gets_empty_dependency_context(
        self, superstore_catalog, monkeypatch
    ):
        from adia.tools.run_sql import RunSqlArgs

        captured = {}

        def _fake_generate_tool_arguments(
            step, catalog, dataset_id, *, dependency_context="", **kwargs
        ):
            captured["dependency_context"] = dependency_context
            return RunSqlArgs(query="SELECT Sales FROM superstore LIMIT 1")

        monkeypatch.setattr(
            "adia.graph.nodes.generate_tool_arguments", _fake_generate_tool_arguments
        )
        step = PlanStep(
            id="step_1",
            intent="Aggregate.",
            tool_family="run_sql",
            expected_output="rows",
            success_criteria="ok",
        )
        state = create_initial_state("Total sales?", "superstore")
        state = state.model_copy(update={"catalog": superstore_catalog, "plan": [step]})
        execute_tools_node(state)
        assert captured["dependency_context"] == ""

    def test_cyclic_plan_produces_typed_errors_without_crash(self, superstore_catalog):
        step_a = PlanStep(
            id="a",
            intent="a",
            tool_family="run_sql",
            depends_on=["b"],
            expected_output="rows",
            success_criteria="ok",
        )
        step_b = PlanStep(
            id="b",
            intent="b",
            tool_family="run_sql",
            depends_on=["a"],
            expected_output="rows",
            success_criteria="ok",
        )
        state = create_initial_state("Why?", "superstore")
        state = state.model_copy(
            update={"catalog": superstore_catalog, "plan": [step_a, step_b]}
        )
        update = execute_tools_node(state)
        assert update["evidence"] == {}
        assert len(update["errors"]) == 2
        assert all(e.kind == ToolErrorKind.VALIDATION for e in update["errors"])


class TestSynthesizerNode:
    def test_no_evidence_produces_honest_placeholder(self):
        # No evidence short-circuits inside synthesize_answer itself, before any LLM call, so
        # this is safe to run without mocking anything.
        state = create_initial_state("How many rows?", "superstore")
        update = synthesizer_node(state)
        assert "No evidence was collected" in update["draft_answer"]
        assert update["draft_answer"] == update["rendered_answer"]

    def test_calls_synthesizer_agent_with_rendered_context(self, superstore_catalog, monkeypatch):
        store = EvidenceStore()
        from adia.tools.profile_dataset import profile_dataset

        result = profile_dataset("superstore", "data/superstore.csv", store)
        evidence = {result.evidence_id: store.get(result.evidence_id)}
        expected_answer = f"The dataset has been profiled [[{result.evidence_id}]]."

        captured = {}

        def _fake_synthesize(question, evidence_context, evidence_map):
            captured["question"] = question
            captured["evidence_context"] = evidence_context
            captured["evidence_map"] = evidence_map
            return expected_answer

        monkeypatch.setattr("adia.graph.nodes.synthesize_answer", _fake_synthesize)
        state = create_initial_state("How many rows?", "superstore")
        state = state.model_copy(update={"catalog": superstore_catalog, "evidence": evidence})
        update = synthesizer_node(state)

        assert update["draft_answer"] == expected_answer
        assert update["draft_answer"] == update["rendered_answer"]
        assert captured["question"] == "How many rows?"
        assert result.evidence_id in captured["evidence_context"]
        assert captured["evidence_map"] == evidence


class TestValidationNode:
    def test_grounded_answer_passes_and_sets_final_answer(self):
        state = create_initial_state("How many rows?", "superstore")
        state = state.model_copy(update={"rendered_answer": "No numeric claims here."})
        update = validation_node(state)
        assert update["validation"].passed is True
        assert update["final_answer"] == "No numeric claims here."

    def test_ungrounded_answer_fails_and_produces_deterministic_fallback(self):
        state = create_initial_state("How many rows?", "superstore")
        state = state.model_copy(update={"rendered_answer": "Revenue was 999999.99."})
        update = validation_node(state)
        assert update["validation"].passed is False
        assert update["final_answer"] == VALIDATION_FALLBACK_ANSWER

    def test_handles_missing_rendered_answer(self):
        state = create_initial_state("How many rows?", "superstore")
        update = validation_node(state)
        assert update["validation"].passed is True


class TestRefusalNode:
    def test_composes_refusal_from_feasibility_reason(self):
        state = create_initial_state("Will sales grow next year?", "superstore")
        state = state.model_copy(
            update={
                "feasibility": FeasibilityResult(
                    verdict=FeasibilityVerdict.INFEASIBLE,
                    reason="No forecasting data exists.",
                )
            }
        )
        update = refusal_node(state)
        assert "No forecasting data exists." in update["draft_answer"]
        assert update["draft_answer"] == update["rendered_answer"]
        assert update["refusal"].verdict == FeasibilityVerdict.INFEASIBLE

    def test_includes_missing_columns_and_capabilities(self):
        state = create_initial_state("Sales by employee?", "superstore")
        state = state.model_copy(
            update={
                "feasibility": FeasibilityResult(
                    verdict=FeasibilityVerdict.INFEASIBLE,
                    reason="Column does not exist.",
                    missing_columns=["Employee Name"],
                    missing_capabilities=["forecasting"],
                )
            }
        )
        update = refusal_node(state)
        assert "Employee Name" in update["draft_answer"]
        assert "forecasting" in update["draft_answer"]

    def test_no_feasibility_result_is_a_no_op(self):
        state = create_initial_state("How many rows?", "superstore")
        assert refusal_node(state) == {}

    def test_refusal_answer_passes_static_validation(self):
        state = create_initial_state("Will sales grow next year?", "superstore")
        state = state.model_copy(
            update={
                "feasibility": FeasibilityResult(
                    verdict=FeasibilityVerdict.INFEASIBLE,
                    reason="No forecasting data exists.",
                )
            }
        )
        update = refusal_node(state)
        state = state.model_copy(update=update)
        validation_update = validation_node(state)
        assert validation_update["validation"].passed is True
        assert validation_update["final_answer"] == update["rendered_answer"]
