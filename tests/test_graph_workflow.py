"""Tests for adia.graph.workflow and adia.graph.state.

Every end-to-end run below mocks `adia.graph.nodes.assess_feasibility` and, where a plan is
needed, `adia.graph.nodes.create_plan` — no real OpenAI call is made, and no OPENAI_API_KEY
is required to run this suite.
"""

from langgraph.graph.state import CompiledStateGraph

from adia.graph.nodes import VALIDATION_FALLBACK_ANSWER
from adia.graph.state import create_initial_state
from adia.graph.workflow import build_graph, route_after_feasibility, run_graph, stream_graph
from adia.models.plan import PlanStep
from adia.models.state import AgentState, FeasibilityResult, FeasibilityVerdict


def _mock_feasible(monkeypatch):
    """Patch the feasibility agent to always report FEASIBLE, with no LLM call at all."""

    def _assess(question, catalog):
        return FeasibilityResult(verdict=FeasibilityVerdict.FEASIBLE, reason="mocked for test")

    monkeypatch.setattr("adia.graph.nodes.assess_feasibility", _assess)


def _mock_planner(monkeypatch, *, tool_family="profile_dataset"):
    """Patch the planner agent to always return one fixed step, with no LLM call at all."""

    def _plan(question, catalog, feasibility):
        return [
            PlanStep(
                id="step_1",
                intent="Mocked plan step for test.",
                tool_family=tool_family,
                expected_output="stats",
                success_criteria="ok",
            )
        ]

    monkeypatch.setattr("adia.graph.nodes.create_plan", _plan)


def _mock_synthesizer(monkeypatch):
    """Patch the synthesizer agent to deterministically cite whatever evidence it's given, with
    no LLM call at all -- avoids asserting byte-for-byte equality against real LLM prose, which
    is not guaranteed to be identical between two separate calls even at temperature=0.
    """

    def _synthesize(question, context, evidence):
        evidence_id = next(iter(evidence))
        return f"Mocked answer citing [[{evidence_id}]]."

    monkeypatch.setattr("adia.graph.nodes.synthesize_answer", _synthesize)


def _fail_if_planner_called(monkeypatch):
    """Patch the planner agent to blow up if it's ever invoked -- proves routing skipped it."""

    def _plan(question, catalog, feasibility):
        raise AssertionError("create_plan should not be called on the refusal path")

    monkeypatch.setattr("adia.graph.nodes.create_plan", _plan)


def _mock_infeasible(monkeypatch, **kwargs):
    """Patch the feasibility agent to always report INFEASIBLE, with no LLM call at all."""

    def _assess(question, catalog):
        return FeasibilityResult(
            verdict=FeasibilityVerdict.INFEASIBLE, reason="mocked for test", **kwargs
        )

    monkeypatch.setattr("adia.graph.nodes.assess_feasibility", _assess)


class TestCreateInitialState:
    def test_generates_a_run_id_when_omitted(self):
        state = create_initial_state("q?", "superstore")
        assert state.run_id.startswith("run_")

    def test_uses_explicit_run_id(self):
        state = create_initial_state("q?", "superstore", run_id="fixed_id")
        assert state.run_id == "fixed_id"

    def test_two_calls_get_different_run_ids(self):
        a = create_initial_state("q?", "superstore")
        b = create_initial_state("q?", "superstore")
        assert a.run_id != b.run_id

    def test_budget_caps_are_set(self):
        state = create_initial_state("q?", "superstore", max_llm_calls=3, max_tool_calls=7)
        assert state.budget.max_llm_calls == 3
        assert state.budget.max_tool_calls == 7


class TestBuildGraph:
    def test_returns_a_compiled_graph(self):
        graph = build_graph()
        assert isinstance(graph, CompiledStateGraph)

    def test_graph_has_all_six_nodes(self):
        graph = build_graph()
        node_names = set(graph.get_graph().nodes) - {"__start__", "__end__"}
        assert node_names == {
            "feasibility",
            "planner",
            "execute_tools",
            "synthesizer",
            "validation",
            "refusal",
        }


class TestRouteAfterFeasibility:
    def test_feasible_verdict_routes_to_planner(self):
        state = create_initial_state("q?", "superstore")
        state = state.model_copy(
            update={
                "feasibility": FeasibilityResult(
                    verdict=FeasibilityVerdict.FEASIBLE, reason="ok"
                )
            }
        )
        assert route_after_feasibility(state) == "planner"

    def test_infeasible_verdict_routes_to_refusal(self):
        state = create_initial_state("q?", "superstore")
        state = state.model_copy(
            update={
                "feasibility": FeasibilityResult(
                    verdict=FeasibilityVerdict.INFEASIBLE, reason="no"
                )
            }
        )
        assert route_after_feasibility(state) == "refusal"

    def test_needs_clarification_verdict_routes_to_refusal(self):
        state = create_initial_state("q?", "superstore")
        state = state.model_copy(
            update={
                "feasibility": FeasibilityResult(
                    verdict=FeasibilityVerdict.NEEDS_CLARIFICATION, reason="ambiguous"
                )
            }
        )
        assert route_after_feasibility(state) == "refusal"

    def test_missing_feasibility_result_routes_to_refusal(self):
        state = create_initial_state("q?", "superstore")
        assert route_after_feasibility(state) == "refusal"


class TestRunGraphEndToEnd:
    def test_returns_a_typed_agent_state(self, monkeypatch):
        _mock_feasible(monkeypatch)
        result = run_graph(create_initial_state("How many rows are there?", "superstore"))
        assert isinstance(result, AgentState)

    def test_feasible_run_produces_grounded_final_answer(self, monkeypatch):
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        result = run_graph(create_initial_state("How many rows are there?", "superstore"))
        assert result.feasibility.verdict == FeasibilityVerdict.FEASIBLE
        assert result.catalog is not None
        assert len(result.plan) == 1
        assert len(result.evidence) == 1
        assert result.errors == []
        assert result.validation is not None
        assert result.validation.passed is True
        assert result.final_answer is not None

    def test_state_carries_the_original_question_through_every_node(self, monkeypatch):
        _mock_feasible(monkeypatch)
        question = "What does the Superstore dataset contain?"
        result = run_graph(create_initial_state(question, "superstore"))
        assert result.question == question

    def test_validation_node_is_always_called(self, monkeypatch):
        _mock_feasible(monkeypatch)
        result = run_graph(create_initial_state("How many rows?", "superstore"))
        assert result.validation is not None

    def test_draft_answer_cites_the_evidence_it_produced(self, monkeypatch):
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        result = run_graph(create_initial_state("How many rows?", "superstore"))
        evidence_id = next(iter(result.evidence))
        assert f"[[{evidence_id}]]" in result.draft_answer

    def test_infeasible_dataset_does_not_crash_and_still_validates(self):
        # No mocking needed: an unregistered dataset short-circuits before the feasibility
        # agent is ever called (see test_graph_nodes.py), so this exercises the real path.
        result = run_graph(create_initial_state("How many rows?", "nonexistent_dataset"))
        assert result.feasibility.verdict == FeasibilityVerdict.INFEASIBLE
        assert result.catalog is None
        assert result.plan == []
        assert result.evidence == {}
        assert result.validation is not None
        assert result.validation.passed is True
        assert result.final_answer is not None
        assert result.refusal is not None
        assert "cannot be answered" in result.final_answer

    def test_feasible_question_follows_planner_path(self, monkeypatch):
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        result = run_graph(create_initial_state("How many rows?", "superstore"))
        assert result.feasibility.verdict == FeasibilityVerdict.FEASIBLE
        assert len(result.plan) == 1
        assert result.plan[0].tool_family == "profile_dataset"
        assert len(result.evidence) == 1  # execute_tools_node ran the mocked plan for real
        assert result.refusal is None

    def test_infeasible_question_skips_planner_entirely(self, monkeypatch):
        # Proves the *routing* skips planner_node, not just that create_plan happens to
        # refuse internally -- create_plan would raise if it were ever called at all.
        _mock_infeasible(monkeypatch, missing_capabilities=["forecasting"])
        _fail_if_planner_called(monkeypatch)
        result = run_graph(create_initial_state("Will sales grow next year?", "superstore"))
        assert result.feasibility.verdict == FeasibilityVerdict.INFEASIBLE
        assert result.feasibility.missing_capabilities == ["forecasting"]
        assert result.plan == []
        assert result.evidence == {}
        assert result.validation is not None
        assert result.validation.passed is True
        assert result.refusal is not None
        assert result.refusal.missing_capabilities == ["forecasting"]

    def test_validation_failure_produces_deterministic_fallback(self, monkeypatch):
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        # An ungrounded synthesizer proposal -- a bare numeral with no citation -- must never
        # reach final_answer; validation_node must catch it and substitute the fixed fallback.
        monkeypatch.setattr(
            "adia.graph.nodes.synthesize_answer",
            lambda question, context, evidence: "Revenue was 999999.99.",
        )
        result = run_graph(create_initial_state("How many rows?", "superstore"))
        assert result.validation is not None
        assert result.validation.passed is False
        assert result.final_answer == VALIDATION_FALLBACK_ANSWER

    def test_planner_proposing_unsupported_tool_family_is_caught_downstream(self, monkeypatch):
        # create_plan itself would reject an unsupported tool_family, but this proves
        # execute_tools_node's own guard also holds if a plan step ever slips through with
        # one -- defense in depth, not reliance on a single validation point.
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch, tool_family="delete_dataset")
        result = run_graph(create_initial_state("How many rows?", "superstore"))
        assert len(result.plan) == 1
        assert result.evidence == {}
        assert len(result.errors) == 1

    def test_repeated_runs_are_deterministic_in_content(self, monkeypatch):
        # Mocks the synthesizer too, not just feasibility/planner: a real LLM call is not
        # guaranteed to return byte-identical prose across two separate calls even at
        # temperature=0, which made this assertion flake whenever a real OPENAI_API_KEY was
        # set in the environment running the suite.
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        _mock_synthesizer(monkeypatch)
        result1 = run_graph(create_initial_state("How many rows?", "superstore", run_id="r"))
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        _mock_synthesizer(monkeypatch)
        result2 = run_graph(create_initial_state("How many rows?", "superstore", run_id="r"))
        assert result1.draft_answer == result2.draft_answer
        assert result1.validation == result2.validation


class TestStreamGraph:
    def test_feasible_run_yields_every_node_in_order(self, monkeypatch):
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        nodes = [
            node_name
            for node_name, _, _ in stream_graph(
                create_initial_state("How many rows?", "superstore")
            )
        ]
        assert nodes == ["feasibility", "planner", "execute_tools", "synthesizer", "validation"]

    def test_refusal_run_skips_planner_and_execute_tools(self, monkeypatch):
        _mock_infeasible(monkeypatch)
        _fail_if_planner_called(monkeypatch)
        nodes = [
            node_name
            for node_name, _, _ in stream_graph(
                create_initial_state("How many rows?", "superstore")
            )
        ]
        assert nodes == ["feasibility", "refusal", "validation"]

    def test_each_partial_matches_the_node_that_produced_it(self, monkeypatch):
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        updates = dict(
            (node_name, set(partial))
            for node_name, partial, _ in stream_graph(
                create_initial_state("How many rows?", "superstore")
            )
        )
        assert updates["feasibility"] == {"catalog", "feasibility"}
        assert updates["planner"] == {"plan"}
        assert updates["execute_tools"] == {"evidence", "errors"}
        assert updates["synthesizer"] == {"draft_answer", "rendered_answer"}
        assert updates["validation"] == {"validation", "final_answer"}

    def test_last_state_so_far_matches_run_graph(self, monkeypatch):
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        _mock_synthesizer(monkeypatch)
        state = create_initial_state("How many rows?", "superstore", run_id="stream-run")
        *_, (last_node, _, last_state) = stream_graph(state)

        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        _mock_synthesizer(monkeypatch)
        invoked = run_graph(
            create_initial_state("How many rows?", "superstore", run_id="stream-run")
        )

        assert last_node == "validation"
        assert last_state.final_answer == invoked.final_answer
        assert last_state.validation == invoked.validation
        assert set(last_state.evidence) == set(invoked.evidence)

    def test_state_so_far_accumulates_across_nodes(self, monkeypatch):
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        by_node = {
            node_name: state
            for node_name, _, state in stream_graph(
                create_initial_state("How many rows?", "superstore")
            )
        }
        # By the time planner completes, feasibility's own result is still visible.
        assert by_node["planner"].feasibility is not None
        assert by_node["planner"].feasibility.verdict == FeasibilityVerdict.FEASIBLE
        # By the time validation completes, everything upstream is still visible too.
        assert len(by_node["validation"].plan) == 1
        assert len(by_node["validation"].evidence) == 1
