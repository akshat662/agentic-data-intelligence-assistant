"""Tests for adia.graph.workflow and adia.graph.state."""

from langgraph.graph.state import CompiledStateGraph

from adia.graph.state import create_initial_state
from adia.graph.workflow import build_graph, run_graph
from adia.models.state import AgentState, FeasibilityVerdict


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

    def test_graph_has_all_five_nodes(self):
        graph = build_graph()
        node_names = set(graph.get_graph().nodes) - {"__start__", "__end__"}
        assert node_names == {
            "feasibility",
            "planner",
            "execute_tools",
            "synthesizer",
            "validation",
        }


class TestRunGraphEndToEnd:
    def test_returns_a_typed_agent_state(self):
        result = run_graph(create_initial_state("How many rows are there?", "superstore"))
        assert isinstance(result, AgentState)

    def test_feasible_run_produces_grounded_final_answer(self):
        result = run_graph(create_initial_state("How many rows are there?", "superstore"))
        assert result.feasibility.verdict == FeasibilityVerdict.FEASIBLE
        assert result.catalog is not None
        assert len(result.plan) == 1
        assert len(result.evidence) == 1
        assert result.errors == []
        assert result.validation is not None
        assert result.validation.passed is True
        assert result.final_answer is not None

    def test_state_carries_the_original_question_through_every_node(self):
        question = "What does the Superstore dataset contain?"
        result = run_graph(create_initial_state(question, "superstore"))
        assert result.question == question

    def test_validation_node_is_always_called(self):
        result = run_graph(create_initial_state("How many rows?", "superstore"))
        assert result.validation is not None

    def test_draft_answer_cites_the_evidence_it_produced(self):
        result = run_graph(create_initial_state("How many rows?", "superstore"))
        evidence_id = next(iter(result.evidence))
        assert f"[[{evidence_id}]]" in result.draft_answer

    def test_infeasible_dataset_does_not_crash_and_still_validates(self):
        result = run_graph(create_initial_state("How many rows?", "nonexistent_dataset"))
        assert result.feasibility.verdict == FeasibilityVerdict.INFEASIBLE
        assert result.catalog is None
        assert result.plan == []
        assert result.evidence == {}
        assert result.validation is not None
        assert result.validation.passed is True
        assert result.final_answer is not None  # the honest "no evidence" placeholder

    def test_repeated_runs_are_deterministic_in_content(self):
        result1 = run_graph(create_initial_state("How many rows?", "superstore", run_id="r"))
        result2 = run_graph(create_initial_state("How many rows?", "superstore", run_id="r"))
        assert result1.draft_answer == result2.draft_answer
        assert result1.validation == result2.validation
