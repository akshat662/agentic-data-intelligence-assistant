"""Tests for adia.graph.workflow and adia.graph.state.

Every end-to-end run below mocks `adia.graph.nodes.assess_feasibility` and, where a plan is
needed, `adia.graph.nodes.create_plan` — no real OpenAI call is made, and no OPENAI_API_KEY
is required to run this suite.
"""

from langgraph.graph.state import CompiledStateGraph

from adia.graph.state import create_initial_state
from adia.graph.workflow import build_graph, run_graph
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
        assert result.final_answer is not None  # the honest "no evidence" placeholder

    def test_agent_infeasible_verdict_still_completes_the_graph(self, monkeypatch):
        # The graph stays strictly linear in this phase (no conditional routing on
        # feasibility yet) -- an INFEASIBLE agent verdict must not crash or skip nodes. No
        # planner mock is needed: create_plan itself refuses to plan for a non-FEASIBLE
        # verdict, without calling the LLM at all -- the empty plan below is that guard
        # firing, not a fallback from a missing API key.
        _mock_infeasible(monkeypatch, missing_capabilities=["forecasting"])
        result = run_graph(create_initial_state("Will sales grow next year?", "superstore"))
        assert result.feasibility.verdict == FeasibilityVerdict.INFEASIBLE
        assert result.feasibility.missing_capabilities == ["forecasting"]
        assert result.plan == []
        assert result.validation is not None
        assert result.validation.passed is True

    def test_graph_execution_with_mocked_planner_produces_expected_plan(self, monkeypatch):
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        result = run_graph(create_initial_state("How many rows?", "superstore"))
        assert len(result.plan) == 1
        assert result.plan[0].tool_family == "profile_dataset"
        assert len(result.evidence) == 1  # execute_tools_node ran the mocked plan for real

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
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        result1 = run_graph(create_initial_state("How many rows?", "superstore", run_id="r"))
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        result2 = run_graph(create_initial_state("How many rows?", "superstore", run_id="r"))
        assert result1.draft_answer == result2.draft_answer
        assert result1.validation == result2.validation
