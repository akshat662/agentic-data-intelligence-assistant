"""Tests for bench.runner.

Every test here mocks `adia.graph.nodes.assess_feasibility` and, where a plan is needed,
`adia.graph.nodes.create_plan` -- the same pattern `test_graph_workflow.py` uses -- so no real
OpenAI call is ever made and no `OPENAI_API_KEY` is required to run this suite.
"""

import json

import pytest
from bench.runner import QuestionResult, run_benchmark, run_question, save_results
from bench.schema import BenchmarkQuestion

from adia.models.plan import PlanStep
from adia.models.state import FeasibilityResult, FeasibilityVerdict


def _mock_feasible(monkeypatch):
    def _assess(question, catalog):
        return FeasibilityResult(verdict=FeasibilityVerdict.FEASIBLE, reason="mocked for test")

    monkeypatch.setattr("adia.graph.nodes.assess_feasibility", _assess)


def _mock_infeasible(monkeypatch):
    def _assess(question, catalog):
        return FeasibilityResult(
            verdict=FeasibilityVerdict.INFEASIBLE, reason="mocked infeasible for test"
        )

    monkeypatch.setattr("adia.graph.nodes.assess_feasibility", _assess)


def _mock_planner(monkeypatch, *, tool_family="profile_dataset"):
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


def _mock_partial_plan(monkeypatch):
    """Two-step plan where only the first step's tool_family is supported -- exercises
    partial evidence_coverage without needing any LLM/ArgGen call."""

    def _plan(question, catalog, feasibility):
        return [
            PlanStep(
                id="step_1",
                intent="Supported step.",
                tool_family="profile_dataset",
                expected_output="stats",
                success_criteria="ok",
            ),
            PlanStep(
                id="step_2",
                intent="Unsupported step.",
                tool_family="magic_tool",
                expected_output="n/a",
                success_criteria="ok",
            ),
        ]

    monkeypatch.setattr("adia.graph.nodes.create_plan", _plan)


def _question(**overrides) -> BenchmarkQuestion:
    defaults = dict(
        id="q001",
        dataset_id="superstore",
        category="descriptive",
        question="How many rows are there?",
        answerable=True,
        gold_tools=["profile_dataset"],
    )
    defaults.update(overrides)
    return BenchmarkQuestion(**defaults)


class TestRunQuestion:
    def test_feasible_question_records_success_and_tools_used(self, monkeypatch):
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        result = run_question(_question())
        assert result.success is True
        assert result.feasibility_verdict == "feasible"
        assert result.tools_used == ["profile_dataset"]
        assert result.validation_passed is True
        assert result.final_answer is not None
        assert result.error is None
        assert result.duration_ms >= 0
        assert result.plan_step_count == 1
        assert result.executed_step_count == 1
        assert result.evidence_count == 1
        assert result.evidence_coverage == 1.0

    def test_unanswerable_question_records_refusal_without_tools(self, monkeypatch):
        _mock_infeasible(monkeypatch)
        result = run_question(
            _question(
                id="q018",
                category="unanswerable",
                question="Will next quarter's revenue increase?",
                answerable=False,
                gold_tools=[],
            )
        )
        assert result.success is True
        assert result.feasibility_verdict == "infeasible"
        assert result.tools_used == []
        assert result.final_answer is not None
        # Refusal routes around the planner entirely -- there is no plan to have a coverage of.
        assert result.plan_step_count == 0
        assert result.executed_step_count == 0
        assert result.evidence_count == 0
        assert result.evidence_coverage is None

    def test_partial_plan_execution_reports_fractional_coverage(self, monkeypatch):
        _mock_feasible(monkeypatch)
        _mock_partial_plan(monkeypatch)
        result = run_question(_question())
        assert result.plan_step_count == 2
        assert result.executed_step_count == 1
        assert result.evidence_count == 1
        assert result.evidence_coverage == 0.5

    def test_unregistered_dataset_does_not_crash_the_runner(self):
        result = run_question(_question(dataset_id="nonexistent_dataset"))
        assert result.success is True
        assert result.feasibility_verdict == "infeasible"

    def test_unhandled_exception_is_caught_and_recorded(self, monkeypatch):
        def _boom(initial_state):
            raise RuntimeError("simulated graph crash")

        monkeypatch.setattr("bench.runner.run_graph", _boom)
        result = run_question(_question())
        assert result.success is False
        assert "simulated graph crash" in result.error
        assert result.final_answer is None


class TestRunBenchmark:
    def test_executes_every_question_in_file(self, tmp_path, monkeypatch):
        _mock_feasible(monkeypatch)
        _mock_planner(monkeypatch)
        path = tmp_path / "questions.json"
        path.write_text(
            json.dumps(
                [
                    _question(id="q001").model_dump(),
                    _question(id="q002", question="How many distinct regions?").model_dump(),
                ]
            )
        )
        results = run_benchmark(path)
        assert [r.question_id for r in results] == ["q001", "q002"]
        assert all(r.success for r in results)

    def test_missing_questions_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            run_benchmark(tmp_path / "does_not_exist.json")


class TestSaveResults:
    def test_creates_parent_directory_and_writes_json(self, tmp_path):
        results = [
            QuestionResult(
                question_id="q001",
                dataset_id="superstore",
                category="descriptive",
                question="How many rows?",
                expected_answerable=True,
                success=True,
                feasibility_verdict="feasible",
                feasibility_reason="ok",
                tools_used=["profile_dataset"],
                validation_passed=True,
                final_answer="There are 9994 rows [[ev_profile_dataset_deadbeef]].",
                duration_ms=12.3,
            )
        ]
        target = tmp_path / "nested" / "results.json"
        save_results(results, target)

        assert target.exists()
        saved = json.loads(target.read_text())
        assert len(saved) == 1
        assert saved[0]["question_id"] == "q001"
        assert saved[0]["success"] is True
        assert saved[0]["tools_used"] == ["profile_dataset"]

    def test_overwrites_existing_file(self, tmp_path):
        target = tmp_path / "results.json"
        target.write_text("stale content")
        save_results([], target)
        saved = json.loads(target.read_text())
        assert saved == []
