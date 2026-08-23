"""Tests for adia.agents.planner. No real OpenAI call is ever made here."""

import pytest

from adia.agents.planner import _build_messages, _PlannerLLMOutput, _PlanStepLLMOutput, create_plan
from adia.models.catalog import ColumnProfile, DatasetCatalog, SemanticType
from adia.models.state import FeasibilityResult, FeasibilityVerdict


@pytest.fixture
def catalog() -> DatasetCatalog:
    columns = [
        ColumnProfile(
            name="Sales",
            dtype="float64",
            semantic_type=SemanticType.NUMERIC,
            non_null_count=100,
            null_count=0,
            null_rate=0.0,
            unique_count=87,
            min_value=1.5,
            max_value=999.0,
        ),
        ColumnProfile(
            name="Region",
            dtype="str",
            semantic_type=SemanticType.CATEGORICAL,
            non_null_count=100,
            null_count=0,
            null_rate=0.0,
            unique_count=4,
        ),
    ]
    return DatasetCatalog(
        dataset_id="orders", source_path="data/orders.parquet", row_count=100, columns=columns
    )


@pytest.fixture
def feasible() -> FeasibilityResult:
    return FeasibilityResult(verdict=FeasibilityVerdict.FEASIBLE, reason="Answerable.")


def _fake_llm_call(output: _PlannerLLMOutput):
    """Build an `llm_call` that ignores its arguments and returns a fixed response."""

    def _call(question, catalog, feasibility):  # noqa: ARG001 - signature must match LLMCall
        return output

    return _call


class TestValidPlanGeneration:
    def test_single_step_plan(self, catalog, feasible):
        output = _PlannerLLMOutput(
            steps=[
                _PlanStepLLMOutput(
                    step_id="step_1",
                    tool_family="profile_dataset",
                    purpose="Understand the dataset's shape before deeper analysis.",
                )
            ]
        )
        plan = create_plan(
            "Tell me about this data", catalog, feasible, llm_call=_fake_llm_call(output)
        )
        assert len(plan) == 1
        assert plan[0].id == "step_1"
        assert plan[0].tool_family == "profile_dataset"
        assert plan[0].intent == output.steps[0].purpose
        assert plan[0].expected_output
        assert plan[0].success_criteria

    def test_multi_step_plan_with_dependency(self, catalog, feasible):
        output = _PlannerLLMOutput(
            steps=[
                _PlanStepLLMOutput(
                    step_id="profile", tool_family="profile_dataset", purpose="Profile first."
                ),
                _PlanStepLLMOutput(
                    step_id="aggregate",
                    tool_family="run_sql",
                    purpose="Aggregate sales by region.",
                    depends_on=["profile"],
                ),
            ]
        )
        plan = create_plan("Sales by region?", catalog, feasible, llm_call=_fake_llm_call(output))
        assert [step.id for step in plan] == ["profile", "aggregate"]
        assert plan[1].depends_on == ["profile"]

    def test_all_supported_tool_families_accepted(self, catalog, feasible):
        families = [
            "profile_dataset",
            "run_sql",
            "compare_groups",
            "compute_correlation",
            "train_model",
        ]
        output = _PlannerLLMOutput(
            steps=[
                _PlanStepLLMOutput(step_id=f"s{i}", tool_family=fam, purpose="Do something.")
                for i, fam in enumerate(families)
            ]
        )
        plan = create_plan("...", catalog, feasible, llm_call=_fake_llm_call(output))
        assert [step.tool_family for step in plan] == families

    def test_not_feasible_question_produces_no_plan_without_calling_llm(self, catalog):
        def _fail_if_called(question, catalog, feasibility):
            raise AssertionError("llm_call should not be invoked for a non-FEASIBLE verdict")

        infeasible = FeasibilityResult(
            verdict=FeasibilityVerdict.INFEASIBLE, reason="No forecasting data."
        )
        plan = create_plan("Will sales grow?", catalog, infeasible, llm_call=_fail_if_called)
        assert plan == []

    def test_needs_clarification_produces_no_plan(self, catalog):
        needs_clarification = FeasibilityResult(
            verdict=FeasibilityVerdict.NEEDS_CLARIFICATION, reason="Ambiguous time range."
        )
        output = _PlannerLLMOutput(
            steps=[_PlanStepLLMOutput(step_id="s1", tool_family="run_sql", purpose="x")]
        )
        plan = create_plan(
            "Recent sales?", catalog, needs_clarification, llm_call=_fake_llm_call(output)
        )
        assert plan == []

    def test_empty_steps_produces_empty_plan(self, catalog, feasible):
        plan = create_plan(
            "...", catalog, feasible, llm_call=_fake_llm_call(_PlannerLLMOutput(steps=[]))
        )
        assert plan == []


class TestInvalidToolRejection:
    def test_unsupported_tool_family_rejects_whole_plan(self, catalog, feasible):
        output = _PlannerLLMOutput(
            steps=[
                _PlanStepLLMOutput(
                    step_id="s1", tool_family="profile_dataset", purpose="Fine."
                ),
                _PlanStepLLMOutput(
                    step_id="s2", tool_family="delete_dataset", purpose="Not a real tool."
                ),
            ]
        )
        plan = create_plan("...", catalog, feasible, llm_call=_fake_llm_call(output))
        assert plan == []


class TestMalformedOutputHandling:
    def test_duplicate_step_ids_rejected(self, feasible):
        with pytest.raises(Exception, match="duplicate"):
            _PlannerLLMOutput(
                steps=[
                    _PlanStepLLMOutput(step_id="s1", tool_family="run_sql", purpose="a"),
                    _PlanStepLLMOutput(step_id="s1", tool_family="run_sql", purpose="b"),
                ]
            )

    def test_more_than_max_steps_rejected(self):
        with pytest.raises(Exception, match="at most|List should have at most"):
            _PlannerLLMOutput(
                steps=[
                    _PlanStepLLMOutput(step_id=f"s{i}", tool_family="run_sql", purpose="x")
                    for i in range(7)
                ]
            )

    def test_dependency_on_nonexistent_step_rejects_whole_plan(self, catalog, feasible):
        output = _PlannerLLMOutput(
            steps=[
                _PlanStepLLMOutput(
                    step_id="s1",
                    tool_family="run_sql",
                    purpose="x",
                    depends_on=["does_not_exist"],
                )
            ]
        )
        plan = create_plan("...", catalog, feasible, llm_call=_fake_llm_call(output))
        assert plan == []

    def test_self_dependency_rejects_whole_plan(self, catalog, feasible):
        output = _PlannerLLMOutput(
            steps=[
                _PlanStepLLMOutput(
                    step_id="s1", tool_family="run_sql", purpose="x", depends_on=["s1"]
                )
            ]
        )
        plan = create_plan("...", catalog, feasible, llm_call=_fake_llm_call(output))
        assert plan == []

    def test_llm_call_returning_wrong_type_degrades_to_empty_plan(self, catalog, feasible):
        def _bad_call(question, catalog, feasibility):
            return {"steps": "not even the right shape"}

        plan = create_plan("...", catalog, feasible, llm_call=_bad_call)
        assert plan == []


class TestBuildMessages:
    def test_includes_question_columns_and_feasibility_reason(self, catalog, feasible):
        messages = _build_messages("Sales by region?", catalog, feasible)
        human = messages[1].content
        assert "Sales by region?" in human
        assert "Sales" in human
        assert "Region" in human
        assert feasible.reason in human


class TestLLMFailureHandling:
    def test_llm_call_exception_degrades_to_empty_plan_not_a_crash(self, catalog, feasible):
        def _raising_call(question, catalog, feasibility):
            raise RuntimeError("simulated network failure")

        plan = create_plan("...", catalog, feasible, llm_call=_raising_call)
        assert plan == []

    def test_missing_api_key_degrades_to_empty_plan_not_a_crash(
        self, catalog, feasible, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        # Prevent load_llm_settings() from picking up a real developer .env file, so this
        # test's outcome doesn't depend on whether one happens to exist on this machine.
        monkeypatch.setattr("adia.agents.llm_config.load_dotenv", lambda *args, **kwargs: None)
        # No llm_call override -- exercises the real default path, which must reach
        # load_llm_settings(), fail on the missing key, and be caught, never raised.
        plan = create_plan("Sales by region?", catalog, feasible)
        assert plan == []
