"""Validation, serialization, and invalid-payload tests for adia.models."""

import pytest
from pydantic import ValidationError

from adia.models import (
    AgentState,
    Budget,
    ColumnProfile,
    DatasetCatalog,
    Evidence,
    PlanStep,
    Provenance,
    SemanticType,
    ToolError,
    ToolErrorKind,
    ToolResult,
)


def _provenance(**overrides) -> Provenance:
    defaults = dict(tool_name="run_sql", args={"query": "select 1"}, args_hash="abc123")
    defaults.update(overrides)
    return Provenance(**defaults)


class TestProvenance:
    def test_round_trip(self):
        prov = _provenance(seed=42, library_versions={"duckdb": "1.5.5"})
        restored = Provenance.model_validate_json(prov.model_dump_json())
        assert restored == prov

    def test_defaults(self):
        prov = _provenance()
        assert prov.seed is None
        assert prov.args == {"query": "select 1"}

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            Provenance(args_hash="abc123")  # missing tool_name

    def test_is_frozen(self):
        prov = _provenance()
        with pytest.raises(ValidationError):
            prov.tool_name = "other_tool"


class TestToolError:
    def test_round_trip(self):
        err = ToolError(kind=ToolErrorKind.EXECUTION, message="boom", retryable=True)
        restored = ToolError.model_validate_json(err.model_dump_json())
        assert restored == err

    def test_invalid_kind_rejected(self):
        with pytest.raises(ValidationError):
            ToolError(kind="not_a_real_kind", message="boom")

    def test_missing_message_rejected(self):
        with pytest.raises(ValidationError):
            ToolError(kind=ToolErrorKind.UNKNOWN)


class TestToolResult:
    def test_ok_result_round_trip(self):
        result = ToolResult(
            ok=True,
            tool="run_sql",
            evidence_id="ev_01",
            data={"rows": 3},
            provenance=_provenance(),
            duration_ms=12.5,
        )
        restored = ToolResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_error_result_round_trip(self):
        result = ToolResult(
            ok=False,
            tool="run_sql",
            error=ToolError(kind=ToolErrorKind.GUARD_REJECTED, message="blocked"),
            duration_ms=1.0,
        )
        restored = ToolResult.model_validate_json(result.model_dump_json())
        assert restored == result

    def test_ok_true_requires_data_evidence_and_provenance(self):
        with pytest.raises(ValidationError):
            ToolResult(ok=True, tool="run_sql", duration_ms=1.0)

    def test_ok_true_rejects_error_present(self):
        with pytest.raises(ValidationError):
            ToolResult(
                ok=True,
                tool="run_sql",
                evidence_id="ev_01",
                data={"rows": 1},
                provenance=_provenance(),
                error=ToolError(kind=ToolErrorKind.UNKNOWN, message="x"),
                duration_ms=1.0,
            )

    def test_ok_false_requires_error(self):
        with pytest.raises(ValidationError):
            ToolResult(ok=False, tool="run_sql", duration_ms=1.0)

    def test_negative_duration_rejected(self):
        with pytest.raises(ValidationError):
            ToolResult(
                ok=False,
                tool="run_sql",
                error=ToolError(kind=ToolErrorKind.UNKNOWN, message="x"),
                duration_ms=-1.0,
            )


class TestEvidence:
    def test_round_trip(self):
        ev = Evidence(
            id="ev_01",
            tool="profile_dataset",
            data={"mean": 42.17},
            provenance=_provenance(tool_name="profile_dataset"),
            plan_step_id="step_1",
        )
        restored = Evidence.model_validate_json(ev.model_dump_json())
        assert restored == ev

    def test_missing_data_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(id="ev_01", tool="profile_dataset", provenance=_provenance())


class TestPlanStep:
    def test_round_trip(self):
        step = PlanStep(
            id="step_1",
            intent="Profile the orders table",
            tool_family="profile_dataset",
            depends_on=[],
            expected_output="column-level summary statistics",
            success_criteria="all columns present with non-null counts",
        )
        restored = PlanStep.model_validate_json(step.model_dump_json())
        assert restored == step

    def test_self_dependency_rejected(self):
        with pytest.raises(ValidationError):
            PlanStep(
                id="step_1",
                intent="x",
                tool_family="run_sql",
                depends_on=["step_1"],
                expected_output="x",
                success_criteria="x",
            )

    def test_missing_required_field_rejected(self):
        with pytest.raises(ValidationError):
            PlanStep(id="step_1", intent="x", tool_family="run_sql")


class TestColumnProfileAndCatalog:
    def test_column_profile_round_trip(self):
        col = ColumnProfile(
            name="price",
            dtype="float64",
            semantic_type=SemanticType.NUMERIC,
            non_null_count=98,
            null_count=2,
            null_rate=0.02,
            unique_count=87,
            min_value=1.5,
            max_value=999.0,
        )
        restored = ColumnProfile.model_validate_json(col.model_dump_json())
        assert restored == col

    def test_null_rate_out_of_range_rejected(self):
        with pytest.raises(ValidationError):
            ColumnProfile(
                name="price",
                dtype="float64",
                semantic_type=SemanticType.NUMERIC,
                non_null_count=98,
                null_count=2,
                null_rate=1.5,
                unique_count=87,
            )

    def test_catalog_round_trip_and_column_names(self):
        col = ColumnProfile(
            name="price",
            dtype="float64",
            semantic_type=SemanticType.NUMERIC,
            non_null_count=98,
            null_count=2,
            null_rate=0.02,
            unique_count=87,
            min_value=1.5,
            max_value=999.0,
        )
        catalog = DatasetCatalog(
            dataset_id="ecommerce", source_path="data/processed/ecommerce.parquet",
            row_count=100, columns=[col],
        )
        restored = DatasetCatalog.model_validate_json(catalog.model_dump_json())
        assert restored == catalog
        assert catalog.column_names() == ["price"]

    def test_negative_row_count_rejected(self):
        with pytest.raises(ValidationError):
            DatasetCatalog(
                dataset_id="ecommerce", source_path="x.parquet", row_count=-1, columns=[]
            )


class TestAgentState:
    def _state(self, **overrides) -> AgentState:
        defaults = dict(
            run_id="run_1",
            question="What drove the Q3 revenue drop?",
            dataset_id="ecommerce",
            budget=Budget(max_llm_calls=10, max_tool_calls=20),
        )
        defaults.update(overrides)
        return AgentState(**defaults)

    def test_minimal_construction_and_round_trip(self):
        state = self._state()
        restored = AgentState.model_validate_json(state.model_dump_json())
        assert restored == state
        assert state.evidence == {}
        assert state.errors == []
        assert state.repair_attempts == 0

    def test_missing_required_field_rejected(self):
        budget = Budget(max_llm_calls=1, max_tool_calls=1)
        with pytest.raises(ValidationError):
            AgentState(question="x", dataset_id="ecommerce", budget=budget)

    def test_missing_budget_rejected(self):
        with pytest.raises(ValidationError):
            AgentState(run_id="run_1", question="x", dataset_id="ecommerce")

    def test_evidence_dict_holds_evidence_models(self):
        ev = Evidence(id="ev_01", tool="run_sql", data={"n": 3}, provenance=_provenance())
        state = self._state(evidence={"ev_01": ev})
        assert state.evidence["ev_01"].data == {"n": 3}

    def test_budget_exhausted(self):
        budget = Budget(max_llm_calls=2, max_tool_calls=5, used_llm_calls=2)
        assert budget.exhausted() is True

    def test_budget_not_exhausted(self):
        budget = Budget(max_llm_calls=2, max_tool_calls=5, used_llm_calls=1, used_tool_calls=1)
        assert budget.exhausted() is False
