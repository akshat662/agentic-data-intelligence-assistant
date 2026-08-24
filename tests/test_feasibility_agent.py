"""Tests for adia.agents.feasibility. No real OpenAI call is ever made here."""

import pytest

from adia.agents.feasibility import (
    _SYSTEM_PROMPT,
    _build_messages,
    _FeasibilityLLMOutput,
    assess_feasibility,
)
from adia.models.catalog import ColumnProfile, DatasetCatalog, SemanticType
from adia.models.state import FeasibilityVerdict


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


def _fake_llm_call(output: _FeasibilityLLMOutput):
    """Build an `llm_call` that ignores its arguments and returns a fixed response."""

    def _call(question, catalog):  # noqa: ARG001 - signature must match LLMCall
        return output

    return _call


class TestValidFeasibilityOutput:
    def test_feasible_verdict_with_real_columns_passes_through(self, catalog):
        output = _FeasibilityLLMOutput(
            verdict="feasible",
            relevant_columns=["Sales", "Region"],
            missing_capabilities=[],
            reasoning="Sales can be aggregated by Region using the available columns.",
        )
        result = assess_feasibility(
            "Total sales by region?", catalog, llm_call=_fake_llm_call(output)
        )
        assert result.verdict == FeasibilityVerdict.FEASIBLE
        assert result.reason == output.reasoning
        assert result.missing_columns == []

    def test_needs_clarification_passes_through(self, catalog):
        output = _FeasibilityLLMOutput(
            verdict="needs_clarification",
            relevant_columns=["Sales"],
            reasoning="It's unclear which time period 'recent' refers to.",
        )
        result = assess_feasibility(
            "What was recent revenue?", catalog, llm_call=_fake_llm_call(output)
        )
        assert result.verdict == FeasibilityVerdict.NEEDS_CLARIFICATION

    def test_missing_capabilities_preserved(self, catalog):
        output = _FeasibilityLLMOutput(
            verdict="infeasible",
            relevant_columns=[],
            missing_capabilities=["forecasting", "future data"],
            reasoning="No forecasting capability or future data exists in this dataset.",
        )
        result = assess_feasibility(
            "Will sales grow next year?", catalog, llm_call=_fake_llm_call(output)
        )
        assert result.verdict == FeasibilityVerdict.INFEASIBLE
        assert result.missing_capabilities == ["forecasting", "future data"]


class TestHallucinatedColumnRejection:
    def test_hallucinated_column_forces_infeasible_even_when_llm_said_feasible(self, catalog):
        output = _FeasibilityLLMOutput(
            verdict="feasible",
            relevant_columns=["Sales", "Employee Name"],  # "Employee Name" is not in catalog
            missing_capabilities=[],
            reasoning="Employee Name can be used to break down sales performance.",
        )
        result = assess_feasibility(
            "Sales by employee?", catalog, llm_call=_fake_llm_call(output)
        )
        assert result.verdict == FeasibilityVerdict.INFEASIBLE
        assert result.missing_columns == ["Employee Name"]

    def test_multiple_hallucinated_columns_all_reported(self, catalog):
        output = _FeasibilityLLMOutput(
            verdict="feasible",
            relevant_columns=["Email", "Sales Rep", "Region"],
            reasoning="Uses email and sales rep to segment sales.",
        )
        result = assess_feasibility("...", catalog, llm_call=_fake_llm_call(output))
        assert result.verdict == FeasibilityVerdict.INFEASIBLE
        assert result.missing_columns == ["Email", "Sales Rep"]

    def test_valid_columns_do_not_trigger_rejection(self, catalog):
        output = _FeasibilityLLMOutput(
            verdict="feasible", relevant_columns=["Sales", "Region"], reasoning="Fine."
        )
        result = assess_feasibility("...", catalog, llm_call=_fake_llm_call(output))
        assert result.verdict == FeasibilityVerdict.FEASIBLE
        assert result.missing_columns == []


class TestInvestigationQuestions:
    """"Why" questions that can be built from existing dataset columns must not be blocked."""

    @pytest.fixture
    def investigation_catalog(self) -> DatasetCatalog:
        columns = [
            ColumnProfile(
                name="Category",
                dtype="str",
                semantic_type=SemanticType.CATEGORICAL,
                non_null_count=9994,
                null_count=0,
                null_rate=0.0,
                unique_count=3,
            ),
            ColumnProfile(
                name="Sales",
                dtype="float64",
                semantic_type=SemanticType.NUMERIC,
                non_null_count=9994,
                null_count=0,
                null_rate=0.0,
                unique_count=5825,
                min_value=0.44,
                max_value=22638.48,
            ),
            ColumnProfile(
                name="Quantity",
                dtype="int64",
                semantic_type=SemanticType.NUMERIC,
                non_null_count=9994,
                null_count=0,
                null_rate=0.0,
                unique_count=14,
                min_value=1,
                max_value=14,
            ),
            ColumnProfile(
                name="Discount",
                dtype="float64",
                semantic_type=SemanticType.NUMERIC,
                non_null_count=9994,
                null_count=0,
                null_rate=0.0,
                unique_count=12,
                min_value=0.0,
                max_value=0.8,
            ),
            ColumnProfile(
                name="Region",
                dtype="str",
                semantic_type=SemanticType.CATEGORICAL,
                non_null_count=9994,
                null_count=0,
                null_rate=0.0,
                unique_count=4,
            ),
        ]
        return DatasetCatalog(
            dataset_id="superstore",
            source_path="data/superstore.csv",
            row_count=9994,
            columns=columns,
        )

    def test_analytical_why_question_can_be_accepted(self, investigation_catalog):
        """1. Analytical "why" question accepted.

        The Python layer has no hardcoded rejection of "why" phrasing: when the LLM proposes
        'feasible' for a question investigable from real, existing columns, that verdict
        passes through unchanged, exactly like any other feasible verdict.
        """
        output = _FeasibilityLLMOutput(
            verdict="feasible",
            relevant_columns=["Category", "Sales", "Quantity", "Discount", "Region"],
            missing_capabilities=[],
            reasoning=(
                "Category and Sales establish which category has the lowest total Sales; "
                "Quantity, Discount, and Region are available to investigate what is "
                "associated with the difference, though the dataset cannot prove a "
                "definitive cause."
            ),
        )
        result = assess_feasibility(
            "Why does Office Supplies have the lowest total Sales?",
            investigation_catalog,
            llm_call=_fake_llm_call(output),
        )
        assert result.verdict == FeasibilityVerdict.FEASIBLE
        assert result.reason == output.reasoning
        assert result.missing_columns == []

    def test_truly_causal_external_question_is_rejected(self, investigation_catalog):
        """2. Truly causal/external question rejected.

        A "why" question that needs data the dataset can't provide -- customer psychology,
        preference, or motivation -- must still resolve to INFEASIBLE.
        """
        output = _FeasibilityLLMOutput(
            verdict="infeasible",
            relevant_columns=["Category"],
            missing_capabilities=["customer preference data", "behavioral/survey data"],
            reasoning=(
                "Explaining customer preference requires behavioral or survey data about "
                "why customers choose products, which this dataset does not contain."
            ),
        )
        result = assess_feasibility(
            "Why do customers prefer Technology products?",
            investigation_catalog,
            llm_call=_fake_llm_call(output),
        )
        assert result.verdict == FeasibilityVerdict.INFEASIBLE
        assert result.missing_capabilities == [
            "customer preference data",
            "behavioral/survey data",
        ]

    def test_forecasting_why_question_remains_infeasible(self, investigation_catalog):
        """3. Existing refusal behavior remains unchanged.

        A question needing data that doesn't exist yet (forecasting) must still be refused --
        the broadened "why" guidance must not make the agent over-eager to accept everything.
        """
        output = _FeasibilityLLMOutput(
            verdict="infeasible",
            relevant_columns=["Sales"],
            missing_capabilities=["forecasting", "future data"],
            reasoning="No forecasting capability or future data exists in this dataset.",
        )
        result = assess_feasibility(
            "Why will next quarter's sales be higher?",
            investigation_catalog,
            llm_call=_fake_llm_call(output),
        )
        assert result.verdict == FeasibilityVerdict.INFEASIBLE
        assert result.missing_capabilities == ["forecasting", "future data"]

    def test_hallucinated_column_still_forces_infeasible_for_a_why_question(
        self, investigation_catalog
    ):
        """3. Existing refusal behavior remains unchanged.

        Column-hallucination verification is untouched: even a "why" question the LLM claims
        is feasible must still be rejected if it names a column that doesn't exist.
        """
        output = _FeasibilityLLMOutput(
            verdict="feasible",
            relevant_columns=["Category", "Customer Satisfaction Score"],
            missing_capabilities=[],
            reasoning="Customer Satisfaction Score explains the sales difference.",
        )
        result = assess_feasibility(
            "Why does Office Supplies have the lowest total Sales?",
            investigation_catalog,
            llm_call=_fake_llm_call(output),
        )
        assert result.verdict == FeasibilityVerdict.INFEASIBLE
        assert result.missing_columns == ["Customer Satisfaction Score"]


class TestSystemPromptContent:
    """Directly verifies the prompt text this task changed, since agent judgment quality on
    real questions can only otherwise be evaluated through the benchmark (real LLM calls)."""

    def test_prompt_describes_data_supported_investigation_as_feasible(self):
        assert "Data-supported investigation" in _SYSTEM_PROMPT
        assert "comparing or aggregating" in _SYSTEM_PROMPT

    def test_prompt_describes_unsupported_causal_questions_as_infeasible(self):
        assert "Unsupported causal question" in _SYSTEM_PROMPT
        assert "customer" in _SYSTEM_PROMPT.lower()

    def test_prompt_does_not_require_certainty_for_feasibility(self):
        assert "definitive cause" in _SYSTEM_PROMPT


class TestBuildMessages:
    def test_includes_question_and_column_names(self, catalog):
        messages = _build_messages("Total sales by region?", catalog)
        assert len(messages) == 2
        human = messages[1].content
        assert "Total sales by region?" in human
        assert "Sales" in human
        assert "Region" in human

    def test_includes_dataset_id_and_row_count(self, catalog):
        messages = _build_messages("...", catalog)
        human = messages[1].content
        assert catalog.dataset_id in human
        assert str(catalog.row_count) in human


class TestLLMFailureHandling:
    def test_llm_call_exception_degrades_to_infeasible_not_a_crash(self, catalog):
        def _raising_call(question, catalog):
            raise RuntimeError("simulated network failure")

        result = assess_feasibility("...", catalog, llm_call=_raising_call)
        assert result.verdict == FeasibilityVerdict.INFEASIBLE
        assert "could not be assessed" in result.reason

    def test_missing_api_key_degrades_to_infeasible_not_a_crash(self, catalog, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        # Prevent load_llm_settings() from picking up a real developer .env file, so this
        # test's outcome doesn't depend on whether one happens to exist on this machine.
        monkeypatch.setattr("adia.agents.llm_config.load_dotenv", lambda *args, **kwargs: None)
        # No llm_call override -- exercises the real default path, which must reach
        # load_llm_settings(), fail on the missing key, and be caught, never raised.
        result = assess_feasibility("Total sales by region?", catalog)
        assert result.verdict == FeasibilityVerdict.INFEASIBLE
        assert "could not be assessed" in result.reason
