"""Tests for adia.agents.synthesizer. No real OpenAI call is ever made here."""

import pytest

from adia.agents.synthesizer import (
    _build_messages,
    _mechanical_fallback,
    _select_fallback_values,
    _SynthesizerLLMOutput,
    synthesize_answer,
)
from adia.evidence.ids import compute_args_hash, generate_evidence_id
from adia.evidence.renderer import render_evidence, render_evidence_context
from adia.models.evidence import Evidence
from adia.models.provenance import Provenance
from adia.validate.static import validate_answer


def _make_evidence(tool: str, args: dict, data: object) -> Evidence:
    return Evidence(
        id=generate_evidence_id(tool, args),
        tool=tool,
        data=data,
        provenance=Provenance(tool_name=tool, args=args, args_hash=compute_args_hash(args)),
    )


@pytest.fixture
def evidence() -> Evidence:
    return _make_evidence(
        "run_sql",
        {"query": "select avg(price) from orders"},
        {"rows": [{"avg_price": 42.17}], "row_count": 1},
    )


def _fake_llm_call(output: _SynthesizerLLMOutput):
    """Build an `llm_call` that ignores its arguments and returns a fixed response."""

    def _call(question, evidence_context):  # noqa: ARG001 - signature must match LLMCall
        return output

    return _call


class TestAnswerGenerationFromEvidence:
    def test_grounded_answer_is_returned_verbatim(self, evidence):
        output = _SynthesizerLLMOutput(
            answer=f"The average price is 42.17 [[{evidence.id}]]."
        )
        answer = synthesize_answer(
            "What's the average price?",
            render_evidence_context([evidence]),
            {evidence.id: evidence},
            llm_call=_fake_llm_call(output),
        )
        assert answer == output.answer

    def test_citation_marker_is_preserved(self, evidence):
        output = _SynthesizerLLMOutput(
            answer=f"The average price is 42.17 [[{evidence.id}]]."
        )
        answer = synthesize_answer(
            "What's the average price?",
            render_evidence_context([evidence]),
            {evidence.id: evidence},
            llm_call=_fake_llm_call(output),
        )
        assert f"[[{evidence.id}]]" in answer

    def test_citation_only_answer_with_no_numbers_passes(self, evidence):
        output = _SynthesizerLLMOutput(
            answer=f"See the query result for details [[{evidence.id}]]."
        )
        answer = synthesize_answer(
            "What's the average price?",
            render_evidence_context([evidence]),
            {evidence.id: evidence},
            llm_call=_fake_llm_call(output),
        )
        assert answer == output.answer


class TestEmptyEvidenceHandling:
    def test_no_evidence_returns_fallback_without_calling_llm(self):
        def _fail_if_called(question, evidence_context):
            raise AssertionError("llm_call should not be invoked with no evidence")

        answer = synthesize_answer("Any question?", "", {}, llm_call=_fail_if_called)
        assert "No evidence was collected" in answer


class TestHallucinatedNumberRejection:
    def test_uncited_number_falls_back_to_mechanical_answer(self, evidence):
        # The LLM invents a number that isn't in the evidence at all, and cites nothing.
        output = _SynthesizerLLMOutput(answer="The average price is 999.99.")
        answer = synthesize_answer(
            "What's the average price?",
            render_evidence_context([evidence]),
            {evidence.id: evidence},
            llm_call=_fake_llm_call(output),
        )
        assert answer != output.answer
        assert "999.99" not in answer
        assert f"[[{evidence.id}]]" in answer

    def test_cited_but_unsupported_number_falls_back_to_mechanical_answer(self, evidence):
        # Citation is well-formed and real, but the number doesn't match anything in it.
        output = _SynthesizerLLMOutput(answer=f"The average price is 999.99 [[{evidence.id}]].")
        answer = synthesize_answer(
            "What's the average price?",
            render_evidence_context([evidence]),
            {evidence.id: evidence},
            llm_call=_fake_llm_call(output),
        )
        assert answer != output.answer
        assert "999.99" not in answer

    def test_hallucinated_evidence_id_falls_back_to_mechanical_answer(self, evidence):
        output = _SynthesizerLLMOutput(answer="The average price is 42.17 [[ev_run_sql_deadbeef]].")
        answer = synthesize_answer(
            "What's the average price?",
            render_evidence_context([evidence]),
            {evidence.id: evidence},
            llm_call=_fake_llm_call(output),
        )
        assert answer != output.answer
        assert f"[[{evidence.id}]]" in answer

    def test_blank_answer_falls_back_to_mechanical_answer(self, evidence):
        output = _SynthesizerLLMOutput(answer="   ")
        answer = synthesize_answer(
            "What's the average price?",
            render_evidence_context([evidence]),
            {evidence.id: evidence},
            llm_call=_fake_llm_call(output),
        )
        assert f"[[{evidence.id}]]" in answer


class TestMechanicalFallback:
    def test_evidence_with_no_scalar_values_is_reported_honestly(self):
        empty_evidence = _make_evidence(
            "profile_dataset", {"dataset_id": "superstore"}, {}
        )

        def _raising_call(question, evidence_context):
            raise RuntimeError("simulated network failure")

        answer = synthesize_answer(
            "How many rows?",
            render_evidence_context([empty_evidence]),
            {empty_evidence.id: empty_evidence},
            llm_call=_raising_call,
        )
        assert "reported no scalar values" in answer
        assert f"[[{empty_evidence.id}]]" in answer


class TestMechanicalFallbackValueSelection:
    """Covers the headline-selection fix: bookkeeping/config fields no longer win by default."""

    def test_run_sql_fallback_prefers_row_values_over_rows_count(self):
        # rows_count is deprioritized, not excluded -- with a real row value available, it
        # must no longer be the *headline* (top-ranked) choice, though it may still fill a
        # remaining slot in the top-3 if nothing better is left.
        run_sql_evidence = _make_evidence(
            "run_sql",
            {"query": "select * from orders order by sales desc limit 10"},
            {
                "rows": [{"Product Name": "Widget", "Sales": 5000.0}],
                "row_count": 10,
            },
        )
        selected = _select_fallback_values(render_evidence(run_sql_evidence).key_values)
        assert selected[0] == ("rows[0].Sales", 5000.0)
        answer = _mechanical_fallback(
            "Top products by sales?", {run_sql_evidence.id: run_sql_evidence}
        )
        assert "rows[0].Sales = 5000.0" in answer

    def test_train_model_fallback_prefers_metric_value_over_model_type(self):
        # Same principle: model_type is deprioritized (non-numeric), not banned outright.
        # With real metric values present, they must rank ahead of it as the headline.
        train_model_evidence = _make_evidence(
            "train_model",
            {"target_column": "Region"},
            {
                "model_type": "random_forest_classifier",
                "task_type": "classification",
                "metric_name": "accuracy",
                "metric_value": 0.83,
                "baseline_metric_name": "accuracy",
                "baseline_metric_value": 0.5,
            },
        )
        selected = _select_fallback_values(render_evidence(train_model_evidence).key_values)
        assert selected[0] == ("metric_value", 0.83)
        assert selected[1] == ("baseline_metric_value", 0.5)
        answer = _mechanical_fallback(
            "How well can Region be predicted?", {train_model_evidence.id: train_model_evidence}
        )
        assert "metric_value = 0.83" in answer

    def test_select_fallback_values_ranks_numeric_non_count_first(self):
        key_values = {
            "rows_count": 10,
            "model_type": "random_forest_classifier",
            "metric_value": 0.83,
        }
        selected = _select_fallback_values(key_values)
        assert selected[0] == ("metric_value", 0.83)

    def test_select_fallback_values_falls_back_to_count_when_nothing_else_exists(self):
        # A count key is still better than nothing: it's not excluded, only deprioritized.
        key_values = {"rows_count": 10}
        selected = _select_fallback_values(key_values)
        assert selected == [("rows_count", 10)]

    def test_select_fallback_values_caps_at_three(self):
        key_values = {f"metric_{i}": float(i) for i in range(6)}
        selected = _select_fallback_values(key_values)
        assert len(selected) == 3

    def test_fallback_with_multiple_selected_values_passes_validate_answer(self):
        train_model_evidence = _make_evidence(
            "train_model",
            {"target_column": "Region"},
            {
                "model_type": "random_forest_classifier",
                "metric_name": "accuracy",
                "metric_value": 0.83,
                "baseline_metric_value": 0.5,
            },
        )
        answer = _mechanical_fallback(
            "How well can Region be predicted?", {train_model_evidence.id: train_model_evidence}
        )
        result = validate_answer(answer, {train_model_evidence.id: train_model_evidence})
        assert result.passed is True
        assert result.issues == []

    def test_fallback_output_is_deterministic_across_repeated_calls(self):
        train_model_evidence = _make_evidence(
            "train_model",
            {"target_column": "Region"},
            {
                "model_type": "random_forest_classifier",
                "task_type": "classification",
                "metric_name": "accuracy",
                "metric_value": 0.83,
                "baseline_metric_value": 0.5,
            },
        )
        evidence_map = {train_model_evidence.id: train_model_evidence}
        first = _mechanical_fallback("How well can Region be predicted?", evidence_map)
        second = _mechanical_fallback("How well can Region be predicted?", evidence_map)
        third = _mechanical_fallback("How well can Region be predicted?", evidence_map)
        assert first == second == third

    def test_selection_is_derived_only_from_rendered_evidence_values(self):
        # No architecture change: selection still reads exclusively from render_evidence's
        # own key_values -- nothing new is invented, and citations are untouched.
        train_model_evidence = _make_evidence(
            "train_model",
            {"target_column": "Region"},
            {"model_type": "random_forest_classifier", "metric_value": 0.83},
        )
        rendered = render_evidence(train_model_evidence)
        selected = _select_fallback_values(rendered.key_values)
        for key, value in selected:
            assert rendered.key_values[key] == value


class TestBuildMessages:
    def test_includes_question_and_evidence_context(self):
        messages = _build_messages("What's the average price?", "some rendered context")
        human = messages[1].content
        assert "What's the average price?" in human
        assert "some rendered context" in human


class TestLLMFailureHandling:
    def test_llm_call_exception_falls_back_not_a_crash(self, evidence):
        def _raising_call(question, evidence_context):
            raise RuntimeError("simulated network failure")

        answer = synthesize_answer(
            "What's the average price?",
            render_evidence_context([evidence]),
            {evidence.id: evidence},
            llm_call=_raising_call,
        )
        assert f"[[{evidence.id}]]" in answer

    def test_llm_call_returning_wrong_type_falls_back(self, evidence):
        def _bad_call(question, evidence_context):
            return {"answer": "not even the right shape"}

        answer = synthesize_answer(
            "What's the average price?",
            render_evidence_context([evidence]),
            {evidence.id: evidence},
            llm_call=_bad_call,
        )
        assert f"[[{evidence.id}]]" in answer

    def test_missing_api_key_falls_back_not_a_crash(self, evidence, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        # Prevent load_llm_settings() from picking up a real developer .env file, so this
        # test's outcome doesn't depend on whether one happens to exist on this machine.
        monkeypatch.setattr("adia.agents.llm_config.load_dotenv", lambda *args, **kwargs: None)
        # No llm_call override -- exercises the real default path, which must reach
        # load_llm_settings(), fail on the missing key, and be caught, never raised.
        answer = synthesize_answer(
            "What's the average price?",
            render_evidence_context([evidence]),
            {evidence.id: evidence},
        )
        assert f"[[{evidence.id}]]" in answer
