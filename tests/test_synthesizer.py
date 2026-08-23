"""Tests for adia.agents.synthesizer. No real OpenAI call is ever made here."""

import pytest

from adia.agents.synthesizer import _build_messages, _SynthesizerLLMOutput, synthesize_answer
from adia.evidence.ids import compute_args_hash, generate_evidence_id
from adia.evidence.renderer import render_evidence_context
from adia.models.evidence import Evidence
from adia.models.provenance import Provenance


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
