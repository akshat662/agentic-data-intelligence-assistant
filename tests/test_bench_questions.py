"""Tests for bench.schema."""

import json
from pathlib import Path

import pytest
from bench.schema import BenchmarkQuestion, QuestionCategory, load_questions
from pydantic import ValidationError

_SHIPPED_QUESTIONS_PATH = Path(__file__).parent.parent / "bench" / "questions.json"


def _question(**overrides) -> dict:
    defaults = dict(
        id="q001",
        dataset_id="template_dataset",
        category="descriptive",
        question="How many rows are there?",
        answerable=True,
        gold_tools=["profile_dataset"],
    )
    defaults.update(overrides)
    return defaults


class TestBenchmarkQuestionValidation:
    def test_valid_answerable_question(self):
        question = BenchmarkQuestion(**_question())
        assert question.category == QuestionCategory.DESCRIPTIVE

    def test_valid_unanswerable_question(self):
        question = BenchmarkQuestion(
            **_question(category="unanswerable", answerable=False, gold_tools=[])
        )
        assert question.answerable is False

    def test_unanswerable_category_with_answerable_true_rejected(self):
        with pytest.raises(ValidationError, match="cannot be answerable"):
            BenchmarkQuestion(**_question(category="unanswerable", answerable=True))

    def test_answerable_category_with_answerable_false_rejected(self):
        with pytest.raises(ValidationError, match="Only 'unanswerable'"):
            BenchmarkQuestion(**_question(category="descriptive", answerable=False))

    def test_invalid_category_rejected(self):
        with pytest.raises(ValidationError):
            BenchmarkQuestion(**_question(category="forecasting"))

    def test_missing_question_text_rejected(self):
        payload = _question()
        del payload["question"]
        with pytest.raises(ValidationError):
            BenchmarkQuestion(**payload)

    def test_gold_tools_defaults_to_empty_list(self):
        payload = _question()
        del payload["gold_tools"]
        question = BenchmarkQuestion(**payload)
        assert question.gold_tools == []


class TestLoadQuestions:
    def test_loads_bare_array(self, tmp_path):
        path = tmp_path / "questions.json"
        path.write_text(json.dumps([_question(), _question(id="q002")]))
        questions = load_questions(path)
        assert [q.id for q in questions] == ["q001", "q002"]

    def test_loads_wrapped_object(self, tmp_path):
        path = tmp_path / "questions.json"
        path.write_text(json.dumps({"schema_version": 1, "questions": [_question()]}))
        questions = load_questions(path)
        assert len(questions) == 1

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_questions(tmp_path / "does_not_exist.json")

    def test_duplicate_id_rejected(self, tmp_path):
        path = tmp_path / "questions.json"
        path.write_text(json.dumps([_question(), _question()]))
        with pytest.raises(ValueError, match="Duplicate question id"):
            load_questions(path)


class TestShippedQuestionsFile:
    def test_shipped_file_is_valid(self):
        questions = load_questions(_SHIPPED_QUESTIONS_PATH)
        assert len(questions) > 0

    def test_shipped_file_covers_every_category(self):
        questions = load_questions(_SHIPPED_QUESTIONS_PATH)
        categories = {q.category for q in questions}
        assert categories == set(QuestionCategory)

    def test_shipped_file_references_registered_dataset_only(self):
        questions = load_questions(_SHIPPED_QUESTIONS_PATH)
        assert {q.dataset_id for q in questions} == {"superstore"}

    def test_shipped_questions_dataset_is_registered(self):
        from adia.data.registry import load_registry

        questions = load_questions(_SHIPPED_QUESTIONS_PATH)
        registry = load_registry(_SHIPPED_QUESTIONS_PATH.parent.parent / "data" / "registry.json")
        for question in questions:
            assert question.dataset_id in registry

    def test_shipped_file_has_expected_category_counts(self):
        questions = load_questions(_SHIPPED_QUESTIONS_PATH)
        counts: dict[str, int] = {}
        for question in questions:
            counts[question.category] = counts.get(question.category, 0) + 1
        assert counts[QuestionCategory.SQL_AGGREGATION] == 5
        assert counts[QuestionCategory.STATISTICAL_COMPARISON] == 5
        assert counts[QuestionCategory.PREDICTIVE] == 5
        assert counts[QuestionCategory.UNANSWERABLE] == 5

    def test_shipped_file_unanswerable_questions_cite_no_gold_tools(self):
        questions = load_questions(_SHIPPED_QUESTIONS_PATH)
        for question in questions:
            if question.category == QuestionCategory.UNANSWERABLE:
                assert question.gold_tools == []
