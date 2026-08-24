"""Tests for bench.evaluation_report.

Pure aggregation logic over already-built `QuestionResult`/`BenchmarkQuestion` objects --
no graph run, no LLM call, no real files beyond what a test itself writes to tmp_path.
"""

import json

import pytest
from bench.evaluation_report import (
    generate_summary,
    load_results,
    render_markdown,
    save_report,
)
from bench.runner import QuestionResult
from bench.schema import BenchmarkQuestion, InvestigationExpectation


def _question(**overrides) -> BenchmarkQuestion:
    defaults = dict(
        id="q001",
        dataset_id="superstore",
        category="sql_aggregation",
        question="Total sales by category?",
        answerable=True,
        gold_tools=["run_sql"],
    )
    defaults.update(overrides)
    return BenchmarkQuestion(**defaults)


def _investigation_question(**overrides) -> BenchmarkQuestion:
    investigation = InvestigationExpectation(
        expected_observation="Office Supplies has the lowest total Sales.",
        expected_analysis_dimensions=["order volume", "average order value"],
        acceptable_conclusion_style="States the fact plainly; hedges any explanation.",
        forbidden_causal_phrases=["causes", "due to"],
    )
    defaults = dict(
        id="q017",
        dataset_id="superstore",
        category="root_cause",
        question="Why does Office Supplies have the lowest total Sales?",
        answerable=True,
        gold_tools=["run_sql", "compare_groups"],
        investigation=investigation,
    )
    defaults.update(overrides)
    return BenchmarkQuestion(**defaults)


def _result(**overrides) -> QuestionResult:
    defaults = dict(
        question_id="q001",
        dataset_id="superstore",
        category="sql_aggregation",
        question="Total sales by category?",
        expected_answerable=True,
        success=True,
        feasibility_verdict="feasible",
        feasibility_reason="ok",
        tools_used=["run_sql"],
        validation_passed=True,
        final_answer="Office Supplies has the lowest total Sales [[ev_run_sql_deadbeef]].",
        duration_ms=10.0,
        plan_step_count=1,
        executed_step_count=1,
        evidence_count=1,
        evidence_coverage=1.0,
    )
    defaults.update(overrides)
    return QuestionResult(**defaults)


class TestGenerateSummaryTierGrouping:
    def test_direct_and_investigation_and_refusal_tiers_are_separated(self):
        direct_q = _question(id="q001")
        investigation_q = _investigation_question(id="q017")
        refusal_q = _question(
            id="q018", category="unanswerable", answerable=False, gold_tools=[]
        )
        results = [
            _result(question_id="q001", plan_step_count=1, executed_step_count=1),
            _result(
                question_id="q017",
                category="root_cause",
                plan_step_count=5,
                executed_step_count=5,
                evidence_count=5,
                evidence_coverage=1.0,
            ),
            _result(
                question_id="q018",
                category="unanswerable",
                expected_answerable=False,
                feasibility_verdict="infeasible",
                plan_step_count=0,
                executed_step_count=0,
                evidence_count=0,
                evidence_coverage=None,
                final_answer="This question cannot be answered.",
            ),
        ]
        summary = generate_summary(results, [direct_q, investigation_q, refusal_q])
        by_tier = {m.tier: m for m in summary.tier_metrics}

        assert by_tier["direct"].question_count == 1
        assert by_tier["direct"].avg_plan_step_count == 1.0
        assert by_tier["investigation"].question_count == 1
        assert by_tier["investigation"].avg_plan_step_count == 5.0
        assert by_tier["refusal"].question_count == 1
        assert by_tier["refusal"].avg_plan_step_count == 0.0

    def test_investigation_tier_shows_more_steps_than_direct_tier(self):
        # The headline comparison this whole module exists to make measurable.
        direct_q = _question(id="q001")
        investigation_q = _investigation_question(id="q017")
        results = [
            _result(question_id="q001", plan_step_count=1, executed_step_count=1),
            _result(
                question_id="q017",
                category="root_cause",
                plan_step_count=5,
                executed_step_count=5,
                evidence_count=5,
            ),
        ]
        summary = generate_summary(results, [direct_q, investigation_q])
        by_tier = {m.tier: m for m in summary.tier_metrics}
        assert by_tier["investigation"].avg_plan_step_count > by_tier["direct"].avg_plan_step_count

    def test_empty_tier_reports_zero_not_a_crash(self):
        direct_q = _question(id="q001")
        results = [_result(question_id="q001")]
        summary = generate_summary(results, [direct_q])
        by_tier = {m.tier: m for m in summary.tier_metrics}
        assert by_tier["investigation"].question_count == 0
        assert by_tier["investigation"].avg_plan_step_count == 0.0
        assert by_tier["investigation"].avg_evidence_coverage is None


class TestRefusalMetrics:
    def test_refusal_recall_counts_correctly_refused_questions(self):
        refusal_q = _question(
            id="q018", category="unanswerable", answerable=False, gold_tools=[]
        )
        results = [
            _result(
                question_id="q018",
                expected_answerable=False,
                feasibility_verdict="infeasible",
                plan_step_count=0,
                executed_step_count=0,
                evidence_count=0,
                evidence_coverage=None,
            )
        ]
        summary = generate_summary(results, [refusal_q])
        assert summary.refusal_metrics.refusal_recall == 1.0
        assert summary.refusal_metrics.refusal_recall_count == "1/1"

    def test_false_refusal_rate_counts_wrongly_refused_answerable_questions(self):
        direct_q = _question(id="q001")
        results = [
            _result(question_id="q001", feasibility_verdict="infeasible", success=True)
        ]
        summary = generate_summary(results, [direct_q])
        assert summary.refusal_metrics.false_refusal_rate == 1.0
        assert summary.refusal_metrics.false_refusal_count == "1/1"

    def test_correctly_answered_question_has_zero_false_refusal_rate(self):
        direct_q = _question(id="q001")
        results = [_result(question_id="q001", feasibility_verdict="feasible")]
        summary = generate_summary(results, [direct_q])
        assert summary.refusal_metrics.false_refusal_rate == 0.0

    def test_no_unanswerable_questions_reports_none_not_a_crash(self):
        direct_q = _question(id="q001")
        results = [_result(question_id="q001")]
        summary = generate_summary(results, [direct_q])
        assert summary.refusal_metrics.refusal_recall is None
        assert summary.refusal_metrics.refusal_recall_count == "0/0"


class TestForbiddenCausalPhraseScan:
    def test_forbidden_phrase_present_is_reported(self):
        investigation_q = _investigation_question(id="q017")
        results = [
            _result(
                question_id="q017",
                category="root_cause",
                final_answer="Lower average order value causes the lower total Sales.",
            )
        ]
        summary = generate_summary(results, [investigation_q])
        assert len(summary.causal_phrase_findings) == 1
        assert summary.causal_phrase_findings[0].question_id == "q017"
        assert "causes" in summary.causal_phrase_findings[0].phrases_found

    def test_clean_answer_produces_no_findings(self):
        investigation_q = _investigation_question(id="q017")
        results = [
            _result(
                question_id="q017",
                category="root_cause",
                final_answer="Lower average order value is associated with lower total Sales.",
            )
        ]
        summary = generate_summary(results, [investigation_q])
        assert summary.causal_phrase_findings == []

    def test_scan_is_case_insensitive(self):
        investigation_q = _investigation_question(id="q017")
        results = [
            _result(
                question_id="q017",
                category="root_cause",
                final_answer="This DUE TO the difference in order value.",
            )
        ]
        summary = generate_summary(results, [investigation_q])
        assert len(summary.causal_phrase_findings) == 1

    def test_non_investigation_question_is_never_scanned(self):
        direct_q = _question(id="q001")
        results = [
            _result(question_id="q001", final_answer="Sales are 100 because of high demand.")
        ]
        summary = generate_summary(results, [direct_q])
        assert summary.causal_phrase_findings == []


class TestRenderMarkdown:
    def test_report_contains_tier_table_and_sections(self):
        direct_q = _question(id="q001")
        results = [_result(question_id="q001")]
        summary = generate_summary(results, [direct_q])
        markdown = render_markdown(summary)
        assert "# ADIA Benchmark Evaluation Report" in markdown
        assert "## Metrics by Tier" in markdown
        assert "## Refusal Correctness" in markdown
        assert "## Forbidden Causal Phrase Scan" in markdown
        assert "direct" in markdown

    def test_findings_are_listed_by_question_id(self):
        investigation_q = _investigation_question(id="q017")
        results = [
            _result(
                question_id="q017",
                category="root_cause",
                final_answer="This causes the difference.",
            )
        ]
        summary = generate_summary(results, [investigation_q])
        markdown = render_markdown(summary)
        assert "q017" in markdown
        assert "causes" in markdown


class TestSaveReportAndLoadResults:
    def test_save_report_writes_both_files(self, tmp_path):
        direct_q = _question(id="q001")
        results = [_result(question_id="q001")]
        summary = generate_summary(results, [direct_q])
        markdown_path = tmp_path / "report.md"
        json_path = tmp_path / "report.json"
        save_report(summary, markdown_path=markdown_path, json_path=json_path)

        assert markdown_path.exists()
        assert json_path.exists()
        saved = json.loads(json_path.read_text())
        assert saved["total_questions"] == 1

    def test_load_results_round_trips(self, tmp_path):
        from bench.runner import save_results

        results = [_result(question_id="q001")]
        path = tmp_path / "results.json"
        save_results(results, path)

        loaded = load_results(path)
        assert len(loaded) == 1
        assert loaded[0].question_id == "q001"

    def test_load_results_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path / "does_not_exist.json")
