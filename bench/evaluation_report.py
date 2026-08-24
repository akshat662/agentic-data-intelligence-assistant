"""Evaluation report: turns `bench/results/results.json` into tier-grouped comparison metrics.

This is the piece that makes ADIA's "more than NL-to-SQL" claim checkable rather than
asserted: it groups the same `QuestionResult` records `bench/runner.py` already produces by
`bench.schema.EvaluationTier` (direct / investigation / refusal) and reports, per tier, how
many plan steps were proposed and executed, how much evidence was produced, and whether
validation passed -- so "direct questions average ~1 step, investigation questions average
several" is a number this module prints, not a claim made in prose. It also computes refusal
recall, false-refusal rate, and a deterministic forbidden-causal-phrase scan against each
investigation question's `InvestigationExpectation` -- see that model's own docstring for why
this is the one piece of investigation metadata that's checked automatically today, and why
the rest (`expected_observation`, `acceptable_conclusion_style`) is a rubric for a human
reviewer, not something graded here.

Purely a second pass over already-produced data: no graph call, no LLM call, no new judgment
about correctness beyond string-matching a fixed phrase list. Mirrors `bench/runner.py`'s own
shape (pure functions plus a thin `main()`), not a different pattern.
"""

import json
from datetime import UTC, datetime
from pathlib import Path

from bench.runner import QuestionResult
from bench.schema import BenchmarkQuestion, EvaluationTier, load_questions
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

_QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
_RESULTS_PATH = Path(__file__).resolve().parent / "results" / "results.json"
_REPORT_MARKDOWN_PATH = Path(__file__).resolve().parent / "results" / "evaluation_report.md"
_REPORT_JSON_PATH = Path(__file__).resolve().parent / "results" / "evaluation_report.json"

_results_list_adapter: TypeAdapter[list[QuestionResult]] = TypeAdapter(list[QuestionResult])


class TierMetrics(BaseModel):
    """Aggregate metrics for every question in one `EvaluationTier`."""

    model_config = ConfigDict(frozen=True)

    tier: str
    question_count: int = Field(..., ge=0)
    avg_plan_step_count: float = Field(..., ge=0)
    avg_executed_step_count: float = Field(..., ge=0)
    avg_evidence_count: float = Field(..., ge=0)
    avg_evidence_coverage: float | None = Field(
        default=None, description="Mean of evidence_coverage over questions where it's set."
    )
    success_rate: float = Field(..., ge=0, le=1)
    validation_pass_rate: float = Field(
        ..., ge=0, le=1, description="Fraction with validation_passed=True."
    )


class RefusalMetrics(BaseModel):
    """Whether the system refused exactly the questions it should have, and no others."""

    model_config = ConfigDict(frozen=True)

    refusal_recall: float | None = Field(
        default=None,
        description="Of expected-unanswerable questions, fraction actually refused "
        "(feasibility_verdict != 'feasible'). None if there are none.",
    )
    refusal_recall_count: str = Field(..., description="'correct/total' for refusal_recall.")
    false_refusal_rate: float | None = Field(
        default=None,
        description="Of expected-answerable questions, fraction wrongly refused. None if "
        "there are none.",
    )
    false_refusal_count: str = Field(..., description="'wrong/total' for false_refusal_rate.")


class CausalPhraseFinding(BaseModel):
    """A forbidden causal phrase found verbatim in an investigation question's final answer."""

    model_config = ConfigDict(frozen=True)

    question_id: str
    phrases_found: list[str]


class EvaluationSummary(BaseModel):
    """The full evaluation report, as both the JSON summary and the source for the Markdown."""

    model_config = ConfigDict(frozen=True)

    generated_at: str
    total_questions: int = Field(..., ge=0)
    tier_metrics: list[TierMetrics]
    refusal_metrics: RefusalMetrics
    causal_phrase_findings: list[CausalPhraseFinding] = Field(
        default_factory=list,
        description="Only questions where a forbidden phrase was actually found -- an empty "
        "list means the scan found nothing, not that it didn't run.",
    )


def generate_summary(
    results: list[QuestionResult], questions: list[BenchmarkQuestion]
) -> EvaluationSummary:
    """Compute the full evaluation summary from already-run results and their question defs.

    Args:
        results: Benchmark results, e.g. from `bench.runner.run_benchmark`.
        questions: The question definitions those results were produced from, e.g. from
            `bench.schema.load_questions`. Joined to `results` by `question_id`/`id`.

    Returns:
        The computed `EvaluationSummary`. Pure function -- no I/O, no LLM call.
    """
    questions_by_id = {question.id: question for question in questions}

    tier_metrics = [
        _tier_metrics(tier, [r for r in results if _tier_of(r, questions_by_id) == tier])
        for tier in EvaluationTier
    ]

    return EvaluationSummary(
        generated_at=datetime.now(UTC).isoformat(),
        total_questions=len(results),
        tier_metrics=tier_metrics,
        refusal_metrics=_refusal_metrics(results),
        causal_phrase_findings=_causal_phrase_findings(results, questions_by_id),
    )


def _tier_of(result: QuestionResult, questions_by_id: dict[str, BenchmarkQuestion]) -> str:
    """Look up a result's evaluation tier via its question definition."""
    question = questions_by_id.get(result.question_id)
    return question.evaluation_tier.value if question is not None else "unknown"


def _tier_metrics(tier: EvaluationTier, tier_results: list[QuestionResult]) -> TierMetrics:
    """Aggregate one tier's worth of results. Empty tiers report all-zero, not a crash."""
    if not tier_results:
        return TierMetrics(
            tier=tier.value,
            question_count=0,
            avg_plan_step_count=0.0,
            avg_executed_step_count=0.0,
            avg_evidence_count=0.0,
            avg_evidence_coverage=None,
            success_rate=0.0,
            validation_pass_rate=0.0,
        )

    coverages = [r.evidence_coverage for r in tier_results if r.evidence_coverage is not None]
    return TierMetrics(
        tier=tier.value,
        question_count=len(tier_results),
        avg_plan_step_count=_mean(r.plan_step_count for r in tier_results),
        avg_executed_step_count=_mean(r.executed_step_count for r in tier_results),
        avg_evidence_count=_mean(r.evidence_count for r in tier_results),
        avg_evidence_coverage=_mean(coverages) if coverages else None,
        success_rate=_mean(1.0 if r.success else 0.0 for r in tier_results),
        validation_pass_rate=_mean(1.0 if r.validation_passed else 0.0 for r in tier_results),
    )


def _refusal_metrics(results: list[QuestionResult]) -> RefusalMetrics:
    """Refusal recall (over expected-unanswerable questions) and false-refusal rate (over
    expected-answerable ones), computed purely from each result's own recorded fields."""
    expected_unanswerable = [r for r in results if not r.expected_answerable]
    expected_answerable = [r for r in results if r.expected_answerable]

    refused_correctly = sum(1 for r in expected_unanswerable if r.feasibility_verdict != "feasible")
    refused_wrongly = sum(1 for r in expected_answerable if r.feasibility_verdict != "feasible")

    return RefusalMetrics(
        refusal_recall=(
            refused_correctly / len(expected_unanswerable) if expected_unanswerable else None
        ),
        refusal_recall_count=f"{refused_correctly}/{len(expected_unanswerable)}",
        false_refusal_rate=(
            refused_wrongly / len(expected_answerable) if expected_answerable else None
        ),
        false_refusal_count=f"{refused_wrongly}/{len(expected_answerable)}",
    )


def _causal_phrase_findings(
    results: list[QuestionResult], questions_by_id: dict[str, BenchmarkQuestion]
) -> list[CausalPhraseFinding]:
    """Scan each investigation question's final answer for its own forbidden causal phrases.

    Deterministic and independent of `adia.validate.static`'s own causal-language check: this
    re-checks the final rendered text against a fixed phrase list defined per-question in
    `bench/questions.json`, not the validator's regex -- a second, separate check in the same
    spirit as this project's "independent oracles, not self-grading" evaluation philosophy.
    """
    findings: list[CausalPhraseFinding] = []
    for result in results:
        question = questions_by_id.get(result.question_id)
        if question is None or question.investigation is None or not result.final_answer:
            continue
        answer_lower = result.final_answer.lower()
        found = [
            phrase
            for phrase in question.investigation.forbidden_causal_phrases
            if phrase.lower() in answer_lower
        ]
        if found:
            findings.append(
                CausalPhraseFinding(question_id=result.question_id, phrases_found=found)
            )
    return findings


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def render_markdown(summary: EvaluationSummary) -> str:
    """Render an `EvaluationSummary` as a human-readable Markdown report."""
    lines = [
        "# ADIA Benchmark Evaluation Report",
        "",
        f"Generated: {summary.generated_at}",
        f"Total questions: {summary.total_questions}",
        "",
        "## Metrics by Tier",
        "",
        "| Tier | Questions | Avg Plan Steps | Avg Executed Steps | Avg Evidence Count | "
        "Avg Evidence Coverage | Success Rate | Validation Pass Rate |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for metrics in summary.tier_metrics:
        coverage = (
            f"{metrics.avg_evidence_coverage:.1%}"
            if metrics.avg_evidence_coverage is not None
            else "N/A"
        )
        lines.append(
            f"| {metrics.tier} | {metrics.question_count} | "
            f"{metrics.avg_plan_step_count:.2f} | {metrics.avg_executed_step_count:.2f} | "
            f"{metrics.avg_evidence_count:.2f} | {coverage} | "
            f"{metrics.success_rate:.1%} | {metrics.validation_pass_rate:.1%} |"
        )

    lines += ["", "## Refusal Correctness", ""]
    refusal = summary.refusal_metrics
    recall_text = (
        f"{refusal.refusal_recall:.1%}" if refusal.refusal_recall is not None else "N/A"
    )
    false_refusal_text = (
        f"{refusal.false_refusal_rate:.1%}" if refusal.false_refusal_rate is not None else "N/A"
    )
    lines.append(
        f"- Refusal recall (unanswerable questions correctly refused): "
        f"{refusal.refusal_recall_count} ({recall_text})"
    )
    lines.append(
        f"- False refusal rate (answerable questions incorrectly refused): "
        f"{refusal.false_refusal_count} ({false_refusal_text})"
    )

    lines += ["", "## Forbidden Causal Phrase Scan (investigation questions)", ""]
    if summary.causal_phrase_findings:
        for finding in summary.causal_phrase_findings:
            phrases = ", ".join(f"'{p}'" for p in finding.phrases_found)
            lines.append(f"- **{finding.question_id}**: found {phrases}")
    else:
        lines.append("No forbidden causal phrases found in any investigation-tier answer.")

    return "\n".join(lines) + "\n"


def save_report(
    summary: EvaluationSummary,
    *,
    markdown_path: str | Path = _REPORT_MARKDOWN_PATH,
    json_path: str | Path = _REPORT_JSON_PATH,
) -> None:
    """Write the Markdown report and the JSON summary, creating parent directories as needed."""
    markdown_target = Path(markdown_path)
    markdown_target.parent.mkdir(parents=True, exist_ok=True)
    markdown_target.write_text(render_markdown(summary))

    json_target = Path(json_path)
    json_target.parent.mkdir(parents=True, exist_ok=True)
    json_target.write_text(json.dumps(summary.model_dump(mode="json"), indent=2))


def load_results(path: str | Path = _RESULTS_PATH) -> list[QuestionResult]:
    """Load previously-saved benchmark results, e.g. from `bench.runner.save_results`.

    Raises:
        FileNotFoundError: If `path` does not exist -- run `python -m bench.runner` first.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(
            f"Benchmark results not found: {resolved}. Run `python -m bench.runner` first."
        )
    return _results_list_adapter.validate_json(resolved.read_bytes())


def main() -> None:
    """CLI entry point: build the evaluation report from the last benchmark run and save it."""
    results = load_results()
    questions = load_questions(_QUESTIONS_PATH)
    summary = generate_summary(results, questions)
    save_report(summary)
    print(render_markdown(summary))
    print(f"Report written to {_REPORT_MARKDOWN_PATH} and {_REPORT_JSON_PATH}")


if __name__ == "__main__":
    main()
