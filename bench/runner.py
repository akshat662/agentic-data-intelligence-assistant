"""Benchmark runner: drives the full ADIA graph over `bench/questions.json`.

Purely a driver over already-built pieces -- `bench.schema.load_questions` for the question
set, `adia.graph.workflow.run_graph` for the actual pipeline, `adia.graph.state` to seed each
run. It adds no judgment of its own about correctness: whether an answer matches an
independent oracle is Phase 3 scope (see `bench/README.md`). What this records is narrower and
purely observational -- did the run complete without crashing, what verdict feasibility
reached, which tools actually produced evidence, how many plan steps were proposed versus
actually executed, did the answer pass static validation, and what the final answer text was
-- one `QuestionResult` per question, saved as one JSON array. Turning these observations into
tier-grouped, direct-vs-investigation comparison metrics is `bench/evaluation_report.py`'s job,
not this module's -- this module only records what happened, once, per question.
"""

import json
import time
from pathlib import Path

from bench.schema import BenchmarkQuestion, load_questions
from pydantic import BaseModel, ConfigDict, Field

from adia.graph.state import create_initial_state
from adia.graph.workflow import run_graph

_QUESTIONS_PATH = Path(__file__).resolve().parent / "questions.json"
_RESULTS_PATH = Path(__file__).resolve().parent / "results" / "results.json"


class QuestionResult(BaseModel):
    """The observed outcome of running one benchmark question through the full graph.

    Records what happened, not whether it was *correct* -- there is no independent oracle to
    grade against yet (see `bench/README.md`'s "Implemented vs. Planned").
    """

    model_config = ConfigDict(frozen=True)

    question_id: str
    dataset_id: str
    category: str
    question: str
    expected_answerable: bool
    success: bool = Field(
        ..., description="Whether the graph ran to completion without an unhandled exception."
    )
    feasibility_verdict: str | None = None
    feasibility_reason: str | None = None
    tools_used: list[str] = Field(
        default_factory=list, description="Distinct tool names that produced evidence."
    )
    validation_passed: bool | None = None
    final_answer: str | None = None
    error: str | None = Field(
        default=None, description="Exception message from the graph run, if `success` is False."
    )
    duration_ms: float = Field(..., ge=0)

    plan_step_count: int = Field(
        default=0, ge=0, description="Number of steps the planner proposed."
    )
    executed_step_count: int = Field(
        default=0,
        ge=0,
        description="Number of those plan steps that actually produced evidence -- bounded "
        "by plan_step_count; lower than it means a step was rejected or failed.",
    )
    evidence_count: int = Field(
        default=0, ge=0, description="Total evidence records produced by the run."
    )
    evidence_coverage: float | None = Field(
        default=None,
        description="executed_step_count / plan_step_count, or None when no plan was "
        "proposed (e.g. a refused question) -- there is nothing to cover.",
    )


def run_question(question: BenchmarkQuestion) -> QuestionResult:
    """Run one benchmark question through the full graph and record its outcome.

    Never raises: every node in the graph already catches its own failures into typed state
    (a `ToolError`, an `INFEASIBLE` verdict, a validation-fallback answer), so an exception
    here means something outside that contract broke (e.g. the dataset registry itself). That
    is caught too, and recorded as `success=False`, so one bad question can't abort the rest
    of a benchmark run.

    Args:
        question: The benchmark question to run.

    Returns:
        A `QuestionResult` describing what happened.
    """
    started = time.perf_counter()
    try:
        initial_state = create_initial_state(question.question, question.dataset_id)
        final_state = run_graph(initial_state)

        plan_step_ids = {step.id for step in final_state.plan}
        executed_step_ids = {
            evidence.plan_step_id
            for evidence in final_state.evidence.values()
            if evidence.plan_step_id in plan_step_ids
        }
        plan_step_count = len(plan_step_ids)
        executed_step_count = len(executed_step_ids)

        return QuestionResult(
            question_id=question.id,
            dataset_id=question.dataset_id,
            category=question.category.value,
            question=question.question,
            expected_answerable=question.answerable,
            success=True,
            feasibility_verdict=(
                final_state.feasibility.verdict.value if final_state.feasibility else None
            ),
            feasibility_reason=(
                final_state.feasibility.reason if final_state.feasibility else None
            ),
            tools_used=sorted({e.tool for e in final_state.evidence.values()}),
            validation_passed=(
                final_state.validation.passed if final_state.validation else None
            ),
            final_answer=final_state.final_answer,
            duration_ms=_elapsed_ms(started),
            plan_step_count=plan_step_count,
            executed_step_count=executed_step_count,
            evidence_count=len(final_state.evidence),
            evidence_coverage=(
                executed_step_count / plan_step_count if plan_step_count else None
            ),
        )
    except Exception as exc:  # one bad question must never abort the whole benchmark run
        return QuestionResult(
            question_id=question.id,
            dataset_id=question.dataset_id,
            category=question.category.value,
            question=question.question,
            expected_answerable=question.answerable,
            success=False,
            error=str(exc),
            duration_ms=_elapsed_ms(started),
        )


def run_benchmark(questions_path: str | Path = _QUESTIONS_PATH) -> list[QuestionResult]:
    """Run every question in `questions_path` through the full graph.

    Args:
        questions_path: Path to a benchmark questions JSON file, in `bench.schema`'s format.
            Defaults to the shipped `bench/questions.json`.

    Returns:
        One `QuestionResult` per question, in file order.
    """
    questions = load_questions(questions_path)
    return [run_question(question) for question in questions]


def save_results(results: list[QuestionResult], path: str | Path = _RESULTS_PATH) -> None:
    """Persist benchmark results to a JSON file, creating its parent directory if needed.

    Args:
        results: Results to save, e.g. from `run_benchmark`.
        path: Destination file path. Defaults to `bench/results/results.json`.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    payload = [result.model_dump(mode="json") for result in results]
    target.write_text(json.dumps(payload, indent=2))


def _elapsed_ms(started: float) -> float:
    """Milliseconds elapsed since a `time.perf_counter()` reading."""
    return (time.perf_counter() - started) * 1000


def main() -> None:
    """CLI entry point: run the full shipped benchmark and save results.

    Runs every question in `bench/questions.json` through the real graph -- including real
    LLM calls at every agent -- and writes `bench/results/results.json`.
    """
    results = run_benchmark()
    save_results(results)
    succeeded = sum(1 for r in results if r.success)
    validated = sum(1 for r in results if r.validation_passed)
    print(f"Ran {len(results)} questions: {succeeded} completed, {validated} passed validation.")
    print(f"Results written to {_RESULTS_PATH}")


if __name__ == "__main__":
    main()
