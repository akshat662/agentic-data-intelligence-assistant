"""Tests for adia.cli. No real graph run (and therefore no real LLM call) is ever made here."""

from adia.cli import answer_question, format_result, main, run_interactive
from adia.evidence.ids import compute_args_hash, generate_evidence_id
from adia.graph.state import create_initial_state
from adia.models.evidence import Evidence
from adia.models.provenance import Provenance
from adia.models.state import AgentState, ValidationResult


def _make_evidence(tool: str, args: dict) -> Evidence:
    return Evidence(
        id=generate_evidence_id(tool, args),
        tool=tool,
        data={"row_count": 9994},
        provenance=Provenance(tool_name=tool, args=args, args_hash=compute_args_hash(args)),
    )


def _fake_run_graph(final_answer, *, passed=True, evidence=None, validation=True):
    """Build a `run_graph_fn` that ignores its state and returns a fixed, finished state."""
    evidence = evidence or {}

    def _run(initial_state: AgentState) -> AgentState:
        return initial_state.model_copy(
            update={
                "final_answer": final_answer,
                "validation": ValidationResult(passed=passed) if validation else None,
                "evidence": evidence,
            }
        )

    return _run


class TestAnswerQuestion:
    def test_formats_passed_validation_with_evidence(self):
        ev = _make_evidence("run_sql", {"query": "select 1"})
        output = answer_question(
            "superstore",
            "How many rows?",
            run_graph_fn=_fake_run_graph(
                f"There are 9994 rows [[{ev.id}]].", passed=True, evidence={ev.id: ev}
            ),
        )
        assert "Answer:" in output
        assert "There are 9994 rows" in output
        assert "Validation: PASSED" in output
        assert f"Evidence used: {ev.id}" in output

    def test_formats_failed_validation(self):
        output = answer_question(
            "superstore",
            "How many rows?",
            run_graph_fn=_fake_run_graph("fallback text", passed=False),
        )
        assert "Validation: FAILED" in output

    def test_no_evidence_omits_evidence_line(self):
        output = answer_question(
            "superstore",
            "How many rows?",
            run_graph_fn=_fake_run_graph("some answer", evidence={}),
        )
        assert "Evidence used" not in output

    def test_passes_dataset_id_and_question_through(self):
        captured = {}

        def _run(initial_state: AgentState) -> AgentState:
            captured["dataset_id"] = initial_state.dataset_id
            captured["question"] = initial_state.question
            return initial_state.model_copy(
                update={"final_answer": "ok", "validation": ValidationResult(passed=True)}
            )

        answer_question("superstore", "Which category has the highest sales?", run_graph_fn=_run)
        assert captured["dataset_id"] == "superstore"
        assert captured["question"] == "Which category has the highest sales?"


class TestFormatResult:
    def test_missing_final_answer_is_reported_honestly(self):
        state = create_initial_state("q?", "superstore")
        output = format_result(state)
        assert "No answer was produced." in output

    def test_missing_validation_result_is_reported_as_not_run(self):
        state = create_initial_state("q?", "superstore")
        state = state.model_copy(update={"final_answer": "some answer"})
        output = format_result(state)
        assert "Validation: NOT RUN" in output

    def test_validation_failure_is_clearly_reported(self):
        state = create_initial_state("q?", "superstore")
        state = state.model_copy(
            update={
                "final_answer": "The generated answer could not be verified.",
                "validation": ValidationResult(passed=False),
            }
        )
        output = format_result(state)
        assert "Validation: FAILED" in output
        assert "could not be verified" in output

    def test_multiple_evidence_ids_are_sorted(self):
        ev1 = _make_evidence("run_sql", {"query": "a"})
        ev2 = _make_evidence("profile_dataset", {"dataset_id": "superstore"})
        state = create_initial_state("q?", "superstore")
        state = state.model_copy(
            update={
                "final_answer": "answer",
                "validation": ValidationResult(passed=True),
                "evidence": {ev1.id: ev1, ev2.id: ev2},
            }
        )
        output = format_result(state)
        expected = ", ".join(sorted([ev1.id, ev2.id]))
        assert f"Evidence used: {expected}" in output


class TestRunInteractive:
    def test_reads_dataset_and_question_and_prints_answer(self, monkeypatch, capsys):
        inputs = iter(["superstore", "Which category has the highest sales?"])
        monkeypatch.setattr("builtins.input", lambda *args: next(inputs))
        monkeypatch.setattr(
            "adia.cli.answer_question",
            lambda dataset_id, question, **kwargs: f"Answer:\nmocked answer for {dataset_id}",
        )
        run_interactive()
        captured = capsys.readouterr()
        assert "Dataset:" in captured.out
        assert "Question:" in captured.out
        assert "mocked answer for superstore" in captured.out

    def test_strips_whitespace_from_inputs(self, monkeypatch):
        inputs = iter(["  superstore  ", "  How many rows?  "])
        monkeypatch.setattr("builtins.input", lambda *args: next(inputs))
        captured_args = {}

        def _fake_answer_question(dataset_id, question, **kwargs):
            captured_args["dataset_id"] = dataset_id
            captured_args["question"] = question
            return "Answer:\nok"

        monkeypatch.setattr("adia.cli.answer_question", _fake_answer_question)
        run_interactive()
        assert captured_args["dataset_id"] == "superstore"
        assert captured_args["question"] == "How many rows?"

    def test_never_calls_the_real_graph_runner(self, monkeypatch):
        def _fail_if_called(*args, **kwargs):
            raise AssertionError("the real graph runner must never be invoked in tests")

        monkeypatch.setattr("adia.cli.run_graph", _fail_if_called)
        inputs = iter(["superstore", "How many rows?"])
        monkeypatch.setattr("builtins.input", lambda *args: next(inputs))
        monkeypatch.setattr(
            "adia.cli.answer_question", lambda dataset_id, question, **kwargs: "Answer:\nok"
        )
        run_interactive()  # would raise via _fail_if_called if the real path were reachable


class TestMain:
    def test_normal_run_returns_zero(self, monkeypatch):
        monkeypatch.setattr("adia.cli.run_interactive", lambda: None)
        assert main() == 0

    def test_eof_during_input_returns_one(self, monkeypatch):
        def _raise_eof():
            raise EOFError

        monkeypatch.setattr("adia.cli.run_interactive", _raise_eof)
        assert main() == 1

    def test_keyboard_interrupt_returns_one(self, monkeypatch):
        def _raise_interrupt():
            raise KeyboardInterrupt

        monkeypatch.setattr("adia.cli.run_interactive", _raise_interrupt)
        assert main() == 1
