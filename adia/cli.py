"""Minimal interactive CLI for the ADIA graph -- `python -m adia`.

This is a thin interface layer only: it collects a `dataset_id` and a question, hands them to
the same path `bench/runner.py` already drives the benchmark through
(`adia.graph.state.create_initial_state` -> `adia.graph.workflow.run_graph`), and formats what
the graph produced for the terminal. No feasibility, planning, tool execution, synthesis, or
validation logic lives here or is duplicated here -- every decision that matters is still made
inside the graph; this module only asks a question and prints the answer.
"""

import sys
from collections.abc import Callable

from adia.graph.state import create_initial_state
from adia.graph.workflow import run_graph
from adia.models.state import AgentState

#: Signature every graph runner -- the real one or a test's fake -- must satisfy.
GraphRunner = Callable[[AgentState], AgentState]


def answer_question(
    dataset_id: str,
    question: str,
    *,
    run_graph_fn: GraphRunner = run_graph,
) -> str:
    """Run one question through the ADIA graph and format the result for the terminal.

    Args:
        dataset_id: Dataset to answer the question against. Resolved against the dataset
            registry by `feasibility_node`, not here -- an unregistered ID surfaces as a
            grounded refusal from the graph itself, not a CLI-level error.
        question: The user's question, verbatim.
        run_graph_fn: Override for the graph runner, e.g. a fake for tests. Defaults to the
            real `adia.graph.workflow.run_graph`.

    Returns:
        Formatted, human-readable text: the final answer, validation status, and (if any)
        the evidence IDs the answer cites.
    """
    initial_state = create_initial_state(question, dataset_id)
    final_state = run_graph_fn(initial_state)
    return format_result(final_state)


def format_result(state: AgentState) -> str:
    """Format a finished `AgentState` into the CLI's answer/validation/evidence block.

    Args:
        state: The `AgentState` returned by `adia.graph.workflow.run_graph`.

    Returns:
        Text with three parts: the answer (or an honest "no answer" line), the validation
        status derived from `state.validation`, and, only when evidence exists, the sorted
        list of evidence IDs the run produced.
    """
    lines = ["Answer:", state.final_answer or "No answer was produced."]

    lines.append("")
    if state.validation is not None:
        status = "PASSED" if state.validation.passed else "FAILED"
        lines.append(f"Validation: {status}")
    else:
        lines.append("Validation: NOT RUN")

    if state.evidence:
        lines.append(f"Evidence used: {', '.join(sorted(state.evidence))}")

    return "\n".join(lines)


def run_interactive() -> None:
    """Prompt for a dataset ID and question on stdin, then print the formatted answer."""
    print("Dataset:")
    dataset_id = input().strip()
    print()
    print("Question:")
    question = input("> ").strip()
    print()
    print(answer_question(dataset_id, question))


def main() -> int:
    """CLI entry point for `python -m adia`.

    Returns:
        `0` on a normal run, `1` if input was interrupted (Ctrl-D/Ctrl-C) before both prompts
        were answered.
    """
    try:
        run_interactive()
    except (EOFError, KeyboardInterrupt):
        print()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
