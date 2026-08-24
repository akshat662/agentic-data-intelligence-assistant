"""The LangGraph workflow: one conditional branch on feasibility, otherwise linear.

                          +--> planner -> execute_tools -> synthesizer --+
                          |                                              |
    START -> feasibility -+                                              +-> validation -> END
                          |                                              |
                          +--> refusal ---------------------------------+

`route_after_feasibility` is the one place this graph reacts to a node's judgment rather than
just recording it: a `state.feasibility.verdict` other than `FEASIBLE` (infeasible, or needs
clarification) routes straight to `refusal_node`, skipping `planner_node` and
`execute_tools_node` entirely — there is nothing to plan or execute for a question already
ruled out. Both branches rejoin at `validation_node`, which stays the single gate on
`final_answer` regardless of which path produced `rendered_answer`. No repair loop or replan
branch exists yet; a validation failure resolves to a fixed fallback answer inside
`validation_node` itself, not a cycle back into the graph.
"""

from collections.abc import Iterator
from typing import Any

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from adia.graph.nodes import (
    execute_tools_node,
    feasibility_node,
    planner_node,
    refusal_node,
    synthesizer_node,
    validation_node,
)
from adia.graph.state import finalize_state
from adia.models.state import AgentState, FeasibilityVerdict


def route_after_feasibility(state: AgentState) -> str:
    """Decide whether `feasibility_node`'s verdict warrants planning or an outright refusal.

    Args:
        state: State as returned by `feasibility_node` — `state.feasibility` is expected to
            be populated; a missing result (shouldn't happen in practice) is treated the same
            as a non-`FEASIBLE` verdict, since there is nothing to plan against either way.

    Returns:
        `"planner"` if `state.feasibility.verdict == FeasibilityVerdict.FEASIBLE`, else
        `"refusal"`.
    """
    if state.feasibility is not None and state.feasibility.verdict == FeasibilityVerdict.FEASIBLE:
        return "planner"
    return "refusal"


def build_graph() -> CompiledStateGraph:
    """Construct and compile the ADIA workflow graph.

    `AgentState` is used directly as the graph's state schema (see `adia/graph/state.py`).

    Returns:
        A compiled LangGraph graph, ready for `.invoke(some_agent_state)`.
    """
    builder = StateGraph(AgentState)

    builder.add_node("feasibility", feasibility_node)
    builder.add_node("planner", planner_node)
    builder.add_node("execute_tools", execute_tools_node)
    builder.add_node("synthesizer", synthesizer_node)
    builder.add_node("validation", validation_node)
    builder.add_node("refusal", refusal_node)

    builder.add_edge(START, "feasibility")
    builder.add_conditional_edges(
        "feasibility",
        route_after_feasibility,
        {"planner": "planner", "refusal": "refusal"},
    )
    builder.add_edge("planner", "execute_tools")
    builder.add_edge("execute_tools", "synthesizer")
    builder.add_edge("synthesizer", "validation")
    builder.add_edge("refusal", "validation")
    builder.add_edge("validation", END)

    return builder.compile()


def run_graph(initial_state: AgentState) -> AgentState:
    """Run the workflow end-to-end and return a fully-typed `AgentState`.

    Args:
        initial_state: The state to start from — see
            `adia.graph.state.create_initial_state` for a convenient way to build one.

    Returns:
        The final `AgentState` after every node has run.
    """
    graph = build_graph()
    raw_result = graph.invoke(initial_state)
    return finalize_state(raw_result)


def stream_graph(initial_state: AgentState) -> Iterator[tuple[str, dict[str, Any], AgentState]]:
    """Run the workflow node-by-node, yielding progress as each node completes.

    Uses the same compiled graph `run_graph` runs (`build_graph()`), just consumed through
    LangGraph's `.stream(..., stream_mode="updates")` instead of `.invoke()` -- no node or agent
    code is touched to support this. `stream_mode="updates"` yields one `{node_name: partial}`
    dict per completed node, `partial` being exactly what that node function returned (see
    `adia/graph/nodes.py`); each is folded into a running copy of the state with the same
    "later value wins" merge `.invoke()` already performs for this graph, which has no parallel
    branches yet.

    Args:
        initial_state: The state to start from, e.g. from
            `adia.graph.state.create_initial_state`.

    Yields:
        `(node_name, partial_update, state_so_far)` once per completed node, in the graph's
        execution order. `state_so_far` is the fully merged, validated `AgentState` as of that
        point -- the last yielded tuple's `state_so_far` is equivalent to what `run_graph`
        returns for the same `initial_state`.
    """
    graph = build_graph()
    accumulated: dict[str, Any] = {
        name: getattr(initial_state, name) for name in AgentState.model_fields
    }
    for update in graph.stream(initial_state, stream_mode="updates"):
        for node_name, partial in update.items():
            accumulated.update(partial)
            state = AgentState.model_validate(accumulated)
            yield node_name, partial, state
