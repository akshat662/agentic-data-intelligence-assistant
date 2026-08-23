"""The LangGraph workflow: a strictly linear skeleton, no branching or cycles yet.

    START -> feasibility -> planner -> execute_tools -> synthesizer -> validation -> END

Every node is a deterministic placeholder (`adia/graph/nodes.py`) — no LLM call exists in
this graph. Conditional routing (an early refusal terminal when infeasible, a tool-repair
loop, a replan) requires real judgment from a node to route *on*; none of these nodes has
any yet, so adding branching now would mean routing on nothing. That's Phase 2D's job, once
an LLM can actually produce a feasibility verdict or a validation failure worth reacting to
rather than just recording.
"""

from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from adia.graph.nodes import (
    execute_tools_node,
    feasibility_node,
    planner_node,
    synthesizer_node,
    validation_node,
)
from adia.graph.state import finalize_state
from adia.models.state import AgentState


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

    builder.add_edge(START, "feasibility")
    builder.add_edge("feasibility", "planner")
    builder.add_edge("planner", "execute_tools")
    builder.add_edge("execute_tools", "synthesizer")
    builder.add_edge("synthesizer", "validation")
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
