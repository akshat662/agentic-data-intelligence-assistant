"""Orchestration layer: a LangGraph workflow over deterministic placeholder nodes.

No LLM call exists anywhere in this package yet. `AgentState` (`adia.models.state`) is used
directly as the graph's state schema — nothing here duplicates or wraps its contract, only
adapts around the boundary of running it through LangGraph (see `adia/graph/state.py`).
"""

from adia.graph.nodes import (
    execute_tools_node,
    feasibility_node,
    planner_node,
    synthesizer_node,
    validation_node,
)
from adia.graph.state import create_initial_state, finalize_state
from adia.graph.workflow import build_graph, run_graph

__all__ = [
    "build_graph",
    "create_initial_state",
    "execute_tools_node",
    "feasibility_node",
    "finalize_state",
    "planner_node",
    "run_graph",
    "synthesizer_node",
    "validation_node",
]
