"""Graph-specific state handling: how `AgentState` is initialized and read back from LangGraph.

`AgentState` (`adia.models.state`) is used directly as the graph's state schema — LangGraph
accepts a pydantic `BaseModel` natively (verified against the installed `langgraph` version),
so there is no separate graph-only state type to define or keep in sync with it. This module
holds only the two things that are genuinely specific to running `AgentState` through
LangGraph, not part of the model's own contract: constructing a fresh run, and converting the
plain dict LangGraph's `.invoke()` returns back into a typed `AgentState`.
"""

import uuid

from adia.models.state import AgentState, Budget

DEFAULT_MAX_LLM_CALLS = 10
DEFAULT_MAX_TOOL_CALLS = 20


def create_initial_state(
    question: str,
    dataset_id: str,
    *,
    run_id: str | None = None,
    max_llm_calls: int = DEFAULT_MAX_LLM_CALLS,
    max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
) -> AgentState:
    """Build a fresh `AgentState` to start a graph run.

    Args:
        question: The user's question.
        dataset_id: Dataset to answer it against; resolved against the dataset registry by
            `feasibility_node`, not here — this function does no I/O.
        run_id: Stable identifier for this run. A random one is generated if omitted.
        max_llm_calls: Budget cap for future LLM calls. Unused by any node yet, but the cap
            is a required part of `AgentState`'s contract and must be set on every run.
        max_tool_calls: Budget cap for tool calls.

    Returns:
        A new `AgentState` with every other field at its default (empty plan, no evidence,
        no errors, no catalog).
    """
    return AgentState(
        run_id=run_id or f"run_{uuid.uuid4().hex[:12]}",
        question=question,
        dataset_id=dataset_id,
        budget=Budget(max_llm_calls=max_llm_calls, max_tool_calls=max_tool_calls),
    )


def finalize_state(raw_state: dict) -> AgentState:
    """Convert the plain dict a compiled LangGraph run returns back into an `AgentState`.

    LangGraph's `.invoke()` returns a plain `dict` (and omits any key still at its field
    default), not the pydantic model instance that was passed in. This is the one adapter
    point needed to keep everything past the graph boundary working with a real, validated
    `AgentState` rather than an untyped dict.

    Args:
        raw_state: The dict returned by `CompiledStateGraph.invoke(...)`.

    Returns:
        A validated `AgentState`.
    """
    return AgentState.model_validate(raw_state)
