"""Node functions for the ADIA workflow graph.

`feasibility_node` and `planner_node` call real LLM agents
(`adia.agents.feasibility.assess_feasibility`, `adia.agents.planner.create_plan`); every
other node here remains a deterministic placeholder doing real work with the existing
Phase 1/2 infrastructure (dataset registry, tools, evidence store, renderer, static
validator) — none of it is a no-op stub. What's still missing from those is *judgment*:
deciding what a number means, writing prose. Each node's docstring says exactly what an LLM
will replace it with next, and why the current version is a defensible placeholder rather
than a fake one.

No node here raises: every external call — LLM or otherwise — is wrapped so a failure
becomes a state update (an error, an infeasible verdict, an empty plan), never an exception
the graph has to handle.
"""

from pathlib import Path
from typing import Any

from adia.agents.argument_generator import generate_tool_arguments
from adia.agents.feasibility import assess_feasibility
from adia.agents.planner import create_plan
from adia.data.catalog import build_catalog
from adia.data.loader import load_dataset
from adia.data.registry import get_dataset_config, load_registry
from adia.evidence.renderer import render_evidence
from adia.evidence.store import EvidenceStore
from adia.models.errors import ToolError, ToolErrorKind
from adia.models.state import AgentState, FeasibilityResult, FeasibilityVerdict
from adia.tools.profile_dataset import profile_dataset
from adia.tools.run_sql import run_sql
from adia.validate.static import validate_answer

#: `adia/graph/nodes.py` -> `adia/graph` -> `adia` -> repo root -> `data/registry.json`.
#: Resolved from the package location, not the working directory, so it holds regardless of
#: where the graph is invoked from.
_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "registry.json"

#: `tool_family` values `execute_tools_node` currently knows how to run. Every other value in
#: a plan step produces a typed error rather than being silently skipped — dispatching a tool
#: with no way to fill in its arguments would mean guessing them, which is exactly the "no
#: real agent reasoning" line this phase must not cross. `run_sql` arguments come from
#: `generate_tool_arguments` (LLM proposes, Python validates via `sql_guard`); the other
#: tool families still have no ArgGen support.
_SUPPORTED_TOOL_FAMILIES = frozenset({"profile_dataset", "run_sql"})


def feasibility_node(state: AgentState) -> dict[str, Any]:
    """Verify the dataset loads, then ask the feasibility agent whether the question fits it.

    The registry lookup and dataset load are deterministic Python, unchanged from before —
    if the dataset itself doesn't exist or won't load, there is nothing for an LLM to
    usefully reason about, so this short-circuits to `INFEASIBLE` before any LLM call.
    Once a catalog is available, `assess_feasibility` (`adia.agents.feasibility`) makes the
    real verdict: it asks an LLM which columns the question needs and whether it's
    answerable, then cross-checks every column the LLM named against this exact catalog in
    Python before trusting any of it — a hallucinated column forces `INFEASIBLE` regardless
    of what the LLM claimed.

    Populates `catalog` on success, so every later node can read dataset shape without
    touching the registry or filesystem again.
    """
    try:
        registry = load_registry(_REGISTRY_PATH)
        config = get_dataset_config(registry, state.dataset_id)
    except (FileNotFoundError, KeyError) as exc:
        return {
            "feasibility": FeasibilityResult(
                verdict=FeasibilityVerdict.INFEASIBLE,
                reason=f"Dataset '{state.dataset_id}' is not registered: {exc}",
            )
        }

    try:
        df = load_dataset(config.file_path)
    except (FileNotFoundError, ValueError) as exc:
        return {
            "feasibility": FeasibilityResult(
                verdict=FeasibilityVerdict.INFEASIBLE,
                reason=f"Dataset '{state.dataset_id}' is registered but failed to load: {exc}",
            )
        }

    catalog = build_catalog(df, dataset_id=state.dataset_id, source_path=config.file_path)
    feasibility = assess_feasibility(state.question, catalog)
    return {"catalog": catalog, "feasibility": feasibility}


def planner_node(state: AgentState) -> dict[str, Any]:
    """Ask the planner agent for a plan, already validated against the tool surface.

    `create_plan` (`adia.agents.planner`) does the real work: it asks an LLM for a short
    sequence of steps and validates every one of them in Python (supported tool family,
    resolvable dependencies) before any `PlanStep` reaches this return value. It also skips
    the LLM call entirely — returning an empty plan — when `state.feasibility.verdict` isn't
    `FEASIBLE`, so a question already ruled out never gets planned for.

    Produces no plan if `feasibility_node` did not populate a catalog or feasibility result.
    """
    if state.catalog is None or state.feasibility is None:
        return {}
    plan = create_plan(state.question, state.catalog, state.feasibility)
    return {"plan": plan}


def execute_tools_node(state: AgentState) -> dict[str, Any]:
    """Dispatch each plan step to its tool, deterministically, via the real tool layer.

    `profile_dataset` needs nothing beyond the dataset itself and runs directly. `run_sql`
    needs a query, which `generate_tool_arguments` (`adia.agents.argument_generator`) proposes
    via an LLM and then validates in Python — non-blank query, then a full pass through
    `sql_guard.check_sql` — before anything is trusted; a rejected or unreachable proposal
    becomes a typed `ToolError` (`kind=VALIDATION`), never a guessed argument. A plan step
    naming any other `tool_family` (no ArgGen support yet, e.g. `compare_groups`) becomes a
    typed `ToolError` (`kind=UNKNOWN`) instead.

    Builds a fresh in-memory `EvidenceStore` seeded from `state.evidence` so this node stays
    idempotent across repeated calls, and writes every resulting evidence ID back into state.
    """
    if not state.plan or state.catalog is None:
        return {}

    store = EvidenceStore()
    for existing in state.evidence.values():
        store.add(existing)

    new_errors: list[ToolError] = []
    for step in state.plan:
        if step.tool_family not in _SUPPORTED_TOOL_FAMILIES:
            new_errors.append(
                ToolError(
                    kind=ToolErrorKind.UNKNOWN,
                    message=(
                        f"No argument-generation support yet for tool_family "
                        f"'{step.tool_family}' (plan step '{step.id}')."
                    ),
                    retryable=False,
                )
            )
            continue

        if step.tool_family == "profile_dataset":
            result = profile_dataset(
                state.catalog.dataset_id,
                state.catalog.source_path,
                store,
                plan_step_id=step.id,
            )
        else:  # run_sql
            args = generate_tool_arguments(step, state.catalog, state.catalog.dataset_id)
            if args is None:
                new_errors.append(
                    ToolError(
                        kind=ToolErrorKind.VALIDATION,
                        message=(
                            "Argument generation failed or was rejected for tool_family "
                            f"'run_sql' (plan step '{step.id}')."
                        ),
                        retryable=False,
                    )
                )
                continue
            result = run_sql(args.query, state.catalog, store, plan_step_id=step.id)

        if not result.ok:
            new_errors.append(result.error)

    return {
        "evidence": {evidence.id: evidence for evidence in store.list()},
        "errors": [*state.errors, *new_errors],
    }


def synthesizer_node(state: AgentState) -> dict[str, Any]:
    """Mechanically compose a citation-bearing draft from collected evidence — no prose.

    An LLM Synthesizer will replace this with prose that reasons about what the numbers
    mean, written as placeholders the deterministic renderer substitutes (per the project's
    grounding design). This placeholder cannot do that — it has no language model — so it
    does the only thing that's still honest without one: state each evidence record's first
    reported value verbatim, cited by evidence ID, and nothing it didn't get from a tool.

    Sets `rendered_answer` directly to the same text as `draft_answer`, since there are no
    `{{ev.field}}` placeholders here to substitute — this placeholder never writes a value it
    didn't already have in hand.
    """
    if not state.evidence:
        text = "No evidence was collected; unable to produce a grounded answer."
        return {"draft_answer": text, "rendered_answer": text}

    lines = [f"Draft answer for: {state.question}", ""]
    for evidence in sorted(state.evidence.values(), key=lambda e: e.id):
        rendered = render_evidence(evidence)
        headline = next(iter(rendered.key_values.items()), None)
        if headline is None:
            lines.append(f"{rendered.tool} ran but reported no scalar values [[{evidence.id}]].")
        else:
            key, value = headline
            lines.append(f"{rendered.tool} reports {key} = {value} [[{evidence.id}]].")

    text = "\n".join(lines)
    return {"draft_answer": text, "rendered_answer": text}


def validation_node(state: AgentState) -> dict[str, Any]:
    """Run the static grounding validator and gate `final_answer` on the result.

    Reuses `adia.validate.static.validate_answer` unchanged — this node is only the graph
    wiring around it. `final_answer` is only ever set to a passing, grounded answer;
    validation failure leaves it `None` rather than surfacing unverified text. A future
    Critic node handles semantic overreach this validator doesn't check; that node does not
    exist yet, so nothing here claims to catch it.
    """
    result = validate_answer(state.rendered_answer or "", state.evidence)
    return {
        "validation": result,
        "final_answer": state.rendered_answer if result.passed else None,
    }
