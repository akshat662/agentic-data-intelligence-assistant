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
from adia.agents.synthesizer import synthesize_answer
from adia.data.catalog import build_catalog
from adia.data.loader import load_dataset
from adia.data.registry import get_dataset_config, load_registry
from adia.evidence.renderer import render_evidence_context
from adia.evidence.store import EvidenceStore
from adia.models.errors import ToolError, ToolErrorKind
from adia.models.plan import PlanStep
from adia.models.state import AgentState, FeasibilityResult, FeasibilityVerdict
from adia.tools.compare_groups import compare_groups
from adia.tools.correlation import compute_correlation
from adia.tools.ml_model import train_model
from adia.tools.profile_dataset import profile_dataset
from adia.tools.run_sql import run_sql
from adia.tools.segment_contribution import segment_contribution
from adia.validate.static import validate_answer

#: `adia/graph/nodes.py` -> `adia/graph` -> `adia` -> repo root -> `data/registry.json`.
#: Resolved from the package location, not the working directory, so it holds regardless of
#: where the graph is invoked from.
_REGISTRY_PATH = Path(__file__).resolve().parents[2] / "data" / "registry.json"

#: `tool_family` values `execute_tools_node` currently knows how to run. Every other value in
#: a plan step produces a typed error rather than being silently skipped — dispatching a tool
#: with no way to fill in its arguments would mean guessing them, which is exactly the "no
#: real agent reasoning" line this phase must not cross. `run_sql`, `compare_groups`,
#: `compute_correlation`, `train_model`, and `segment_contribution` arguments come from
#: `generate_tool_arguments` (LLM proposes, Python validates); `profile_dataset` runs directly.
_SUPPORTED_TOOL_FAMILIES = frozenset({
    "profile_dataset",
    "run_sql",
    "compare_groups",
    "compute_correlation",
    "train_model",
    "segment_contribution",
})

#: Returned as `final_answer` when `validation_node` rejects the synthesized answer. A fixed,
#: known-safe string rather than `None`: the graph always terminates with *something* to show
#: the user, and this text can never itself carry an ungrounded claim because it states none.
VALIDATION_FALLBACK_ANSWER = (
    "The generated answer could not be verified against the collected evidence, so no "
    "answer is being returned. Please try rephrasing the question."
)


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


def refusal_node(state: AgentState) -> dict[str, Any]:
    """Compose a grounded refusal answer for a question that was ruled out at feasibility.

    Reached only via `adia.graph.workflow.route_after_feasibility` when
    `state.feasibility.verdict` isn't `FEASIBLE` — skips `planner_node` and
    `execute_tools_node` entirely, since there is nothing to plan or execute for a question
    already ruled out. States only what `feasibility_node` already determined and verified in
    Python (`reason`, `missing_columns`, `missing_capabilities`); it invents nothing and cites
    no evidence, so `validation_node` downstream has nothing ungrounded to catch.

    Sets `refusal` to the triggering `FeasibilityResult`, per that field's own contract
    ("set when the run terminates via refusal/clarification, not an answer").
    """
    if state.feasibility is None:
        return {}

    parts = [
        f"This question cannot be answered from the '{state.dataset_id}' dataset: "
        f"{state.feasibility.reason}"
    ]
    if state.feasibility.missing_columns:
        parts.append(f"Missing column(s): {', '.join(state.feasibility.missing_columns)}.")
    if state.feasibility.missing_capabilities:
        parts.append(
            f"Missing capability(ies): {', '.join(state.feasibility.missing_capabilities)}."
        )
    text = " ".join(parts)
    return {"draft_answer": text, "rendered_answer": text, "refusal": state.feasibility}


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


def _topological_order(plan: list[PlanStep]) -> tuple[list[PlanStep], list[ToolError]]:
    """Order plan steps so every step runs after everything it `depends_on` (Kahn's algorithm).

    A step whose `depends_on` names a step_id absent from this same plan can't be ordered at
    all -- defense in depth, since `create_plan` already rejects this at proposal time, but a
    directly-constructed or mocked plan (tests, a future caller) might not go through that
    check. A dependency cycle leaves one or more steps permanently unready; both cases produce
    a typed `ToolError` (`kind=VALIDATION`) per affected step rather than raising or silently
    dropping them.

    Args:
        plan: The plan steps to order, in their original (proposed) order.

    Returns:
        `(ordered_steps, errors)`. `ordered_steps` holds every step that could be placed, each
        appearing after all of its own dependencies. `errors` holds one `ToolError` per step
        that couldn't be placed (unknown dependency or cycle membership) -- those steps are
        simply absent from `ordered_steps`, not retried or guessed at.
    """
    by_id = {step.id: step for step in plan}
    errors: list[ToolError] = []

    valid_steps: list[PlanStep] = []
    for step in plan:
        unknown_deps = [d for d in step.depends_on if d not in by_id]
        if unknown_deps:
            errors.append(
                ToolError(
                    kind=ToolErrorKind.VALIDATION,
                    message=(
                        f"Plan step '{step.id}' depends on unknown step(s): {unknown_deps}."
                    ),
                    retryable=False,
                )
            )
        else:
            valid_steps.append(step)

    in_degree = {step.id: len(step.depends_on) for step in valid_steps}
    dependents: dict[str, list[str]] = {step.id: [] for step in valid_steps}
    for step in valid_steps:
        for dep in step.depends_on:
            dependents[dep].append(step.id)

    queue = [step.id for step in valid_steps if in_degree[step.id] == 0]
    ordered: list[PlanStep] = []
    while queue:
        step_id = queue.pop(0)
        ordered.append(by_id[step_id])
        for dependent_id in dependents[step_id]:
            in_degree[dependent_id] -= 1
            if in_degree[dependent_id] == 0:
                queue.append(dependent_id)

    ordered_ids = {step.id for step in ordered}
    for step in valid_steps:
        if step.id not in ordered_ids:
            errors.append(
                ToolError(
                    kind=ToolErrorKind.VALIDATION,
                    message=f"Plan step '{step.id}' is part of a dependency cycle.",
                    retryable=False,
                )
            )

    return ordered, errors


def execute_tools_node(state: AgentState) -> dict[str, Any]:
    """Dispatch each plan step to its tool, in dependency order, via the real tool layer.

    Steps run in topological order (`_topological_order`, Kahn's algorithm) so that every
    step's dependencies have already executed -- and already written their evidence to
    `store` -- before it runs. A step with `depends_on` gets that evidence rendered
    (`render_evidence_context`, unchanged) and passed to `generate_tool_arguments` as
    `dependency_context`, so its proposed arguments can be grounded in a prior step's finding
    (e.g. filtering on a category a dependency step identified) rather than only the dataset
    catalog. A step with no dependencies gets `dependency_context=""`, the same as before this
    existed. An unresolvable dependency or a cycle becomes a typed `ToolError`
    (`kind=VALIDATION`) for the affected step(s) rather than an exception.

    `profile_dataset` needs nothing beyond the dataset itself and runs directly. Other tools
    need arguments, which `generate_tool_arguments` (`adia.agents.argument_generator`)
    proposes via an LLM and then validates in Python before anything is trusted. A rejected
    or unreachable proposal becomes a typed `ToolError` (`kind=VALIDATION`), never a guessed
    argument. A plan step naming any unsupported `tool_family` becomes a typed `ToolError`
    (`kind=UNKNOWN`) instead.

    Builds a fresh in-memory `EvidenceStore` seeded from `state.evidence` so this node stays
    idempotent across repeated calls, and writes every resulting evidence ID back into state.
    """
    if not state.plan or state.catalog is None:
        return {}

    store = EvidenceStore()
    for existing in state.evidence.values():
        store.add(existing)

    ordered_steps, new_errors = _topological_order(state.plan)
    for step in ordered_steps:
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
        else:
            dependency_context = ""
            if step.depends_on:
                dependency_evidence = [
                    evidence
                    for dep_id in step.depends_on
                    for evidence in store.list_evidence(plan_step_id=dep_id)
                ]
                if dependency_evidence:
                    dependency_context = render_evidence_context(dependency_evidence)

            args = generate_tool_arguments(
                step,
                state.catalog,
                state.catalog.dataset_id,
                dependency_context=dependency_context,
            )
            if args is None:
                new_errors.append(
                    ToolError(
                        kind=ToolErrorKind.VALIDATION,
                        message=(
                            "Argument generation failed or was rejected for tool_family "
                            f"'{step.tool_family}' (plan step '{step.id}')."
                        ),
                        retryable=False,
                    )
                )
                continue

            if step.tool_family == "run_sql":
                result = run_sql(args.query, state.catalog, store, plan_step_id=step.id)
            elif step.tool_family == "compare_groups":
                result = compare_groups(
                    args.dataset_id,
                    args.source_path,
                    args.group_column,
                    args.metric_column,
                    store,
                    plan_step_id=step.id,
                )
            elif step.tool_family == "compute_correlation":
                result = compute_correlation(
                    args.dataset_id,
                    args.source_path,
                    store,
                    columns=args.columns,
                    plan_step_id=step.id,
                )
            elif step.tool_family == "train_model":
                result = train_model(
                    args.dataset_id,
                    args.source_path,
                    args.target_column,
                    args.feature_columns,
                    args.task_type,
                    args.model_type,
                    store,
                    plan_step_id=step.id,
                )
            elif step.tool_family == "segment_contribution":
                result = segment_contribution(
                    args.dataset_id,
                    args.source_path,
                    args.entity_column,
                    args.metric_column,
                    store,
                    parent_column=args.parent_column,
                    parent_value=args.parent_value,
                    plan_step_id=step.id,
                )

        if not result.ok:
            new_errors.append(result.error)

    return {
        "evidence": {evidence.id: evidence for evidence in store.list_evidence()},
        "errors": [*state.errors, *new_errors],
    }


def synthesizer_node(state: AgentState) -> dict[str, Any]:
    """Render collected evidence and ask the synthesizer agent to explain it in prose.

    `synthesize_answer` (`adia.agents.synthesizer`) does the real work: it asks an LLM to
    write a citation-bearing answer from the rendered evidence context built here via
    `render_evidence_context`, then re-validates that proposal itself through the same
    `validate_answer` check `validation_node` applies below, falling back to a mechanical,
    evidence-only answer if the LLM is unreachable or its proposal doesn't hold up. Either way
    the result is already grounded by the time it leaves this node.

    Sets `rendered_answer` directly to the same text as `draft_answer` — the synthesizer
    writes real values inline (cited, not templated), so there is no separate placeholder
    substitution step left to perform here.
    """
    context = render_evidence_context(list(state.evidence.values()))
    answer = synthesize_answer(state.question, context, state.evidence)
    return {"draft_answer": answer, "rendered_answer": answer}


def validation_node(state: AgentState) -> dict[str, Any]:
    """Run the static grounding validator and gate `final_answer` on the result.

    Reuses `adia.validate.static.validate_answer` unchanged — this node is only the graph
    wiring around it. `final_answer` is only ever set to a passing, grounded answer, or to the
    fixed `VALIDATION_FALLBACK_ANSWER` on failure — never to unverified text, and never
    `None`, so the graph always ends with a deterministic response to show the user rather
    than a repair loop. A future Critic node handles semantic overreach this validator doesn't
    check; that node does not exist yet, so nothing here claims to catch it.
    """
    result = validate_answer(state.rendered_answer or "", state.evidence)
    final_answer = state.rendered_answer if result.passed else VALIDATION_FALLBACK_ANSWER
    return {"validation": result, "final_answer": final_answer}
