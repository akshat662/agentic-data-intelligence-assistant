# ADIA Architecture

This document describes ADIA's actual, implemented architecture — every component, contract,
and code path named below exists in this repository today. It is a technical reference, not a
roadmap; for the reasoning behind individual design choices, see `docs/DECISIONS.md`. For a
presentation-oriented overview with example runs, see the root [`README.md`](../README.md).

## 1. System Overview

ADIA answers natural-language questions about a registered tabular dataset by separating two
concerns that a "chat with your CSV" system conflates: **deciding what to compute** and
**actually computing it**. An LLM is used only for the first — proposing a feasibility
verdict, a plan, tool arguments, or a sentence of prose — and every proposal is verified or
executed by deterministic Python before it can affect the final answer. No LLM output reaches
the user unchecked.

This discipline is enforced structurally, not by convention:
- A hallucinated column name is caught by comparing the LLM's claim against the dataset's real
  catalog (`adia/agents/feasibility.py`).
- A hallucinated or dangerous SQL query is caught by parsing and guarding it before it ever
  reaches DuckDB (`adia/tools/sql_guard.py`).
- A hallucinated or unsupported number in a synthesized answer is caught by re-extracting
  every numeral in the text and checking it against the actual evidence
  (`adia/validate/static.py`).

The system is driven by a LangGraph state machine (`adia/graph/workflow.py`) over a single
shared, typed state object (`adia.models.state.AgentState`), and is reachable through one thin
interface, `adia/cli.py` (`python -m adia`). `bench/runner.py` drives the same graph over a
fixed question set for evaluation; both callers use the identical
`create_initial_state` → `run_graph` path, so nothing about how the system answers a question
differs between an interactive session and a benchmark run.

## 2. Component Architecture

```
adia/
    agents/       # LLM reasoning only -- every claim verified in Python before it's trusted
        feasibility.py         # assess_feasibility()
        planner.py              # create_plan()
        argument_generator.py   # generate_tool_arguments()
        synthesizer.py          # synthesize_answer()
    graph/
        nodes.py                 # the six node functions + _topological_order()
        workflow.py               # build_graph(), route_after_feasibility(), run_graph()
        state.py                  # create_initial_state(), finalize_state()
    tools/        # deterministic computation -- no LLM call anywhere in this package
        profile_dataset.py, run_sql.py, compare_groups.py, correlation.py, ml_model.py
        sql_guard.py             # parses and guards SQL before execution
        duckdb_client.py         # in-memory DuckDB connection over a registered DataFrame
    evidence/
        store.py                  # EvidenceStore -- content-addressed, cache-aware
        ids.py                     # generate_evidence_id(), compute_args_hash()
        renderer.py                # render_evidence(), render_evidence_context()
        persistence.py             # JSON load/save for Evidence lists
    validate/
        static.py                  # validate_answer() -- the one grounding gate
    models/       # shared pydantic contracts used by every layer above
        plan.py (PlanStep), evidence.py (Evidence), provenance.py (Provenance),
        tool_result.py (ToolResult), errors.py (ToolError/ToolErrorKind),
        catalog.py (DatasetCatalog/ColumnProfile), state.py (AgentState and friends)
    data/          # dataset registry + loading
    cli.py         # python -m adia -- a thin interface, no business logic

bench/
    schema.py, questions.json, runner.py, evaluation_report.py
```

Every arrow of dependency in this system points one way: `graph` calls `agents` and `tools`
and `evidence` and `validate`; `agents` call `evidence` (to render context) and never call
`tools` directly; `tools` call `evidence` (to write results) and never call `agents`. No tool
imports an agent, and no agent executes a tool — the graph nodes are the only place dispatch
happens.

## 3. LangGraph Workflow

```
                          +--> planner -> execute_tools -> synthesizer --+
                          |                                              |
    START -> feasibility -+                                              +-> validation -> END
                          |                                              |
                          +--> refusal ---------------------------------+
```

Six nodes (`adia/graph/nodes.py`), wired by `build_graph()` (`adia/graph/workflow.py`):

| Node | Function | Purpose |
|---|---|---|
| `feasibility` | `feasibility_node` | Load the dataset, build its catalog, call `assess_feasibility` |
| `planner` | `planner_node` | Call `create_plan`; only reached when feasible |
| `execute_tools` | `execute_tools_node` | Run the plan's steps, in dependency order |
| `synthesizer` | `synthesizer_node` | Call `synthesize_answer` over the collected evidence |
| `validation` | `validation_node` | Run `validate_answer`; gate `final_answer` |
| `refusal` | `refusal_node` | Compose a grounded refusal from the feasibility verdict |

`route_after_feasibility` is the graph's only conditional edge: it inspects
`state.feasibility.verdict` and returns `"planner"` only when it equals
`FeasibilityVerdict.FEASIBLE`; any other verdict (`INFEASIBLE`, `NEEDS_CLARIFICATION`, or a
missing result) routes to `"refusal"`, skipping `planner` and `execute_tools` entirely. Both
branches converge on `validation`, which is unconditionally the last node before `END` — every
answer, refused or not, passes through the same grounding gate.

`AgentState` (`adia.models.state`) is used directly as the graph's state schema; LangGraph
merges each node's returned `dict` into it. `adia/graph/state.py` supplies the two adapter
functions the graph boundary needs: `create_initial_state(question, dataset_id)` builds a
fresh state (no I/O), and `finalize_state(raw_dict)` converts LangGraph's plain-dict return
value back into a validated `AgentState`.

## 4. Agent Responsibilities

Every agent below follows the same contract: it accepts an optional `llm_call` override (used
exclusively by tests, never in production code), defaults to a real `ChatOpenAI` call at
`temperature=0` via `adia.agents.llm_config.load_llm_settings`, and never lets an exception
escape — an unreachable LLM, a malformed response, or a failed verification all degrade to a
safe, typed fallback rather than raising into the graph.

### Feasibility Agent (`adia/agents/feasibility.py`)

`assess_feasibility(question, catalog)` asks an LLM for a verdict
(`feasible`/`infeasible`/`needs_clarification`), the columns it believes are relevant, and any
missing capabilities. Python then cross-checks every column the LLM named against
`catalog.column_names()`; any column not present forces the verdict to `INFEASIBLE`
regardless of what the LLM claimed, and the offending names are recorded in
`FeasibilityResult.missing_columns`. The system prompt explicitly distinguishes two shapes of
"why" question: one answerable by comparing/aggregating columns already in the catalog
(marked feasible — the investigation itself is the planner's job, not resolved here) versus
one needing information no column records even indirectly, such as customer psychology or
external market data (marked infeasible, with the missing capability named). No dataset can
prove causation with certainty; the prompt states this is a limitation of the eventual
answer's confidence, not a reason to refuse the question.

### Planner Agent (`adia/agents/planner.py`)

`create_plan(question, catalog, feasibility)` is only called when
`feasibility.verdict == FEASIBLE` — for anything else it returns an empty plan without
invoking the LLM at all. It asks for a list of steps, each with a `step_id`, a `tool_family`
(one of `profile_dataset`, `run_sql`, `compare_groups`, `compute_correlation`, `train_model`),
a one-sentence purpose, and `depends_on` (other step IDs in the same plan that must run
first). It never proposes tool arguments, SQL text, or column selections — only plan shape.
Python validates every proposed step before any `PlanStep` is built: an unsupported
`tool_family`, a dependency on a step ID absent from the same plan, or a duplicate `step_id`
collapses the *whole* plan to empty, never a partially-trusted one.
`expected_output`/`success_criteria` — required fields on `PlanStep` but not a judgment call —
are filled in deterministically per `tool_family` in Python, not asked of the LLM. The system
prompt additionally guides "why"/"how" questions toward an investigation shape: one
observation step with no dependencies, followed by several supporting-analysis steps that
each `depends_on` the observation step and each test one distinct candidate explanation.

### Argument Generator (`adia/agents/argument_generator.py`)

`generate_tool_arguments(step, catalog, dataset_id, *, dependency_context="", llm_call=None)`
fills in the concrete arguments a plan step's tool needs. It supports the four tool families
that take LLM-proposed arguments (`profile_dataset` needs none and is dispatched directly by
the graph, bypassing this agent entirely). Each tool family has its own private, unvalidated
LLM output schema (`_RunSqlLLMOutput`, `_CompareGroupsLLMOutput`, `_ComputeCorrelationLLMOutput`,
`_TrainModelLLMOutput`); Python then converts it into the tool's own real argument type
(`RunSqlArgs`, `CompareGroupsArgs`, `ComputeCorrelationArgs`, `TrainModelArgs`), rejecting
anything that doesn't check out: a blank SQL query, a `group_column`/`metric_column`/
`target_column`/`feature_column` not present in the catalog, or (for `run_sql`) a query that
fails `adia.tools.sql_guard.check_sql` — the same guard the `run_sql` tool itself applies. Any
rejection or LLM failure returns `None`; the caller (`execute_tools_node`) turns that into a
typed `ToolError`, never a guessed argument. `dependency_context` — rendered evidence from a
step's own dependencies, built by the caller via `render_evidence_context` — is threaded only
into the real LLM call path; a test's `llm_call` override still receives just
`(step, catalog)`, so this parameter is fully additive to the existing contract.

### Synthesizer (`adia/agents/synthesizer.py`)

`synthesize_answer(question, evidence_context, evidence, *, llm_call=None)` asks an LLM to
write prose explaining the collected evidence, citing every numeric claim with an inline
`[[evidence_id]]` marker. It never trusts that prose as-is: before returning it, it runs the
candidate back through `adia.validate.static.validate_answer` — the identical check
`validation_node` applies later — against the same evidence. If the LLM is unreachable,
produces blank text, or its answer fails that check, `synthesize_answer` falls back to
`_mechanical_fallback`: a deterministic composition that restates each evidence record's most
meaningful reported value (`_select_fallback_values`, ranked by a penalty score preferring
non-bookkeeping, numeric fields over configuration/identifier fields), with any list-index
digit in a flattened key name stripped from the *displayed* label only
(`_sanitize_label`) so it can't be misread as an unsupported numeric claim. Because it only
ever restates a value already present in evidence, the fallback can never itself fail
`validate_answer`. The system prompt instructs the LLM to keep three things visibly distinct
when evidence includes more than one analysis step: the observed fact itself, stated plainly;
what other evidence is merely *associated with* it, in non-causal language; and what remains
unsupported, stated explicitly rather than implied.

## 5. Tool Execution Layer

Five deterministic tools, each a plain function taking validated arguments plus an
`EvidenceStore`, returning a `ToolResult` (`adia.models.tool_result`) — never raising into its
caller. `ToolResult.ok` selects between two mutually exclusive shapes, enforced by the model's
own validator: `data`/`evidence_id`/`provenance` when `True`, `error: ToolError` when `False`.

| Tool | Computes | Notable output fields |
|---|---|---|
| `profile_dataset` | Dataset shape + per-column stats (`adia.data.catalog.build_catalog`, enriched) | `row_count`, `column_count`, `memory_bytes`, per-column `top_values` |
| `run_sql` | A single guarded, read-only `SELECT` over the dataset via DuckDB | `rows`, `row_count`, `columns` |
| `compare_groups` | Per-group count/mean/median/std plus pairwise mean differences | `groups`, `pairwise_differences`, `causal_claim_allowed: False` |
| `compute_correlation` | Pairwise Pearson correlation between numeric columns | `matrix`, `pairs`, `causal_claim_allowed: False` |
| `train_model` | One fixed-hyperparameter scikit-learn model vs. a naive baseline on a held-out split | `metric_value`, `baseline_metric_value`, `feature_importance` |

`run_sql` is the only tool that accepts free-form input; every query passes through
`adia.tools.sql_guard.check_sql` first, which parses it with `sqlglot`, rejects anything that
isn't a single read-only `SELECT` (optionally with CTEs), rejects any table reference other
than the one dataset table (and the query's own CTEs), rejects any column not in the catalog
or a query-defined alias, and injects a default `LIMIT` if none was given.
`adia.tools.duckdb_client` only ever registers one in-memory DataFrame under one fixed table
name, so a query has no filesystem or network surface to reach even if the guard were bypassed.
`train_model` never tunes or selects a model on the caller's behalf: `model_type` is a
required argument from a small fixed registry (`logistic_regression`,
`random_forest_classifier`, `linear_regression`, `random_forest_regressor`), a single seeded
`train_test_split` is used (not cross-validation), and a naive baseline
(`DummyClassifier`/`DummyRegressor`) is always fit and reported alongside — a model score with
no baseline next to it is not treated as evidence.

Every tool writes exactly one `Evidence` record on success, via `EvidenceStore.add`. Evidence
IDs are content-addressed: `generate_evidence_id(tool_name, args)` (`adia/evidence/ids.py`)
hashes the tool's canonicalized arguments into an ID of the form
`ev_<tool_name>_<8 hex chars>`, so calling the same tool with the same arguments twice is a
cache hit — `EvidenceStore.add` returns the existing record rather than recomputing — and
raises only if the same ID is claimed by genuinely different arguments (a real collision).

## 6. Evidence Lifecycle

```
question --> feasibility --> plan --> tool execution --> evidence store --> synthesis --> validation --> final_answer
```

1. **Question**: `create_initial_state(question, dataset_id)` builds a fresh `AgentState`; no
   I/O happens yet.
2. **Feasibility**: `feasibility_node` resolves the dataset from the registry, loads it,
   builds its `DatasetCatalog`, and calls `assess_feasibility`. The catalog is stored on
   `state.catalog` so no later node touches the registry or filesystem again.
3. **Plan**: if feasible, `planner_node` calls `create_plan`, producing a validated
   `list[PlanStep]` stored on `state.plan`.
4. **Tool execution**: `execute_tools_node` builds a fresh `EvidenceStore`, seeded from
   `state.evidence` (so repeated invocation is idempotent), orders `state.plan` topologically,
   and dispatches each step to its tool — `profile_dataset` directly, everything else via
   `generate_tool_arguments` first. Every resulting `Evidence` record is written into the
   store, keyed by its content-addressed ID.
5. **Evidence store**: `state.evidence` becomes `{evidence.id: evidence for evidence in store.list()}` —
   a flat, ID-keyed map available to every later node.
6. **Synthesis**: `synthesizer_node` renders every evidence record via
   `render_evidence_context` (`adia/evidence/renderer.py`) and passes that text, plus the
   question and the full evidence map, to `synthesize_answer`.
7. **Validation**: `validation_node` runs `validate_answer(state.rendered_answer, state.evidence)`
   and sets `state.final_answer` to the rendered answer if it passed, or to a fixed
   `VALIDATION_FALLBACK_ANSWER` if not — never to unverified text, and never `None`.

`Evidence.data` holds the tool's full-precision output; `Evidence.provenance` (a `Provenance`
record) holds the exact arguments used, their hash, relevant library versions, an optional
random seed, and — for `run_sql` — the exact guarded SQL text executed. `Evidence.plan_step_id`
links a record back to the plan step that produced it, which is what makes both dependency
evidence handoff (§7) and the benchmark's `executed_step_count`/`evidence_coverage` metrics
(`bench/runner.py`) possible without any additional bookkeeping.

## 7. Multi-Step Investigation Flow

### Dependency-Aware Plans

`PlanStep.depends_on` (`adia/models/plan.py`) has existed since the Planner's earliest
version, but `execute_tools_node` originally executed `state.plan` in list order, ignoring it.
It now orders the plan via `_topological_order` (`adia/graph/nodes.py`), a Kahn's-algorithm
implementation: an in-degree count per step, a FIFO queue seeded with every zero-dependency
step, repeatedly dequeuing a step and decrementing its dependents' counts. A step whose
`depends_on` names a step ID absent from the same plan, or that's part of a dependency cycle,
is never placed into the executed order — each becomes its own typed
`ToolError(kind=VALIDATION)` instead, never an exception and never a guess at execution order.

### Evidence Handoff

Because steps execute in topological order, a step's own dependencies have always already run
— and already written their evidence to the store — by the time `execute_tools_node` reaches
it. For any step with a non-empty `depends_on`, the node collects that evidence
(`store.list(plan_step_id=dep_id)` for each dependency), renders it with the same
`render_evidence_context` the synthesizer uses, and passes the result as
`generate_tool_arguments`'s `dependency_context` parameter. This is what lets a
supporting-analysis step's SQL or column choice be grounded in what an earlier step actually
found, rather than only in the dataset's static catalog.

### Investigation Example

For a "why" question, the Planner typically proposes one **observation** step with no
dependencies (e.g. a `run_sql` aggregation establishing which category has the lowest total
Sales), followed by several **supporting-analysis** steps that each `depends_on` the
observation step and each test one distinct candidate explanation (e.g. separate
`compare_groups` calls on order quantity, discount, and profit, split by the same category
column). `execute_tools_node` runs the observation step first regardless of list order, then
the dependent steps, each receiving the observation's evidence as context. The Synthesizer
then sees all of it — the observation and every supporting analysis — as one evidence map, and
composes an answer that states the observation plainly, describes each supporting result in
non-causal language, and, where the evidence doesn't settle the question, says so explicitly.

## 8. Grounding and Safety Mechanisms

### Validation

`adia.validate.static.validate_answer(text, evidence)` (unchanged by, and independent of, the
investigation work above) is the single mechanical gate every answer passes through. It:
extracts every `[[...]]` citation marker and classifies each as valid, malformed (not a real
evidence-ID shape), or dangling (well-formed but matching no record in `evidence`); extracts
every number-shaped token in the text (masking citation markers first so digits inside an ID
aren't mistaken for a claim); if any numbers are claimed with zero valid citations at all,
fails outright; otherwise checks each claimed number against every numeric leaf value found by
walking the cited evidence's own `data`, within a fixed absolute/relative tolerance; and scans
for causal language (`causes`, `led to`, `due to`, and similar), failing only if a *cited*
record's `data` explicitly sets `causal_claim_allowed: False`. `ValidationResult.passed` is
`False` if any check produced a failing issue — every issue this layer raises is a hard
failure, not advice.

### Refusal Handling

A non-`FEASIBLE` verdict never reaches the planner or the tool layer at all —
`route_after_feasibility` sends it straight to `refusal_node`, which composes its answer
purely from what `feasibility_node` already determined and verified in Python
(`FeasibilityResult.reason`, `.missing_columns`, `.missing_capabilities`). It invents nothing
and cites no evidence, so it has nothing ungrounded for `validation_node` to catch, and it
still passes through that same node like every other answer. `AgentState.refusal` is set to
the triggering `FeasibilityResult`, distinguishing a refusal from an answer even though both
populate `final_answer`.

### Unsupported Causal Claim Handling

Two tools mark their own output as unable to support a causal claim by setting
`causal_claim_allowed: False` in the `Evidence.data` they write: `compute_correlation`
(a correlation coefficient is not evidence of cause) and `compare_groups` (a difference in
group means is not evidence of what caused it). `validate_answer`'s causal-language check
reads this flag directly off whatever evidence a given answer actually cites — it fires only
when the answer both uses causal language *and* cites a record that opted out, so a tool that
simply says nothing about causality (e.g. `run_sql`) is not treated as forbidding it, since
that would be guessing at intent the tool never expressed. This makes the "associations, not
causes" framing in the Synthesizer's prompt (§4) an enforced property of the final answer, not
only a request made of the LLM: an answer that ignores the prompt and claims causation from a
`compare_groups` or `compute_correlation` result fails validation and is replaced by the fixed
fallback answer, the same as any other ungrounded claim.
