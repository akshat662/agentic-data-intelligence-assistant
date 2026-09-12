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
shared, typed state object (`adia.models.state.AgentState`), and is reachable through four
thin interfaces that all drive the identical `create_initial_state` → `run_graph` (or
`stream_graph`) path: `adia/cli.py` (`python -m adia`), the Streamlit UI (`app.py`), the
FastAPI backend (`adia/api/`), and `bench/runner.py` for evaluation. Nothing about how the
system answers a question differs between an interactive session, the browser UI, and a
benchmark run — none of them contain feasibility, planning, tool-dispatch, or validation logic
of their own.

### Full-Stack Topology

The graph above is the same regardless of caller. `app.py` (Streamlit) is the primary way a
browser reaches it today — a single process, calling `create_initial_state`/`stream_graph`
directly in-process, with no separate backend to run:

```
        User
         │
         ▼
   Streamlit UI (app.py)
         │  create_initial_state() -> stream_graph()  (in-process, same Python interpreter)
         ▼
   LangGraph (adia/graph/)
         │
  ┌──────┴──────────────────────────────────────┐
  │  Feasibility -> Planner -> Argument Generator │
  │       -> Tool Executor -> Synthesizer          │
  │            -> Validator                        │
  └──────┬──────────────────────────────────────┘
         │  every tool call
         ▼
   Evidence Store (adia/evidence/)
```

The FastAPI backend (`adia/api/`) still exists as an independent, headless interface over the
exact same graph — for a non-Streamlit client, or a future separate frontend — reachable the
same way a browser-based Next.js frontend originally was, over `fetch`/SSE:

```
   Browser client
         │  fetch + SSE: POST /chat, POST /chat/stream, POST /datasets
         ▼
   FastAPI (adia/api/ -- app.py, routes.py, service.py, schemas.py)
         │  create_initial_state() -> run_graph() / stream_graph()
         ▼
   LangGraph (adia/graph/)   -- same as above
```

`adia/api/` is a thin interface layer only, by the same discipline as `adia/cli.py`: it has no
feasibility, planning, tool-dispatch, or validation logic of its own (§9 has the full endpoint
list and the SSE event contract). "Argument Generator" and "Tool Executor" are not separate
LangGraph nodes — they're `generate_tool_arguments` and the tool-dispatch half of
`execute_tools_node` respectively (§3's six real node names are the ones that matter for the
graph itself); they're broken out here because they're the two places tool-specific reasoning
and execution actually happen, which the six-node diagram in §3 doesn't distinguish.

**History:** this system originally shipped with a Next.js frontend (`web/`) calling the
FastAPI backend over `fetch`/SSE, matching the second diagram above. It was removed entirely
in favor of `app.py` — see `docs/DECISIONS.md`'s entry on the Streamlit rewrite for why — but
the FastAPI backend itself was kept as-is, so that second path is still real and usable, just
no longer the primary one.

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
        profile_dataset.py, run_sql.py, compare_groups.py, correlation.py, ml_model.py,
        segment_contribution.py
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
    api/           # FastAPI backend -- app.py, routes.py, service.py, schemas.py
                   # (POST /chat, POST /chat/stream, POST /datasets, GET /health) -- headless,
                   # independent of the Streamlit UI below

app.py            # Streamlit UI -- the primary browser interface; drives the graph in-process,
                   # no separate backend process required (see README.md "Running the
                   # Streamlit UI"). Replaced a Next.js frontend (web/) that called adia/api/
                   # over fetch/SSE -- removed entirely, see docs/DECISIONS.md.

scripts/
    download_dataset.py   # fetches a real ~540k-row CSV to try the Streamlit uploader with

bench/
    schema.py, questions.json, tough_questions.json, runner.py, evaluation_report.py
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
(one of `profile_dataset`, `run_sql`, `compare_groups`, `compute_correlation`, `train_model`,
`segment_contribution`), a one-sentence purpose, and `depends_on` (other step IDs in the same
plan that must run
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
fills in the concrete arguments a plan step's tool needs. It supports the five tool families
that take LLM-proposed arguments (`profile_dataset` needs none and is dispatched directly by
the graph, bypassing this agent entirely). Each tool family has its own private, unvalidated
LLM output schema (`_RunSqlLLMOutput`, `_CompareGroupsLLMOutput`, `_ComputeCorrelationLLMOutput`,
`_TrainModelLLMOutput`, `_SegmentContributionLLMOutput`); Python then converts it into the
tool's own real argument type (`RunSqlArgs`, `CompareGroupsArgs`, `ComputeCorrelationArgs`,
`TrainModelArgs`, `SegmentContributionArgs`), rejecting anything that doesn't check out: a
blank SQL query, a `group_column`/`metric_column`/`target_column`/`feature_column`/
`entity_column`/`parent_column` not present in the catalog, a `parent_column` given without a
matching `parent_value` (or vice versa — `SegmentContributionArgs`'s own validator rejects
this), or (for `run_sql`) a query that fails `adia.tools.sql_guard.check_sql` — the same guard
the `run_sql` tool itself applies. Any
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

Six deterministic tools, each a plain function taking validated arguments plus an
`EvidenceStore`, returning a `ToolResult` (`adia.models.tool_result`) — never raising into its
caller. `ToolResult.ok` selects between two mutually exclusive shapes, enforced by the model's
own validator: `data`/`evidence_id`/`provenance` when `True`, `error: ToolError` when `False`.

| Tool | Computes | Notable output fields |
|---|---|---|
| `profile_dataset` | Dataset shape + per-column stats (`adia.data.catalog.build_catalog`, enriched) | `row_count`, `column_count`, `memory_bytes`, per-column `top_values`, `column_names_preview` |
| `run_sql` | A single guarded, read-only `SELECT` over the dataset via DuckDB | `rows`, `row_count`, `columns`, `rows_preview` |
| `compare_groups` | Per-group count/mean/median/std plus pairwise mean differences, for a `group_column` with at most 50 distinct values | `groups`, `pairwise_differences`, `causal_claim_allowed: False` |
| `compute_correlation` | Pairwise Pearson correlation between numeric columns | `matrix`, `pairs`, `causal_claim_allowed: False` |
| `train_model` | One fixed-hyperparameter scikit-learn model vs. a naive baseline on a held-out split | `metric_value`, `baseline_metric_value`, `feature_importance` |
| `segment_contribution` | Ranks each entity's count/total/mean/share of a metric's total, optionally scoped to one parent value (e.g. Sub-Category within `Category == "Technology"`) | `entities` (`rank`, `total`, `share_of_total`, ...), `overall_total`, `causal_claim_allowed: False` |

`rows_preview` and `column_names_preview` exist because of a subtle interaction with the
evidence renderer (§6): `adia.evidence.renderer`'s generic list-summarization deliberately
collapses any list over 10 items down to a bare count (so one tool's huge result can't blow up
every other evidence record's rendered size) — which silently left the Synthesizer with a row
or column *count* and no actual values for any `run_sql` result over 10 rows, or any dataset
with more than 10 columns, no matter how small the result genuinely was. Both fields are plain
strings (a Markdown table, and a comma-separated name list respectively, each capped
separately from the renderer's own limit), so they survive that generic collapsing untouched
and always reach the Synthesizer's prompt. `data["rows"]`/`data["column_names"]` are left
completely unbounded — the previews are additive, not a replacement, and grounding validation
(§8) checks claims against the real, unbounded data, not the preview text.

`compare_groups`'s 50-group cap exists because `pairwise_differences` is one entry per
*unordered pair* of groups — O(n²) in `group_column`'s distinct-value count. A genuinely
categorical business dimension (Region, Category, Segment, Country) fits comfortably under 50;
a near-unique identifier (a customer ID, an order ID) does not — one real run against a
541k-row dataset grouped by `CustomerID` (4,372 distinct values) produced **9.5 million**
pairwise differences from a single tool call, expensive to compute, huge to store as evidence,
and large enough on its own to exhaust the grounding validator's numeric-comparison budget
(§8) before any other cited evidence was ever checked. Rejected outright with an actionable
`ToolError`, not silently truncated.

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
5. **Evidence store**: `state.evidence` becomes `{evidence.id: evidence for evidence in store.list_evidence()}` —
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
(`store.list_evidence(plan_step_id=dep_id)` for each dependency), renders it with the same
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
investigation work above) is the single mechanical gate every real answer passes through. It:
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

The numeric walk shares one bounded budget (`_MAX_WALK_NODES`, 5,000) across every cited
evidence record, but only charges it for a value genuinely new to the comparison set, never for
revisiting one already collected. This matters because `citation_ids` is a `set` — the order
its members get walked in is unspecified — and a large, duplicate-heavy record (a `run_sql`
result with thousands of rows but a handful of distinct values in most columns) could otherwise
exhaust the whole budget on repeat visits before a different, smaller cited record's genuinely
distinct values were ever reached, nondeterministically flagging that second record's correctly
cited number as unsupported depending on iteration order alone. A cross-dataset stress run
(`bench/tough_questions.json`) is what surfaced this; see `docs/DECISIONS.md`.

### Refusal Handling

A non-`FEASIBLE` verdict never reaches the planner or the tool layer at all —
`route_after_feasibility` sends it straight to `refusal_node`, which composes its answer
purely from what `feasibility_node` already determined and verified in Python
(`FeasibilityResult.reason`, `.missing_columns`, `.missing_capabilities`). It invents nothing
and cites no evidence, so there is nothing ungrounded in it for a *grounding* check to catch —
and `validation_node` now enforces exactly that by construction: when `AgentState.refusal` is
set, it skips `validate_answer` entirely and reports a trivially-passing `ValidationResult`,
rather than running the refusal text through the same numeric-claim check a real answer gets.
This isn't just an optimization: with zero cited evidence, that check has no way to tell a
digit incidentally present in the feasibility agent's free-form `reason` prose (e.g. explaining
there's no data for "the next 6 months," quoting the question) from a genuine ungrounded claim
— every such digit used to be flagged, intermittently breaking otherwise-correct refusals for
reasons unrelated to their actual correctness. `AgentState.refusal` is set to the triggering
`FeasibilityResult`, distinguishing a refusal from an answer even though both populate
`final_answer`.

### Unsupported Causal Claim Handling

Three tools mark their own output as unable to support a causal claim by setting
`causal_claim_allowed: False` in the `Evidence.data` they write: `compute_correlation`
(a correlation coefficient is not evidence of cause), `compare_groups` (a difference in
group means is not evidence of what caused it), and `segment_contribution` (a share of a
total says nothing about why that segment is larger). `validate_answer`'s causal-language check
reads this flag directly off whatever evidence a given answer actually cites — it fires only
when the answer both uses causal language *and* cites a record that opted out, so a tool that
simply says nothing about causality (e.g. `run_sql`) is not treated as forbidding it, since
that would be guessing at intent the tool never expressed. This makes the "associations, not
causes" framing in the Synthesizer's prompt (§4) an enforced property of the final answer, not
only a request made of the LLM: an answer that ignores the prompt and claims causation from a
`compare_groups` or `compute_correlation` result fails validation and is replaced by the fixed
fallback answer, the same as any other ungrounded claim.

## 9. UI, API, and Frontend Layer

### Streamlit UI (`app.py`) — the primary browser interface

`app.py` calls `create_initial_state`/`stream_graph` directly in the same Python process
Streamlit runs in — no HTTP hop, no separate backend to keep running alongside it. Dataset
upload reuses `adia.api.service.register_dataset` directly for the same reason. It streams the
graph's node-by-node progress into an expandable "agent steps" panel the same way
`POST /chat/stream` does for an HTTP client (see below) — same events, different transport —
and renders the final, already-validated answer plus its cited evidence once `validation_node`
completes. See the root [`README.md`](../README.md)'s "Running the Streamlit UI" section.

### FastAPI backend (`adia/api/`) — headless, independent of the UI above

`adia/api/` is a thin FastAPI interface over the exact same graph, following the same
"no business logic in the interface" discipline as `adia/cli.py`: `routes.py` only validates
input and translates exceptions to HTTP responses, `service.py` calls
`create_initial_state`/`run_graph`/`stream_graph` and shapes the result, `schemas.py` holds
request/response contracts. `app.py` (Streamlit) does not call this API and does not require it
to be running — it exists for a non-Streamlit client, or a future separate frontend, not as a
dependency of the UI above.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness check |
| `POST /chat` | Run one question through the graph; returns the full result as one JSON response |
| `POST /chat/stream` | Same, but as Server-Sent Events — see below |
| `POST /datasets` | Upload a CSV, register it (`adia/data/registry.py`), available to future `/chat` calls immediately |

`POST /chat/stream` streams one `data: <json>\n\n` frame per event as the graph runs, each
being exactly one of:
- `{"type":"phase", "node": ..., "data": {...}}` — one per completed graph node, `data` a
  small curated summary (never a raw state dump).
- `{"type":"evidence", "evidence": {...}}` — one per new evidence record, carrying the same
  bounded `RenderedEvidence` (`adia/evidence/renderer.py`) the Synthesizer itself sees, never
  raw tool output.
- `{"type":"final", "answer": ..., "evidence": [...], "validation_passed": ..., ...}` — exactly
  once, last, on success.
- `{"type":"error", "detail": "Internal server error."}` — instead of `final`, on failure; the
  HTTP response itself is still 200 (the stream started fine), so a client must check event
  `type`, not just response status.

**Token-level streaming of the answer text is deliberately not implemented.** The Synthesizer
can silently discard its own LLM draft and substitute the mechanical fallback if grounding
validation fails (§4, §8) — streaming raw tokens to the browser as they're generated would
mean sometimes showing text that gets retracted a moment later, undermining the one guarantee
this system exists to make. The stream instead narrates *progress* (which node just completed,
which evidence was just produced) live, and delivers the final answer as a single,
already-validated chunk once `validation` completes.

Either interface's "chat session" is stateless server-side: every `/chat/stream` call, and
every question asked through `app.py`, is an independent graph run — no conversational memory
is fed back into the LLM between turns. `app.py` keeps its own visible transcript in
`st.session_state` purely for display, the same role a Next.js frontend's client-side
`useReducer` transcript originally played against this same API before it was replaced (see
`docs/DECISIONS.md`). See the root [`README.md`](../README.md)'s "Project Structure" and
"Deployment" sections for the full component list and how to run or deploy either interface.
