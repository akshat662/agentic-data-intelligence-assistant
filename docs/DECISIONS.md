# Architecture Decision Records

This document tracks significant technical decisions made during ADIA's
development, in reverse-chronological order. Each entry should capture the
decision, the alternatives considered, and the reasoning — not just the
outcome.

Template for new entries:

```
## YYYY-MM-DD: <short decision title>

**Status:** Proposed | Accepted | Superseded by <link>

**Context:**
What problem or question prompted this decision?

**Decision:**
What was decided?

**Alternatives considered:**
What else was on the table, and why was it not chosen?

**Consequences:**
What does this decision commit us to? What does it rule out?
```

---

## 2026-09-11: Streamlit UI replaces the Next.js frontend

**Status:** Accepted

**Context:**
The original browser interface was a Next.js frontend (`web/`) calling the FastAPI backend
(`adia/api/`) over `fetch`/SSE — two processes, two languages, two deploy targets (Vercel +
Render), kept in sync by hand-mirrored TypeScript types (`web/lib/types.ts`) against
`adia/api/schemas.py`. That split earns its cost when the frontend and backend genuinely need
to scale or deploy independently; for a project whose entire point is to demonstrate one
Python system's reasoning, it added a second language, a second dependency tree
(`web/package.json`), and a second deploy target for no capability the graph itself needed.

**Decision:**
Delete `web/` entirely. Add `app.py`, a single-file Streamlit UI that imports
`adia.graph.state`/`adia.graph.workflow`/`adia.api.service` directly and drives the graph
in-process — no HTTP hop, no second server to start or keep running, one deploy target
(Streamlit Community Cloud or any host that runs one Python process). Dataset upload reuses
`adia.api.service.register_dataset` directly for the same reason `POST /datasets` did. The
FastAPI backend was kept, unmodified, as an independent, headless interface — `app.py` does not
call it and does not require it running; it remains available for a non-Streamlit client or a
future separate frontend.

**Alternatives considered:**
- Keep `web/`, add `app.py` alongside it as a second frontend — rejected: two frontends against
  one backend is more surface to keep in sync than this project needs, for a capability
  (a second UI framework) nothing about the system's core claim depends on demonstrating.
- Delete the FastAPI backend too, since `app.py` doesn't need it — rejected: `adia/api/` costs
  nothing to keep (it's a thin, already-tested interface layer with its own test suite,
  `tests/test_api.py`), and keeping it preserves a real, working HTTP/SSE path for any future
  client that isn't Streamlit, without reintroducing the two-frontend problem above.

**Consequences:**
One process to run for the full browser demo (`uv run streamlit run app.py`), one deploy
target, no TypeScript in the repository, no hand-mirrored type contract to keep in sync. The
tradeoff: the FastAPI backend's SSE contract (§9 of `docs/ARCHITECTURE.md`) is now demonstrated
by its own test suite rather than by a live frontend consuming it end-to-end — acceptable,
since that contract was already independently tested and unchanged by this decision.

---

## 2026-09-11: Planner must route derivable metrics through `run_sql`, never `compare_groups`/`segment_contribution`

**Status:** Accepted

**Context:**
Tested against a raw dataset lacking a pre-aggregated metric column — the UCI Online Retail
dataset (`data/real_dataset.csv`: `InvoiceNo`, `StockCode`, `Quantity`, `UnitPrice`, ... but no
`Sales`/`revenue` column) — the Planner correctly reasoned in its feasibility step that
"Sales = Quantity × UnitPrice" was derivable, but then proposed a `compare_groups` or
`segment_contribution` step with `metric_column="Sales"` anyway. Neither tool can compute a
derived expression; `metric_column` must name a real column in the dataset, so the step failed
outright with a column-not-found error.

**Decision:**
Add an explicit, high-visibility rule to the Planner's system prompt
(`adia/agents/planner.py`): if a question's metric isn't a standalone column but is
mathematically derivable from columns that do exist, the step **must** use `run_sql` (which can
express the derivation inline, e.g. `SUM(Quantity * UnitPrice)`), never `compare_groups` or
`segment_contribution`. The rule is stated once in the "Rules" section and repeated as a
one-line caveat directly on both tools' own descriptions in the same prompt — repetition next
to the specific tools being misused, not just a rule buried in a list, is what actually changed
the Planner's behavior in testing.

**Alternatives considered:**
- Teach `compare_groups`/`segment_contribution` to accept a SQL expression as `metric_column`
  instead of a bare column name — rejected: it would blur the boundary these tools intentionally
  keep ("propose a validated column, not a validated expression"), and reopens exactly the kind
  of free-form-input surface `adia.tools.sql_guard` exists to contain for `run_sql` alone.
- Have `feasibility_node` pre-compute derived columns onto the DataFrame before the Planner ever
  sees it — rejected: this is real, undisclosed data transformation happening outside any tool
  call, with no evidence record and no citation trail — exactly the kind of ungrounded
  computation this system's entire design exists to prevent.

**Consequences:**
A question about a metric the dataset doesn't store as a column, but can compute, now reaches
`run_sql` reliably instead of crashing a step. This is a prompt-level fix, not a code-level
guarantee — it is not unit-tested against the LLM's actual behavior (only that the prompt text
itself contains the rule, `tests/test_planner_agent.py`), so a future model swap or prompt
regression could reopen this gap without a failing test catching it directly; the mitigation is
the same repeated-near-the-tool phrasing that fixed it once already, still in place.

---

## 2026-09-11: `run_sql`/`profile_dataset` result previews, and a fallback that stopped echoing the question

**Status:** Accepted

**Context:**
Three related failures surfaced in close succession, all traceable to one blind spot:
`adia.evidence.renderer`'s generic list-summarization deliberately collapses any list over 10
items down to a bare count, so no single tool's huge result can blow up every other evidence
record's rendered size. That's the right call for, say, `train_model`'s `feature_importance` —
but it meant:
1. A `run_sql` query returning more than 10 rows (e.g. one row per country) left the Synthesizer
   with `rows_count: 38` and zero actual fetched values — it could see *how many* rows came
   back, never *what was in them*.
2. `profile_dataset` on any dataset with more than 10 columns (the common case) left the
   Synthesizer with `column_names_count: 21` and no actual names — asked "what are 5 of the
   columns," the honest, correct response was "the evidence doesn't say."
3. Diagnosing (2) surfaced a third, independent bug: `_mechanical_fallback`
   (`adia/agents/synthesizer.py`) — the safety net documented as "can never itself fail
   grounding validation" — echoed the raw question into its header line. A question containing
   any digit not also present in evidence (e.g. "provide names of any **5** columns") made the
   fallback state an ungrounded "5," breaking the one guarantee it exists to provide, on exactly
   the runs where it was needed most.

**Decision:**
- Add `rows_preview` (`run_sql`) and `column_names_preview` (`profile_dataset`): plain strings
  — a Markdown table capped at 50 rows, and a comma-separated name list capped at 200 —
  computed alongside the existing unbounded `rows`/`column_names` fields. A string is a leaf
  value to the renderer regardless of how many rows or columns went into building it, so both
  survive the generic collapsing untouched and always reach the Synthesizer's prompt.
- Stop echoing `question` in `_mechanical_fallback`'s output entirely; replaced with a fixed,
  content-free header line.

**Alternatives considered:**
- Raise the renderer's small-list threshold above 10 — rejected: it's shared by every tool's
  evidence, and raising it for `run_sql`/`profile_dataset` would raise it for everything,
  reintroducing the exact "one huge result crowds out everything else" problem the threshold
  exists to prevent (and is directly what the next entry's shared-budget bug is about).
- Mask or strip digits out of the echoed question in the fallback instead of removing it —
  rejected: a mangled restatement ("Provide names of any columns") reads as broken, is no
  more informative to a reader than no restatement at all (the question is already visible as
  the user's own chat turn / CLI prompt in every real caller), and a masking regex is one more
  thing that could itself have edge cases.

**Consequences:**
Both tools' full, unbounded data remains available to the grounding validator exactly as
before — these previews are additive rendering fixes, not new tool capability. Verified against
the real UCI dataset and the real Telco Churn dataset (both regularly exceed 10 rows/columns)
with a real LLM call, not just unit tests. Any *future* tool whose result can exceed 10 items
and needs to be citable in prose inherits the same blind spot until it gets an equivalent
preview field — this fix is per-tool, not a change to the renderer's own default behavior.

---

## 2026-09-11: A 30-question, cross-dataset stress benchmark surfaced three validation-layer bugs

**Status:** Accepted

**Context:**
`bench/questions.json` (26 questions, one dataset) had reached 100% validation pass rate,
which risked reflecting a well-worn happy path more than genuine robustness. A new,
deliberately harder suite (`bench/tough_questions.json`, 30 questions across `superstore`,
`telco_churnn`, and the 541k-row `real_dataset`, leaning into null handling, multi-condition
filters, derived metrics, and refusal edge cases) was run end-to-end against the real graph.
The first run scored 28/30 on validation. Diagnosing the two failures — and a third that
appeared on a subsequent rerun of the same suite — found three independent bugs, none of them
in the LLM's reasoning:
1. **Shared numeric-walk budget starvation** (`adia/validate/static.py`): the grounding
   validator's numeric-claim check shared one 5,000-node budget across every cited evidence
   record, decrementing it on every *visit* to a numeric leaf, not every new one. A large,
   duplicate-heavy `run_sql` result (1,871 rows, heavy repetition in most columns) could burn
   the entire budget on repeat visits to values already collected before a different, smaller
   cited record's genuinely distinct values were ever walked — nondeterministically rejecting
   that second record's correctly cited number, depending on `citation_ids`' unspecified `set`
   iteration order alone.
2. **Refusals failing their own grounding check** (`adia/graph/nodes.py`): `validation_node` ran
   every answer, refusals included, through the same numeric-claim check. A refusal cites zero
   evidence by design, so any digit incidentally present in the feasibility agent's free-form
   `reason` text (e.g. quoting "the next 6 months" from the question) had no possible citation
   to match against and was flagged as an ungrounded claim — breaking an otherwise-correct
   refusal and replacing it with the generic fallback message.
3. **`compare_groups` O(n²) blowup on high-cardinality columns** (`adia/tools/compare_groups.py`):
   grouping by a near-unique identifier (a real `CustomerID` column, 4,372 distinct values)
   computed `pairwise_differences` for every unordered pair — 9.5 million entries from one call.
   Beyond being wasteful, it was large enough (nearly all distinct floats, so bug (1)'s fix
   alone didn't neutralize it) to still exhaust the validator's budget on its own.

**Decision:**
- `_walk_numeric` now spends budget only on a value genuinely new to the comparison set, never
  on a revisit — a record's *distinct* value count is what's bounded, which is also the only
  thing that could ever change whether some claimed number matches.
- `validation_node` skips `validate_answer` entirely when `AgentState.refusal` is set, reporting
  a trivially-passing `ValidationResult` instead — enforcing in code the invariant
  `refusal_node`'s own docstring already claimed ("cites no evidence, so there is nothing
  ungrounded to catch") rather than leaving it incidentally true only when the reason text
  happened to be digit-free.
- `compare_groups` now rejects a `group_column` with more than 50 distinct values outright, with
  an actionable `ToolError` pointing at `segment_contribution` for a high-cardinality breakdown,
  instead of silently computing a pathological result.

**Alternatives considered:**
- Raise `_MAX_WALK_NODES` instead of changing what it charges for — rejected: it delays the
  failure mode rather than fixing it (a large enough record still starves everything else,
  exactly as `compare_groups`-on-`CustomerID` proved even after the per-record accounting fix),
  and a bigger shared budget makes a pathological single record slower to detect, not safer.
- Special-case the validator to skip the numeric check whenever zero evidence is cited, for any
  answer, not just refusals — rejected: that's a broader relaxation than the actual invariant
  being restored (a *real* answer citing zero evidence while stating numbers is exactly the
  `missing_evidence_reference` failure mode this validator is supposed to catch); gating the
  skip on `state.refusal is not None` targets only the one path structurally guaranteed to
  never need the check, per `refusal_node`'s own contract.
- Cap `pairwise_differences` at some fixed count instead of rejecting the call — rejected:
  a truncated, arbitrary subset of pairwise differences is a misleading partial result for a
  comparison that's analytically meaningless in the first place (comparing thousands of
  effectively single-row "groups" pairwise was never a sound operation); an explicit error that
  suggests the right tool is more honest and more useful to a repair loop than a silently
  incomplete answer.

**Consequences:**
All three fixes are covered by regression tests reproducing the exact failure shape (a
duplicate-heavy large record starving a small one regardless of walk order; a refusal whose
reason quotes a digit from the question; a 60-distinct-value column rejected, a 50-value one
still accepted) — not just re-running the benchmark and hoping. Re-running
`bench/tough_questions.json` after all three fixes: **30/30 completed, 30/30 passed
validation**, 8/8 refusals correctly refused, 0/22 false refusals, zero forbidden causal
phrases. Two further, real findings from the same run were reported but deliberately left
unfixed here as LLM-output-quality issues rather than engine bugs: a generated `run_sql` query
used `JULIANDAY(...)`, a SQLite function DuckDB doesn't have; and a churn-rate investigation
chose `compare_groups` against the categorical `Churn` column, which requires a numeric metric.
Both are candidates for a future Planner/Argument-Generator prompt refinement, in the same
spirit as the derivable-metric rule above, not a validation-layer fix.

---

## 2026-08-24: A tier-grouped evaluation report, not an LLM judge

**Status:** Accepted

**Context:**
The system could now demonstrably plan and execute multi-step investigations (see the
"Investigation DAGs" entry below), but nothing measured that — `bench/runner.py` recorded
`tools_used` as a deduplicated set of tool *names*, which can't distinguish "one
`compare_groups` call" from "four `compare_groups` calls investigating four dimensions." The
project's own claim to be "more than a simple NL-to-SQL assistant" was asserted in prose, not
shown as a number.

**Decision:**
- `bench/schema.py` gained `EvaluationTier` (`direct`/`investigation`/`refusal`) as a
  **computed property** on `BenchmarkQuestion`, derived from the existing `category` via a
  fixed mapping — not a stored field. Every question in `bench/questions.json` needed zero
  changes for this to work.
- `BenchmarkQuestion` gained an optional `investigation: InvestigationExpectation` field,
  set only on `root_cause` questions (enforced by a validator mirroring the existing
  `answerable`/`category` one). It carries `expected_observation`,
  `expected_analysis_dimensions`, and `acceptable_conclusion_style` as rubric prose for a
  human reviewer, and `forbidden_causal_phrases` as the one part that's actually
  machine-checked.
- `bench/runner.py`'s `QuestionResult` gained four additive fields, all with defaults so
  every existing construction (in code and in already-saved `results.json` files) stays
  valid: `plan_step_count`, `executed_step_count`, `evidence_count`, `evidence_coverage`.
  These are read directly off the `AgentState` the graph already returns
  (`final_state.plan`, `final_state.evidence`) — no new graph output was needed.
- New `bench/evaluation_report.py`: a second, pure pass over already-saved `results.json` (no
  graph call, no LLM call) that groups by tier and computes per-tier averages, refusal recall,
  false-refusal rate, and the forbidden-causal-phrase scan, writing both a Markdown report and
  a JSON summary.

**Why the existing benchmark runner was extended, not replaced:**
`bench/runner.py` already did exactly the right thing — drive the real graph, record what
happened, never grade correctness it can't independently verify. The gap was narrower than a
new component: it just wasn't reading two fields (`final_state.plan`, `final_state.evidence`)
it already had in hand. Splitting "run and record" from "aggregate and compare" into two
files (`runner.py` / `evaluation_report.py`) mirrors how `bench/runner.py` already separates
from `bench/schema.py` — one file drives, one defines contracts, and now one aggregates; no
file took on a second responsibility it didn't already have hints of.

**Why no LLM-based evaluator was introduced:**
`expected_observation` and `acceptable_conclusion_style` are exactly the kind of judgment an
LLM-as-judge could plausibly score — and exactly the kind this project's own evaluation
philosophy (`bench/README.md`: "independent oracles, not self-grading") warns against
introducing casually. An LLM judge grading an LLM-produced answer is not independent in the
sense that phrase means here; it would need its own validation (does the judge's own scoring
agree with a human's, is it stable across runs) before its numbers could be trusted, which is
a real, separate piece of scope this task didn't ask for and that the constraints ("no new
agents") explicitly ruled out. What's checkable *without* that risk shipped now
(`forbidden_causal_phrases`, plan/execution/evidence counts); what genuinely needs either a
human rubric pass or a validated LLM judge stays honestly unscored rather than pretend-scored.

**Alternatives considered:**
- Storing `evaluation_tier` directly on each question in `bench/questions.json` — rejected:
  it's a pure function of `category`, so storing it risks the two drifting out of sync: a
  question's category could be edited without its tier being updated to match.
- Auto-grading `expected_observation` by checking whether its stated fact's numbers appear
  among the cited evidence's values — considered, but this would need the same regex-based
  number extraction `adia.validate.static` already does, applied a second time for a
  different purpose (checking a specific claim, not just any grounded number), and would
  silently pass a technically-grounded but wrong observation (e.g. citing the *second*-lowest
  category's numbers with correct arithmetic). Left as an honestly-unscored rubric field
  rather than a check that would look more rigorous than it is.

**Consequences:**
`bench/results/evaluation_report.md` now prints, from a real benchmark run: `direct` tier
averages 1.00 plan steps and `investigation` tier averages 5.00 — the actual measured
difference between a lookup and an investigation, not a claim. Refusal recall and
false-refusal rate are now tracked as an explicit regression guard for the exact bug fixed
two phases ago (the feasibility agent over-refusing "why" questions) — if it regresses,
`false_refusal_rate` will move off `0.0` and the report will show it directly.

---

## 2026-08-24: Investigation DAGs reuse PlanStep; no new RootCauseAgent

**Status:** Accepted

**Context:**
The system could answer direct questions ("which category has the highest sales?") but not
investigate a "why" question the way a human analyst would: establish the fact, propose
several candidate explanations, test each with real evidence, and conclude honestly about
what's supported versus what would be pure speculation. A prior architecture review (see the
conversation history) confirmed the *data structure* for this already existed —
`adia.models.plan.PlanStep` already carries `depends_on` and its own docstring already calls
it "one node in the Planner's step DAG" — but nothing in the system actually executed a plan
in dependency order or let one step's arguments be grounded in an earlier step's finding.

**Decision:**
- No new agent, no new graph node, no new top-level contract. `PlanStep`, `Evidence`,
  `ToolResult`, and `ValidationResult` are reused exactly as they were.
- `execute_tools_node` (`adia/graph/nodes.py`) gained a private `_topological_order` helper
  (textbook Kahn's algorithm: in-degree table + FIFO queue) so steps run in dependency order
  regardless of the order the planner listed them in. A step whose `depends_on` names an
  unknown step_id, or that's part of a cycle, becomes a typed `ToolError(kind=VALIDATION)`
  for that step alone — never an exception, never a guess.
- `generate_tool_arguments` (`adia.agents.argument_generator`) gained one new optional
  parameter, `dependency_context: str = ""`. `execute_tools_node` renders the evidence
  produced by a step's own dependencies with the *existing* `render_evidence_context` (no new
  rendering logic, no new evidence model) and passes that text through, so a dependent step's
  LLM-proposed arguments can reference a prior step's actual finding instead of only the
  dataset catalog. The default keeps every dependency-free step's prompt byte-identical to
  before this parameter existed.
- The Planner's and Synthesizer's system prompts (prose only, no schema change) now guide
  "why"/"how" questions toward an observation-step-then-supporting-analyses shape, and guide
  final answers to keep observed facts, evidence-supported associations, and unsupported
  causal claims visibly distinct.
- `compare_groups` now sets `causal_claim_allowed: False` on its evidence, mirroring
  `compute_correlation`'s existing precedent exactly. This was the one real, pre-existing gap
  found during review: `adia.validate.static.validate_answer`'s causal-language check already
  existed and already worked, but only `compute_correlation` evidence could ever trigger it —
  a root-cause conclusion built on a `compare_groups` result had no structural protection
  against overclaiming causation until now. `run_sql` is deliberately left unchanged (a
  descriptive query result isn't inherently more or less causal than any other tool's output,
  and touching it wasn't in this phase's scope).
- `bench/questions.json` gained three `root_cause`-category questions (the category already
  existed — no schema change), including the Office-Supplies-sales example from the review.

**Why PlanStep was reused instead of a new "investigation plan" type:**
The DAG shape was never missing — `depends_on` already existed and was already validated
(unresolvable dependency, self-dependency, duplicate step_id) by `create_plan`. What was
missing was purely mechanical: the executor didn't read it, and no step could see a prior
step's output. Both gaps are addable to the existing types without touching their contracts.
Introducing a parallel "InvestigationStep" type would have meant two plan representations to
keep in sync, two things `execute_tools_node` and `argument_generator` would need to handle,
and no benefit — a "why" plan and a direct-lookup plan differ only in whether `depends_on` is
populated and how many steps there are, not in kind.

**Why no new RootCauseAgent:**
An investigation isn't a different kind of reasoning that needs its own agent — it's the same
three agents (Planner, ArgGen, Synthesizer) doing their existing jobs with two new inputs
(dependency order, dependency evidence) they didn't have before. A dedicated agent would have
meant a fourth LLM call to route between "direct" and "investigation" modes, plus a decision
about who owns that boundary — this system already has a place questions like that route from
(the Planner, via its own judgment about plan shape), so adding a router agent on top would
have duplicated that judgment rather than extended it.

**Alternatives considered:**
- A `PlanStep.role` field (`observation`/`hypothesis`/...) — deferred, not rejected: role is
  inferable from topology alone (no incoming `depends_on` = anchor; has `depends_on` = tests
  the anchor) for a first pass, and adding a field later is additive. Skipping it now avoids a
  schema change every existing plan, test, and mocked fixture would otherwise need to tolerate.
- A replan/repair loop reacting to what a supporting-analysis step finds — explicitly out of
  scope. `AgentState.replan_count`/`repair_attempts` already exist, unused, for exactly this
  kind of future work; adding a loop now would mean new conditional graph edges and
  termination conditions well beyond "execute the DAG that already exists."
- Setting `causal_claim_allowed: False` on `run_sql` too — deferred per this phase's explicit
  scope; `compare_groups` was the one gap directly relevant to root-cause investigation
  (comparison is the tool that most naturally invites a causal misreading).

**Consequences:**
`execute_tools_node` is no longer a flat loop — it's one extra (still small, still
dependency-free of any new library) private function away from being one, and every existing
dependency-free plan (100% of plans before this phase) executes identically to before, since
the empty-`depends_on` case reduces to list order for both the topological sort and the
dependency-context default. The causal-guard fix changes real behavior: an answer citing
`compare_groups` evidence that used causal language will now be rejected by
`validation_node` where it previously wasn't — re-run the full benchmark after this change to
confirm no regressions among the previously-passing 22 questions, not just the new ones.

---

## 2026-08-24: A thin `python -m adia` CLI, no business logic, no UI framework

**Status:** Accepted

**Context:**
Until now the only way to run a question through the full graph was `bench/runner.py`, which
exists to batch-execute the fixed question set in `bench/questions.json` and record outcomes
— it has no notion of "ask one arbitrary question interactively." That gap made the system
hard to demonstrate or manually probe without either writing a throwaway script or editing the
benchmark file.

**Decision:**
- Added `adia/cli.py` and `adia/__main__.py` so `python -m adia` runs an interactive prompt:
  it asks for a `dataset_id`, then a question, then prints the answer, a `PASSED`/`FAILED`/
  `NOT RUN` validation status, and (only when evidence exists) the sorted evidence IDs cited.
- The CLI is a pure interface layer: `adia.cli.answer_question` does nothing but call
  `adia.graph.state.create_initial_state` and `adia.graph.workflow.run_graph` — the exact same
  two calls `bench/runner.py::run_question` already makes — and format the resulting
  `AgentState`. It contains no feasibility, planning, tool-argument, execution, synthesis, or
  validation logic of its own; every one of those decisions is still made inside the graph,
  by the same nodes the benchmark runner exercises. An unregistered `dataset_id` is not
  special-cased here either — it flows into `feasibility_node` exactly as it does for a
  benchmark question, and comes back as the same grounded refusal.
- No UI framework (Streamlit, a web server, anything with its own dependency and lifecycle) was
  added. A terminal prompt was the smallest interface that satisfies "let a person ask an
  arbitrary question" — it needs no new dependency (`input()`/`print()` only), no process
  model, and no design work beyond formatting three already-existing `AgentState` fields
  (`final_answer`, `validation`, `evidence`).

**Alternatives considered:**
- Argument-parsing (`argparse`) flags for non-interactive one-shot invocation — deferred, not
  rejected outright: the task asked for a minimal interface matching a specific interactive
  transcript, and `answer_question(dataset_id, question)` is already a plain function any
  future flag-based entry point can call directly without touching graph code, so adding flags
  later is additive, not a rework.
- Putting the CLI under `adia/ui/` — rejected: that package is reserved for a real (likely
  web-based) UI layer that doesn't exist yet; conflating a terminal prompt with that future
  component would make "UI phase" ambiguous later. `adia/cli.py` names what it is.
- Formatting output inside `adia.graph.nodes` or `adia.graph.workflow` so the CLI could just
  print a pre-built string — rejected: the graph's job is to produce a typed `AgentState`, not
  presentation text for one particular caller; `bench/runner.py` needs the same state shaped
  into JSON, not terminal-formatted prose, so the formatting belongs at each caller's edge, not
  inside the graph.

**Consequences:**
Both the benchmark runner and the interactive CLI are now two thin, independent callers of the
same `create_initial_state` -> `run_graph` -> `AgentState` contract, with zero shared
presentation code and zero duplicated business logic between them — a future third caller (an
API endpoint, a real UI) follows the identical pattern. `adia/ui/` remains an intentionally
empty placeholder: nothing about today's change decides what that future layer looks like.

---

## 2026-08-24: Sanitize mechanical-fallback display labels instead of loosening the validator

**Status:** Accepted

**Context:**
`adia.agents.synthesizer._mechanical_fallback` is the safety net `synthesize_answer` falls
back to whenever the LLM's own proposal is unreachable, blank, or fails
`adia.validate.static.validate_answer` — and it is documented, and relied on elsewhere, as a
path that "can never itself fail grounding validation," since it only ever restates a value
already present in evidence, cited by ID.

A live benchmark run (`bench/runner.py`) surfaced a case where that guarantee didn't hold:
benchmark question q004 ("Which 10 products have the highest total Sales?") failed
`validation_passed` even after falling back. Diagnosis (see conversation history) found two
independent problems:
1. The real LLM's own answer was correctly grounded — every stated figure matched evidence —
   but `validate_answer`'s regex-based number extractor mistook digits embedded in product
   names ("Canon imageCLASS **2200** Advanced Copier") for claimed numeric facts with no
   matching evidence.
2. The mechanical fallback, invoked after (1), *also* failed: it prints the evidence
   renderer's flattened dotted-path key names verbatim (e.g. `rows[0].total_sales`), and the
   same regex misread the list-index digit (`0`) sitting between `[` and `]` as a second,
   unrelated claimed number with no matching evidence.

(1) is a validator false positive against genuinely correct text; (2) is what actually broke
the previously-relied-on "fallback always passes" invariant, since it's the path everything
else assumes is unconditionally safe.

**Decision:**
- Fix only the *presentation* of the mechanical fallback, not the validator. Added
  `_sanitize_label` in `adia/agents/synthesizer.py`, which strips every `[<digits>]` segment
  from a key name before it is printed (`rows[0].total_sales` → `rows.total_sales`,
  `feature_importance[0].importance` → `feature_importance.importance`). The key used to look
  up the value, and the value itself, are never touched — only the label shown to the reader.
- `adia.validate.static.validate_answer`'s number-extraction regex, its citation format, and
  every other grounding rule are unchanged. `adia/evidence/renderer.py` (the source of the
  bracketed key names) is unchanged too — the index notation is still exactly how evidence is
  addressed internally; it just no longer leaks into rendered answer text.

**Alternatives considered:**
- Loosening `_extract_claimed_numbers`'s regex to ignore digits adjacent to `[`/`]` (or inside
  any bracketed token) — rejected: the validator is the one place in this system that is
  supposed to be conservative on purpose (per its own docstring, "this layer gates, it doesn't
  advise"); carving out an exception for one presentation quirk of one caller risks quietly
  widening what the validator accepts for every other answer path in the system, including the
  real LLM path, for a problem that is really about what `_mechanical_fallback` chooses to
  print, not about what the validator should tolerate.
- Fixing question (1) — the LLM tripping over digits in product names — with a system-prompt
  instruction telling the LLM to be careful around numeric-looking proper nouns — considered,
  but deferred as a secondary, lower-confidence mitigation: an LLM instruction can reduce this
  class of false positive but can't reliably eliminate it, whereas fixing the fallback's label
  formatting is a complete, deterministic fix for the actual observed failure (which was the
  fallback failing, not the LLM's prose reaching the user in a broken state).
- Excluding bracket-indexed keys from `_select_fallback_values`'s ranking entirely — rejected:
  that would silently make the fallback worse (it would refuse to report exactly the granular
  row/list values that make the fallback answer meaningful in the first place, e.g. individual
  product sales figures) to work around what is really a display-formatting problem, not a
  ranking problem.

**Consequences:**
`_mechanical_fallback`'s "always grounded, always passes `validate_answer`" guarantee is
restored and now verified directly by test (`tests/test_synthesizer.py`,
`TestMechanicalFallbackLabelSanitization`), including the exact q004 shape that exposed the
break. A narrower, separate issue was found but deliberately left unfixed here, out of this
change's scope: a renderer-*synthesized* count value (e.g. `rows_count`, inserted by
`adia.evidence.renderer._walk_summary` and not a real field any tool reported) can, in
principle, fail `validate_answer` if nothing else in that evidence record's real data happens
to share its numeric value — every real tool output checked so far coincidentally avoids this
(e.g. `run_sql` always reports a real `row_count` leaf equal to its row list's length), but
that's a property of today's tools, not a guarantee `_select_fallback_values` currently
enforces. Worth a follow-up if a future tool's output shape doesn't share that coincidence.

---

## 2026-08-23: Benchmark and dataset-registry structure precedes any real dataset

**Status:** Accepted

**Context:**
Phase 2B needed to establish the benchmark and dataset-registration infrastructure before
any real dataset is acquired, so that adding one later is a data-entry task rather than a
design task.

**Decision:**
- `bench/questions.json` ships with six *template* questions — one per category — against a
  placeholder `dataset_id` (`"template_dataset"`) that is registered nowhere. They fix the
  schema and categories; they are not live benchmark cases.
- `BenchmarkQuestion` (`bench/schema.py`) enforces that only the `unanswerable` category may
  have `answerable: False` — category and answerability can't silently drift apart.
- Dataset registration (`DatasetConfig` in `adia/models/dataset.py`, loader in
  `adia/data/registry.py`) is purely declarative: `load_registry` never touches the
  filesystem paths it records. Whether a registered dataset's file actually exists is only
  checked when something calls `adia.data.loader.load_dataset` on it.
- `BenchmarkQuestion` lives in `bench/schema.py`, not `adia/models/` — it's a benchmarking
  concern the runtime system doesn't need to know about, unlike `DatasetConfig`, which the
  data layer itself consumes.

**Alternatives considered:**
- Shipping `questions.json` empty until a real dataset exists — rejected: an empty file
  doesn't document the schema or categories, and the task asked for both.
- Validating `file_path` existence at registration time — rejected: would make the registry
  mechanism impossible to demonstrate or test before any real dataset is acquired, and
  couples a purely declarative operation to filesystem state it doesn't need.
- Putting `BenchmarkQuestion` in `adia/models/` alongside `DatasetConfig` — rejected: the
  question schema has no runtime consumer yet (no agent, no benchmark runner), so it doesn't
  belong in the layer the actual system depends on.

**Consequences:**
Adding a real dataset later means: write a `DatasetConfig` entry, replace the
`"template_dataset"` questions with real ones referencing it, and nothing about the schema,
loaders, or validators needs to change. Running a question against a dataset still requires
the agent/graph layer, which does not exist yet.
