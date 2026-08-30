# Talking About ADIA in an Interview

This is a companion to [`ARCHITECTURE.md`](ARCHITECTURE.md) (the technical reference) and
[`DECISIONS.md`](DECISIONS.md) (the full decision log). It's written for the conversation, not
the codebase: the questions an interviewer is likely to actually ask, answered the way you'd
say them out loud, with a pointer to the real code and a real, current benchmark number behind
each claim — nothing here is aspirational.

## Architecture Explanation

### Why LangGraph, instead of a single prompt or a plain function pipeline?

Because the system needs **branching and repair loops that persist typed state**, not just a
sequence of function calls. A question that turns out to be infeasible has to skip planning
and tool execution entirely and jump straight to a refusal (`route_after_feasibility`,
`adia/graph/workflow.py`) — that's a real conditional edge, not an `if` statement bolted onto a
script. A plain chain of function calls could technically do this too, but LangGraph gives it
for free: a typed shared state object (`AgentState`) threaded through every node, a compiled
graph you can introspect (`build_graph().get_graph().nodes`, which is literally how
`tests/test_graph_workflow.py::TestBuildGraph` asserts the six nodes exist), and — the part
that mattered most for Phase 6B — `.stream(state, stream_mode="updates")` for free, which is
what `POST /chat/stream`'s live progress narration is built on with **zero changes to any
node**. That last point is worth having ready: it's a concrete example of a framework choice
paying for itself later, not just theoretical.

### Why multiple agents (Feasibility, Planner, Argument Generator, Synthesizer) instead of one?

Each one answers a **different kind of question**, and splitting them makes each individually
small, testable, and safe to fail. Feasibility asks "can this be answered from this dataset at
all" — a yes/no/needs-clarification judgment cross-checked against the real catalog
(`adia/agents/feasibility.py`). Planner asks "what steps, in what order" — never touching
column names or SQL. Argument Generator asks "what exact arguments for *this* tool" — the only
place that sees a specific tool's contract. Synthesizer asks "how do I explain what was found"
— the only place that produces prose. If this were one agent, one bad call couldn't fail
safely in isolation; as four, each has its own narrow failure mode and its own typed,
independently-tested fallback (`docs/ARCHITECTURE.md` §4). This isn't "more agents for their
own sake" — Phase 7 (adding `segment_contribution`) deliberately did **not** add a fifth agent;
it reused all four exactly as they were, because the new capability was a *tool*, not a new
*kind* of judgment.

### Why tools instead of just letting the LLM write and run pandas/SQL directly?

Because an LLM asked to "write code that computes X" will occasionally write code that computes
something close to X, and there is no way to check that from the code alone — you'd have to
re-derive the right answer to know it was wrong. A fixed tool (`adia/tools/compare_groups.py`,
`segment_contribution.py`, ...) computes one well-specified thing the same way every time; the
LLM's job shrinks to *choosing which tool and which columns*, both of which are checkable
against the real dataset catalog before anything executes (`adia/agents/argument_generator.py`
rejects any column the LLM names that isn't actually in `catalog.column_names()`). The one
partial exception, `run_sql`, still isn't "arbitrary code execution" — every query is parsed
and guarded (`adia/tools/sql_guard.py`: single read-only `SELECT`, one table, only real
columns) before it reaches DuckDB.

### Why an evidence layer, instead of the Synthesizer just reading tool output directly?

Because "the LLM saw the right numbers" and "the LLM's answer is provably grounded in them" are
different guarantees, and only the second one is checkable after the fact. Every tool call
writes one immutable `Evidence` record with a content-addressed ID
(`generate_evidence_id(tool, args)` — same args always hash to the same ID, so a repeated call
is a cache hit, not a new fact); the Synthesizer is shown a rendered summary of these records
and required to cite the exact ID next to every number it states
(`[[ev_run_sql_1a2b3c4d]]`). That citation marker is what turns "trust the model" into
"re-extract every number the model claimed and check it against the actual record it says
backs it up" — a mechanical check, not a matter of the prose sounding credible.

### Why a separate validation layer, if the Synthesizer is already told to cite evidence?

Because an instruction in a prompt is a request, not a guarantee — an LLM can ignore it, and
does. `adia/validate/static.py::validate_answer` is a second, independent, code-only pass: it
re-extracts every numeral in the answer text and confirms each one matches a real value in the
*cited* evidence (within a small tolerance), and it separately scans for causal language
(`leads to`, `due to`, `causes`, ...) and fails the answer if it's paired with a citation whose
tool explicitly marked itself `causal_claim_allowed: False`. This is not hypothetical: while
building Phase 7's benchmark example, the real Synthesizer proposed *"...the higher average
sales amount per transaction **leads to** a total sales of 836,154..."*, citing a
`segment_contribution` record. That tool's output is `causal_claim_allowed: False` (a share of
a total says nothing about why one segment is larger) — the validator caught it, rejected the
draft, and the system fell back to `_mechanical_fallback` (a plain, citation-only restatement
of the evidence) rather than ship the causal claim. That's the validator doing real, measured
work on a real run, not a hypothetical safety net — see `adia/agents/synthesizer.py`'s
`_mechanical_fallback` for what "fails safe" actually looks like in code.

## Design Decisions (Q&A)

**Q: Why not just use text-to-SQL?**
Because a "why" question isn't answerable by one query. *"Why does Technology have the highest
total Sales?"* needs an observation (which category is highest), then a decomposition (which
Sub-Categories drive that total), then supporting checks (is it order volume, price, discount)
— a small dependency graph of steps, not a single `SELECT`. `bench/questions.json`'s
`root_cause` category exists specifically to measure this: those questions average **5.00**
plan steps versus **1.44** for direct lookups, in the last real benchmark run
(`bench/results/evaluation_report.md`) — a measured structural difference, not a claim.

**Q: Why validate LLM answers instead of trusting a well-written prompt?**
Because fluent, well-formatted prose can still contain a claim nothing supports — and a good
prompt reduces how *often* this happens, but can't make it impossible to check. See the real
`segment_contribution` causal-language example above: the prompt already tells the Synthesizer
not to claim causation, and it did anyway, on a real run. The validator is what actually
stopped it from reaching a user.

**Q: Why deterministic tools instead of letting the LLM reason its way to the number?**
Because "reasoning to a number" and "computing a number" have different failure rates, and only
one of them is checkable. The LLM is good at the parts that need judgment — which columns are
relevant, which comparison would test a candidate explanation, how to phrase the conclusion
honestly. It is not the thing computing `SUM(Sales)` or a group mean; `adia/tools/*.py` always
is, with the exact same pandas/DuckDB call every time. This is the one-line pitch for the whole
project: **the LLM decides what to compute and how to say it; it never *is* the computation.**

**Q: Why did Phase 7 add a tool (`segment_contribution`) instead of a smarter prompt for the
existing tools?**
Because the underlying capability — "which sub-groups actually drive this total, ranked by
share" — genuinely didn't exist in any existing tool's output; no prompt change could make
`compare_groups` (whole-dataset group means) produce a parent-scoped, ranked contribution
breakdown, because it isn't computing that. Once the real gap was identified (see
`bench/README.md`'s own "Not implemented" list before Phase 7), the fix was a new deterministic
function, wired through the exact same Planner/Argument-Generator/dispatch pattern every other
tool already used — not a new agent, not a graph change. Good practice for the "how do you
decide what's a prompt fix vs. a real change" question in general: if the LLM is being asked to
produce a number no tool actually computes, that's a missing tool, not a missing instruction.

**Q: What's the biggest real limitation of this system?**
It cannot perform actual causal inference, and says so rather than pretending otherwise — no
experimental data, no instrumental variables, nothing in a retail transactions table could
support a causal claim regardless of how the system were built. What it *can* do is state an
observation precisely, test several candidate explanations against real computed evidence, and
describe what's merely *associated with* the observation versus what remains unexplained. The
"Limitations" section of the root README says this plainly rather than burying it in a caveat,
which is itself a deliberate choice — see `docs/DECISIONS.md` for the reasoning behind
treating limitations as a stated design property, not marketing to work around.

**Q: How do you know the grounding/validation actually changes anything, versus just being
extra code that never fires?**
Because it's measured, not assumed — `bench/evaluation_report.py` reports a real
`validation_pass_rate` per tier from `bench/results/results.json`, and separately scans every
investigation-tier answer for forbidden causal phrases as an independent check of the same
property. The real example above (a genuine validation rejection on a real run) is the honest
answer to "has this ever actually caught anything" — yes, during this very project's
development, not in a contrived test case.
