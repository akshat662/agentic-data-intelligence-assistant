# ADIA Benchmark

> **Status: real dataset, real question set, real runner, real evaluation report.** The
> `superstore` dataset is registered and profiled; `bench/questions.json` holds 25 real
> questions against its actual schema; `bench/runner.py` drives every question through the
> full LangGraph agent (real LLM calls); `bench/evaluation_report.py` turns the results into
> tier-grouped comparison metrics. What's still not implemented is listed in
> [Implemented vs. Planned](#implemented-vs-planned) — most notably, independent gold/oracle
> answers for numeric correctness grading.

## Purpose

A demo that answers five hand-picked questions correctly proves nothing — it's as likely to
reflect a lucky prompt as a working system. This benchmark exists to make ADIA's central
claim checkable rather than asserted: that grounding numbers in deterministic tool output,
instead of letting an LLM state them, measurably changes how often the system is wrong — and,
as of the evaluation framework below, that the system does more than translate a question into
one SQL call: it plans multi-step investigations, threads evidence between steps, and hedges
conclusions it can't fully support.

That claim only means something if it survives contact with:
- questions the system should refuse, not just questions it can answer
- questions that need more than one tool call to answer honestly, not just lookups
- an independently computed answer to check against (never the pipeline grading itself)
- a comparison against a simpler architecture, so "agentic" is shown to earn its complexity
  rather than assumed to

## Question Categories and Evaluation Tiers

Every question belongs to exactly one of six `QuestionCategory` values (`bench/schema.py`):

| Category | What it tests | Example gold tool(s) |
|---|---|---|
| `descriptive` | Can the system describe the dataset itself — shape, columns, types? | `profile_dataset` |
| `sql_aggregation` | Can it answer a question that requires filtering/grouping/aggregating rows? | `run_sql` |
| `statistical_comparison` | Can it compare a metric across groups correctly? | `compare_groups` |
| `root_cause` | Can it *investigate* why a metric is what it is — not just state it? | `run_sql`, `compare_groups`, `segment_contribution` |
| `predictive` | Can it fit a model and honestly report how much better than a naive baseline it is? | `train_model` |
| `unanswerable` | Does it decline instead of guessing, when the data can't support an answer? | *(none — refusal is correct)* |

`unanswerable` is not an afterthought slice. A system that never refuses looks impressive
right up until it confidently answers a question the data cannot support. Measuring this
means measuring **refusal recall** (does it actually decline these?) *and* **false-refusal
rate** (does it wrongly decline questions it could have answered?) — a system that refuses
everything scores perfectly on the first metric and is useless.

`bench/schema.py` also defines `EvaluationTier`, a **derived, not stored** grouping of those
six categories into three evaluation groups — `direct` (`descriptive`, `sql_aggregation`,
`statistical_comparison`, `predictive`: answerable with one tool call by construction),
`investigation` (`root_cause`: answerable only by planning and executing several
dependency-linked steps), and `refusal` (`unanswerable`). `BenchmarkQuestion.evaluation_tier`
computes this from `category`, so no question's JSON needs to name its own tier, and existing
question data never needs to change when the tier concept was added.

### Investigation metadata

`root_cause` questions may carry an optional `investigation: InvestigationExpectation` block:
`expected_observation` and `acceptable_conclusion_style` are rubric prose for a human
reviewer (grading them automatically needs an independent oracle, which doesn't exist yet —
see below); `expected_analysis_dimensions` names the candidate explanations a good plan
should test; `forbidden_causal_phrases` is the one part that's checked automatically —
`bench/evaluation_report.py` scans each investigation question's final answer for these
phrases verbatim (case-insensitive), independent of and in addition to
`adia.validate.static.validate_answer`'s own causal-language check.

## Evaluation Philosophy

- **Independent oracles, not self-grading.** A correct answer is only meaningful if it was
  computed a second, independent way (plain pandas/scipy, written without looking at the
  pipeline's own code). Grading the system against its own tool output would prove nothing
  except that the tool ran. This applies to the causal-phrase scan too: it's a second,
  separate text check against `bench/questions.json`'s own phrase list, not a re-run of the
  validator's own regex.
- **Tolerance-banded, not exact-match, numerical correctness.** Floating-point arithmetic and
  reasonable rounding mean "correct" has to mean "within tolerance of the oracle," not
  bit-identical. (Not yet implemented — see below.)
- **Grounding is measured, not assumed.** `validation_passed` in every `QuestionResult` comes
  from actually running `adia/validate/static.py` over the answer this benchmark produced.
- **Structure is measured, not assumed.** `plan_step_count`, `executed_step_count`,
  `evidence_count`, and `evidence_coverage` (`bench/runner.py`) make "the investigation tier
  actually plans and executes more steps than the direct tier" a number
  `bench/evaluation_report.py` prints, not a claim made in prose.
- **A comparison arm, not just a pass rate.** The interesting number isn't "how many questions
  did the full system get right" — it's "what changed when grounding enforcement was turned
  off," and, now, "what changed between a direct lookup and a multi-step investigation." That
  comparison is what turns "agentic" from a claim into a measured result.
- **Cost and latency are reported, not hidden.** `duration_ms` is recorded per question.

## How the Pieces Fit Together

- `bench/questions.json` — the question set (25 questions), validated by `bench/schema.py`'s
  `load_questions`. Every question names `dataset_id: "superstore"`, which resolves through
  the dataset registry at `data/registry.json`.
- `bench/schema.py` — `BenchmarkQuestion`, `QuestionCategory`, `EvaluationTier`,
  `InvestigationExpectation`, and `load_questions`.
- `bench/runner.py` — `run_benchmark` drives every question through the real
  `adia.graph.workflow.run_graph` (real LLM calls at every agent) and writes one
  `QuestionResult` per question to `bench/results/results.json` (gitignored — regenerate with
  `python -m bench.runner`).
- `bench/evaluation_report.py` — `generate_summary` groups `results.json` by
  `evaluation_tier` and computes the metrics above; `python -m bench.evaluation_report` writes
  `bench/results/evaluation_report.md` (human-readable) and `evaluation_report.json`
  (machine-readable), both gitignored — regenerate after every `bench.runner` run.
- `data/registry.json`, `data/superstore.csv`, `data/catalog/superstore.json` — the
  registered, profiled evaluation dataset (see `data/README.md`).

## Implemented vs. Planned

**Implemented:**
- `bench/schema.py` — question contract, evaluation tiers, investigation metadata.
- `bench/questions.json` — 26 questions: 1 `descriptive`, 5 `sql_aggregation`, 5
  `statistical_comparison`, 5 `predictive`, 5 `root_cause` (with investigation metadata), 5
  `unanswerable`.
- `bench/runner.py` — drives the real graph end-to-end; records feasibility verdict, tools
  used, plan/execution/evidence counts, validation outcome, and the final answer.
- `bench/evaluation_report.py` — tier-grouped comparison metrics, refusal recall/false-refusal
  rate, forbidden-causal-phrase scan, Markdown + JSON report.
- `data/superstore.csv`, `data/registry.json`, `data/catalog/superstore.json` — a real
  registered, profiled evaluation dataset (see `data/README.md`).

**Not implemented (planned):**
- Gold/oracle answers, or independently-written oracle scripts, for numeric correctness
  grading — `evaluation_report.py` measures structure (steps, coverage, grounding, refusal
  correctness) and a deterministic phrase scan, not whether a stated number is the *right*
  number.
- Automatic grading of `expected_observation`/`acceptable_conclusion_style` — these remain a
  rubric for a human reviewer, deliberately not auto-scored (see `docs/DECISIONS.md`'s entry
  on why no LLM-based evaluator was introduced for this).
- The 3-arm ablation comparison (grounding on vs. off) and a `EVAL.md`-style narrative report.

**Phase 7 addition:** `adia/tools/segment_contribution.py` — ranks how much each sub-group of
a category contributes to a metric's total (row count, summed/mean metric, share of the
in-scope total, ranked), optionally scoped to one parent value (e.g. break Sales down by
Sub-Category within `Category == "Technology"`). Wired through the same
planner/argument-generator/`execute_tools_node` allowlist every other tool uses — no new agent,
no graph change. `q026` is the first question written specifically to exercise it.
