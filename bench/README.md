# ADIA Benchmark

> **Status: infrastructure only.** No real dataset is registered, no oracle scripts exist,
> and nothing in this directory has ever been run end-to-end. This documents the shape the
> benchmark will take, not a working evaluation. See [Implemented vs. Planned](#implemented-vs-planned).

## Purpose

A demo that answers five hand-picked questions correctly proves nothing — it's as likely to
reflect a lucky prompt as a working system. This benchmark exists to make ADIA's central
claim checkable rather than asserted: that grounding numbers in deterministic tool output,
instead of letting an LLM state them, measurably changes how often the system is wrong.

That claim only means something if it survives contact with:
- questions the system should refuse, not just questions it can answer
- an independently computed answer to check against (never the pipeline grading itself)
- a comparison against a simpler architecture, so "agentic" is shown to earn its complexity
  rather than assumed to

None of that exists yet. This phase only fixes the *shape* — the question schema and the
dataset registration mechanism — so that adding real questions and real datasets later is a
data-entry task, not a design task.

## Question Categories

Every question belongs to exactly one of six categories (`bench/schema.py`,
`QuestionCategory`):

| Category | What it tests | Example gold tool(s) |
|---|---|---|
| `descriptive` | Can the system describe the dataset itself — shape, columns, types? | `profile_dataset` |
| `sql_aggregation` | Can it answer a question that requires filtering/grouping/aggregating rows? | `run_sql` |
| `statistical_comparison` | Can it compare a metric across groups correctly? | `compare_groups` |
| `root_cause` | Can it explain *why* a metric changed, not just *that* it changed? | `run_sql`, `compare_groups` |
| `predictive` | Can it fit a model and honestly report how much better than a naive baseline it is? | `train_model` |
| `unanswerable` | Does it decline instead of guessing, when the data can't support an answer? | *(none — refusal is correct)* |

The five answerable categories map directly onto the deterministic tools already built in
Phase 1 (`adia/tools/`) — a question that doesn't fit one of these categories doesn't fit the
tool surface either, which is a signal to fix the tool surface, not to write the question
anyway (see the "notebook gate" thinking in `docs/DECISIONS.md`).

`unanswerable` is not an afterthought slice. A system that never refuses looks impressive
right up until it confidently answers a question the data cannot support. Measuring this
later means measuring **refusal recall** (does it actually decline these?) *and*
**false-refusal rate** (does it wrongly decline questions it could have answered?) — a
system that refuses everything scores perfectly on the first metric and is useless.

## Evaluation Philosophy

- **Independent oracles, not self-grading.** A correct answer is only meaningful if it was
  computed a second, independent way (plain pandas/scipy, written without looking at the
  pipeline's own code). Grading the system against its own tool output would prove nothing
  except that the tool ran.
- **Tolerance-banded, not exact-match, numerical correctness.** Floating-point arithmetic and
  reasonable rounding mean "correct" has to mean "within tolerance of the oracle," not
  bit-identical.
- **Grounding is measured, not assumed.** Every claim in `docs/DECISIONS.md`'s and the
  README's "why isn't this ChatGPT with a CSV" answer is falsifiable: run the static validator
  (`adia/validate/static.py`) over the answers this benchmark produces and count what fails.
- **A comparison arm, not just a pass rate.** The interesting number isn't "how many questions
  did the full system get right" — it's "what changed when grounding enforcement was turned
  off." That comparison is what turns "agentic" from a claim into a measured result.
- **Cost and latency are reported, not hidden.** A system that is only impressive when nobody
  asks what it costs to run isn't finished.

## How the Pieces Fit Together

- `bench/questions.json` — the question set. Loaded and validated by `bench/schema.py`'s
  `load_questions`. Every question names a `dataset_id`, which must resolve through the
  dataset registry (`adia/data/registry.py`) before the question can actually be run.
- `adia/data/registry.py` — where a dataset gets a `dataset_id`, a file path, a description,
  and (optionally) its target column(s), independent of whether the file exists yet. See its
  module docstring for the exact JSON shape.
- Neither of these executes anything. Running a question against a dataset — dispatching
  tools, collecting evidence, checking the result — needs the agent/graph layer this project
  hasn't built yet.

## Implemented vs. Planned

**Implemented:**
- `bench/schema.py` — `BenchmarkQuestion` contract (with a validator enforcing that only the
  `unanswerable` category may be marked non-answerable) and `load_questions`.
- `bench/questions.json` — six template questions, one per category, referencing a
  placeholder `dataset_id` (`"template_dataset"`) that is not registered anywhere. These fix
  the schema and categories; they are not live benchmark cases.
- `adia/data/registry.py` / `adia/models/dataset.py` — the dataset registration mechanism.

**Not implemented (planned, per `docs/DECISIONS.md` and the project README's roadmap):**
- A real registered dataset.
- Gold/oracle answers, or oracle scripts.
- A benchmark runner that dispatches questions through the (not yet built) agent graph.
- The 3-arm ablation comparison and `EVAL.md` report.
