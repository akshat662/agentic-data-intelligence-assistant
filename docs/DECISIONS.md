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
