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
