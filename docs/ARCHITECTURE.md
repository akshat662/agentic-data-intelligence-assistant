# Architecture

> Placeholder. This document will describe ADIA's actual, implemented
> architecture as it is built. For the current design intent, see
> "Planned High-Level Architecture" in the root [README.md](../README.md) —
> nothing there is implemented yet.

## Status

Repository-initialization stage only. No graph, agent, tool, or API code
exists in this repository yet.

## Sections to fill in as implementation proceeds

- **Data contracts** (`adia/models`) — pydantic schemas for plans, tool
  calls/results, evidence records, validation outcomes.
- **Data layer** (`adia/data`) — how DuckDB/pandas/pyarrow are used to load,
  query, and describe datasets.
- **Tools** (`adia/tools`) — the deterministic tool interface and the set of
  available analytical tools.
- **Validation** (`adia/validate`) — what is checked, and how a failed
  validation affects the plan/graph.
- **Evidence** (`adia/evidence`) — how provenance is recorded and linked to
  reported results.
- **Orchestration** (`adia/graph`, `adia/agents`) — the LangGraph structure,
  node responsibilities, and agent roles.
- **API** (`adia/api`) — FastAPI routes and request/response contracts.
- **UI** (`adia/ui`) — Streamlit app structure.
- **Deployment** — Docker/compose layout.
