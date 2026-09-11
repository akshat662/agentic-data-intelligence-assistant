"""Streamlit UI for ADIA -- `streamlit run app.py`.

A thin interface layer, same spirit as `adia/cli.py`: it collects a dataset and a question and
hands them to the same graph the CLI, the benchmark, and the FastAPI backend all drive
(`adia.graph.state.create_initial_state` -> `adia.graph.workflow.stream_graph`). No feasibility,
planning, tool execution, synthesis, or validation logic is duplicated here.

Dataset upload reuses `adia.api.service.register_dataset` directly rather than going over HTTP
to a running FastAPI process -- this file *is* the whole app, so there is no separate backend
to keep running alongside it.
"""

import os
import re
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from adia.api.service import (
    REGISTRY_PATH,
    UPLOAD_DIR,
    DatasetAlreadyRegisteredError,
    InvalidCsvError,
    register_dataset,
)
from adia.data.registry import load_registry
from adia.evidence.renderer import render_evidence
from adia.graph.state import create_initial_state
from adia.graph.workflow import stream_graph
from adia.models.dataset import DatasetConfig
from adia.models.state import AgentState

_REPO_ROOT = Path(__file__).resolve().parent
_DATASET_ID_PATTERN = re.compile(r"^[a-zA-Z0-9_-]+$")

#: Session-state key the sidebar `st.selectbox` itself is bound to (via `key=`) -- this, not a
#: separate shadow variable, is the single source of truth for which dataset is active. A
#: keyless/unbound selectbox remembers its own last-clicked value across reruns and silently
#: ignores a fresh `index=` argument on every later run, which is what let a previously
#: uploaded dataset stay "stuck" as active even after `index` was recomputed to point at a
#: newly registered one -- binding the widget directly to this key is what actually lets code
#: change its value.
_DATASET_KEY = "dataset_selectbox"

load_dotenv()

st.set_page_config(
    page_title="ADIA — Agentic Data Intelligence Assistant", page_icon="🔎", layout="wide"
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of {"role", "content", "steps", "evidence"}

if "_pending_active_dataset" in st.session_state:
    # Set by a just-completed dataset registration below, then `st.rerun()`-ed here. Applied
    # at the very top of the script, before the selectbox is instantiated: Streamlit forbids
    # writing to a widget's own session-state key *after* that widget has already rendered in
    # the same run, so the switch is deferred to the start of this fresh run instead of being
    # written directly from inside the button handler that requested it.
    st.session_state[_DATASET_KEY] = st.session_state.pop("_pending_active_dataset")

#: A `st.success`/`st.error`/... call made right before `st.rerun()` never actually reaches the
#: browser -- `st.rerun()` aborts that script run immediately, discarding everything rendered
#: in it, message included. This is why registering a dataset could look like it "did nothing":
#: it had succeeded, but its confirmation was thrown away by the very rerun that switched to it.
#: The fix is the same deferred pattern as `_pending_active_dataset` above: stash `(method,
#: text)` before rerunning, then render it for real on the next run, where it will actually
#: paint before anything else can clear it.
_flash = st.session_state.pop("_flash_message", None)


def _resolve(file_path: str) -> Path:
    """Resolve a `DatasetConfig.file_path` (relative or absolute) against the repo root."""
    path = Path(file_path)
    return path if path.is_absolute() else _REPO_ROOT / path


def _available_datasets() -> dict[str, DatasetConfig]:
    """Registered datasets whose backing file actually exists on disk.

    The registry is purely declarative (`adia.data.registry`) and can contain entries for
    files that were never committed or have since moved -- filtering here keeps a stale
    registration from turning into a confusing dead entry in the picker.
    """
    if not REGISTRY_PATH.exists():
        return {}
    registry = load_registry(REGISTRY_PATH)
    return {
        dataset_id: config
        for dataset_id, config in registry.items()
        if _resolve(config.file_path).exists()
    }


def _slugify(stem: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9_-]+", "_", stem).strip("_")
    return slug or "dataset"


def _summarize_node(node_name: str, partial: dict, state: AgentState) -> str:
    """One-line, human-readable summary of a just-completed graph node, for the step trace."""
    if node_name == "feasibility" and state.feasibility is not None:
        return f"verdict=`{state.feasibility.verdict.value}` — {state.feasibility.reason}"
    if node_name == "planner":
        steps = ", ".join(f"`{s.tool_family}`" for s in state.plan) or "(no steps)"
        return f"{len(state.plan)}-step plan: {steps}"
    if node_name == "execute_tools":
        new_evidence = partial.get("evidence", {})
        errors = partial.get("errors", [])
        return f"{len(new_evidence)} evidence record(s) produced, {len(errors)} error(s)"
    if node_name == "synthesizer":
        return "draft answer written from collected evidence"
    if node_name == "validation" and state.validation is not None:
        status = "PASSED" if state.validation.passed else "FAILED"
        return f"grounding check {status} ({len(state.validation.issues)} issue(s))"
    if node_name == "refusal" and state.refusal is not None:
        return f"refused — {state.refusal.reason}"
    return "done"


def _run_question(dataset_id: str, question: str, *, show_steps: bool) -> dict:
    """Run one question through the graph, streaming node-by-node progress into a status box.

    Returns a chat-history entry: `{"role": "assistant", "content", "steps", "evidence"}`.
    """
    initial_state = create_initial_state(question, dataset_id)
    steps: list[tuple[str, str]] = []
    final_state: AgentState | None = None

    with st.status("Running the ADIA graph...", expanded=show_steps) as status_box:
        try:
            for node_name, partial, state in stream_graph(initial_state):
                summary = _summarize_node(node_name, partial, state)
                steps.append((node_name, summary))
                status_box.write(f"**{node_name}** — {summary}")
                final_state = state
        except Exception as exc:  # noqa: BLE001 -- surfaced to the user, not swallowed
            status_box.update(label="Graph run failed", state="error")
            return {
                "role": "assistant",
                "content": f"The graph run failed: {exc}",
                "steps": steps,
                "evidence": [],
            }
        status_box.update(label="Graph run complete", state="complete")

    if final_state is None:
        return {
            "role": "assistant",
            "content": "No answer was produced.",
            "steps": steps,
            "evidence": [],
        }

    evidence = [
        render_evidence(e) for e in sorted(final_state.evidence.values(), key=lambda e: e.id)
    ]
    validation = (
        "PASSED" if final_state.validation and final_state.validation.passed
        else "FAILED" if final_state.validation
        else "NOT RUN"
    )
    return {
        "role": "assistant",
        "content": final_state.final_answer or "No answer was produced.",
        "steps": steps,
        "evidence": evidence,
        "validation": validation,
    }


# --- Sidebar: dataset selection + upload -----------------------------------------------------

with st.sidebar:
    st.header("Dataset")

    datasets = _available_datasets()
    dataset_ids = sorted(datasets)
    if dataset_ids:
        if st.session_state.get(_DATASET_KEY) not in dataset_ids:
            # First-ever load, or the previously active dataset's file no longer resolves --
            # prefer the shipped demo dataset if present, else whatever sorts first. Safe to
            # write here: this runs before the selectbox below is instantiated this run.
            st.session_state[_DATASET_KEY] = (
                "superstore" if "superstore" in dataset_ids else dataset_ids[0]
            )
        active_dataset = st.selectbox(
            "Registered datasets",
            dataset_ids,
            key=_DATASET_KEY,
            format_func=lambda d: f"{d} — {datasets[d].description[:40]}",
        )
        st.caption(datasets[active_dataset].description)

        previously_active = st.session_state.get("_last_active_dataset")
        if previously_active is not None and previously_active != active_dataset:
            st.session_state.chat_history = []
            st.toast(f"Switched to '{active_dataset}' — chat history cleared.", icon="🔄")
        st.session_state["_last_active_dataset"] = active_dataset
    else:
        active_dataset = None
        st.info("No registered datasets found yet — upload one below.")

    st.divider()
    st.subheader("Upload your own CSV")

    # Rendered here, in context, rather than at the top of the script -- see `_flash` above.
    if _flash is not None:
        getattr(st, _flash[0])(_flash[1])

    uploaded = st.file_uploader("Dataset file", type=["csv"], label_visibility="collapsed")

    if uploaded is not None:
        default_id = _slugify(Path(uploaded.name).stem)
        new_id = st.text_input(
            "Dataset ID", value=default_id, help="Letters, digits, `_`, `-` only."
        )
        new_description = st.text_area(
            "Description", value=f"Uploaded dataset from {uploaded.name}.", height=70
        )
        if st.button("Register dataset", type="primary", use_container_width=True):
            print(f"[ADIA] Register button clicked: dataset_id={new_id!r}, file={uploaded.name!r}")

            if not _DATASET_ID_PATTERN.match(new_id or ""):
                print(f"[ADIA] Rejected: dataset_id {new_id!r} fails pattern check.")
                st.error("Dataset ID must match `[a-zA-Z0-9_-]+`.")
            else:
                # Belt-and-suspenders: `register_dataset` itself already creates this
                # directory, but ensuring it here too means a permissions/missing-parent
                # problem surfaces at this explicit line, not deep inside the service call.
                UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
                print(f"[ADIA] Ensured upload directory exists: {UPLOAD_DIR}")

                try:
                    file_bytes = uploaded.getvalue()
                    print(f"[ADIA] Read {len(file_bytes)} bytes from '{uploaded.name}'.")

                    # register_dataset must fully succeed -- file written, registry.json
                    # updated on disk -- before any state-switching (`_pending_active_dataset`)
                    # or `st.rerun()` below is allowed to run.
                    response = register_dataset(
                        new_id,
                        new_description,
                        uploaded.name,
                        file_bytes,
                        registry_path=REGISTRY_PATH,
                        upload_dir=UPLOAD_DIR,
                    )
                    print(
                        f"[ADIA] Registry updated: dataset_id={response.dataset_id!r}, "
                        f"rows={response.row_count}, columns={response.column_count}"
                    )
                except DatasetAlreadyRegisteredError as exc:
                    print(f"[ADIA] Registration skipped -- already registered: {exc}")
                    if new_id in _available_datasets():
                        # The existing registration's file genuinely resolves -- safe to
                        # switch the chat to it, same as before.
                        st.session_state["_flash_message"] = (
                            "warning",
                            f"'{new_id}' is already registered — selecting it for chat.",
                        )
                        st.session_state["_pending_active_dataset"] = new_id
                        st.rerun()
                    else:
                        # The ID is taken, but by a stale registration whose file doesn't
                        # exist (e.g. left over from before this project's folder was moved)
                        # -- switching to it would silently no-op, same bug as before under a
                        # different trigger. Say so plainly instead of pretending to switch.
                        print(
                            f"[ADIA] '{new_id}' is registered but its file is missing -- "
                            "cannot select it."
                        )
                        st.error(
                            f"'{new_id}' is already registered, but its file no longer "
                            "exists on disk (a stale entry) and can't be selected. Pick a "
                            "different Dataset ID above and click Register again."
                        )
                except InvalidCsvError as exc:
                    print(f"[ADIA] Registration failed -- invalid CSV: {exc}")
                    st.error(str(exc))
                except Exception as exc:  # noqa: BLE001 -- always surfaced, never swallowed
                    print(f"[ADIA] Registration failed -- unexpected error: {exc!r}")
                    st.error(f"Registration failed: {exc}")
                else:
                    print(
                        f"[ADIA] Registration succeeded; switching active dataset to "
                        f"{response.dataset_id!r} and rerunning."
                    )
                    st.session_state["_flash_message"] = (
                        "success",
                        f"Registered '{response.dataset_id}': "
                        f"{response.row_count:,} rows x {response.column_count} columns.",
                    )
                    st.session_state["_pending_active_dataset"] = response.dataset_id
                    st.rerun()

    st.divider()
    show_steps = st.toggle(
        "Show LangGraph steps (Planner → Tools)",
        value=True,
        help="Expand the agent's step-by-step trace (feasibility, planner, tool execution, "
        "synthesizer, validation) for every answer.",
    )

    if not os.environ.get("OPENAI_API_KEY"):
        st.warning("`OPENAI_API_KEY` is not set — copy `.env.example` to `.env` and fill it in.")


# --- Main: chat --------------------------------------------------------------------------------

st.title("🔎 ADIA — Agentic Data Intelligence Assistant")
active = active_dataset  # set in the sidebar block above -- the selectbox's own bound value
if active:
    st.caption(f"Asking questions about dataset: **{active}**")
else:
    st.caption("Upload or select a dataset from the sidebar to get started.")

for turn in st.session_state.chat_history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])
        if turn["role"] == "assistant" and turn.get("steps"):
            with st.expander("Agent steps", expanded=False):
                for node_name, summary in turn["steps"]:
                    st.markdown(f"**{node_name}** — {summary}")
        if turn.get("evidence"):
            with st.expander(f"Evidence ({len(turn['evidence'])})", expanded=False):
                for ev in turn["evidence"]:
                    st.code(ev.summary, language=None)
        if turn.get("validation"):
            st.caption(f"Validation: {turn['validation']}")

question = st.chat_input(
    f"Ask a question about '{active}'..." if active else "Select a dataset first",
    disabled=not active,
)

if question and active:
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        result = _run_question(active, question, show_steps=show_steps)
        st.markdown(result["content"])
        if result.get("steps"):
            with st.expander("Agent steps", expanded=show_steps):
                for node_name, summary in result["steps"]:
                    st.markdown(f"**{node_name}** — {summary}")
        if result.get("evidence"):
            with st.expander(f"Evidence ({len(result['evidence'])})", expanded=False):
                for ev in result["evidence"]:
                    st.code(ev.summary, language=None)
        if result.get("validation"):
            st.caption(f"Validation: {result['validation']}")

    st.session_state.chat_history.append(result)
