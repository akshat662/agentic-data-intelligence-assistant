"""Deterministic rendering of Evidence records into human-readable context.

This is the piece a future Synthesizer will use to see what it's allowed to cite, and a
future evidence inspector UI will use to show a human what backs a given number. It performs
no substitution into answer text (that's a later phase, once a Synthesizer exists to produce
placeholder-bearing drafts) — this module only turns one or more `Evidence` records into a
structured, deterministic, human-readable form.
"""

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from adia.models.evidence import Evidence

_MAX_SUMMARY_VALUES = 20


class RenderedEvidence(BaseModel):
    """The structured, human-readable rendering of a single `Evidence` record."""

    model_config = ConfigDict(frozen=True)

    evidence_id: str
    tool: str
    arguments: dict[str, Any] = Field(..., description="The tool call's arguments, verbatim.")
    generated_at: str = Field(..., description="ISO 8601 timestamp the evidence was produced.")
    key_values: dict[str, Any] = Field(
        ...,
        description=(
            "A bounded, deterministic flattening of the evidence's result data: scalar "
            "leaves under dotted-path keys, with lists summarized by length rather than "
            "expanded."
        ),
    )
    summary: str = Field(..., description="A ready-to-inject human-readable text block.")


def render_evidence(evidence: Evidence) -> RenderedEvidence:
    """Render a single `Evidence` record into structured, human-readable context.

    Preserves the evidence ID, tool name, call arguments, generation timestamp, and a bounded
    set of the result's important values — everything a future Synthesizer needs to describe
    and cite this evidence without touching the raw tool output directly.

    Args:
        evidence: The evidence record to render.

    Returns:
        A `RenderedEvidence` with a deterministic `summary` text block.
    """
    key_values = _summarize_data(evidence.data)
    generated_at = evidence.provenance.generated_at.isoformat()

    lines = [
        f"[{evidence.id}] tool={evidence.tool}",
        f"  args: {_format_args(evidence.provenance.args)}",
        f"  generated_at: {generated_at}",
    ]
    if key_values:
        lines.append("  values:")
        for key, value in key_values.items():
            lines.append(f"    {key}: {value}")
    else:
        lines.append("  values: (none)")

    return RenderedEvidence(
        evidence_id=evidence.id,
        tool=evidence.tool,
        arguments=evidence.provenance.args,
        generated_at=generated_at,
        key_values=key_values,
        summary="\n".join(lines),
    )


def render_evidence_context(evidence_records: list[Evidence]) -> str:
    """Render several `Evidence` records into one combined context block.

    Records are ordered by evidence ID before rendering, regardless of input order, so the
    same evidence set always produces byte-identical context text.

    Args:
        evidence_records: The evidence records to render.

    Returns:
        The rendered summaries, ordered by evidence ID, joined by blank lines.
    """
    ordered = sorted(evidence_records, key=lambda e: e.id)
    return "\n\n".join(render_evidence(e).summary for e in ordered)


def _format_args(args: dict[str, Any]) -> str:
    """Render call arguments as a stable, sorted `key=value` list."""
    if not args:
        return "(none)"
    return ", ".join(f"{key}={args[key]!r}" for key in sorted(args))


def _summarize_data(data: Any, *, max_items: int = _MAX_SUMMARY_VALUES) -> dict[str, Any]:
    """Flatten the scalar leaves of a tool's result data into a bounded, dotted-path dict.

    Deliberately tool-agnostic: it knows nothing about any specific tool's output shape, only
    that a dict's scalar values are worth surfacing directly and a list is worth surfacing by
    length (recording every row of a 10,000-row `run_sql` result would make this unusable as
    a compact, human-readable summary). Traversal order matches the data's own key order, so
    output is deterministic for a given evidence record.
    """
    values: dict[str, Any] = {}
    _walk_summary(data, "", values, max_items)
    return values


def _walk_summary(node: Any, prefix: str, values: dict[str, Any], max_items: int) -> None:
    if len(values) >= max_items:
        return
    if isinstance(node, dict):
        for key, value in node.items():
            _walk_summary(value, f"{prefix}.{key}" if prefix else str(key), values, max_items)
            if len(values) >= max_items:
                return
    elif isinstance(node, list):
        values[f"{prefix}_count" if prefix else "count"] = len(node)
    elif isinstance(node, bool | int | float | str) or node is None:
        values[prefix] = node
