"""The synthesizer agent: turns collected evidence into prose, never into new facts.

`synthesize_answer` asks an LLM to explain a question's collected evidence in plain prose,
citing every numeric claim with the same `[[evidence_id]]` marker
`adia.validate.static.validate_answer` already checks for. It never trusts that prose as-is:
before it is returned, it is run back through `validate_answer` -- the exact same grounding
check the graph's own `validation_node` applies later -- against the same evidence. Anything
that fails (an invented number, a dangling or malformed citation, a number with no citation at
all) is discarded, never handed back. Any failure to reach the LLM at all is treated the same
way. In every one of those cases, this module falls back to a mechanical, citation-bearing
answer built directly from the evidence with no LLM involved -- restating a value it already
has in hand can never itself fail grounding validation, so a caller of `synthesize_answer`
always gets *some* grounded text back, never `None` and never untrusted prose.

The LLM only ever explains evidence that already exists; it performs no calculation, and every
number in its output must already appear in the evidence it was shown.
"""

import re
from collections.abc import Callable, Mapping
from typing import Any

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, ConfigDict, Field

from adia.agents.llm_config import load_llm_settings
from adia.evidence.renderer import render_evidence
from adia.models.evidence import Evidence
from adia.validate.static import validate_answer

_NO_EVIDENCE_ANSWER = "No evidence was collected; unable to produce a grounded answer."

#: Cap on how many key-values `_mechanical_fallback` reports per evidence record -- enough to
#: be informative without turning the fallback into a raw data dump.
_MAX_FALLBACK_VALUES_PER_RECORD = 3

#: Matches a list-index segment in a renderer-flattened dotted-path key, e.g. the `[0]` in
#: `rows[0].total_sales`. Used only to sanitize what `_mechanical_fallback` *displays* --
#: never touches the key used to look up a value, or the value itself.
_INDEX_SEGMENT_PATTERN = re.compile(r"\[\d+\]")

_SYSTEM_PROMPT = (
    "You are the synthesis component of a data analysis system. You are given a user's "
    "question and a rendered summary of evidence already computed by deterministic tools. "
    "Write a concise, direct answer to the question using ONLY the facts in the evidence "
    "below.\n\n"
    "Rules:\n"
    "- Every numeric claim you make must be a value that appears verbatim in the evidence, "
    "immediately followed by a citation marker in the form [[evidence_id]] naming the exact "
    "evidence record that value came from.\n"
    "- Never perform arithmetic, aggregation, unit conversion, or estimation yourself -- if a "
    "number is not already present in the evidence exactly as given, do not state it.\n"
    "- Never introduce facts, numbers, or claims that are not present in the evidence below.\n"
    "- Only cite evidence IDs that appear in the evidence below; never invent one.\n"
    "- Write plain prose explaining what the evidence shows, not a raw data dump.\n\n"
    "When the evidence includes more than one analysis step (e.g. an observation plus "
    "several supporting analyses that were run to investigate it), keep three things "
    "distinct in your answer:\n"
    "- The observed fact itself (e.g. which group has the extreme value), stated plainly.\n"
    "- What the supporting evidence is associated with -- describe a pattern the other "
    "evidence shows using non-causal language ('is associated with', 'coincides with', "
    "'the data suggests'), never asserting that one thing caused another.\n"
    "- What remains unsupported -- if the evidence cannot establish why something happened "
    "(no cited evidence explicitly permits causal language -- see the rules on causal "
    "claims already enforced downstream), say so explicitly rather than implying a cause. "
    "Correlation or comparison evidence describes what co-occurs, not what causes what."
)


class _SynthesizerLLMOutput(BaseModel):
    """Raw, unverified shape the LLM is asked to produce.

    Never returned to a caller directly -- `synthesize_answer` checks `answer` is non-blank
    and then re-validates it in full through `validate_answer` before trusting it.
    """

    model_config = ConfigDict(frozen=True)

    answer: str = Field(
        default="", description="The synthesized answer, with [[evidence_id]] citations."
    )


#: Signature every `llm_call` -- real or a test's fake -- must satisfy.
LLMCall = Callable[[str, str], _SynthesizerLLMOutput]


def synthesize_answer(
    question: str,
    evidence_context: str,
    evidence: Mapping[str, Evidence],
    *,
    llm_call: LLMCall | None = None,
) -> str:
    """Synthesize a grounded, citation-bearing answer from already-collected evidence.

    Args:
        question: The user's question, verbatim.
        evidence_context: Rendered evidence text (e.g. from
            `adia.evidence.renderer.render_evidence_context`) shown to the LLM -- the only
            description of the evidence it sees.
        evidence: The evidence records themselves, keyed by ID -- used both to validate the
            LLM's proposed answer and, if that fails, to build the mechanical fallback. Should
            be the full evidence set for this run, not just what the LLM chose to cite.
        llm_call: Override for the LLM call, e.g. a fake for tests. Defaults to a real OpenAI
            call built from `adia.agents.llm_config.load_llm_settings`.

    Returns:
        A grounded answer string that always passes `validate_answer` against `evidence`: the
        LLM's own proposal if it passes, or a mechanical, evidence-only fallback otherwise.
        Never `None` and never raises.
    """
    if not evidence:
        return _NO_EVIDENCE_ANSWER

    resolved_call = llm_call or _call_openai
    try:
        raw = resolved_call(question, evidence_context)
        candidate = raw.answer.strip()
        if not candidate:
            raise ValueError("LLM produced no answer text.")
        result = validate_answer(candidate, evidence)
        if not result.passed:
            raise ValueError(f"Synthesized answer failed grounding validation: {result.issues}")
        return candidate
    except Exception:  # the LLM is never trusted to be reachable, well-behaved, or grounded
        return _mechanical_fallback(question, evidence)


def _mechanical_fallback(question: str, evidence: Mapping[str, Evidence]) -> str:
    """Deterministically compose a citation-bearing answer with no LLM involved.

    Used whenever the LLM is unreachable or its proposed answer fails grounding validation.
    Restates each evidence record's most meaningful reported values verbatim (see
    `_select_fallback_values`), cited by its evidence ID -- it states nothing it didn't
    already have in hand, so it can never itself fail `validate_answer`. Displayed labels are
    sanitized (`_sanitize_label`) so a list index embedded in a flattened key name (e.g. the
    `0` in `rows[0].total_sales`) can't be misread as an unsupported claimed number; the
    numeric value printed after `=` is always the real, untouched evidence value.
    """
    lines = [f"Draft answer for: {question}", ""]
    for record in sorted(evidence.values(), key=lambda e: e.id):
        rendered = render_evidence(record)
        selected = _select_fallback_values(rendered.key_values)
        if not selected:
            lines.append(f"{rendered.tool} ran but reported no scalar values [[{record.id}]].")
        else:
            pairs = ", ".join(f"{_sanitize_label(key)} = {value}" for key, value in selected)
            lines.append(f"{rendered.tool} reports {pairs} [[{record.id}]].")
    return "\n".join(lines)


def _sanitize_label(key: str) -> str:
    """Strip list-index brackets from a dotted-path key, for display only.

    The renderer's flattened keys (`adia.evidence.renderer._walk_summary`) embed a list's
    numeric index directly in the key name, e.g. `rows[0].total_sales` or, for nested lists,
    `feature_importance[0].importance`. Printing that key verbatim puts a bare digit between
    non-word characters (`[`/`]`), which `validate_answer`'s regex-based number extractor
    reads as a standalone claimed numeric value -- a false positive, since an array index was
    never a claim about the data. Stripping every `[<digits>]` segment leaves a genuinely
    readable label (`rows.total_sales`) without changing which value it names or the value
    itself; the key used to look up `value` in `_select_fallback_values` is never touched.

    Args:
        key: A dotted-path key as produced by the renderer, e.g. `rows[0].total_sales`.

    Returns:
        The key with every `[<digits>]` segment removed and stray leading/trailing/duplicate
        dots cleaned up. Falls back to `"value"` if stripping leaves nothing (a bare top-level
        list index, e.g. `[0]`, has no other text to display).
    """
    stripped = _INDEX_SEGMENT_PATTERN.sub("", key)
    stripped = re.sub(r"\.{2,}", ".", stripped).strip(".")
    return stripped or "value"


def _select_fallback_values(key_values: dict[str, Any]) -> list[tuple[str, Any]]:
    """Rank a rendered evidence record's key-values and return the most meaningful few.

    "First key in the dict" is a poor proxy for "most meaningful value": a tool's result dict
    may put a bookkeeping field (e.g. the renderer's own synthesized `rows_count`) or a
    configuration/identifier field (e.g. `model_type`) ahead of the actual analytical result
    (an aggregated figure, a correlation coefficient, a model's metric value) simply because
    of insertion order. Two independent, tool-agnostic signals correct for that, combined
    additively so neither strictly overrides the other:

    - `count_penalty`: 1 if the key name contains "count" -- catches both the renderer's own
      synthesized list-length keys and count-flavored tool fields, deprioritized (not
      excluded) since a count is still informative when nothing else is available.
    - `type_penalty`: 1 if the value isn't numeric -- a question is almost always asking for a
      number (a total, a rate, a score), while configuration/identifier fields (model type,
      dataset ID, column names) are almost always strings or lists.

    Keys are otherwise kept in their original (insertion) order, so this is a stable
    re-ranking, not a re-sort by value -- and, for a fixed evidence record, always produces
    the same output.

    Args:
        key_values: A `RenderedEvidence.key_values` mapping, already bounded and flattened by
            `adia.evidence.renderer`.

    Returns:
        Up to `_MAX_FALLBACK_VALUES_PER_RECORD` `(key, value)` pairs, best first. Empty if
        `key_values` is empty.
    """

    def _penalty(indexed_item: tuple[int, tuple[str, Any]]) -> tuple[int, int]:
        index, (key, value) = indexed_item
        count_penalty = 1 if "count" in key.lower() else 0
        is_numeric = isinstance(value, int | float) and not isinstance(value, bool)
        type_penalty = 0 if is_numeric else 1
        return (count_penalty + type_penalty, index)

    ranked = sorted(enumerate(key_values.items()), key=_penalty)
    return [pair for _, pair in ranked[:_MAX_FALLBACK_VALUES_PER_RECORD]]


def _call_openai(question: str, evidence_context: str) -> _SynthesizerLLMOutput:
    """The real LLM call: build a `ChatOpenAI` client from the environment and invoke it.

    Raises whatever `load_llm_settings` or the API call itself raises -- `synthesize_answer`
    is responsible for catching that, not this function.
    """
    settings = load_llm_settings()
    model = ChatOpenAI(model=settings.model, api_key=settings.api_key, temperature=0)
    structured_model = model.with_structured_output(_SynthesizerLLMOutput)
    messages = _build_messages(question, evidence_context)
    return structured_model.invoke(messages)


def _build_messages(question: str, evidence_context: str) -> list[BaseMessage]:
    """Build the prompt: the fixed system rules plus the question and rendered evidence."""
    human_content = f"Question: {question}\n\nEvidence:\n{evidence_context}"
    return [SystemMessage(content=_SYSTEM_PROMPT), HumanMessage(content=human_content)]
