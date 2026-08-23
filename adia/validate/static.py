"""Static grounding validation: code, not a model, deciding whether an answer is trustworthy.

This is the mechanical half of the grounding story described in the project's own design
goal: a validator that "fails the run on any numeral it can't trace." It has no LLM
dependency and no graph dependency — it is a set of pure functions over plain text and
`Evidence` records, testable and usable long before a Synthesizer exists to produce that
text. A future Critic agent's job is the *other* half — semantic overreach a regex can't
catch (confounders, over-generalization) — and is deliberately out of scope here.

Answers cite evidence with an inline `[[evidence_id]]` marker, e.g.
`"Revenue grew 12% [[ev_run_sql_1a2b3c4d]]."`. This is the one convention a future
Synthesizer must follow for its output to be checkable by this module.
"""

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

from adia.models.evidence import Evidence
from adia.models.state import ValidationIssue, ValidationResult, ValidationSeverity

#: Inline citation marker syntax a future Synthesizer's output is expected to use.
CITATION_PATTERN = re.compile(r"\[\[([^\[\]]*)\]\]")

#: Shape produced by `adia.evidence.ids.generate_evidence_id`: `ev_<tool_name>_<8 hex chars>`.
EVIDENCE_ID_PATTERN = re.compile(r"^ev_.+_[0-9a-f]{8}$")

_NUMBER_PATTERN = re.compile(r"(?<!\w)-?\d[\d,]*\.?\d*(?!\w)")
_CAUSAL_PATTERN = re.compile(
    r"\b(causes?|caused|causing|leads? to|led to|results? in|resulted in|"
    r"due to|because of|drives?|driven by)\b",
    re.IGNORECASE,
)

_ABS_TOLERANCE = 0.01
_REL_TOLERANCE = 1e-3
_MAX_WALK_NODES = 5000


def validate_answer(
    text: str, evidence: Sequence[Evidence] | Mapping[str, Evidence]
) -> ValidationResult:
    """Run every static grounding check against a candidate answer.

    Args:
        text: The candidate answer text, using `[[evidence_id]]` inline citations.
        evidence: Every evidence record the answer is allowed to cite — typically the full
            contents of the run's evidence store, not just the records actually cited (this
            is what lets a dangling or malformed citation be told apart from a real one).

    Returns:
        A `ValidationResult`: `passed` is `False` if any check produced a `FAIL`-severity
        issue. Every issue currently raised is `FAIL` — this layer gates, it doesn't advise.
    """
    evidence_by_id = _as_evidence_map(evidence)
    issues: list[ValidationIssue] = []

    citation_ids, citation_issues = _extract_citations(text, evidence_by_id)
    issues.extend(citation_issues)

    numbers = _extract_claimed_numbers(text)
    if numbers and not citation_ids:
        issues.append(
            ValidationIssue(
                code="missing_evidence_reference",
                message="The answer states numeric values but cites no evidence at all.",
                severity=ValidationSeverity.FAIL,
            )
        )
    elif numbers:
        cited_values = _collect_numeric_values(evidence_by_id, citation_ids)
        issues.extend(_check_numbers_supported(numbers, cited_values))

    issues.extend(_check_causal_claims(text, evidence_by_id, citation_ids))

    passed = not any(issue.severity == ValidationSeverity.FAIL for issue in issues)
    return ValidationResult(passed=passed, issues=issues)


def _as_evidence_map(evidence: Sequence[Evidence] | Mapping[str, Evidence]) -> dict[str, Evidence]:
    """Normalize either a list or an ID-keyed mapping of evidence into an ID-keyed dict."""
    if isinstance(evidence, Mapping):
        return dict(evidence)
    return {record.id: record for record in evidence}


def _extract_citations(
    text: str, evidence_by_id: dict[str, Evidence]
) -> tuple[set[str], list[ValidationIssue]]:
    """Find every `[[...]]` citation marker and classify each as valid or malformed.

    A citation is malformed either because its inner text isn't a well-formed evidence ID
    (`ev_<tool>_<8 hex chars>`), or because it *is* well-formed but doesn't match any record
    in `evidence_by_id` — a dangling reference to evidence that was never actually produced.
    """
    valid_ids: set[str] = set()
    issues: list[ValidationIssue] = []
    for match in CITATION_PATTERN.finditer(text):
        raw = match.group(1).strip()
        if not EVIDENCE_ID_PATTERN.match(raw):
            issues.append(
                ValidationIssue(
                    code="malformed_evidence_reference",
                    message=f"Citation '[[{raw}]]' is not a valid evidence ID.",
                    severity=ValidationSeverity.FAIL,
                )
            )
            continue
        if raw not in evidence_by_id:
            issues.append(
                ValidationIssue(
                    code="malformed_evidence_reference",
                    message=f"Citation '[[{raw}]]' does not match any available evidence.",
                    severity=ValidationSeverity.FAIL,
                )
            )
            continue
        valid_ids.add(raw)
    return valid_ids, issues


def _extract_claimed_numbers(text: str) -> list[float]:
    """Extract numeric literals from answer text, ignoring digits inside citation markers.

    A pragmatic heuristic, not full NLP: it finds number-shaped tokens by regex. It is not
    expected to parse every conceivable numeric phrasing (spelled-out numbers, fractions),
    only the plain decimal figures a Synthesizer's rendered output is expected to contain.
    `_NUMBER_PATTERN` requires at least one digit, so every match is guaranteed parseable by
    `float()` once commas are stripped.
    """
    masked = CITATION_PATTERN.sub("[EVIDENCE]", text)
    return [
        float(match.group(0).replace(",", "")) for match in _NUMBER_PATTERN.finditer(masked)
    ]


def _collect_numeric_values(
    evidence_by_id: dict[str, Evidence], citation_ids: set[str]
) -> set[float]:
    """Collect every numeric leaf value found anywhere in the cited evidence records."""
    values: set[float] = set()
    budget = [_MAX_WALK_NODES]
    for evidence_id in citation_ids:
        _walk_numeric(evidence_by_id[evidence_id].data, values, budget)
    return values


def _walk_numeric(node: Any, values: set[float], budget: list[int]) -> None:
    """Recursively collect numeric leaves from a tool's result data, bounded by `budget`."""
    if budget[0] <= 0:
        return
    if isinstance(node, dict):
        for child in node.values():
            _walk_numeric(child, values, budget)
    elif isinstance(node, list):
        for item in node:
            _walk_numeric(item, values, budget)
    elif isinstance(node, bool):
        return
    elif isinstance(node, int | float):
        values.add(float(node))
        budget[0] -= 1


def _check_numbers_supported(
    numbers: list[float], cited_values: set[float]
) -> list[ValidationIssue]:
    """Flag any claimed number that doesn't approximately match a cited evidence value."""
    issues: list[ValidationIssue] = []
    checked: set[float] = set()
    for number in numbers:
        if number in checked:
            continue
        checked.add(number)
        if not any(
            math.isclose(number, value, rel_tol=_REL_TOLERANCE, abs_tol=_ABS_TOLERANCE)
            for value in cited_values
        ):
            issues.append(
                ValidationIssue(
                    code="unsupported_numerical_claim",
                    message=f"The value {number!r} does not match any cited evidence value.",
                    severity=ValidationSeverity.FAIL,
                )
            )
    return issues


def _check_causal_claims(
    text: str, evidence_by_id: dict[str, Evidence], citation_ids: set[str]
) -> list[ValidationIssue]:
    """Flag causal language backed by evidence that explicitly disallows causal claims.

    Only fires when a cited tool's result opts itself out via `data["causal_claim_allowed"]
    is False` (e.g. `compute_correlation`) — a tool that simply says nothing about causality
    is not treated as forbidding it, since that would be guessing at intent the tool never
    expressed.
    """
    if not _CAUSAL_PATTERN.search(text):
        return []
    forbidding = sorted(
        evidence_id
        for evidence_id in citation_ids
        if isinstance(evidence_by_id[evidence_id].data, dict)
        and evidence_by_id[evidence_id].data.get("causal_claim_allowed") is False
    )
    if not forbidding:
        return []
    return [
        ValidationIssue(
            code="unsupported_causal_claim",
            message=(
                "The answer uses causal language, but cited evidence "
                f"{forbidding} explicitly does not support causal claims."
            ),
            severity=ValidationSeverity.FAIL,
        )
    ]
