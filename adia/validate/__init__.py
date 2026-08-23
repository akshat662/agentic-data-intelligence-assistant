"""Validation layer: code-level grounding checks over answer text and evidence.

No dependency on `adia.graph` or `adia.agents` — these are pure functions over `Evidence`
records and plain text, usable and testable before any Synthesizer or orchestration exists.
"""

from adia.validate.static import (
    CITATION_PATTERN,
    EVIDENCE_ID_PATTERN,
    validate_answer,
)

__all__ = [
    "CITATION_PATTERN",
    "EVIDENCE_ID_PATTERN",
    "validate_answer",
]
