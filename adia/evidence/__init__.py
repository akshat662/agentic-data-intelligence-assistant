"""Evidence layer: content-addressed storage and provenance-preserving persistence.

Independent of `adia.graph` and `adia.agents` — this package only depends on `adia.models`,
so it can be built and tested before any orchestration code exists. Every record here is a
`adia.models.evidence.Evidence`, and every record's `provenance` field already carries tool
name, input arguments, and execution timestamp; see `adia/evidence/store.py` for how those are
put to use for caching, lookup, and search.
"""

from adia.evidence.ids import compute_args_hash, generate_evidence_id
from adia.evidence.persistence import load_evidence_list, save_evidence_list
from adia.evidence.store import EvidenceStore

__all__ = [
    "EvidenceStore",
    "compute_args_hash",
    "generate_evidence_id",
    "load_evidence_list",
    "save_evidence_list",
]
