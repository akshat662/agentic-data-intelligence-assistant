"""Deterministic evidence ID and argument-hash generation.

Content-addressing evidence by (tool name, arguments) is what lets the store deduplicate
repeated tool calls and lets tests assert that calling a tool twice with identical arguments
produces identical evidence — a strong signal that the tool has no unaccounted randomness.
"""

import hashlib
import json
from typing import Any

_HASH_LENGTH = 8


def compute_args_hash(args: dict[str, Any]) -> str:
    """Compute a stable hash of a tool's arguments.

    Args are canonicalized via sorted-key JSON serialization before hashing, so key order
    never affects the result. Non-JSON-serializable values (e.g. numpy scalars, sets) fall
    back to their `str()` representation.

    Args:
        args: The exact keyword arguments a tool was called with.

    Returns:
        A hex-encoded SHA-256 digest of the canonicalized arguments.
    """
    canonical = json.dumps(args, sort_keys=True, default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def generate_evidence_id(tool_name: str, args: dict[str, Any]) -> str:
    """Generate a deterministic evidence ID from a tool name and its arguments.

    The same tool called with the same arguments always yields the same ID, which is what
    lets `EvidenceStore` treat a repeated call as a cache hit rather than a new record.

    Args:
        tool_name: Registered name of the tool.
        args: The exact keyword arguments the tool was called with.

    Returns:
        An ID of the form `ev_<tool_name>_<hash prefix>`.
    """
    args_hash = compute_args_hash(args)
    return f"ev_{tool_name}_{args_hash[:_HASH_LENGTH]}"
