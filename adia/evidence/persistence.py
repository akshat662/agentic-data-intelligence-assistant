"""JSON persistence for evidence records."""

from pathlib import Path

from pydantic import TypeAdapter

from adia.models.evidence import Evidence

_evidence_list_adapter: TypeAdapter[list[Evidence]] = TypeAdapter(list[Evidence])


def save_evidence_list(records: list[Evidence], path: str | Path) -> None:
    """Serialize a list of evidence records to a JSON file.

    Args:
        records: Evidence records to persist, in any order.
        path: Destination file path; overwritten if it already exists.
    """
    Path(path).write_bytes(_evidence_list_adapter.dump_json(records, indent=2))


def load_evidence_list(path: str | Path) -> list[Evidence]:
    """Load evidence records previously written by `save_evidence_list`.

    Args:
        path: Path to a JSON file produced by `save_evidence_list`.

    Returns:
        The list of evidence records, in file order.

    Raises:
        FileNotFoundError: If `path` does not exist.
        pydantic.ValidationError: If the file's contents don't match the `Evidence` schema.
    """
    resolved = Path(path)
    if not resolved.exists():
        raise FileNotFoundError(f"Evidence file not found: {resolved}")
    return _evidence_list_adapter.validate_json(resolved.read_bytes())
