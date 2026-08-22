"""In-memory evidence store with content-addressed caching and optional JSON persistence."""

from pathlib import Path

from adia.evidence.ids import generate_evidence_id
from adia.evidence.persistence import load_evidence_list, save_evidence_list
from adia.models.evidence import Evidence


class EvidenceStore:
    """Append-only store of `Evidence` records, keyed by deterministic evidence ID.

    Evidence is addressed by content: `generate_evidence_id(tool, args)` means calling the
    same tool with the same arguments always resolves to the same ID, so `add` is idempotent
    for repeated tool calls rather than creating duplicate records.
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        """Create a store, optionally backed by a JSON file.

        Args:
            persist_path: If given and the file exists, records are loaded from it on
                construction, and every subsequent `add` re-persists the full store to it.
        """
        self._records: dict[str, Evidence] = {}
        self.persist_path = Path(persist_path) if persist_path is not None else None
        if self.persist_path is not None and self.persist_path.exists():
            self.load()

    def add(self, evidence: Evidence) -> Evidence:
        """Add an evidence record to the store.

        If a record with the same ID already exists, this is treated as a cache hit: the
        existing record is returned unchanged, unless the new record's argument hash differs
        from the existing one's, which indicates a genuine ID collision rather than a
        legitimate repeat call.

        Args:
            evidence: The evidence record to add.

        Returns:
            The stored record — the newly added one, or the pre-existing cached one.

        Raises:
            ValueError: If `evidence.id` already exists with a different `args_hash`.
        """
        existing = self._records.get(evidence.id)
        if existing is not None:
            if existing.provenance.args_hash != evidence.provenance.args_hash:
                raise ValueError(
                    f"Evidence ID collision: '{evidence.id}' already exists with a "
                    "different args_hash."
                )
            return existing
        self._records[evidence.id] = evidence
        if self.persist_path is not None:
            self.save()
        return evidence

    def get(self, evidence_id: str) -> Evidence | None:
        """Retrieve an evidence record by ID, or `None` if it doesn't exist."""
        return self._records.get(evidence_id)

    def get_or_raise(self, evidence_id: str) -> Evidence:
        """Retrieve an evidence record by ID.

        Raises:
            KeyError: If no record with this ID exists.
        """
        try:
            return self._records[evidence_id]
        except KeyError:
            raise KeyError(f"No evidence found with id '{evidence_id}'.") from None

    def find_cached(self, tool_name: str, args: dict) -> Evidence | None:
        """Look up an existing record for this exact tool and arguments, without recomputing."""
        return self._records.get(generate_evidence_id(tool_name, args))

    def list(
        self, *, tool: str | None = None, plan_step_id: str | None = None
    ) -> list[Evidence]:
        """List stored evidence, optionally filtered by tool name and/or plan step.

        Args:
            tool: If given, only return records produced by this tool.
            plan_step_id: If given, only return records tied to this plan step.

        Returns:
            Matching records, ordered by creation time.
        """
        records = self._records.values()
        if tool is not None:
            records = (e for e in records if e.tool == tool)
        if plan_step_id is not None:
            records = (e for e in records if e.plan_step_id == plan_step_id)
        return sorted(records, key=lambda e: e.created_at)

    def search(self, keyword: str) -> list[Evidence]:
        """Case-insensitive search over tool name, plan step ID, and stringified data.

        Args:
            keyword: Substring to search for.

        Returns:
            Matching records, ordered by creation time.
        """
        needle = keyword.lower()
        matches = [
            e
            for e in self._records.values()
            if needle in e.tool.lower()
            or needle in (e.plan_step_id or "").lower()
            or needle in str(e.data).lower()
        ]
        return sorted(matches, key=lambda e: e.created_at)

    def save(self, path: str | Path | None = None) -> None:
        """Persist all records to a JSON file.

        Args:
            path: Destination path. Defaults to the store's configured `persist_path`.

        Raises:
            ValueError: If no path is given and none was configured at construction.
        """
        target = Path(path) if path is not None else self.persist_path
        if target is None:
            raise ValueError("No persist path configured or provided.")
        save_evidence_list(list(self._records.values()), target)

    def load(self, path: str | Path | None = None) -> None:
        """Load records from a JSON file, merging them into the current store.

        Args:
            path: Source path. Defaults to the store's configured `persist_path`.

        Raises:
            ValueError: If no path is given and none was configured at construction.
        """
        target = Path(path) if path is not None else self.persist_path
        if target is None:
            raise ValueError("No persist path configured or provided.")
        for record in load_evidence_list(target):
            self._records[record.id] = record

    def __len__(self) -> int:
        return len(self._records)

    def __contains__(self, evidence_id: str) -> bool:
        return evidence_id in self._records
