"""Tests for adia.evidence.store.EvidenceStore."""

import json

import pytest
from pydantic import ValidationError

from adia.evidence.ids import compute_args_hash, generate_evidence_id
from adia.evidence.store import EvidenceStore
from adia.models.evidence import Evidence
from adia.models.provenance import Provenance


def _evidence(
    tool: str = "run_sql",
    args: dict | None = None,
    data: object | None = None,
    plan_step_id: str | None = None,
) -> Evidence:
    args = args if args is not None else {"query": "select 1"}
    return Evidence(
        id=generate_evidence_id(tool, args),
        tool=tool,
        data=data if data is not None else {"rows": 3},
        provenance=Provenance(tool_name=tool, args=args, args_hash=compute_args_hash(args)),
        plan_step_id=plan_step_id,
    )


class TestAddAndGet:
    def test_add_then_get_returns_same_record(self):
        store = EvidenceStore()
        ev = _evidence()
        store.add(ev)
        assert store.get(ev.id) == ev

    def test_get_missing_returns_none(self):
        store = EvidenceStore()
        assert store.get("does_not_exist") is None

    def test_get_or_raise_missing_raises_keyerror(self):
        store = EvidenceStore()
        with pytest.raises(KeyError):
            store.get_or_raise("does_not_exist")

    def test_repeated_identical_call_is_cache_hit(self):
        store = EvidenceStore()
        stored1 = store.add(_evidence())
        stored2 = store.add(_evidence())
        assert stored1 is stored2
        assert len(store) == 1

    def test_id_collision_with_different_args_hash_raises(self):
        store = EvidenceStore()
        ev = _evidence()
        store.add(ev)
        conflicting = ev.model_copy(
            update={"provenance": ev.provenance.model_copy(update={"args_hash": "different"})}
        )
        with pytest.raises(ValueError, match="collision"):
            store.add(conflicting)

    def test_contains(self):
        store = EvidenceStore()
        ev = _evidence()
        store.add(ev)
        assert ev.id in store
        assert "missing_id" not in store


class TestFindCached:
    def test_find_cached_hits_for_same_tool_and_args(self):
        store = EvidenceStore()
        args = {"query": "select 1"}
        ev = _evidence(args=args)
        store.add(ev)
        assert store.find_cached("run_sql", args) == ev

    def test_find_cached_misses_for_different_args(self):
        store = EvidenceStore()
        store.add(_evidence(args={"query": "select 1"}))
        assert store.find_cached("run_sql", {"query": "select 2"}) is None


class TestListAndSearch:
    def test_list_filters_by_tool(self):
        store = EvidenceStore()
        store.add(_evidence(tool="run_sql", args={"q": 1}))
        store.add(_evidence(tool="profile_dataset", args={"q": 2}))
        assert [e.tool for e in store.list(tool="profile_dataset")] == ["profile_dataset"]

    def test_list_filters_by_plan_step(self):
        store = EvidenceStore()
        store.add(_evidence(args={"q": 1}, plan_step_id="step_1"))
        store.add(_evidence(args={"q": 2}, plan_step_id="step_2"))
        assert [e.plan_step_id for e in store.list(plan_step_id="step_2")] == ["step_2"]

    def test_list_all_returns_everything(self):
        store = EvidenceStore()
        store.add(_evidence(args={"q": 1}))
        store.add(_evidence(args={"q": 2}))
        assert len(store.list()) == 2

    def test_search_matches_tool_name(self):
        store = EvidenceStore()
        store.add(_evidence(tool="segment_contribution", args={"q": 1}))
        assert len(store.search("segment")) == 1

    def test_search_matches_data_contents(self):
        store = EvidenceStore()
        store.add(_evidence(args={"q": 1}, data={"mean": 42.17}))
        assert len(store.search("42.17")) == 1

    def test_search_no_match_returns_empty(self):
        store = EvidenceStore()
        store.add(_evidence(args={"q": 1}))
        assert store.search("nonexistent_keyword_xyz") == []


class TestPersistence:
    def test_save_and_load_round_trip(self, tmp_path):
        path = tmp_path / "evidence.json"
        store = EvidenceStore()
        store.add(_evidence(args={"q": 1}))
        store.add(_evidence(args={"q": 2}, tool="profile_dataset"))
        store.save(path)

        reloaded = EvidenceStore()
        reloaded.load(path)
        assert len(reloaded) == 2
        assert {e.id for e in reloaded.list()} == {e.id for e in store.list()}

    def test_persist_path_autoloads_on_construction(self, tmp_path):
        path = tmp_path / "evidence.json"
        store = EvidenceStore(persist_path=path)
        store.add(_evidence(args={"q": 1}))

        reopened = EvidenceStore(persist_path=path)
        assert len(reopened) == 1

    def test_save_without_path_raises(self):
        store = EvidenceStore()
        with pytest.raises(ValueError):
            store.save()

    def test_load_without_path_raises(self):
        store = EvidenceStore()
        with pytest.raises(ValueError):
            store.load()


class TestInvalidEvidenceHandling:
    def test_load_missing_file_raises(self, tmp_path):
        store = EvidenceStore()
        with pytest.raises(FileNotFoundError):
            store.load(tmp_path / "does_not_exist.json")

    def test_load_malformed_json_raises(self, tmp_path):
        path = tmp_path / "corrupt.json"
        path.write_text("{ this is not valid json")
        store = EvidenceStore()
        with pytest.raises(ValidationError):
            store.load(path)

    def test_load_json_not_matching_schema_raises(self, tmp_path):
        path = tmp_path / "wrong_shape.json"
        path.write_text(json.dumps([{"id": "ev_1", "tool": "run_sql"}]))  # missing data/provenance
        store = EvidenceStore()
        with pytest.raises(ValidationError):
            store.load(path)

    def test_constructing_evidence_without_data_rejected(self):
        with pytest.raises(ValidationError):
            Evidence(
                id="ev_run_sql_deadbeef",
                tool="run_sql",
                provenance=Provenance(
                    tool_name="run_sql", args={}, args_hash=compute_args_hash({})
                ),
            )
