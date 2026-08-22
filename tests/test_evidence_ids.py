"""Tests for adia.evidence.ids (deterministic hashing and evidence ID generation)."""

from adia.evidence.ids import compute_args_hash, generate_evidence_id


class TestComputeArgsHash:
    def test_deterministic_for_same_args(self):
        args = {"table": "orders", "limit": 10}
        assert compute_args_hash(args) == compute_args_hash(args)

    def test_key_order_does_not_affect_hash(self):
        assert compute_args_hash({"a": 1, "b": 2}) == compute_args_hash({"b": 2, "a": 1})

    def test_different_args_produce_different_hash(self):
        assert compute_args_hash({"a": 1}) != compute_args_hash({"a": 2})

    def test_non_json_native_values_do_not_raise(self):
        compute_args_hash({"seed": 42, "columns": {"x", "y"}})


class TestGenerateEvidenceId:
    def test_deterministic_for_same_tool_and_args(self):
        args = {"query": "select 1"}
        assert generate_evidence_id("run_sql", args) == generate_evidence_id("run_sql", args)

    def test_different_tool_produces_different_id(self):
        args = {"x": 1}
        assert generate_evidence_id("tool_a", args) != generate_evidence_id("tool_b", args)

    def test_different_args_produce_different_id(self):
        id_a = generate_evidence_id("run_sql", {"q": 1})
        id_b = generate_evidence_id("run_sql", {"q": 2})
        assert id_a != id_b

    def test_id_includes_tool_name(self):
        assert generate_evidence_id("run_sql", {"q": 1}).startswith("ev_run_sql_")
