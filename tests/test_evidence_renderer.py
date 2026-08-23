"""Tests for adia.evidence.renderer."""

from adia.evidence.ids import compute_args_hash, generate_evidence_id
from adia.evidence.renderer import render_evidence, render_evidence_context
from adia.models.evidence import Evidence
from adia.models.provenance import Provenance


def _make_evidence(tool: str, args: dict, data: object) -> Evidence:
    return Evidence(
        id=generate_evidence_id(tool, args),
        tool=tool,
        data=data,
        provenance=Provenance(tool_name=tool, args=args, args_hash=compute_args_hash(args)),
    )


class TestRenderEvidence:
    def test_preserves_id_tool_and_arguments(self):
        ev = _make_evidence(
            "run_sql", {"query": "select 1", "dataset_id": "orders"}, {"row_count": 3}
        )
        rendered = render_evidence(ev)
        assert rendered.evidence_id == ev.id
        assert rendered.tool == "run_sql"
        assert rendered.arguments == {"query": "select 1", "dataset_id": "orders"}

    def test_preserves_generated_timestamp(self):
        ev = _make_evidence("run_sql", {"query": "select 1"}, {"row_count": 1})
        rendered = render_evidence(ev)
        assert rendered.generated_at == ev.provenance.generated_at.isoformat()

    def test_scalar_values_flattened(self):
        ev = _make_evidence(
            "profile_dataset", {"dataset_id": "orders"}, {"row_count": 100, "column_count": 3}
        )
        rendered = render_evidence(ev)
        assert rendered.key_values["row_count"] == 100
        assert rendered.key_values["column_count"] == 3

    def test_lists_summarized_by_count_not_expanded(self):
        ev = _make_evidence(
            "run_sql",
            {"query": "select * from orders"},
            {"rows": [{"price": i} for i in range(50)], "row_count": 50},
        )
        rendered = render_evidence(ev)
        assert rendered.key_values["rows_count"] == 50
        assert "rows" not in rendered.key_values

    def test_nested_scalars_use_dotted_paths(self):
        ev = _make_evidence(
            "train_model", {"dataset_id": "orders"}, {"metadata": {"n_train": 80, "n_test": 20}}
        )
        rendered = render_evidence(ev)
        assert rendered.key_values["metadata.n_train"] == 80
        assert rendered.key_values["metadata.n_test"] == 20

    def test_summary_is_deterministic(self):
        ev = _make_evidence("run_sql", {"query": "select 1"}, {"row_count": 1})
        assert render_evidence(ev).summary == render_evidence(ev).summary

    def test_summary_contains_id_and_tool(self):
        ev = _make_evidence("run_sql", {"query": "select 1"}, {"row_count": 1})
        rendered = render_evidence(ev)
        assert ev.id in rendered.summary
        assert "run_sql" in rendered.summary

    def test_empty_data_reports_no_values(self):
        ev = _make_evidence("run_sql", {"query": "select 1"}, {})
        rendered = render_evidence(ev)
        assert rendered.key_values == {}
        assert "(none)" in rendered.summary

    def test_no_arguments_renders_none(self):
        ev = _make_evidence("run_sql", {}, {"row_count": 1})
        rendered = render_evidence(ev)
        assert "(none)" in rendered.summary

    def test_key_values_truncated_at_max_items(self):
        data = {f"metric_{i}": float(i) for i in range(30)}
        ev = _make_evidence("profile_dataset", {"dataset_id": "orders"}, data)
        rendered = render_evidence(ev)
        assert len(rendered.key_values) == 20


class TestRenderEvidenceContext:
    def test_orders_by_evidence_id_regardless_of_input_order(self):
        ev_a = _make_evidence("run_sql", {"query": "a"}, {"x": 1})
        ev_b = _make_evidence("run_sql", {"query": "b"}, {"x": 2})
        ordered_ids = sorted([ev_a.id, ev_b.id])

        context_1 = render_evidence_context([ev_a, ev_b])
        context_2 = render_evidence_context([ev_b, ev_a])
        assert context_1 == context_2
        assert context_1.index(ordered_ids[0]) < context_1.index(ordered_ids[1])

    def test_empty_list_returns_empty_string(self):
        assert render_evidence_context([]) == ""


class TestSmallListExpansion:
    """Lists with ≤10 items expose scalar values inline for the synthesizer."""

    def test_small_list_exposes_scalar_values(self):
        rows = [
            {"Category": "Technology", "total_sales": 836154},
            {"Category": "Furniture", "total_sales": 741999},
        ]
        ev = _make_evidence("run_sql", {"query": "select * from orders"}, {"rows": rows})
        rendered = render_evidence(ev)
        assert rendered.key_values["rows_count"] == 2
        assert rendered.key_values["rows[0].Category"] == "Technology"
        assert rendered.key_values["rows[0].total_sales"] == 836154
        assert rendered.key_values["rows[1].Category"] == "Furniture"
        assert rendered.key_values["rows[1].total_sales"] == 741999

    def test_large_list_stays_count_only(self):
        rows = [{"price": i} for i in range(50)]
        ev = _make_evidence("run_sql", {"query": "select * from orders"}, {"rows": rows})
        rendered = render_evidence(ev)
        assert rendered.key_values["rows_count"] == 50
        # No expanded items for large lists.
        assert "rows[0].price" not in rendered.key_values

    def test_boundary_exactly_ten_items_expands(self):
        rows = [{"val": i} for i in range(10)]
        ev = _make_evidence("run_sql", {"query": "select * from orders"}, {"rows": rows})
        rendered = render_evidence(ev)
        assert rendered.key_values["rows_count"] == 10
        assert rendered.key_values["rows[0].val"] == 0
        assert rendered.key_values["rows[9].val"] == 9

    def test_boundary_eleven_items_does_not_expand(self):
        rows = [{"val": i} for i in range(11)]
        ev = _make_evidence("run_sql", {"query": "select * from orders"}, {"rows": rows})
        rendered = render_evidence(ev)
        assert rendered.key_values["rows_count"] == 11
        assert "rows[0].val" not in rendered.key_values

    def test_expanded_values_appear_in_summary_text(self):
        rows = [{"Category": "Technology", "total_sales": 836154}]
        ev = _make_evidence("run_sql", {"query": "q"}, {"rows": rows})
        rendered = render_evidence(ev)
        assert "836154" in rendered.summary
        assert "Technology" in rendered.summary

    def test_expansion_is_deterministic(self):
        rows = [
            {"a": 1, "b": 2},
            {"a": 3, "b": 4},
        ]
        ev = _make_evidence("run_sql", {"query": "q"}, {"rows": rows})
        r1 = render_evidence(ev)
        r2 = render_evidence(ev)
        assert r1.summary == r2.summary
        assert r1.key_values == r2.key_values

