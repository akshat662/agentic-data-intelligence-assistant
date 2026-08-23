"""Tests for adia.validate.static."""

from adia.evidence.ids import compute_args_hash, generate_evidence_id
from adia.models.evidence import Evidence
from adia.models.provenance import Provenance
from adia.validate.static import validate_answer


def _make_evidence(tool: str, args: dict, data: object) -> Evidence:
    return Evidence(
        id=generate_evidence_id(tool, args),
        tool=tool,
        data=data,
        provenance=Provenance(tool_name=tool, args=args, args_hash=compute_args_hash(args)),
    )


class TestValidGroundedResponse:
    def test_matching_number_with_valid_citation_passes(self):
        ev = _make_evidence(
            "run_sql",
            {"query": "select avg(price) from orders"},
            {"rows": [{"avg_price": 42.17}], "row_count": 1},
        )
        text = f"The average price is 42.17 [[{ev.id}]]."
        result = validate_answer(text, [ev])
        assert result.passed is True
        assert result.issues == []

    def test_comma_formatted_number_matches(self):
        ev = _make_evidence("run_sql", {"query": "select sum(price)"}, {"total": 1234.5})
        text = f"Total revenue was 1,234.50 [[{ev.id}]]."
        result = validate_answer(text, [ev])
        assert result.passed is True

    def test_citation_without_numbers_passes(self):
        ev = _make_evidence("run_sql", {"query": "select 1"}, {"row_count": 1})
        text = f"See [[{ev.id}]] for details."
        result = validate_answer(text, [ev])
        assert result.passed is True

    def test_empty_text_passes(self):
        result = validate_answer("", [])
        assert result.passed is True

    def test_evidence_as_mapping_works(self):
        ev = _make_evidence("run_sql", {"query": "select 1"}, {"total": 42.17})
        text = f"The total is 42.17 [[{ev.id}]]."
        result = validate_answer(text, {ev.id: ev})
        assert result.passed is True


class TestMissingEvidenceRejection:
    def test_number_with_no_citation_fails(self):
        text = "The average price is 42.17."
        result = validate_answer(text, [])
        assert result.passed is False
        assert any(i.code == "missing_evidence_reference" for i in result.issues)

    def test_number_with_no_citation_fails_even_with_unrelated_evidence_available(self):
        ev = _make_evidence("run_sql", {"query": "select 1"}, {"total": 42.17})
        text = "The average price is 42.17."
        result = validate_answer(text, [ev])
        assert result.passed is False
        assert any(i.code == "missing_evidence_reference" for i in result.issues)


class TestUnsupportedClaimRejection:
    def test_number_not_matching_cited_evidence_fails(self):
        ev = _make_evidence(
            "run_sql",
            {"query": "select avg(price) from orders"},
            {"rows": [{"avg_price": 42.17}], "row_count": 1},
        )
        text = f"The average price is 999.99 [[{ev.id}]]."
        result = validate_answer(text, [ev])
        assert result.passed is False
        assert any(i.code == "unsupported_numerical_claim" for i in result.issues)

    def test_malformed_citation_format_fails(self):
        text = "Revenue was 42.17 [[not-a-real-id]]."
        result = validate_answer(text, [])
        assert result.passed is False
        assert any(i.code == "malformed_evidence_reference" for i in result.issues)

    def test_dangling_citation_to_nonexistent_evidence_fails(self):
        ev = _make_evidence("run_sql", {"query": "select 1"}, {"total": 42.17})
        dangling_id = generate_evidence_id("run_sql", {"query": "select 2"})
        text = f"Revenue was 42.17 [[{dangling_id}]]."
        result = validate_answer(text, [ev])
        assert result.passed is False
        assert any(i.code == "malformed_evidence_reference" for i in result.issues)


class TestCausalClaimRejection:
    def test_causal_language_from_correlation_evidence_fails(self):
        corr_ev = _make_evidence(
            "compute_correlation",
            {"dataset_id": "orders", "columns": ["price", "discount"]},
            {
                "columns": ["price", "discount"],
                "matrix": [[1.0, -1.0], [-1.0, 1.0]],
                "pairs": [
                    {"column_a": "price", "column_b": "discount", "correlation": -1.0, "n": 5}
                ],
                "causal_claim_allowed": False,
            },
        )
        text = f"Higher discounts cause lower prices [[{corr_ev.id}]]."
        result = validate_answer(text, [corr_ev])
        assert result.passed is False
        assert any(i.code == "unsupported_causal_claim" for i in result.issues)

    def test_causal_language_without_forbidding_evidence_passes(self):
        cg_ev = _make_evidence(
            "compare_groups",
            {"group_column": "region", "metric_column": "price"},
            {"groups": [{"group": "north", "mean": 50.0}], "pairwise_differences": []},
        )
        text = f"The north region drives higher prices [[{cg_ev.id}]]."
        result = validate_answer(text, [cg_ev])
        assert not any(i.code == "unsupported_causal_claim" for i in result.issues)

    def test_non_causal_language_with_forbidding_evidence_passes(self):
        corr_ev = _make_evidence(
            "compute_correlation",
            {"dataset_id": "orders", "columns": ["price", "discount"]},
            {
                "columns": ["price", "discount"],
                "matrix": [[1.0, -1.0], [-1.0, 1.0]],
                "pairs": [],
                "causal_claim_allowed": False,
            },
        )
        text = f"Price and discount are strongly correlated [[{corr_ev.id}]]."
        result = validate_answer(text, [corr_ev])
        assert not any(i.code == "unsupported_causal_claim" for i in result.issues)


class TestNumericEdgeCases:
    def test_boolean_values_do_not_support_numeric_claims(self):
        # `True == 1` in Python; a stray boolean field must not be treated as evidence for
        # a claimed "1".
        ev = _make_evidence("run_sql", {"query": "select is_active"}, {"is_active": True})
        text = f"The count is 1 [[{ev.id}]]."
        result = validate_answer(text, [ev])
        assert result.passed is False
        assert any(i.code == "unsupported_numerical_claim" for i in result.issues)

    def test_repeated_unsupported_number_reported_once(self):
        ev = _make_evidence("run_sql", {"query": "select 1"}, {"total": 1.0})
        text = f"Revenue was 999.99, yes 999.99 [[{ev.id}]]."
        result = validate_answer(text, [ev])
        unsupported = [i for i in result.issues if i.code == "unsupported_numerical_claim"]
        assert len(unsupported) == 1


class TestDeterminism:
    def test_validate_answer_is_deterministic(self):
        ev = _make_evidence("run_sql", {"query": "select 1"}, {"total": 42.17})
        text = f"The total is 42.17 [[{ev.id}]]."
        result1 = validate_answer(text, [ev])
        result2 = validate_answer(text, [ev])
        assert result1 == result2
