"""Tests for adia.agents.argument_generator. No real OpenAI call is ever made here."""

import pytest

from adia.agents.argument_generator import (
    _build_messages,
    _RunSqlLLMOutput,
    generate_tool_arguments,
)
from adia.models.catalog import ColumnProfile, DatasetCatalog, SemanticType
from adia.models.plan import PlanStep


@pytest.fixture
def catalog() -> DatasetCatalog:
    columns = [
        ColumnProfile(
            name="price",
            dtype="float64",
            semantic_type=SemanticType.NUMERIC,
            non_null_count=100,
            null_count=0,
            null_rate=0.0,
            unique_count=87,
            min_value=1.5,
            max_value=999.0,
        ),
        ColumnProfile(
            name="region",
            dtype="str",
            semantic_type=SemanticType.CATEGORICAL,
            non_null_count=100,
            null_count=0,
            null_rate=0.0,
            unique_count=4,
        ),
    ]
    return DatasetCatalog(
        dataset_id="orders", source_path="data/orders.parquet", row_count=100, columns=columns
    )


@pytest.fixture
def run_sql_step() -> PlanStep:
    return PlanStep(
        id="step_1",
        intent="Total price by region.",
        tool_family="run_sql",
        expected_output="rows",
        success_criteria="ok",
    )


def _fake_llm_call(output: _RunSqlLLMOutput):
    """Build an `llm_call` that ignores its arguments and returns a fixed response."""

    def _call(step, catalog):  # noqa: ARG001 - signature must match LLMCall
        return output

    return _call


class TestValidArgumentGeneration:
    def test_valid_sql_argument_generation(self, catalog, run_sql_step):
        output = _RunSqlLLMOutput(query="SELECT price, region FROM orders")
        args = generate_tool_arguments(
            run_sql_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is not None
        assert "orders" in args.query
        assert "LIMIT" in args.query  # sql_guard injects a default limit

    def test_existing_limit_is_preserved(self, catalog, run_sql_step):
        output = _RunSqlLLMOutput(query="SELECT price FROM orders LIMIT 5")
        args = generate_tool_arguments(
            run_sql_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is not None
        assert "LIMIT 5" in args.query


class TestMissingQueryRejection:
    def test_blank_query_rejected(self, catalog, run_sql_step):
        output = _RunSqlLLMOutput(query="")
        args = generate_tool_arguments(
            run_sql_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None

    def test_whitespace_only_query_rejected(self, catalog, run_sql_step):
        output = _RunSqlLLMOutput(query="   ")
        args = generate_tool_arguments(
            run_sql_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None

    def test_default_query_rejected(self, catalog, run_sql_step):
        # The LLM output type itself must never require the field to be present -- when the
        # LLM omits it entirely, the schema default kicks in and this must still be rejected.
        output = _RunSqlLLMOutput()
        args = generate_tool_arguments(
            run_sql_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None


class TestDangerousSqlRejection:
    @pytest.mark.parametrize(
        "query",
        [
            "DROP TABLE orders",
            "DELETE FROM orders",
            "INSERT INTO orders VALUES (1.0, 'x')",
            "UPDATE orders SET price = 0",
            "SELECT price FROM orders; DROP TABLE orders;",
        ],
    )
    def test_dangerous_statement_rejected(self, catalog, run_sql_step, query):
        output = _RunSqlLLMOutput(query=query)
        args = generate_tool_arguments(
            run_sql_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None


class TestHallucinatedReferenceRejection:
    def test_hallucinated_table_rejected(self, catalog, run_sql_step):
        output = _RunSqlLLMOutput(query="SELECT price FROM other_table")
        args = generate_tool_arguments(
            run_sql_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None

    def test_hallucinated_column_rejected(self, catalog, run_sql_step):
        output = _RunSqlLLMOutput(query="SELECT nonexistent_column FROM orders")
        args = generate_tool_arguments(
            run_sql_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None


class TestUnsupportedToolFamily:
    def test_non_run_sql_step_returns_none_without_calling_llm(self, catalog):
        def _fail_if_called(step, catalog):
            raise AssertionError("llm_call should not be invoked for a non-run_sql step")

        step = PlanStep(
            id="step_1",
            intent="Profile it.",
            tool_family="profile_dataset",
            expected_output="stats",
            success_criteria="ok",
        )
        args = generate_tool_arguments(step, catalog, "orders", llm_call=_fail_if_called)
        assert args is None


class TestBuildMessages:
    def test_includes_intent_and_columns(self, catalog, run_sql_step):
        messages = _build_messages(run_sql_step, catalog)
        human = messages[1].content
        assert run_sql_step.intent in human
        assert "price" in human
        assert "region" in human
        assert catalog.dataset_id in human


class TestLLMFailureHandling:
    def test_llm_call_exception_degrades_to_none_not_a_crash(self, catalog, run_sql_step):
        def _raising_call(step, catalog):
            raise RuntimeError("simulated network failure")

        args = generate_tool_arguments(run_sql_step, catalog, "orders", llm_call=_raising_call)
        assert args is None

    def test_llm_call_returning_wrong_type_degrades_to_none(self, catalog, run_sql_step):
        def _bad_call(step, catalog):
            return {"query": "not even the right shape"}

        args = generate_tool_arguments(run_sql_step, catalog, "orders", llm_call=_bad_call)
        assert args is None

    def test_missing_api_key_degrades_to_none_not_a_crash(
        self, catalog, run_sql_step, monkeypatch
    ):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("OPENAI_MODEL", raising=False)
        # Prevent load_llm_settings() from picking up a real developer .env file, so this
        # test's outcome doesn't depend on whether one happens to exist on this machine.
        monkeypatch.setattr("adia.agents.llm_config.load_dotenv", lambda *args, **kwargs: None)
        # No llm_call override -- exercises the real default path, which must reach
        # load_llm_settings(), fail on the missing key, and be caught, never raised.
        args = generate_tool_arguments(run_sql_step, catalog, "orders")
        assert args is None
