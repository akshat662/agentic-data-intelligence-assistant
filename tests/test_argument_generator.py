"""Tests for adia.agents.argument_generator. No real OpenAI call is ever made here."""

import pytest

from adia.agents.argument_generator import (
    _build_messages,
    _CompareGroupsLLMOutput,
    _ComputeCorrelationLLMOutput,
    _RunSqlLLMOutput,
    _SegmentContributionLLMOutput,
    _TrainModelLLMOutput,
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

    def test_blank_dependency_context_produces_identical_prompt(self, catalog, run_sql_step):
        # Backward compatibility: the default ("") must produce the exact same prompt as
        # before this parameter existed.
        with_default = _build_messages(run_sql_step, catalog)
        with_explicit_blank = _build_messages(run_sql_step, catalog, "")
        assert with_default[1].content == with_explicit_blank[1].content

    def test_nonblank_dependency_context_is_included(self, catalog, run_sql_step):
        messages = _build_messages(run_sql_step, catalog, "prior finding: region = 'West'")
        human = messages[1].content
        assert "prior finding: region = 'West'" in human
        assert run_sql_step.intent in human  # base content is still present, not replaced


class TestDependencyContextWiring:
    """Covers the dependency-evidence handoff: `execute_tools_node` renders a dependency
    step's evidence and passes it through as `dependency_context` -- this tests that
    `generate_tool_arguments` forwards it to the real LLM call path without disturbing the
    `llm_call` override contract every other test in this file relies on."""

    def test_dependency_context_reaches_the_real_call_path(
        self, catalog, run_sql_step, monkeypatch
    ):
        captured = {}

        def _fake_call_openai(step, catalog, dependency_context=""):
            captured["dependency_context"] = dependency_context
            return _RunSqlLLMOutput(query="SELECT price FROM orders")

        monkeypatch.setattr("adia.agents.argument_generator._call_openai", _fake_call_openai)
        args = generate_tool_arguments(
            run_sql_step, catalog, "orders", dependency_context="region = 'West' [[ev_x]]"
        )
        assert args is not None
        assert captured["dependency_context"] == "region = 'West' [[ev_x]]"

    def test_default_dependency_context_is_blank_on_the_real_call_path(
        self, catalog, run_sql_step, monkeypatch
    ):
        captured = {}

        def _fake_call_openai(step, catalog, dependency_context=""):
            captured["dependency_context"] = dependency_context
            return _RunSqlLLMOutput(query="SELECT price FROM orders")

        monkeypatch.setattr("adia.agents.argument_generator._call_openai", _fake_call_openai)
        generate_tool_arguments(run_sql_step, catalog, "orders")
        assert captured["dependency_context"] == ""

    def test_llm_call_override_is_unaffected_by_dependency_context(
        self, catalog, run_sql_step
    ):
        # The existing llm_call contract (step, catalog) -> raw output must be completely
        # unaffected by dependency_context -- every existing test's fake keeps working as-is.
        output = _RunSqlLLMOutput(query="SELECT price FROM orders")
        args = generate_tool_arguments(
            run_sql_step,
            catalog,
            "orders",
            dependency_context="some prior finding",
            llm_call=_fake_llm_call(output),
        )
        assert args is not None
        assert "orders" in args.query


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


# ---------------------------------------------------------------------------
# Fixtures for the three new tool families
# ---------------------------------------------------------------------------


@pytest.fixture
def compare_groups_step() -> PlanStep:
    return PlanStep(
        id="step_cg",
        intent="Compare price across regions.",
        tool_family="compare_groups",
        expected_output="group_stats",
        success_criteria="ok",
    )


@pytest.fixture
def compute_correlation_step() -> PlanStep:
    return PlanStep(
        id="step_cc",
        intent="Correlate all numeric columns.",
        tool_family="compute_correlation",
        expected_output="correlation_matrix",
        success_criteria="ok",
    )


@pytest.fixture
def train_model_step() -> PlanStep:
    return PlanStep(
        id="step_tm",
        intent="Predict price from region.",
        tool_family="train_model",
        expected_output="model_report",
        success_criteria="ok",
    )


@pytest.fixture
def segment_contribution_step() -> PlanStep:
    return PlanStep(
        id="step_sc",
        intent="Break down price by region within a scope.",
        tool_family="segment_contribution",
        expected_output="ranked_contributions",
        success_criteria="ok",
    )


# ---------------------------------------------------------------------------
# compare_groups tests
# ---------------------------------------------------------------------------


class TestCompareGroupsArgumentGeneration:
    def test_valid_compare_groups_argument_generation(self, catalog, compare_groups_step):
        output = _CompareGroupsLLMOutput(group_column="region", metric_column="price")
        args = generate_tool_arguments(
            compare_groups_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is not None
        assert args.group_column == "region"
        assert args.metric_column == "price"
        assert args.dataset_id == "orders"

    def test_unknown_group_column_rejected(self, catalog, compare_groups_step):
        output = _CompareGroupsLLMOutput(group_column="nonexistent", metric_column="price")
        args = generate_tool_arguments(
            compare_groups_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None

    def test_unknown_metric_column_rejected(self, catalog, compare_groups_step):
        output = _CompareGroupsLLMOutput(group_column="region", metric_column="nonexistent")
        args = generate_tool_arguments(
            compare_groups_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None

    def test_llm_failure_degrades_to_none(self, catalog, compare_groups_step):
        def _raising_call(step, catalog):
            raise RuntimeError("simulated network failure")

        args = generate_tool_arguments(
            compare_groups_step, catalog, "orders", llm_call=_raising_call
        )
        assert args is None


# ---------------------------------------------------------------------------
# compute_correlation tests
# ---------------------------------------------------------------------------


class TestComputeCorrelationArgumentGeneration:
    def test_valid_explicit_columns(self, catalog, compute_correlation_step):
        output = _ComputeCorrelationLLMOutput(columns=["price"])
        args = generate_tool_arguments(
            compute_correlation_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is not None
        assert args.columns == ["price"]
        assert args.dataset_id == "orders"

    def test_null_columns_means_all_numeric(self, catalog, compute_correlation_step):
        output = _ComputeCorrelationLLMOutput(columns=None)
        args = generate_tool_arguments(
            compute_correlation_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is not None
        assert args.columns is None

    def test_unknown_column_rejected(self, catalog, compute_correlation_step):
        output = _ComputeCorrelationLLMOutput(columns=["price", "nonexistent"])
        args = generate_tool_arguments(
            compute_correlation_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None

    def test_llm_failure_degrades_to_none(self, catalog, compute_correlation_step):
        def _raising_call(step, catalog):
            raise RuntimeError("simulated network failure")

        args = generate_tool_arguments(
            compute_correlation_step, catalog, "orders", llm_call=_raising_call
        )
        assert args is None


# ---------------------------------------------------------------------------
# train_model tests
# ---------------------------------------------------------------------------


class TestTrainModelArgumentGeneration:
    def test_valid_train_model_argument_generation(self, catalog, train_model_step):
        output = _TrainModelLLMOutput(
            target_column="price",
            feature_columns=["price"],
            task_type="regression",
            model_type="linear_regression",
        )
        args = generate_tool_arguments(
            train_model_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is not None
        assert args.target_column == "price"
        assert args.feature_columns == ["price"]
        assert args.task_type == "regression"
        assert args.model_type == "linear_regression"
        assert args.dataset_id == "orders"

    def test_unknown_target_column_rejected(self, catalog, train_model_step):
        output = _TrainModelLLMOutput(
            target_column="nonexistent",
            feature_columns=["price"],
            task_type="regression",
            model_type="linear_regression",
        )
        args = generate_tool_arguments(
            train_model_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None

    def test_unknown_feature_column_rejected(self, catalog, train_model_step):
        output = _TrainModelLLMOutput(
            target_column="price",
            feature_columns=["nonexistent"],
            task_type="regression",
            model_type="linear_regression",
        )
        args = generate_tool_arguments(
            train_model_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None

    def test_llm_failure_degrades_to_none(self, catalog, train_model_step):
        def _raising_call(step, catalog):
            raise RuntimeError("simulated network failure")

        args = generate_tool_arguments(
            train_model_step, catalog, "orders", llm_call=_raising_call
        )
        assert args is None


# ---------------------------------------------------------------------------
# segment_contribution tests
# ---------------------------------------------------------------------------


class TestSegmentContributionArgumentGeneration:
    def test_valid_unscoped_argument_generation(self, catalog, segment_contribution_step):
        output = _SegmentContributionLLMOutput(entity_column="region", metric_column="price")
        args = generate_tool_arguments(
            segment_contribution_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is not None
        assert args.entity_column == "region"
        assert args.metric_column == "price"
        assert args.parent_column is None
        assert args.parent_value is None
        assert args.dataset_id == "orders"

    def test_valid_parent_scoped_argument_generation(self, catalog, segment_contribution_step):
        output = _SegmentContributionLLMOutput(
            entity_column="region",
            metric_column="price",
            parent_column="region",
            parent_value="west",
        )
        args = generate_tool_arguments(
            segment_contribution_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is not None
        assert args.parent_column == "region"
        assert args.parent_value == "west"

    def test_unknown_entity_column_rejected(self, catalog, segment_contribution_step):
        output = _SegmentContributionLLMOutput(entity_column="nonexistent", metric_column="price")
        args = generate_tool_arguments(
            segment_contribution_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None

    def test_unknown_metric_column_rejected(self, catalog, segment_contribution_step):
        output = _SegmentContributionLLMOutput(entity_column="region", metric_column="nonexistent")
        args = generate_tool_arguments(
            segment_contribution_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None

    def test_unknown_parent_column_rejected(self, catalog, segment_contribution_step):
        output = _SegmentContributionLLMOutput(
            entity_column="region",
            metric_column="price",
            parent_column="nonexistent",
            parent_value="x",
        )
        args = generate_tool_arguments(
            segment_contribution_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None

    def test_parent_column_without_value_rejected(self, catalog, segment_contribution_step):
        # SegmentContributionArgs' own model_validator requires both together -- the
        # ValueError it raises must degrade to None like any other rejected proposal.
        output = _SegmentContributionLLMOutput(
            entity_column="region", metric_column="price", parent_column="region"
        )
        args = generate_tool_arguments(
            segment_contribution_step, catalog, "orders", llm_call=_fake_llm_call(output)
        )
        assert args is None

    def test_llm_failure_degrades_to_none(self, catalog, segment_contribution_step):
        def _raising_call(step, catalog):
            raise RuntimeError("simulated network failure")

        args = generate_tool_arguments(
            segment_contribution_step, catalog, "orders", llm_call=_raising_call
        )
        assert args is None

