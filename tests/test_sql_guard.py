"""Tests for adia.tools.sql_guard."""

import pytest

from adia.models.catalog import ColumnProfile, DatasetCatalog, SemanticType
from adia.tools.sql_guard import SqlGuardError, check_sql


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
        ColumnProfile(
            name="order_date",
            dtype="datetime64[us]",
            semantic_type=SemanticType.DATETIME,
            non_null_count=100,
            null_count=0,
            null_rate=0.0,
            unique_count=90,
            min_date="2024-01-01T00:00:00",
            max_date="2024-06-01T00:00:00",
        ),
    ]
    return DatasetCatalog(
        dataset_id="orders",
        source_path="data/processed/orders.parquet",
        row_count=100,
        columns=columns,
    )


class TestValidQueries:
    def test_simple_select_passes(self, catalog):
        guarded = check_sql(
            "SELECT price, region FROM orders", catalog=catalog, table_name="orders"
        )
        assert "LIMIT" in guarded.sql
        assert guarded.limit_injected is True

    def test_existing_limit_is_preserved(self, catalog):
        guarded = check_sql(
            "SELECT price FROM orders LIMIT 5", catalog=catalog, table_name="orders"
        )
        assert "LIMIT 5" in guarded.sql
        assert guarded.limit_injected is False

    def test_cte_referencing_own_alias_passes(self, catalog):
        guarded = check_sql(
            "WITH by_region AS (SELECT region, price FROM orders) "
            "SELECT region, SUM(price) AS total_price FROM by_region GROUP BY region",
            catalog=catalog,
            table_name="orders",
        )
        assert "total_price" in guarded.sql

    def test_order_by_output_alias_passes(self, catalog):
        guarded = check_sql(
            "SELECT region, SUM(price) AS total_price FROM orders "
            "GROUP BY region ORDER BY total_price DESC",
            catalog=catalog,
            table_name="orders",
        )
        assert guarded.sql


class TestBlockedStatements:
    @pytest.mark.parametrize(
        ("sql", "keyword"),
        [
            ("INSERT INTO orders VALUES (1.0, 'x', '2024-01-01')", "INSERT"),
            ("UPDATE orders SET price = 0", "UPDATE"),
            ("DELETE FROM orders", "DELETE"),
            ("DROP TABLE orders", "DROP"),
            ("ALTER TABLE orders ADD COLUMN x INT", "ALTER"),
            ("CREATE TABLE evil AS SELECT * FROM orders", "CREATE"),
        ],
    )
    def test_destructive_statement_rejected(self, catalog, sql, keyword):
        with pytest.raises(SqlGuardError, match=keyword):
            check_sql(sql, catalog=catalog, table_name="orders")

    def test_multi_statement_rejected(self, catalog):
        with pytest.raises(SqlGuardError, match="single SQL statement"):
            check_sql(
                "SELECT price FROM orders; DROP TABLE orders;",
                catalog=catalog,
                table_name="orders",
            )

    def test_unparseable_query_rejected(self, catalog):
        with pytest.raises(SqlGuardError):
            check_sql("SELECT FROM FROM WHERE", catalog=catalog, table_name="orders")

    def test_blank_statement_rejected(self, catalog):
        with pytest.raises(SqlGuardError, match="could not be parsed"):
            check_sql("   ;   ", catalog=catalog, table_name="orders")

    def test_pragma_rejected(self, catalog):
        with pytest.raises(SqlGuardError):
            check_sql("PRAGMA table_info('orders')", catalog=catalog, table_name="orders")

    def test_attach_rejected(self, catalog):
        with pytest.raises(SqlGuardError):
            check_sql("ATTACH 'evil.db' AS evil", catalog=catalog, table_name="orders")


class TestUnknownTablesAndColumns:
    def test_unknown_table_rejected(self, catalog):
        with pytest.raises(SqlGuardError, match="Unknown table"):
            check_sql("SELECT price FROM other_table", catalog=catalog, table_name="orders")

    def test_join_on_unknown_table_rejected(self, catalog):
        with pytest.raises(SqlGuardError, match="Unknown table"):
            check_sql(
                "SELECT o.price FROM orders o JOIN customers c ON o.region = c.region",
                catalog=catalog,
                table_name="orders",
            )

    def test_unknown_column_rejected(self, catalog):
        with pytest.raises(SqlGuardError, match="Unknown column"):
            check_sql(
                "SELECT nonexistent_column FROM orders", catalog=catalog, table_name="orders"
            )

    def test_unknown_column_in_where_rejected(self, catalog):
        with pytest.raises(SqlGuardError, match="Unknown column"):
            check_sql(
                "SELECT price FROM orders WHERE ssn = '123'",
                catalog=catalog,
                table_name="orders",
            )
