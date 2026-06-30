"""
Tests for chaksudb/db/connection.py

Tests database connection pool management, async context managers,
and query execution functions based on their docstrings.
"""

import pytest
import psycopg
from psycopg_pool import AsyncConnectionPool
from contextlib import asynccontextmanager

from chaksudb.db import connection as db_conn


class TestGetPool:
    """Tests for get_pool() function."""

    @pytest.mark.asyncio
    async def test_get_pool_returns_pool_instance(self, clean_pool):
        """Test that get_pool returns an AsyncConnectionPool instance."""
        pool = await db_conn.get_pool()
        assert isinstance(pool, AsyncConnectionPool)

    @pytest.mark.asyncio
    async def test_get_pool_creates_pool_if_none_exists(self, clean_pool):
        """Test that get_pool creates a pool when none exists."""
        # Pool should be None initially after clean_pool
        assert db_conn._pool is None
        
        # Calling get_pool should create it
        pool = await db_conn.get_pool()
        assert pool is not None
        assert db_conn._pool is not None

    @pytest.mark.asyncio
    async def test_get_pool_returns_same_instance_on_multiple_calls(self, clean_pool):
        """Test that get_pool returns the same pool instance on subsequent calls."""
        pool1 = await db_conn.get_pool()
        pool2 = await db_conn.get_pool()
        assert pool1 is pool2


class TestInitPool:
    """Tests for init_pool() function."""

    @pytest.mark.asyncio
    async def test_init_pool_creates_new_pool(self, clean_pool):
        """Test that init_pool creates a new AsyncConnectionPool."""
        pool = await db_conn.init_pool()
        assert isinstance(pool, AsyncConnectionPool)
        assert db_conn._pool is not None

    @pytest.mark.asyncio
    async def test_init_pool_returns_existing_pool_if_already_initialized(self, clean_pool):
        """Test that init_pool returns the existing pool if already initialized."""
        pool1 = await db_conn.init_pool()
        pool2 = await db_conn.init_pool()
        assert pool1 is pool2

    @pytest.mark.asyncio
    async def test_init_pool_returns_initialized_pool(self, clean_pool):
        """Test that init_pool returns an initialized AsyncConnectionPool."""
        pool = await db_conn.init_pool()
        assert isinstance(pool, AsyncConnectionPool)

    @pytest.mark.asyncio
    async def test_init_pool_with_bad_config_succeeds_but_connection_fails(self, clean_pool, monkeypatch):
        """Test that pool with bad config initializes but fails when trying to get a connection."""
        class BadConfig:
            async_connection_string = "postgresql://invalid:invalid@nonexistent:99999/invalid"
            min_connections = 0  # Set to 0 to avoid background connection attempts
            max_connections = 1
        
        monkeypatch.setattr(db_conn, "db_config", BadConfig())
        
        # Pool initialization should succeed (psycopg3 design)
        pool = await db_conn.init_pool()
        assert pool is not None
        
        # But trying to use it should fail
        with pytest.raises(Exception):  # Could be OperationalError or PoolTimeout
            async with db_conn.get_connection() as conn:
                await conn.execute("SELECT 1")


class TestClosePool:
    """Tests for close_pool() function."""

    @pytest.mark.asyncio
    async def test_close_pool_closes_existing_pool(self, clean_pool):
        """Test that close_pool closes the global connection pool."""
        # Create a pool first
        await db_conn.init_pool()
        assert db_conn._pool is not None
        
        # Close it
        await db_conn.close_pool()
        assert db_conn._pool is None

    @pytest.mark.asyncio
    async def test_close_pool_when_no_pool_exists(self, clean_pool):
        """Test that close_pool handles the case when no pool exists."""
        # Should not raise an error
        await db_conn.close_pool()
        assert db_conn._pool is None

    @pytest.mark.asyncio
    async def test_close_pool_can_be_called_multiple_times(self, clean_pool):
        """Test that close_pool can be called multiple times safely."""
        await db_conn.init_pool()
        await db_conn.close_pool()
        await db_conn.close_pool()  # Second call should not raise
        assert db_conn._pool is None


class TestGetConnection:
    """Tests for get_connection() async context manager."""

    @pytest.mark.asyncio
    async def test_get_connection_yields_async_connection(self, clean_pool):
        """Test that get_connection yields a psycopg AsyncConnection."""
        async with db_conn.get_connection() as conn:
            assert isinstance(conn, psycopg.AsyncConnection)

    @pytest.mark.asyncio
    async def test_get_connection_can_execute_query(self, clean_pool):
        """Test that connection from get_connection can execute queries."""
        async with db_conn.get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                result = await cur.fetchone()
                assert result == (1,)

    @pytest.mark.asyncio
    async def test_get_connection_initializes_pool_if_needed(self, clean_pool):
        """Test that get_connection initializes pool if it doesn't exist."""
        assert db_conn._pool is None
        
        async with db_conn.get_connection() as conn:
            assert isinstance(conn, psycopg.AsyncConnection)
        
        # Pool should now be initialized
        assert db_conn._pool is not None

    @pytest.mark.asyncio
    async def test_get_connection_returns_connection_to_pool(self, clean_pool):
        """Test that connection is returned to pool after context exits."""
        pool = await db_conn.get_pool()
        
        # Get initial pool stats
        async with db_conn.get_connection() as conn:
            assert conn is not None
        
        # After exiting context, connection should be returned to pool
        # We can verify by getting another connection
        async with db_conn.get_connection() as conn2:
            assert conn2 is not None


class TestCheckConnectionHealth:
    """Tests for check_connection_health() function."""

    @pytest.mark.asyncio
    async def test_check_connection_health_returns_true_when_healthy(self, clean_pool):
        """Test that check_connection_health returns True for a healthy connection."""
        result = await db_conn.check_connection_health()
        assert result is True

    @pytest.mark.asyncio
    async def test_check_connection_health_returns_false_on_failure(self, clean_pool, monkeypatch):
        """Test that check_connection_health returns False when connection fails."""
        @asynccontextmanager
        async def mock_get_connection():
            raise psycopg.OperationalError("Connection failed")
            yield  # Never reached
        
        monkeypatch.setattr(db_conn, "get_connection", mock_get_connection)
        
        result = await db_conn.check_connection_health()
        assert result is False

    @pytest.mark.asyncio
    async def test_check_connection_health_executes_select_1(self, clean_pool):
        """Test that check_connection_health executes SELECT 1 query."""
        # This is implicit in the function, but we can verify it works
        result = await db_conn.check_connection_health()
        assert result is True


class TestExecuteQuery:
    """Tests for execute_query() function."""

    @pytest.mark.asyncio
    async def test_execute_query_returns_list_of_results(self, clean_pool, test_db_schema):
        """Test that execute_query returns a list of query results."""
        result = await db_conn.execute_query("SELECT 1 AS num")
        assert isinstance(result, list)
        assert len(result) == 1
        assert result[0] == (1,)

    @pytest.mark.asyncio
    async def test_execute_query_with_parameters(self, clean_pool, test_db_schema):
        """Test that execute_query works with query parameters."""
        result = await db_conn.execute_query(
            "SELECT %s::int AS num, %s::text AS txt",
            (42, "hello")
        )
        assert len(result) == 1
        assert result[0] == (42, "hello")

    @pytest.mark.asyncio
    async def test_execute_query_returns_multiple_rows(self, clean_pool, test_db_schema):
        """Test that execute_query returns multiple rows."""
        result = await db_conn.execute_query(
            "SELECT generate_series(1, 5) AS num"
        )
        assert len(result) == 5
        assert result == [(1,), (2,), (3,), (4,), (5,)]

    @pytest.mark.asyncio
    async def test_execute_query_returns_empty_list_for_no_results(self, clean_pool, test_db_schema):
        """Test that execute_query returns empty list when no results."""
        result = await db_conn.execute_query("SELECT 1 WHERE FALSE")
        assert result == []

    @pytest.mark.asyncio
    async def test_execute_query_raises_on_sql_error(self, clean_pool, test_db_schema):
        """Test that execute_query raises psycopg.Error on invalid SQL."""
        with pytest.raises(psycopg.Error):
            await db_conn.execute_query("SELECT * FROM nonexistent_table")

    @pytest.mark.asyncio
    async def test_execute_query_with_none_parameters(self, clean_pool, test_db_schema):
        """Test that execute_query works when params is None."""
        result = await db_conn.execute_query("SELECT 1", None)
        assert result == [(1,)]


class TestExecuteOne:
    """Tests for execute_one() function."""

    @pytest.mark.asyncio
    async def test_execute_one_returns_single_result_tuple(self, clean_pool, test_db_schema):
        """Test that execute_one returns a single result tuple."""
        result = await db_conn.execute_one("SELECT 1 AS num, 'hello' AS txt")
        assert isinstance(result, tuple)
        assert result == (1, "hello")

    @pytest.mark.asyncio
    async def test_execute_one_with_parameters(self, clean_pool, test_db_schema):
        """Test that execute_one works with query parameters."""
        result = await db_conn.execute_one(
            "SELECT %s::int AS num, %s::text AS txt",
            (99, "world")
        )
        assert result == (99, "world")

    @pytest.mark.asyncio
    async def test_execute_one_returns_none_for_no_results(self, clean_pool, test_db_schema):
        """Test that execute_one returns None when no results."""
        result = await db_conn.execute_one("SELECT 1 WHERE FALSE")
        assert result is None

    @pytest.mark.asyncio
    async def test_execute_one_returns_first_row_only(self, clean_pool, test_db_schema):
        """Test that execute_one returns only the first row when multiple exist."""
        result = await db_conn.execute_one("SELECT generate_series(1, 5)")
        assert result == (1,)

    @pytest.mark.asyncio
    async def test_execute_one_raises_on_sql_error(self, clean_pool, test_db_schema):
        """Test that execute_one raises psycopg.Error on invalid SQL."""
        with pytest.raises(psycopg.Error):
            await db_conn.execute_one("SELECT * FROM nonexistent_table")

    @pytest.mark.asyncio
    async def test_execute_one_with_none_parameters(self, clean_pool, test_db_schema):
        """Test that execute_one works when params is None."""
        result = await db_conn.execute_one("SELECT 42", None)
        assert result == (42,)