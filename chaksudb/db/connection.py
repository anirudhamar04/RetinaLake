"""
Async database connection pool management using psycopg3.

Provides connection pool management, async context managers, and connection
health checks for PostgreSQL database operations.
"""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator, LiteralString, Optional

import psycopg
from psycopg_pool import AsyncConnectionPool  

from chaksudb.config.config import db_config

logger = logging.getLogger(__name__)

# Global connection pool instance
_pool: Optional[AsyncConnectionPool] = None


async def get_pool() -> AsyncConnectionPool:
    """
    Get or create the global async connection pool.

    Returns:
        AsyncConnectionPool instance

    Raises:
        RuntimeError: If pool initialization fails
    """
    global _pool
    if _pool is None:
        return await init_pool()
    return _pool


async def init_pool() -> AsyncConnectionPool:
    """
    Initialize the global async connection pool.

    Returns:
        Initialized AsyncConnectionPool

    Raises:
        RuntimeError: If pool initialization fails
    """
    global _pool
    if _pool is not None:
        return _pool

    try:
        conninfo = db_config.async_connection_string
        _pool = AsyncConnectionPool(
            conninfo=conninfo,
            min_size=db_config.min_connections,
            max_size=db_config.max_connections,
            open=False,  # We'll open it explicitly
            reconnect_timeout=30,   # give up trying to reconnect after 30s
            max_waiting=2000,       # large queue for concurrent ingest workloads
            timeout=120,            # wait up to 2 min for a connection before failing
        )
        await _pool.open(wait=True)
        logger.info(
            f"Initialized connection pool: "
            f"min={db_config.min_connections}, max={db_config.max_connections}"
        )
        return _pool
    except Exception as e:
        logger.error(f"Failed to initialize connection pool: {e}")
        raise RuntimeError(f"Connection pool initialization failed: {e}") from e


async def close_pool() -> None:
    """
    Close the global connection pool.

    This should be called during application shutdown.
    """
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Connection pool closed")


@asynccontextmanager
async def get_connection() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """
    Get a connection from the pool as an async context manager.

    Usage:
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")

    Yields:
        psycopg.AsyncConnection from the pool

    Raises:
        RuntimeError: If pool is not initialized
        psycopg.OperationalError: If connection cannot be acquired
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        yield conn


@asynccontextmanager
async def get_transaction() -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """
    Get a connection from the pool with an explicit transaction.

    The transaction will automatically commit on success or rollback on error.
    This is useful for ensuring atomic operations across multiple queries.

    Usage:
        async with get_transaction() as conn:
            # All operations here are in a transaction
            async with conn.cursor() as cur:
                await cur.execute("INSERT INTO table1 ...")
                await cur.execute("INSERT INTO table2 ...")
            # Transaction commits here if no exception
            # Transaction rolls back if any exception occurs

    Yields:
        psycopg.AsyncConnection from the pool with active transaction

    Raises:
        RuntimeError: If pool is not initialized
        psycopg.OperationalError: If connection cannot be acquired
        Exception: Any exception will trigger automatic rollback
    """
    pool = await get_pool()
    async with pool.connection() as conn:
        async with conn.transaction():
            logger.debug("Transaction started")
            try:
                yield conn
                logger.debug("Transaction will commit")
            except Exception as e:
                logger.warning(f"Transaction will rollback due to error: {e}")
                raise


async def check_connection_health() -> bool:
    """
    Check if the database connection pool is healthy.

    Returns:
        True if connection is healthy, False otherwise
    """
    try:
        async with get_connection() as conn:
            async with conn.cursor() as cur:
                await cur.execute("SELECT 1")
                result = await cur.fetchone()
                return result is not None and result[0] == 1
    except Exception as e:
        logger.error(f"Connection health check failed: {e}")
        return False


async def execute_query(query: LiteralString, params: Optional[tuple] = None) -> list:
    """
    Execute a query and return results.

    Args:
        query: SQL query string
        params: Optional query parameters

    Returns:
        List of query results

    Raises:
        psycopg.Error: If query execution fails
    """
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(query, params)
            return await cur.fetchall()

