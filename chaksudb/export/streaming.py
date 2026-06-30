"""
Streaming: Server-side cursor support for large exports.

Provides async streaming of query results using PostgreSQL server-side cursors
to avoid loading all data into memory at once.
"""

import logging
import re
from collections.abc import AsyncIterator, Callable
from typing import Optional

import psycopg

from chaksudb.common.progress import ProgressTracker
from chaksudb.db import get_connection
from chaksudb.export.parquet_schema import build_parquet_schema_from_query_description
from chaksudb.export.query_builder import QueryBuilder, QueryPlan
from chaksudb.export.spec import ExportSpec

logger = logging.getLogger(__name__)


async def get_query_schema(spec: ExportSpec):
    """
    Get the PyArrow schema for the export query by running it with LIMIT 0.

    Uses the cursor's description (column names and type OIDs) so the schema
    exactly matches what the database returns. Use this to drive Parquet
    export so columns = DB columns.

    Args:
        spec: The ExportSpec defining the query.

    Returns:
        pyarrow.Schema with one field per query column, in query order.

    Raises:
        ValueError: If query building fails
        psycopg.OperationalError: If database connection fails
        psycopg.ProgrammingError: If query execution fails
    """
    builder = QueryBuilder()
    plan = builder.build_query(spec)
    sql = plan.render_sql() + "\nLIMIT 0"
    params = plan.params

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            if not cur.description:
                # No columns (e.g. non-SELECT) - return empty schema
                import pyarrow as pa

                return pa.schema([])
            return build_parquet_schema_from_query_description(cur.description)


async def stream_rows(
    spec: ExportSpec,
    batch_size: int = 5000,
    progress_tracker: Optional[ProgressTracker] = None,
) -> AsyncIterator[list[dict]]:
    """
    Stream rows from a query using a server-side cursor.

    Uses PostgreSQL server-side cursors to fetch results in batches without
    loading all data into memory. The cursor is created with WITH HOLD to allow
    it to persist beyond the transaction, enabling streaming across multiple
    batches.

    Args:
        spec: The ExportSpec defining the query to execute
        batch_size: Number of rows to fetch per batch (default: 5000)
        progress_tracker: Optional ProgressTracker instance for progress tracking.
                         If None, progress will be logged but not tracked.

    Yields:
        Lists of dictionaries, where each dictionary represents one row
        from the query result. Each batch contains up to batch_size rows.

    Raises:
        ValueError: If query building fails
        psycopg.OperationalError: If database connection fails
        psycopg.ProgrammingError: If query execution fails

    Example:
        >>> from chaksudb.common.progress import ProgressTracker
        >>> spec = ExportSpec(dataset_names=["EYEPACS"])
        >>> total_count = await count_rows(spec)
        >>> tracker = ProgressTracker(total=total_count, description="Exporting rows")
        >>> async for batch in stream_rows(spec, batch_size=1000, progress_tracker=tracker):
        ...     # Process batch
        ...     pass
        >>> tracker.finish()
    """
    # Build query from spec
    builder = QueryBuilder()
    plan = builder.build_query(spec)
    sql = plan.render_sql()
    params = plan.params

    logger.info(
        f"Starting streaming export with batch_size={batch_size}. "
        f"Query has {len(params)} parameters."
    )

    # Use a connection from the pool
    # Server-side cursors need a persistent connection
    async with get_connection() as conn:
        cursor_name = "export_cursor"
        cursor: Optional[psycopg.AsyncCursor] = None
        total_rows_fetched = 0

        try:
            # Create server-side cursor with WITH HOLD
            # WITH HOLD allows the cursor to persist beyond the transaction
            # This is necessary for streaming large result sets
            async with conn.cursor(name=cursor_name, withhold=True) as cur:
                cursor = cur

                # Execute the query
                logger.debug(f"Executing query with cursor '{cursor_name}'")
                await cur.execute(sql, params)

                # Stream results in batches
                while True:
                    batch = await cur.fetchmany(batch_size)
                    if not batch:
                        # No more rows
                        break

                    # Convert rows to dictionaries
                    # cur.description contains column metadata
                    if cur.description:
                        column_names = [desc.name for desc in cur.description]
                        dict_batch = [
                            dict(zip(column_names, row)) for row in batch
                        ]
                    else:
                        # Fallback if description is not available
                        dict_batch = [dict(row) for row in batch]

                    total_rows_fetched += len(dict_batch)

                    # Update progress tracker if provided
                    if progress_tracker:
                        try:
                            progress_tracker.update(count=len(dict_batch), success=True)
                        except Exception as e:
                            logger.warning(
                                f"Progress tracker update raised exception: {e}"
                            )

                    yield dict_batch

                logger.info(
                    f"Streaming complete. Total rows fetched: {total_rows_fetched}"
                )

        except psycopg.Error as e:
            error_msg = f"Database error during streaming: {e}. Fetched {total_rows_fetched} rows before error."
            logger.error(error_msg)
            if progress_tracker:
                progress_tracker.record_error("database", str(e))
            raise

        except Exception as e:
            error_msg = f"Unexpected error during streaming: {e}. Fetched {total_rows_fetched} rows before error."
            logger.error(error_msg)
            if progress_tracker:
                progress_tracker.record_error("unexpected", str(e))
            raise

        finally:
            # Cleanup: Close cursor if it still exists
            # The cursor should be automatically closed by the context manager,
            # but we ensure cleanup in case of errors
            if cursor is not None:
                try:
                    # Cursor is closed automatically by context manager
                    # But we can explicitly close it if needed
                    pass
                except Exception as e:
                    logger.warning(f"Error closing cursor: {e}")

            logger.debug(f"Streaming connection cleanup complete")


async def count_rows(spec: ExportSpec) -> int:
    """
    Count the total number of rows that would be returned by a query.

    Executes a COUNT(*) query based on the ExportSpec to determine the
    total number of rows without fetching all data.

    Args:
        spec: The ExportSpec defining the query

    Returns:
        Total number of rows that would be returned

    Raises:
        ValueError: If query building fails
        psycopg.OperationalError: If database connection fails
        psycopg.ProgrammingError: If query execution fails

    Example:
        >>> spec = ExportSpec(dataset_names=["EYEPACS"])
        >>> count = await count_rows(spec)
        >>> print(f"Query will return {count} rows")
    """
    # Build query from spec
    builder = QueryBuilder()
    plan = builder.build_query(spec)

    # Wrap the full query as a subquery so GROUP BY is preserved correctly.
    # The naive approach of prepending SELECT COUNT(*) to the FROM clause
    # breaks when the query has GROUP BY — fetchone() would return the count
    # for only the first group (always 1 per image) instead of total rows.
    inner_sql = re.sub(r'\nORDER BY [^\n]+', '', plan.render_sql())
    count_sql = f"SELECT COUNT(*) FROM ({inner_sql}) _count_subq"

    params = plan.params

    logger.debug("Executing COUNT query")

    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(count_sql, params)
            result = await cur.fetchone()
            if result is None:
                return 0
            return int(result[0])
