"""
Parquet Export: Export query results to Parquet format.

Combines streaming query execution with Parquet file writing to efficiently
export large datasets without loading all data into memory.
"""

import asyncio
import logging
from pathlib import Path
from typing import Optional

from chaksudb.common.progress import ProgressTracker
from chaksudb.export.parquet_sink import ParquetSink
from chaksudb.export.path_resolution import resolve_paths_in_row
from chaksudb.export.spec import ExportSpec
from chaksudb.export.streaming import count_rows, get_query_schema, stream_rows

logger = logging.getLogger(__name__)


async def export_to_parquet(
    spec: ExportSpec,
    output_path: Path,
    batch_size: int = 5000,
    progress_tracker: Optional[ProgressTracker] = None,
) -> None:
    """
    Export query results to Parquet format.

    Streams rows from the database using server-side cursors and writes them
    to a Parquet file in batches. Handles large datasets efficiently without
    loading all data into memory.

    Args:
        spec: The ExportSpec defining the query to execute
        output_path: Path where the Parquet file will be written
        batch_size: Number of rows to fetch per batch from database (default: 5000)
        progress_tracker: Optional ProgressTracker instance for progress tracking.
                         If None, progress will be logged but not tracked.

    Raises:
        ValueError: If spec validation fails or output_path is invalid
        psycopg.OperationalError: If database connection fails
        psycopg.ProgrammingError: If query execution fails
        IOError: If file writing fails

    Example:
        >>> from pathlib import Path
        >>> from chaksudb.export.spec import ExportSpec
        >>> spec = ExportSpec(
        ...     dataset_names=["EYEPACS"],
        ...     annotation_tasks=["grading"],
        ...     disease_types=["DR"]
        ... )
        >>> await export_to_parquet(spec, Path("output.parquet"))
        >>> print("Export complete")
    """
    output_path = Path(output_path)

    # Validate output path
    if output_path.exists():
        logger.warning(f"Output file already exists: {output_path}. It will be overwritten.")

    # Count total rows for progress tracking
    logger.info("Counting total rows for export...")
    try:
        total_count = await count_rows(spec)
        logger.info(f"Export will contain approximately {total_count} rows")
    except Exception as e:
        logger.warning(f"Could not count rows: {e}. Progress tracking may be inaccurate.")
        total_count = 0

    # Create progress tracker if not provided
    if progress_tracker is None and total_count > 0:
        progress_tracker = ProgressTracker(
            total=total_count,
            description=f"Exporting to {output_path.name}",
        )

    # Schema from query metadata (LIMIT 0 + cursor.description) so Parquet columns match DB
    logger.info("Resolving Parquet schema from export query...")
    schema = await get_query_schema(spec)

    # If caption synthesis is requested, add the computed 'caption' column to the schema
    caption_engine = None
    if spec.caption_mode is not None:
        import pyarrow as pa
        from chaksudb.common.dictionary import abbreviations, definitions
        from chaksudb.export.caption_engine import CaptionEngine
        schema = schema.append(pa.field("caption", pa.string(), nullable=True))
        caption_engine = CaptionEngine(definitions=definitions, abbreviations=abbreviations)

    # Initialize Parquet sink with schema from DB
    logger.info(f"Starting Parquet export to {output_path}")
    sink = ParquetSink(
        output_path=output_path,
        spec=spec,
        schema=schema,
        row_group_size=50000,  # 50k rows per row group
        batch_size=10000,  # Accumulate 10k rows before writing
    )

    try:
        # Stream rows and write to Parquet
        rows_processed = 0
        async for batch in stream_rows(
            spec, batch_size=batch_size, progress_tracker=progress_tracker
        ):
            # Resolve paths so Parquet has full paths for file_path and mask_file_path
            batch = [resolve_paths_in_row(row) for row in batch]
            # Synthesize captions if requested
            if caption_engine is not None:
                mode = spec.caption_mode
                batch = [{**row, "caption": caption_engine.synthesize(row, mode)} for row in batch]
            # Write batch to Parquet
            sink.write_batch(batch)
            rows_processed += len(batch)

            # Log progress periodically
            if rows_processed % (batch_size * 10) == 0:
                logger.info(f"Processed {rows_processed} rows...")
                logger.info(f"Row sample: {batch[0]}")

        # Final flush and close
        sink.close()

        logger.info(
            f"Parquet export complete: {rows_processed} rows written to {output_path}"
        )

        # Finish progress tracking
        if progress_tracker:
            progress_tracker.finish()

    except Exception as e:
        logger.error(f"Error during Parquet export: {e}")
        # Ensure sink is closed even on error
        try:
            sink.close()
        except Exception as close_error:
            logger.warning(f"Error closing Parquet sink: {close_error}")

        # Clean up partial file on error (optional)
        if output_path.exists():
            logger.warning(f"Partial file may exist at {output_path}")

        raise


def export_to_parquet_sync(
    spec: ExportSpec,
    output_path: Path,
    batch_size: int = 5000,
    progress_tracker: Optional[ProgressTracker] = None,
) -> None:
    """
    Synchronous wrapper for export_to_parquet.

    Runs the async Parquet export in an event loop so callers can use it
    without async/await. Used by the public export() facade.

    Args:
        spec: The ExportSpec defining the query to execute
        output_path: Path where the Parquet file will be written
        batch_size: Number of rows to fetch per batch (default: 5000)
        progress_tracker: Optional ProgressTracker for progress tracking
    """
    asyncio.run(export_to_parquet(spec, output_path, batch_size=batch_size, progress_tracker=progress_tracker))
