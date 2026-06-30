"""
ParquetSink: Batch writing to Parquet files with row groups.

Provides efficient batch writing of export rows to Parquet format using PyArrow,
with configurable row group sizes and automatic schema handling.
"""

import logging
from pathlib import Path
from typing import Any, Optional

import pyarrow as pa
import pyarrow.parquet as pq

from chaksudb.export.parquet_schema import build_parquet_schema
from chaksudb.export.spec import ExportSpec

logger = logging.getLogger(__name__)


class ParquetSink:
    """
    Sink for writing export rows to Parquet format.

    Accumulates rows in batches and writes them to Parquet files with configurable
    row group sizes. Handles schema creation, type conversion, and file management.

    Attributes:
        output_path: Path to the output Parquet file
        spec: ExportSpec used to build the schema
        writer: PyArrow ParquetWriter instance
        schema: PyArrow schema for the export
        batch: Accumulated batch of rows (not yet written)
        batch_size: Target batch size before writing
        row_group_size: Target row group size in Parquet file
        total_rows_written: Total number of rows written so far
    """

    def __init__(
        self,
        output_path: Path,
        spec: ExportSpec,
        *,
        schema: Optional[pa.Schema] = None,
        row_group_size: int = 50000,
        batch_size: int = 10000,
    ):
        """
        Initialize ParquetSink.

        Creates the output file and initializes the Parquet writer with the
        appropriate schema based on the ExportSpec.

        Args:
            output_path: Path where the Parquet file will be written
            spec: ExportSpec defining the export query and fields
            schema: Optional PyArrow schema from query metadata (recommended).
                    If None, schema is built from spec.
            row_group_size: Target number of rows per row group (default: 50000)
            batch_size: Target number of rows to accumulate before writing (default: 10000)

        Raises:
            ValueError: If output_path is invalid
            IOError: If file cannot be created
        """
        self.output_path = Path(output_path)
        self.spec = spec
        self.row_group_size = row_group_size
        self.batch_size = batch_size
        self.total_rows_written = 0

        # Ensure output directory exists
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        # Use provided schema (from query metadata) or build from spec
        self.schema = schema if schema is not None else build_parquet_schema(spec)

        # Embed export spec as Parquet file metadata for reproducibility
        import json as _json
        spec_json = _json.dumps(spec.model_dump(mode="json", exclude_none=True))
        metadata = {
            b"chaksudb:export_spec": spec_json.encode("utf-8"),
            b"chaksudb:version": b"0.2.0",
        }
        existing_meta = self.schema.metadata or {}
        existing_meta.update(metadata)
        self.schema = self.schema.with_metadata(existing_meta)

        logger.info(
            f"Initializing ParquetSink: output={self.output_path}, "
            f"schema_fields={len(self.schema)}, row_group_size={row_group_size}"
        )

        # Initialize Parquet writer
        # Use version='1.0' for maximum compatibility with viewers and tools
        self.writer = pq.ParquetWriter(
            self.output_path,
            self.schema,
            version="1.0",
            compression="snappy",
            use_dictionary=True,  # Enable dictionary encoding for better compression
            write_statistics=True,  # Enable statistics for better query performance
        )

        # Accumulate rows in batches before writing
        self.batch: list[dict[str, Any]] = []

        logger.debug("ParquetSink initialized")

    def write_batch(self, rows: list[dict[str, Any]]) -> None:
        """
        Write a batch of rows to the Parquet file.

        Accumulates rows and writes them when batch_size is reached or when
        close() is called. Converts rows to PyArrow Table format and writes
        with appropriate row group sizing.

        Args:
            rows: List of dictionaries, where each dictionary represents one row

        Raises:
            ValueError: If rows don't match the expected schema
            IOError: If writing fails
        """
        if not rows:
            return

        # Add rows to batch
        self.batch.extend(rows)

        # Write if batch is large enough
        if len(self.batch) >= self.batch_size:
            self._flush_batch()

    def _flush_batch(self) -> None:
        """
        Flush accumulated batch to Parquet file.

        Converts the batch to a PyArrow Table and writes it to the file.
        Handles type conversion and schema alignment.
        """
        if not self.batch:
            return

        try:
            # Convert batch to PyArrow Table
            # First, convert each row to match schema types
            converted_rows = [self._convert_row(row) for row in self.batch]

            # Create PyArrow Table from batch
            # Use schema to ensure type consistency
            table = pa.Table.from_pylist(converted_rows, schema=self.schema)

            # Write table to Parquet file
            # PyArrow will handle row group sizing based on row_group_size
            self.writer.write_table(table, row_group_size=self.row_group_size)

            rows_written = len(self.batch)
            self.total_rows_written += rows_written

            logger.debug(
                f"Wrote batch: {rows_written} rows "
                f"(total: {self.total_rows_written})"
            )

            # Clear batch
            self.batch = []

        except Exception as e:
            logger.error(f"Error writing batch to Parquet: {e}")
            raise

    def _convert_row(self, row: dict[str, Any]) -> dict[str, Any]:
        """
        Convert a row to match the Parquet schema types.

        Handles type conversions:
        - UUID objects → strings
        - JSONB/dict objects → JSON strings
        - Lists → PyArrow-compatible lists
        - None values → preserved (nullable fields)

        Args:
            row: Dictionary representing one row

        Returns:
            Dictionary with converted values matching schema types
        """
        converted: dict[str, Any] = {}

        for field in self.schema:
            field_name = field.name
            field_type = field.type

            # Get value from row (may be missing)
            value = row.get(field_name)

            # Convert based on PyArrow type
            if value is None:
                converted[field_name] = None
            elif pa.types.is_string(field_type):
                # Convert to string
                converted[field_name] = str(value) if value is not None else None
            elif pa.types.is_integer(field_type):
                # Convert to int
                converted[field_name] = int(value) if value is not None else None
            elif pa.types.is_floating(field_type):
                # Convert to float
                converted[field_name] = float(value) if value is not None else None
            elif pa.types.is_boolean(field_type):
                converted[field_name] = bool(value) if value is not None else None
            elif pa.types.is_list(field_type):
                # Ensure it's a list
                if isinstance(value, str):
                    import json as _json
                    try:
                        value = _json.loads(value)
                    except (ValueError, TypeError):
                        value = None
                converted[field_name] = (
                    list(value) if isinstance(value, (list, tuple)) else None
                )
            else:
                # For other types (struct, etc.), convert to string representation
                # This handles JSONB fields stored as strings
                if isinstance(value, (dict, list)):
                    import json

                    converted[field_name] = json.dumps(value)
                else:
                    converted[field_name] = str(value) if value is not None else None

        return converted

    def close(self) -> None:
        """
        Close the ParquetSink and finalize the file.

        Flushes any remaining rows in the batch and closes the Parquet writer.
        This should be called after all rows have been written.

        Raises:
            IOError: If closing fails
        """
        # Flush any remaining rows
        if self.batch:
            self._flush_batch()

        # Close writer
        if self.writer:
            self.writer.close()
            self.writer = None  # type: ignore

        logger.info(
            f"ParquetSink closed: {self.total_rows_written} total rows written "
            f"to {self.output_path}"
        )

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures close() is called."""
        self.close()
        return False  # Don't suppress exceptions
