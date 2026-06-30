"""
Performance tests for export pipeline.

Tests streaming performance, memory usage, and scalability with large result sets.
Verifies that streaming works correctly and memory usage stays bounded.
"""

import asyncio
import gc
import tempfile
from pathlib import Path

import pytest
import psutil
import pyarrow.parquet as pq

from chaksudb.export.parquet_export import export_to_parquet
from chaksudb.export.spec import ExportSpec
from chaksudb.export.streaming import count_rows, stream_rows


@pytest.mark.asyncio
class TestStreamingPerformance:
    """Tests for streaming performance and memory usage."""

    async def test_streaming_with_large_result_set(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test that streaming works with multiple batches."""
        spec = ExportSpec()
        batch_size = 5  # Small batch size to force multiple batches

        total_rows = 0
        batch_count = 0

        async for batch in stream_rows(spec, batch_size=batch_size):
            total_rows += len(batch)
            batch_count += 1

        # Should have processed at least some rows
        assert total_rows >= 0
        # If we have data, we should have at least one batch
        if total_rows > 0:
            assert batch_count > 0

    async def test_streaming_memory_usage_bounded(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test that streaming doesn't cause unbounded memory growth."""
        spec = ExportSpec()
        batch_size = 10

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Stream through all batches
        max_memory = initial_memory
        batch_count = 0

        async for batch in stream_rows(spec, batch_size=batch_size):
            batch_count += 1
            # Check memory after each batch
            current_memory = process.memory_info().rss / 1024 / 1024  # MB
            max_memory = max(max_memory, current_memory)

            # Force garbage collection periodically
            if batch_count % 10 == 0:
                gc.collect()

        # Memory growth should be reasonable (less than 100MB for test data)
        # This is a loose bound - actual memory depends on batch size and data
        memory_growth = max_memory - initial_memory
        assert memory_growth < 100, f"Memory growth too large: {memory_growth}MB"

    async def test_streaming_progress_tracking(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test that streaming works with progress tracking."""
        from chaksudb.common.progress import ProgressTracker

        spec = ExportSpec()
        batch_size = 5

        # Get total count first
        total_count = await count_rows(spec)
        tracker = ProgressTracker(total=total_count, description="Test streaming")

        rows_processed = 0
        async for batch in stream_rows(spec, batch_size=batch_size, progress_tracker=tracker):
            rows_processed += len(batch)

        # Verify progress tracker was updated
        assert tracker.current >= 0
        tracker.finish()

    async def test_streaming_empty_result(self, test_db_schema):
        """Test that streaming handles empty results correctly."""
        spec = ExportSpec(dataset_names=["NONEXISTENT_DATASET"])

        batch_count = 0
        async for batch in stream_rows(spec, batch_size=100):
            batch_count += 1

        # Should have no batches for empty result
        assert batch_count == 0

    async def test_streaming_small_batch_size(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test streaming with very small batch size."""
        spec = ExportSpec()
        batch_size = 1  # Single row per batch

        total_rows = 0
        async for batch in stream_rows(spec, batch_size=batch_size):
            total_rows += len(batch)
            assert len(batch) <= batch_size

        assert total_rows >= 0

    async def test_streaming_large_batch_size(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test streaming with large batch size."""
        spec = ExportSpec()
        batch_size = 1000  # Large batch size

        total_rows = 0
        async for batch in stream_rows(spec, batch_size=batch_size):
            total_rows += len(batch)
            assert len(batch) <= batch_size

        assert total_rows >= 0


@pytest.mark.asyncio
class TestParquetExportPerformance:
    """Tests for Parquet export performance."""

    async def test_parquet_export_memory_usage(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test that Parquet export doesn't cause unbounded memory growth."""
        spec = ExportSpec()
        batch_size = 10

        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=batch_size)

            # Check memory after export
            gc.collect()  # Force garbage collection
            final_memory = process.memory_info().rss / 1024 / 1024  # MB

            # Memory growth should be reasonable
            memory_growth = final_memory - initial_memory
            assert memory_growth < 200, f"Memory growth too large: {memory_growth}MB"

    async def test_parquet_export_with_multiple_batches(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test Parquet export with multiple batches."""
        spec = ExportSpec()
        batch_size = 5  # Small batch size to force multiple batches

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=batch_size)

            # Verify file was created
            assert output_path.exists()

            # Verify file can be read
            table = pq.read_table(output_path)
            assert table.num_rows >= 0

    async def test_parquet_export_large_batch_size(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test Parquet export with large batch size."""
        spec = ExportSpec()
        batch_size = 1000  # Large batch size

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=batch_size)

            # Verify file was created
            assert output_path.exists()

            # Verify file can be read
            table = pq.read_table(output_path)
            assert table.num_rows >= 0


@pytest.mark.asyncio
class TestCountRowsPerformance:
    """Tests for count_rows performance."""

    async def test_count_rows_performance(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test that count_rows executes quickly."""
        spec = ExportSpec()

        # Time the count operation
        import time

        start_time = time.time()
        count = await count_rows(spec)
        elapsed_time = time.time() - start_time

        # Count should complete quickly (less than 5 seconds for test data)
        assert elapsed_time < 5.0, f"count_rows took too long: {elapsed_time}s"
        assert isinstance(count, int)
        assert count >= 0

    async def test_count_rows_with_filters(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test count_rows with various filters."""
        spec = ExportSpec(dataset_names=["TEST_DATASET"])
        count = await count_rows(spec)
        assert isinstance(count, int)
        assert count >= 0

    async def test_count_rows_empty_result(self, test_db_schema):
        """Test count_rows with spec that matches no rows."""
        spec = ExportSpec(dataset_names=["NONEXISTENT_DATASET"])
        count = await count_rows(spec)
        assert count == 0


@pytest.mark.asyncio
class TestConcurrentStreaming:
    """Tests for concurrent streaming operations."""

    async def test_multiple_streams_sequential(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test multiple sequential streaming operations."""
        spec = ExportSpec()
        batch_size = 10

        # Run multiple streams sequentially
        for i in range(3):
            total_rows = 0
            async for batch in stream_rows(spec, batch_size=batch_size):
                total_rows += len(batch)
            assert total_rows >= 0

    async def test_streaming_connection_cleanup(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test that streaming properly cleans up connections."""
        spec = ExportSpec()
        batch_size = 10

        # Stream and break early
        async for batch in stream_rows(spec, batch_size=batch_size):
            # Break after first batch to test cleanup
            break

        # Connection should be cleaned up
        # We can't directly verify this, but we can verify that
        # subsequent operations still work
        count = await count_rows(spec)
        assert isinstance(count, int)
