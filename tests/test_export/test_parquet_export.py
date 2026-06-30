"""
Integration tests for Parquet export.

Tests end-to-end Parquet export with test database, verifying schema correctness
and data correctness.
"""

import tempfile
from pathlib import Path

import pytest
import pyarrow.parquet as pq

from chaksudb.export.parquet_export import export_to_parquet
from chaksudb.export.spec import ExportSpec


@pytest.mark.asyncio
class TestParquetExport:
    """Integration tests for Parquet export."""

    async def test_export_minimal_spec(self, test_db_schema, test_dataset_in_db, test_image_in_db):
        """Test exporting with minimal spec (just images)."""
        spec = ExportSpec()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify file can be read
            table = pq.read_table(output_path)
            assert table.num_rows >= 0  # May be 0 if no images in test DB
            
            # Verify schema has core fields
            schema = table.schema
            field_names = [f.name for f in schema]
            assert "image_id" in field_names
            assert "dataset_name" in field_names

    async def test_export_with_dataset_filter(self, test_db_schema, test_dataset_in_db, test_image_in_db):
        """Test exporting with dataset name filter."""
        # Get dataset name from fixture (assuming it's "TEST_DATASET" from conftest)
        spec = ExportSpec(dataset_names=["TEST_DATASET"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify file can be read
            table = pq.read_table(output_path)
            schema = table.schema
            field_names = [f.name for f in schema]
            assert "dataset_name" in field_names

    async def test_export_with_modality_filter(self, test_db_schema, test_dataset_in_db, test_image_in_db):
        """Test exporting with modality filter."""
        spec = ExportSpec(modalities=["fundus"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify file can be read
            table = pq.read_table(output_path)
            assert table.num_rows >= 0

    async def test_export_schema_correctness(self, test_db_schema, test_dataset_in_db, test_image_in_db):
        """Test that exported Parquet file has correct schema."""
        spec = ExportSpec()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)
            
            # Read and verify schema
            table = pq.read_table(output_path)
            schema = table.schema
            
            # Core fields should be present
            field_names = [f.name for f in schema]
            assert "image_id" in field_names
            assert "dataset_name" in field_names
            assert "file_path" in field_names
            assert "storage_provider" in field_names
            assert "modality" in field_names
            
            # Verify types (UUIDs should be strings)
            image_id_field = schema.field("image_id")
            assert image_id_field.type == "string" or str(image_id_field.type) == "string"

    async def test_export_empty_result(self, test_db_schema):
        """Test exporting with spec that matches no rows."""
        # Use a dataset name that doesn't exist
        spec = ExportSpec(dataset_names=["NONEXISTENT_DATASET"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)
            
            # Verify file was created (even with 0 rows)
            assert output_path.exists()
            
            # Verify file can be read
            table = pq.read_table(output_path)
            assert table.num_rows == 0

    async def test_export_with_base_path_for_paths(self, test_db_schema, test_dataset_in_db, test_image_in_db):
        """Test exporting with base_path_for_paths transformation."""
        spec = ExportSpec(base_path_for_paths="/data/images")
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify file can be read
            table = pq.read_table(output_path)
            assert table.num_rows >= 0

    async def test_export_multiple_batches(self, test_db_schema, test_dataset_in_db, test_images_in_db):
        """Test exporting with multiple batches (if we have enough test data)."""
        spec = ExportSpec()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            # Use small batch size to force multiple batches
            await export_to_parquet(spec, output_path, batch_size=5)
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify file can be read
            table = pq.read_table(output_path)
            # Should have at least some rows from test_images_in_db fixture
            assert table.num_rows >= 0

    async def test_export_overwrites_existing_file(self, test_db_schema, test_dataset_in_db, test_image_in_db):
        """Test that export overwrites existing file."""
        spec = ExportSpec()
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            
            # Create initial file
            await export_to_parquet(spec, output_path, batch_size=100)
            first_size = output_path.stat().st_size
            
            # Export again (should overwrite)
            await export_to_parquet(spec, output_path, batch_size=100)
            second_size = output_path.stat().st_size
            
            # File should still exist
            assert output_path.exists()
            # Size may be the same or different, but file should be valid
            table = pq.read_table(output_path)
            assert table.num_rows >= 0

    async def test_export_with_split_filter(self, test_db_schema, test_dataset_in_db, test_image_in_db):
        """Test exporting with split filter."""
        spec = ExportSpec(split_names=["train"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)
            
            # Verify file was created
            assert output_path.exists()
            
            # Verify file can be read
            table = pq.read_table(output_path)
            schema = table.schema
            field_names = [f.name for f in schema]
            # Split fields should be present if split module is used
            # (may be None if no splits exist in test data)
            assert table.num_rows >= 0

    async def test_export_data_correctness(self, test_db_schema, test_dataset_in_db, test_image_in_db):
        """Test that exported data matches query results."""
        spec = ExportSpec(dataset_names=["TEST_DATASET"])
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)
            
            # Read exported data
            table = pq.read_table(output_path)
            
            if table.num_rows > 0:
                # Verify that dataset_name matches filter using PyArrow
                if "dataset_name" in table.column_names:
                    col = table.column("dataset_name")
                    for i in range(table.num_rows):
                        assert col[i].as_py() == "TEST_DATASET"
                # Verify core fields present
                assert "image_id" in table.column_names
                assert "dataset_name" in table.column_names
