"""
High-level tests for the public export API.

Uses only the public API: ExportSpec and export(). Verifies parquet-only,
torch=dataset, torch=dataloader, and transform passed through.
"""

import tempfile
from pathlib import Path

import pytest
import pyarrow.parquet as pq

from chaksudb.export import ExportSpec, export

pytest.importorskip("torch")


class TestExportPublicAPI:
    """Tests for export() using only the public API."""

    def test_export_parquet_only_returns_path(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """export(spec, parquet_path=...) writes Parquet and returns the Path."""
        spec = ExportSpec()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.parquet"
            result = export(spec, parquet_path=out)
            assert result == out
            assert out.exists()
            table = pq.read_table(out)
            assert table.num_rows >= 0
            assert "image_id" in table.column_names

    def test_export_parquet_with_filters(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """export(spec, parquet_path=...) respects ExportSpec filters."""
        spec = ExportSpec(dataset_names=["TEST_DATASET"])
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.parquet"
            result = export(spec, parquet_path=out)
            assert result == out
            assert out.exists()
            table = pq.read_table(out)
            if table.num_rows > 0 and "dataset_name" in table.column_names:
                col = table.column("dataset_name")
                for i in range(table.num_rows):
                    assert col[i].as_py() == "TEST_DATASET"

    def test_export_torch_dataset_returns_query_dataset(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """export(spec, torch='dataset') returns a QueryDataset."""
        spec = ExportSpec()
        result = export(spec, torch="dataset")
        from chaksudb.export.torch_dataset import QueryDataset
        assert isinstance(result, QueryDataset)
        assert result.spec == spec
        assert len(result) >= 0

    def test_export_torch_dataloader_returns_dataloader(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """export(spec, torch='dataloader') returns a DataLoader."""
        import torch.utils.data
        spec = ExportSpec()
        result = export(spec, torch="dataloader", batch_size=4, num_workers=0)
        assert isinstance(result, torch.utils.data.DataLoader)
        assert result.batch_size == 4

    def test_export_torch_dataset_with_transform_passed_through(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """export(spec, torch='dataset', transform=fn) passes transform to QueryDataset."""
        spec = ExportSpec()
        def identity(sample):
            return sample
        result = export(spec, torch="dataset", transform=identity)
        assert result.transform is identity

    def test_export_torch_dataloader_with_transform_passed_through(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """export(spec, torch='dataloader', transform=fn) passes transform to dataset."""
        spec = ExportSpec()
        def identity(sample):
            return sample
        result = export(spec, torch="dataloader", transform=identity, batch_size=4, num_workers=0)
        assert result.dataset.transform is identity

    def test_export_parquet_then_torch_dataloader_uses_file(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """export(spec, parquet_path=..., torch='dataloader') writes Parquet then returns DataLoader from file."""
        import torch.utils.data
        spec = ExportSpec()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.parquet"
            result = export(
                spec,
                parquet_path=out,
                torch="dataloader",
                batch_size=4,
                num_workers=0,
            )
            assert out.exists()
            assert isinstance(result, torch.utils.data.DataLoader)
            from chaksudb.export.torch_dataset import ParquetDataset
            assert isinstance(result.dataset, ParquetDataset)
            assert result.dataset.parquet_path == out

    def test_export_parquet_then_torch_dataset_uses_file(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """export(spec, parquet_path=..., torch='dataset') writes Parquet then returns ParquetDataset."""
        spec = ExportSpec()
        with tempfile.TemporaryDirectory() as tmpdir:
            out = Path(tmpdir) / "out.parquet"
            result = export(spec, parquet_path=out, torch="dataset")
            assert out.exists()
            from chaksudb.export.torch_dataset import ParquetDataset
            assert isinstance(result, ParquetDataset)
            assert result.parquet_path == out

    def test_export_no_args_returns_none(self, test_db_schema):
        """export(spec) with no parquet_path or torch returns None."""
        spec = ExportSpec(dataset_names=["NONEXISTENT"])
        result = export(spec)
        assert result is None

    def test_export_invalid_torch_raises(self, test_db_schema, test_dataset_in_db, test_image_in_db):
        """export(spec, torch='invalid') raises ValueError."""
        spec = ExportSpec()
        with pytest.raises(ValueError, match='torch must be "dataset" or "dataloader"'):
            export(spec, torch="invalid")
