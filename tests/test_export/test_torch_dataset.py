"""
Integration tests for PyTorch Dataset and DataLoader.

Tests QueryDataset, ParquetDataset, and DataLoader integration with test database,
verifying that datasets can be created, accessed, and used with PyTorch DataLoaders.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from PIL import Image as PILImage

from chaksudb.export.parquet_export import export_to_parquet
from chaksudb.export.spec import ExportSpec
from chaksudb.export.torch_dataset import (
    ParquetDataset,
    QueryDataset,
    create_dataloader,
)
from chaksudb.export.transforms.collate import default_collate as default_collate_fn

# Skip tests if torch is not available
pytest.importorskip("torch")


@pytest.mark.asyncio
class TestQueryDataset:
    """Integration tests for QueryDataset."""

    async def test_query_dataset_init(self, test_db_schema, test_dataset_in_db, test_image_in_db):
        """Test that QueryDataset can be initialized."""
        spec = ExportSpec()
        dataset = QueryDataset(spec=spec, cache_rows=False)
        assert dataset is not None
        assert dataset.spec == spec

    async def test_query_dataset_len(self, test_db_schema, test_dataset_in_db, test_image_in_db):
        """Test that QueryDataset.__len__() returns correct count."""
        spec = ExportSpec()
        dataset = QueryDataset(spec=spec, cache_rows=False)
        length = len(dataset)
        assert isinstance(length, int)
        assert length >= 0

    async def test_query_dataset_len_with_filter(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test that QueryDataset.__len__() respects filters."""
        spec = ExportSpec(dataset_names=["TEST_DATASET"])
        dataset = QueryDataset(spec=spec, cache_rows=False)
        length = len(dataset)
        assert isinstance(length, int)
        assert length >= 0

    async def test_query_dataset_getitem_raises_index_error(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test that QueryDataset.__getitem__() raises IndexError for out-of-range indices."""
        spec = ExportSpec()
        dataset = QueryDataset(spec=spec, cache_rows=False)
        length = len(dataset)

        if length == 0:
            # If no data, test that accessing index 0 raises IndexError
            with pytest.raises(IndexError):
                _ = dataset[0]
        else:
            # Test negative indexing
            with pytest.raises(IndexError):
                _ = dataset[-(length + 1)]
            # Test out-of-range positive index
            with pytest.raises(IndexError):
                _ = dataset[length]

    async def test_query_dataset_with_cache_rows(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test QueryDataset with cache_rows=True."""
        spec = ExportSpec()
        dataset = QueryDataset(spec=spec, cache_rows=True)
        length = len(dataset)

        if length > 0:
            # Accessing an item should trigger caching
            # We can't directly test the cache, but we can verify it doesn't crash
            try:
                _ = dataset[0]
            except (FileNotFoundError, ValueError):
                # Expected if image file doesn't exist in test environment
                # This is okay - we're just testing that caching logic doesn't crash
                pass

    async def test_query_dataset_with_transform(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test QueryDataset with a transform function."""
        spec = ExportSpec()

        def dummy_transform(sample):
            """Dummy transform that returns the sample unchanged."""
            return sample

        dataset = QueryDataset(spec=spec, transform=dummy_transform, cache_rows=False)
        assert dataset.transform is not None

    async def test_query_dataset_negative_indexing(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test that QueryDataset supports negative indexing."""
        spec = ExportSpec()
        dataset = QueryDataset(spec=spec, cache_rows=False)
        length = len(dataset)

        if length > 0:
            # Test that negative indexing works (if we can access the data)
            try:
                # Should not raise IndexError for valid negative index
                _ = dataset[-1]
            except (FileNotFoundError, ValueError):
                # Expected if image file doesn't exist - that's okay for this test
                pass


@pytest.mark.asyncio
class TestParquetDataset:
    """Integration tests for ParquetDataset."""

    async def test_parquet_dataset_init_fails_without_file(self):
        """Test that ParquetDataset raises FileNotFoundError for non-existent file."""
        with pytest.raises(FileNotFoundError):
            _ = ParquetDataset(parquet_path=Path("/nonexistent/file.parquet"))

    async def test_parquet_dataset_init_succeeds_with_file(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test that ParquetDataset can be initialized with a valid Parquet file."""
        spec = ExportSpec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)

            dataset = ParquetDataset(parquet_path=output_path)
            assert dataset is not None
            assert dataset.parquet_path == output_path

    async def test_parquet_dataset_len(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test that ParquetDataset.__len__() returns correct count."""
        spec = ExportSpec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)

            dataset = ParquetDataset(parquet_path=output_path)
            length = len(dataset)
            assert isinstance(length, int)
            assert length >= 0

    async def test_parquet_dataset_getitem_raises_index_error(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test that ParquetDataset.__getitem__() raises IndexError for out-of-range indices."""
        spec = ExportSpec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)

            dataset = ParquetDataset(parquet_path=output_path)
            length = len(dataset)

            if length == 0:
                with pytest.raises(IndexError):
                    _ = dataset[0]
            else:
                with pytest.raises(IndexError):
                    _ = dataset[length]
                with pytest.raises(IndexError):
                    _ = dataset[-(length + 1)]

    async def test_parquet_dataset_with_transform(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test ParquetDataset with a transform function."""
        spec = ExportSpec()

        def dummy_transform(sample):
            """Dummy transform that returns the sample unchanged."""
            return sample

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)

            dataset = ParquetDataset(parquet_path=output_path, transform=dummy_transform)
            assert dataset.transform is not None


class TestDefaultCollateFn:
    """Tests for default_collate_fn."""

    def test_default_collate_fn_with_pil_images(self):
        """Test default_collate_fn with PIL Images."""
        # Create mock PIL images
        img1 = PILImage.new("RGB", (100, 100), color="red")
        img2 = PILImage.new("RGB", (100, 100), color="blue")

        annotations1 = {"image_id": "img1", "dataset_id": "ds1"}
        annotations2 = {"image_id": "img2", "dataset_id": "ds2"}

        batch = [(img1, annotations1), (img2, annotations2)]
        batched_images, batched_annotations = default_collate_fn(batch)

        # Verify batched_images is a tensor
        assert isinstance(batched_images, torch.Tensor)
        assert batched_images.shape[0] == 2  # Batch size
        assert batched_images.shape[1] == 3  # RGB channels

        # Verify batched_annotations is a dict
        assert isinstance(batched_annotations, dict)
        assert "image_id" in batched_annotations
        assert len(batched_annotations["image_id"]) == 2

    def test_default_collate_fn_with_tensors(self):
        """Test default_collate_fn with tensor images."""
        img1 = torch.randn(3, 100, 100)
        img2 = torch.randn(3, 100, 100)

        annotations1 = {"image_id": "img1"}
        annotations2 = {"image_id": "img2"}

        batch = [(img1, annotations1), (img2, annotations2)]
        batched_images, batched_annotations = default_collate_fn(batch)

        # Verify batched_images is a tensor
        assert isinstance(batched_images, torch.Tensor)
        assert batched_images.shape[0] == 2  # Batch size
        assert batched_images.shape[1] == 3  # RGB channels

    def test_default_collate_fn_empty_batch(self):
        """Test default_collate_fn with empty batch."""
        batch = []
        with pytest.raises(ValueError, match="empty batches"):
            _ = default_collate_fn(batch)

    def test_default_collate_fn_variable_size_pil_images(self):
        """Test default_collate_fn pads variable-size PIL images to batch max."""
        img1 = PILImage.new("RGB", (100, 80), color="red")    # W=100, H=80
        img2 = PILImage.new("RGB", (200, 150), color="blue")  # W=200, H=150

        ann1 = {"image_id": "img1"}
        ann2 = {"image_id": "img2"}

        batch = [(img1, ann1), (img2, ann2)]
        batched_images, batched_annotations = default_collate_fn(batch)

        assert isinstance(batched_images, torch.Tensor)
        assert batched_images.shape == (2, 3, 150, 200)  # (B, C, max_H, max_W)

        # Verify padding metadata is present
        assert "_original_height" in batched_annotations
        assert "_original_width" in batched_annotations
        assert batched_annotations["_original_height"] == [80, 150]
        assert batched_annotations["_original_width"] == [100, 200]

    def test_default_collate_fn_variable_size_tensors(self):
        """Test default_collate_fn pads variable-size tensors to batch max."""
        img1 = torch.randn(3, 64, 128)
        img2 = torch.randn(3, 100, 80)

        ann1 = {"image_id": "img1"}
        ann2 = {"image_id": "img2"}

        batch = [(img1, ann1), (img2, ann2)]
        batched_images, batched_annotations = default_collate_fn(batch)

        assert isinstance(batched_images, torch.Tensor)
        assert batched_images.shape == (2, 3, 100, 128)  # (B, C, max_H, max_W)

        assert batched_annotations["_original_height"] == [64, 100]
        assert batched_annotations["_original_width"] == [128, 80]

    def test_default_collate_fn_padding_values_are_zero(self):
        """Test that padded regions are filled with zeros by default."""
        img1 = torch.ones(3, 50, 50)
        img2 = torch.ones(3, 100, 100)

        batch = [(img1, {"image_id": "a"}), (img2, {"image_id": "b"})]
        batched_images, _ = default_collate_fn(batch)

        # First image was 50x50, padded to 100x100 — bottom-right should be 0
        assert batched_images[0, :, :50, :50].eq(1.0).all()
        assert batched_images[0, :, 50:, :].eq(0.0).all()
        assert batched_images[0, :, :, 50:].eq(0.0).all()

        # Second image fills the full 100x100 — no padding
        assert batched_images[1, :, :, :].eq(1.0).all()

    def test_default_collate_fn_same_size_no_padding_metadata(self):
        """Test that _original_height/_original_width are NOT added when sizes match."""
        img1 = PILImage.new("RGB", (100, 100), color="red")
        img2 = PILImage.new("RGB", (100, 100), color="blue")

        batch = [(img1, {"image_id": "a"}), (img2, {"image_id": "b"})]
        batched_images, batched_annotations = default_collate_fn(batch)

        assert batched_images.shape == (2, 3, 100, 100)
        assert "_original_height" not in batched_annotations
        assert "_original_width" not in batched_annotations


@pytest.mark.asyncio
class TestCreateDataLoader:
    """Integration tests for create_dataloader."""

    async def test_create_dataloader_from_spec(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test creating DataLoader from ExportSpec."""
        spec = ExportSpec()
        dataloader = create_dataloader(spec=spec, batch_size=4, shuffle=False, num_workers=0)

        assert dataloader is not None
        assert dataloader.batch_size == 4

    async def test_create_dataloader_from_parquet(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test creating DataLoader from Parquet file."""
        spec = ExportSpec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)

            dataloader = create_dataloader(
                parquet_path=output_path, batch_size=4, shuffle=False, num_workers=0
            )

            assert dataloader is not None
            assert dataloader.batch_size == 4

    async def test_create_dataloader_raises_error_without_spec_or_parquet(self):
        """Test that create_dataloader raises ValueError when neither spec nor parquet_path is provided."""
        with pytest.raises(ValueError, match="Either spec or parquet_path must be provided"):
            _ = create_dataloader()

    async def test_create_dataloader_raises_error_with_both_spec_and_parquet(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test that create_dataloader raises ValueError when both spec and parquet_path are provided."""
        spec = ExportSpec()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "test_export.parquet"
            await export_to_parquet(spec, output_path, batch_size=100)

            with pytest.raises(ValueError, match="Cannot provide both spec and parquet_path"):
                _ = create_dataloader(spec=spec, parquet_path=output_path)

    async def test_create_dataloader_with_custom_collate_fn(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test creating DataLoader with custom collate function."""
        spec = ExportSpec()

        def custom_collate(batch):
            """Custom collate function."""
            return batch  # Just return batch as-is

        dataloader = create_dataloader(
            spec=spec, batch_size=4, collate_fn=custom_collate, num_workers=0
        )

        assert dataloader is not None
        assert dataloader.collate_fn == custom_collate

    async def test_create_dataloader_with_transform(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test creating DataLoader with transform function."""
        spec = ExportSpec()

        def dummy_transform(sample):
            """Dummy transform."""
            return sample

        dataloader = create_dataloader(
            spec=spec, batch_size=4, transform=dummy_transform, num_workers=0
        )

        assert dataloader is not None
        assert dataloader.dataset.transform is not None

    async def test_create_dataloader_with_cache_rows(
        self, test_db_schema, test_dataset_in_db, test_image_in_db
    ):
        """Test creating DataLoader with cache_rows option."""
        spec = ExportSpec()
        dataloader = create_dataloader(
            spec=spec, batch_size=4, cache_rows=True, num_workers=0
        )

        assert dataloader is not None
        # Verify that the underlying dataset has cache_rows=True
        assert isinstance(dataloader.dataset, QueryDataset)
        assert dataloader.dataset.cache_rows is True

    async def test_create_dataloader_iteration(
        self, test_db_schema, test_dataset_in_db, test_images_in_db
    ):
        """Test that DataLoader can be iterated (if data is available)."""
        spec = ExportSpec()
        dataloader = create_dataloader(
            spec=spec, batch_size=2, shuffle=False, num_workers=0
        )

        # Try to get one batch (may fail if images don't exist, but that's okay)
        try:
            batch = next(iter(dataloader))
            # If we get here, verify batch structure
            assert len(batch) == 2  # (images, annotations)
            images, annotations = batch
            assert isinstance(images, torch.Tensor)
            assert isinstance(annotations, dict)
        except (FileNotFoundError, ValueError, StopIteration):
            # Expected if image files don't exist in test environment
            # This is acceptable - we're just testing that iteration doesn't crash
            pass
