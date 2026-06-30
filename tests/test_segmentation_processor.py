"""
Tests for segmentation processor.

Tests the segmentation processor functions that wrap mask_converter
and prepare SegmentationAnnotation models for database upsert.
"""

import uuid
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from chaksudb.db.models import SegmentationAnnotation
from chaksudb.ingest.framework.gen_uuid import generate_dataset_uuid, generate_image_uuid
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    get_or_create_annotation_type,
    process_segmentation_from_binary_mask,
    process_segmentation_from_multiclass_mask,
)


@pytest.mark.asyncio
class TestSegmentationProcessor:
    """Test suite for segmentation processor."""

    async def test_get_or_create_annotation_type(self, tmp_path):
        """Test that annotation type can be created and retrieved."""
        annotation_type_id = await get_or_create_annotation_type(
            annotation_type="optic_disc",
            annotation_description="Optic disc segmentation",
        )

        assert isinstance(annotation_type_id, uuid.UUID)

        # Should return same UUID on second call (idempotent)
        annotation_type_id2 = await get_or_create_annotation_type(
            annotation_type="optic_disc",
            annotation_description="Optic disc segmentation",
        )

        assert annotation_type_id == annotation_type_id2

    async def test_process_binary_mask(self, tmp_path):
        """Test processing a binary mask."""
        # Create a test binary mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255  # White square in center

        mask_path = tmp_path / "test_mask.png"
        PILImage.fromarray(mask).save(mask_path)

        # Generate test IDs
        dataset_id = generate_dataset_uuid("TestDataset")
        image_id = generate_image_uuid(dataset_id, "test_image.jpg")

        # Process the mask
        annotation = await process_segmentation_from_binary_mask(
            mask_path=mask_path,
            annotation_type="optic_disc",
            image_id=image_id,
            annotation_description="Optic disc segmentation",
        )

        # Verify the annotation
        assert isinstance(annotation, SegmentationAnnotation)
        assert annotation.image_id == image_id
        assert annotation.unified_format == "binary_mask"
        assert annotation.original_format == "png"
        assert annotation.annotation_method == "manual"
        assert annotation.coordinate_system == "pixel"
        # Path may be full path (when under data root) or fallback "external/<filename>" when outside roots
        assert mask_path.name in annotation.mask_file_path

    async def test_process_binary_mask_with_metadata(self, tmp_path):
        """Test processing a binary mask with additional metadata."""
        # Create a test binary mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255

        mask_path = tmp_path / "test_mask.png"
        PILImage.fromarray(mask).save(mask_path)

        # Generate test IDs
        dataset_id = generate_dataset_uuid("TestDataset")
        image_id = generate_image_uuid(dataset_id, "test_image.jpg")

        # Process with metadata
        annotation = await process_segmentation_from_binary_mask(
            mask_path=mask_path,
            annotation_type="microaneurysms",
            image_id=image_id,
            lesion_subtype="microaneurysms",
            annotation_method="semi_automatic",
            confidence_score=0.95,
        )

        assert annotation.lesion_subtype == "microaneurysms"
        assert annotation.annotation_method == "semi_automatic"
        assert annotation.confidence_score == 0.95

    async def test_process_multiclass_mask(self, tmp_path):
        """Test processing a multi-class mask."""
        # Create a test multi-class mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 20:50] = 1  # Class 1: optic disc
        mask[30:40, 30:40] = 2  # Class 2: optic cup

        mask_path = tmp_path / "test_multiclass_mask.png"
        PILImage.fromarray(mask).save(mask_path)

        # Generate test IDs
        dataset_id = generate_dataset_uuid("TestDataset")
        image_id = generate_image_uuid(dataset_id, "test_image.jpg")

        # Process the multi-class mask
        annotations = await process_segmentation_from_multiclass_mask(
            mask_path=mask_path,
            class_names={1: "optic_disc", 2: "optic_cup"},
            image_id=image_id,
        )

        # Verify we got two annotations
        assert len(annotations) == 2
        assert all(isinstance(ann, SegmentationAnnotation) for ann in annotations)

        # Check annotation types
        lesion_subtypes = {ann.lesion_subtype for ann in annotations}
        assert lesion_subtypes == {"optic_disc", "optic_cup"}

    async def test_invalid_annotation_method(self, tmp_path):
        """Test that invalid annotation method raises error."""
        # Create a test binary mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255

        mask_path = tmp_path / "test_mask.png"
        PILImage.fromarray(mask).save(mask_path)

        # Generate test IDs
        dataset_id = generate_dataset_uuid("TestDataset")
        image_id = generate_image_uuid(dataset_id, "test_image.jpg")

        # Try with invalid annotation method
        with pytest.raises(ValueError, match="Invalid annotation_method"):
            await process_segmentation_from_binary_mask(
                mask_path=mask_path,
                annotation_type="optic_disc",
                image_id=image_id,
                annotation_method="invalid_method",
            )

    async def test_deterministic_uuids(self, tmp_path):
        """Test that UUIDs are deterministic."""
        # Create a test binary mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[25:75, 25:75] = 255

        mask_path = tmp_path / "test_mask.png"
        PILImage.fromarray(mask).save(mask_path)

        # Generate test IDs
        dataset_id = generate_dataset_uuid("TestDataset")
        image_id = generate_image_uuid(dataset_id, "test_image.jpg")

        # Process the same mask twice
        annotation1 = await process_segmentation_from_binary_mask(
            mask_path=mask_path,
            annotation_type="optic_disc",
            image_id=image_id,
        )

        annotation2 = await process_segmentation_from_binary_mask(
            mask_path=mask_path,
            annotation_type="optic_disc",
            image_id=image_id,
        )

        # Should generate identical UUIDs (idempotency)
        assert annotation1.segmentation_id == annotation2.segmentation_id
        assert annotation1.annotation_type_id == annotation2.annotation_type_id

    async def test_extract_class_from_multiclass(self, tmp_path):
        """Test extracting a specific class from multi-class mask."""
        # Create a test multi-class mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[20:50, 20:50] = 1  # Class 1: optic disc
        mask[30:40, 30:40] = 2  # Class 2: optic cup

        mask_path = tmp_path / "test_multiclass_mask.png"
        PILImage.fromarray(mask).save(mask_path)

        # Generate test IDs
        dataset_id = generate_dataset_uuid("TestDataset")
        image_id = generate_image_uuid(dataset_id, "test_image.jpg")

        # Extract only class 1
        annotation = await process_segmentation_from_binary_mask(
            mask_path=mask_path,
            annotation_type="optic_disc",
            image_id=image_id,
            extract_class=1,
        )

        assert isinstance(annotation, SegmentationAnnotation)
        assert annotation.unified_format == "binary_mask"
