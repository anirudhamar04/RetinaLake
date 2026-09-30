"""
Tests for grading processor functionality.

These tests validate:
- process_disease_grade() handles various input formats
- get_or_create_scale() is idempotent
- Auto-registration works correctly
- Validation catches invalid inputs
- UUIDs are deterministic
"""

import uuid

import pytest

from chaksudb.db.models import DiseaseGrading
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_disease_grading_uuid,
    generate_grading_scale_uuid,
    generate_image_uuid,
)
from chaksudb.ingest.framework.task_processors.grading_processor import (
    get_or_create_scale,
    process_disease_grade,
)


@pytest.mark.asyncio
class TestGradingProcessor:
    """Test suite for grading processor."""

    async def test_process_disease_grade_basic(self):
        """Test basic processing of a disease grade."""
        dataset_id = generate_dataset_uuid("TestDataset")
        image_id = generate_image_uuid(dataset_id, "test_image.jpg")

        grading = await process_disease_grade(
            grade_value=2,
            disease_type="DR",
            scale_name="TestScale",
            image_id=image_id,
        )

        assert isinstance(grading, DiseaseGrading)
        assert grading.image_id == image_id
        assert grading.disease_type == "DR"
        assert grading.original_grade == "2"
        assert grading.scaled_grade is None  # Trigger will set this
        assert grading.annotation_method == "manual"

    async def test_process_various_input_formats(self):
        """Test processing grades in various formats."""
        dataset_id = generate_dataset_uuid("TestDataset")

        test_cases = [
            (0, "0"),
            (4, "4"),
            ("2", "2"),
            (3.0, "3"),
            (2.5, "2.5"),
            ("  1  ", "1"),  # Whitespace stripped
        ]

        for input_value, expected_output in test_cases:
            image_id = generate_image_uuid(dataset_id, f"image_{input_value}.jpg")

            grading = await process_disease_grade(
                grade_value=input_value,
                disease_type="DR",
                scale_name="TestScale",
                image_id=image_id,
            )

            assert grading.original_grade == expected_output

    async def test_process_with_metadata(self):
        """Test processing with optional metadata."""
        dataset_id = generate_dataset_uuid("TestDataset")
        image_id = generate_image_uuid(dataset_id, "test.jpg")
        raw_data_id = uuid.uuid4()

        grading = await process_disease_grade(
            grade_value=2,
            disease_type="DR",
            scale_name="TestScale",
            image_id=image_id,
            scale_description="Test scale",
            min_value=0,
            max_value=4,
            value_labels={"0": "None", "1": "Mild", "2": "Moderate"},
            grade_label="Moderate NPDR",
            raw_data_id=raw_data_id,
            annotation_method="manual",
            confidence_score=0.95,
        )

        assert grading.grade_label == "Moderate NPDR"
        assert grading.raw_data_id == raw_data_id
        assert grading.annotation_method == "manual"
        assert grading.confidence_score == 0.95

    async def test_invalid_disease_type(self):
        """Test that invalid disease type raises error."""
        dataset_id = generate_dataset_uuid("TestDataset")
        image_id = generate_image_uuid(dataset_id, "test.jpg")

        with pytest.raises(ValueError, match="Invalid disease_type"):
            await process_disease_grade(
                grade_value=2,
                disease_type="InvalidDisease",
                scale_name="TestScale",
                image_id=image_id,
            )

    async def test_invalid_annotation_method(self):
        """Test that invalid annotation method raises error."""
        dataset_id = generate_dataset_uuid("TestDataset")
        image_id = generate_image_uuid(dataset_id, "test.jpg")

        with pytest.raises(ValueError, match="Invalid annotation_method"):
            await process_disease_grade(
                grade_value=2,
                disease_type="DR",
                scale_name="TestScale",
                image_id=image_id,
                annotation_method="invalid_method",
            )

    async def test_deterministic_uuids(self):
        """Test that UUIDs are deterministic."""
        dataset_id = generate_dataset_uuid("TestDataset")
        image_id = generate_image_uuid(dataset_id, "test.jpg")

        # Process the same grade twice
        grading1 = await process_disease_grade(
            grade_value=2,
            disease_type="DR",
            scale_name="TestScale",
            image_id=image_id,
        )

        grading2 = await process_disease_grade(
            grade_value=2,
            disease_type="DR",
            scale_name="TestScale",
            image_id=image_id,
        )

        # Should generate identical UUIDs (idempotency)
        assert grading1.grading_id == grading2.grading_id
        assert grading1.scale_id == grading2.scale_id

    async def test_get_or_create_scale_idempotent(self):
        """Test that get_or_create_scale is idempotent."""
        scale_id1 = await get_or_create_scale(
            scale_name="TestScale",
            disease_type="DR",
            scale_description="Test scale",
        )

        scale_id2 = await get_or_create_scale(
            scale_name="TestScale",
            disease_type="DR",
            scale_description="Test scale",
        )

        # Should return same UUID
        assert scale_id1 == scale_id2

        # Should match expected UUID
        expected_id = generate_grading_scale_uuid("TestScale", "DR")
        assert scale_id1 == expected_id

    async def test_different_diseases_different_scales(self):
        """Test that same scale name but different disease creates different scales."""
        scale_id_dr = await get_or_create_scale(
            scale_name="TestScale",
            disease_type="DR",
        )

        scale_id_dme = await get_or_create_scale(
            scale_name="TestScale",
            disease_type="DME",
        )

        # Should be different UUIDs
        assert scale_id_dr != scale_id_dme

    async def test_process_grade_with_expert_annotation(self):
        """Test processing with expert annotation ID."""
        dataset_id = generate_dataset_uuid("TestDataset")
        image_id = generate_image_uuid(dataset_id, "test.jpg")
        expert_annotation_id = uuid.uuid4()

        grading = await process_disease_grade(
            grade_value=2,
            disease_type="DR",
            scale_name="TestScale",
            image_id=image_id,
            expert_annotation_id=expert_annotation_id,
        )

        assert grading.expert_annotation_id == expert_annotation_id

        # UUID should include expert_annotation_id for uniqueness
        expected_id = generate_disease_grading_uuid(
            image_id=image_id,
            disease_type="DR",
            scale_id=grading.scale_id,
            expert_annotation_id=expert_annotation_id,
            original_grade="2",
        )
        assert grading.grading_id == expected_id

    async def test_process_grade_with_consensus(self):
        """Test processing with consensus ID."""
        dataset_id = generate_dataset_uuid("TestDataset")
        image_id = generate_image_uuid(dataset_id, "test.jpg")
        consensus_id = uuid.uuid4()

        grading = await process_disease_grade(
            grade_value=3,
            disease_type="DR",
            scale_name="TestScale",
            image_id=image_id,
            consensus_id=consensus_id,
            annotation_method="consensus",
        )

        assert grading.consensus_id == consensus_id
        assert grading.annotation_method == "consensus"

    async def test_all_disease_types(self):
        """Test processing grades for all valid disease types."""
        dataset_id = generate_dataset_uuid("TestDataset")
        disease_types = ["DR", "DME", "Glaucoma", "AMD"]

        for disease_type in disease_types:
            image_id = generate_image_uuid(dataset_id, f"image_{disease_type}.jpg")

            grading = await process_disease_grade(
                grade_value=1,
                disease_type=disease_type,
                scale_name="TestScale",
                image_id=image_id,
            )

            assert grading.disease_type == disease_type

    async def test_all_annotation_methods(self):
        """Test processing grades with all valid annotation methods."""
        dataset_id = generate_dataset_uuid("TestDataset")
        annotation_methods = ["manual", "adjudicated", "consensus", "pseudo"]

        for method in annotation_methods:
            image_id = generate_image_uuid(dataset_id, f"image_{method}.jpg")

            grading = await process_disease_grade(
                grade_value=2,
                disease_type="DR",
                scale_name="TestScale",
                image_id=image_id,
                annotation_method=method,
            )

            assert grading.annotation_method == method


@pytest.mark.asyncio
class TestGradingProcessorIntegration:
    """Integration tests requiring database connection."""

    async def test_auto_registration_creates_scale(self):
        """Test that unknown scale is auto-registered in database."""
        # This test would require database connection
        # Skipping for now - implement when database is available
        pytest.skip("Requires database connection")

    async def test_upsert_and_trigger_conversion(self):
        """Test that triggers convert grades on upsert."""
        # This test would require:
        # 1. Bootstrap mappings
        # 2. Apply triggers
        # 3. Upsert grade
        # 4. Verify scaled_grade is populated
        pytest.skip("Requires database connection and triggers applied")

    async def test_backfill_on_new_mapping(self):
        """Test that backfill trigger updates historical grades."""
        # This test would require:
        # 1. Insert grades with unknown scale (scaled_grade = NULL)
        # 2. Add mapping
        # 3. Verify grades are backfilled
        pytest.skip("Requires database connection and triggers applied")
