"""
Tests for chaksudb/ingest/framework/grading_scales.py

Tests grading scale registration, mapping, and disease grading storage functions based on their docstrings.
"""

import pytest
from uuid import UUID
from datetime import datetime

from chaksudb.ingest.framework.grading_scales import (
    register_grading_scale,
    find_mapping_to_standard_scale,
    find_scale_by_name,
    store_disease_grading,
    update_all_grades,
    create_mapping,
    STANDARD_SCALES,
)
from chaksudb.db.queries import (
    find_grading_scale_by_id,
    upsert_grading_scale_mapping,
    upsert_disease_grading,
)
from chaksudb.db.models import GradingScale, GradingScaleMapping, DiseaseGrading, Dataset, Image
from chaksudb.ingest.framework.gen_uuid import (
    generate_grading_scale_uuid,
    generate_dataset_uuid,
    generate_image_uuid,
)
from chaksudb.db.queries import upsert_dataset, upsert_image


# Note: These tests use the actual database functions which create their own connections.
# The db_connection fixture ensures the test database schema is set up.


async def insert_test_image(db_connection, dataset_id: UUID, original_image_id: str) -> UUID:
    """
    Helper function to insert a test image into the database.
    
    Args:
        db_connection: Database connection
        dataset_id: UUID of the dataset
        original_image_id: Original image identifier
        
    Returns:
        UUID of the inserted image
    """
    image_id = generate_image_uuid(dataset_id=dataset_id, original_image_id=original_image_id)
    image = Image(
        image_id=image_id,
        dataset_id=dataset_id,
        original_image_id=original_image_id,
        storage_provider="local",
        file_path=f"/test/path/{original_image_id}.jpg",
        modality="fundus",
    )
    await upsert_image(image)
    return image_id


class TestRegisterGradingScale:
    """Tests for register_grading_scale function."""

    @pytest.mark.asyncio
    async def test_register_grading_scale_returns_uuid(self, db_connection):
        """Test that register_grading_scale returns a UUID."""
        scale_id = await register_grading_scale(
            scale_name="ICDR",
            disease_type="DR",
        )
        assert isinstance(scale_id, UUID)

    @pytest.mark.asyncio
    async def test_register_grading_scale_creates_new_record(self, db_connection):
        """Test that register_grading_scale creates a new grading scale record."""
        scale_id = await register_grading_scale(
            scale_name="ICDR",
            disease_type="DR",
            scale_description="International Clinical Diabetic Retinopathy scale",
            min_value=0,
            max_value=4,
        )
        
        # Verify scale was created
        scale = await find_grading_scale_by_id(scale_id=scale_id)
        assert scale is not None
        assert scale.scale_name == "ICDR"
        assert scale.disease_type == "DR"
        assert scale.scale_description == "International Clinical Diabetic Retinopathy scale"
        assert scale.min_value == 0
        assert scale.max_value == 4

    @pytest.mark.asyncio
    async def test_register_grading_scale_with_value_labels(self, db_connection):
        """Test that register_grading_scale stores value_labels correctly."""
        value_labels = {
            "0": "No DR",
            "1": "Mild",
            "2": "Moderate",
            "3": "Severe",
            "4": "Proliferative"
        }
        
        scale_id = await register_grading_scale(
            scale_name="ICDR",
            disease_type="DR",
            value_labels=value_labels,
        )
        
        scale = await find_grading_scale_by_id(scale_id=scale_id)
        assert scale.value_labels == value_labels

    @pytest.mark.asyncio
    async def test_register_grading_scale_is_idempotent(self, db_connection):
        """Test that register_grading_scale is idempotent (can be called multiple times)."""
        scale_id1 = await register_grading_scale(
            scale_name="ETDRS",
            disease_type="DR",
        )
        
        scale_id2 = await register_grading_scale(
            scale_name="ETDRS",
            disease_type="DR",
        )
        
        # Should return same UUID
        assert scale_id1 == scale_id2

    @pytest.mark.asyncio
    async def test_register_grading_scale_deterministic_uuid(self, db_connection):
        """Test that register_grading_scale generates deterministic UUID based on name and disease type."""
        scale_id = await register_grading_scale(
            scale_name="AAO",
            disease_type="DR",
        )
        
        expected_id = generate_grading_scale_uuid(scale_name="AAO", disease_type="DR")
        assert scale_id == expected_id

    @pytest.mark.asyncio
    async def test_register_grading_scale_without_optional_fields(self, db_connection):
        """Test that register_grading_scale works with only required fields."""
        scale_id = await register_grading_scale(
            scale_name="CustomScale",
            disease_type="AMD",
        )
        
        scale = await find_grading_scale_by_id(scale_id=scale_id)
        assert scale.scale_name == "CustomScale"
        assert scale.disease_type == "AMD"
        assert scale.scale_description is None
        assert scale.min_value is None
        assert scale.max_value is None
        assert scale.value_labels is None


class TestFindMappingToStandardScale:
    """Tests for find_mapping_to_standard_scale function."""

    @pytest.mark.asyncio
    async def test_find_mapping_returns_none_when_no_mapping_exists(self, db_connection):
        """Test that find_mapping_to_standard_scale returns None when no mapping exists."""
        source_scale_id = await register_grading_scale(
            scale_name="CustomScale",
            disease_type="DR",
        )
        
        mapping = await find_mapping_to_standard_scale(
            source_scale_id=source_scale_id,
            source_value="CustomGrade1",
        )
        
        assert mapping is None

    @pytest.mark.asyncio
    async def test_find_mapping_returns_mapping_when_exists(self, db_connection):
        """Test that find_mapping_to_standard_scale returns mapping when it exists."""
        source_scale_id = await register_grading_scale(
            scale_name="CustomScale",
            disease_type="DR",
        )
        target_scale_id = await register_grading_scale(
            scale_name="ICDR",
            disease_type="DR",
        )
        
        # Create mapping
        await create_mapping(
            source_scale_id=source_scale_id,
            target_scale_id=target_scale_id,
            source_value="Mild",
            target_value=1,
        )
        
        # Find the mapping
        mapping = await find_mapping_to_standard_scale(
            source_scale_id=source_scale_id,
            source_value="Mild",
        )
        
        assert mapping is not None
        assert mapping.source_value == "Mild"
        assert mapping.target_value == 1

    @pytest.mark.asyncio
    async def test_find_mapping_with_target_scale_name(self, db_connection):
        """Test that find_mapping_to_standard_scale can filter by target_scale_name."""
        source_scale_id = await register_grading_scale(
            scale_name="CustomScale",
            disease_type="DR",
        )
        icdr_scale_id = await register_grading_scale(
            scale_name="ICDR",
            disease_type="DR",
        )
        
        await create_mapping(
            source_scale_id=source_scale_id,
            target_scale_id=icdr_scale_id,
            source_value="Mild",
            target_value=1,
        )
        
        # Find mapping to specific scale
        mapping = await find_mapping_to_standard_scale(
            source_scale_id=source_scale_id,
            source_value="Mild",
            target_scale_name="ICDR",
        )
        
        assert mapping is not None
        assert mapping.target_value == 1


class TestFindScaleByName:
    """Tests for find_scale_by_name function."""

    @pytest.mark.asyncio
    async def test_find_scale_by_name_returns_none_when_not_found(self, db_connection):
        """Test that find_scale_by_name returns None when scale doesn't exist."""
        scale_id = await find_scale_by_name(
            scale_name="NonExistentScale",
            disease_type="DR",
        )
        
        assert scale_id is None

    @pytest.mark.asyncio
    async def test_find_scale_by_name_returns_uuid_when_found(self, db_connection):
        """Test that find_scale_by_name returns UUID when scale exists."""
        # Register a scale first
        registered_id = await register_grading_scale(
            scale_name="ETDRS",
            disease_type="DR",
        )
        
        # Find it by name
        found_id = await find_scale_by_name(
            scale_name="ETDRS",
            disease_type="DR",
        )
        
        assert found_id == registered_id

    @pytest.mark.asyncio
    async def test_find_scale_by_name_is_deterministic(self, db_connection):
        """Test that find_scale_by_name uses deterministic UUID generation."""
        await register_grading_scale(
            scale_name="AAO",
            disease_type="DME",
        )
        
        found_id = await find_scale_by_name(
            scale_name="AAO",
            disease_type="DME",
        )
        
        expected_id = generate_grading_scale_uuid(scale_name="AAO", disease_type="DME")
        assert found_id == expected_id


class TestStoreDiseaseGrading:
    """Tests for store_disease_grading function."""

    @pytest.mark.asyncio
    async def test_store_disease_grading_returns_uuid(self, db_connection):
        """Test that store_disease_grading returns a UUID."""
        # Setup: create dataset, image, and scale
        dataset_id = generate_dataset_uuid(dataset_name="TestDataset")
        dataset = Dataset(dataset_id=dataset_id, dataset_name="TestDataset")
        await upsert_dataset(dataset)
        
        image_id = generate_image_uuid(dataset_id=dataset_id, original_image_id="img001")
        image = Image(
            image_id=image_id,
            dataset_id=dataset_id,
            original_image_id="img001",
            modality="fundus",
            file_path="/test/images/img001.jpg",
        )
        await upsert_image(image)
        
        scale_id = await register_grading_scale(
            scale_name="ICDR",
            disease_type="DR",
        )
        
        # Store grading
        grading_id = await store_disease_grading(
            image_id=image_id,
            disease_type="DR",
            scale_id=scale_id,
            original_grade="Mild",
        )
        
        assert isinstance(grading_id, UUID)

    @pytest.mark.asyncio
    async def test_store_disease_grading_without_mapping(self, db_connection):
        """Test that store_disease_grading stores only original_grade when no mapping exists."""
        # Setup
        dataset_id = generate_dataset_uuid(dataset_name="TestDataset2")
        dataset = Dataset(dataset_id=dataset_id, dataset_name="TestDataset2")
        await upsert_dataset(dataset)
        
        image_id = await insert_test_image(db_connection, dataset_id, "img002")
        
        scale_id = await register_grading_scale(
            scale_name="CustomScale",
            disease_type="DR",
        )
        
        # Store grading (no mapping exists)
        grading_id = await store_disease_grading(
            image_id=image_id,
            disease_type="DR",
            scale_id=scale_id,
            original_grade="CustomGrade",
        )
        
        # Verify: check that scaled_grade is None
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT original_grade, scaled_grade FROM disease_grading WHERE grading_id = %s",
                (grading_id,)
            )
            row = await cur.fetchone()
            assert row[0] == "CustomGrade"
            assert row[1] is None

    @pytest.mark.asyncio
    async def test_store_disease_grading_with_mapping(self, db_connection):
        """Test that store_disease_grading stores both original_grade and scaled_grade when mapping exists."""
        # Setup
        dataset_id = generate_dataset_uuid(dataset_name="TestDataset3")
        dataset = Dataset(dataset_id=dataset_id, dataset_name="TestDataset3")
        await upsert_dataset(dataset)
        
        image_id = await insert_test_image(db_connection, dataset_id, "img003")
        
        source_scale_id = await register_grading_scale(
            scale_name="CustomScale",
            disease_type="DR",
        )
        target_scale_id = await register_grading_scale(
            scale_name="ICDR",
            disease_type="DR",
        )
        
        # Create mapping
        await create_mapping(
            source_scale_id=source_scale_id,
            target_scale_id=target_scale_id,
            source_value="Mild",
            target_value=1,
        )
        
        # Store grading
        grading_id = await store_disease_grading(
            image_id=image_id,
            disease_type="DR",
            scale_id=source_scale_id,
            original_grade="Mild",
        )
        
        # Verify both original_grade and scaled_grade are stored
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT original_grade, scaled_grade FROM disease_grading WHERE grading_id = %s",
                (grading_id,)
            )
            row = await cur.fetchone()
            assert row[0] == "Mild"
            assert row[1] == 1

    @pytest.mark.asyncio
    async def test_store_disease_grading_is_idempotent(self, db_connection):
        """Test that store_disease_grading is idempotent (upsert)."""
        # Setup
        dataset_id = generate_dataset_uuid(dataset_name="TestDataset4")
        dataset = Dataset(dataset_id=dataset_id, dataset_name="TestDataset4")
        await upsert_dataset(dataset)
        
        image_id = await insert_test_image(db_connection, dataset_id, "img004")
        
        scale_id = await register_grading_scale(
            scale_name="ICDR",
            disease_type="DR",
        )
        
        # Store same grading twice
        grading_id1 = await store_disease_grading(
            image_id=image_id,
            disease_type="DR",
            scale_id=scale_id,
            original_grade="Moderate",
        )
        
        grading_id2 = await store_disease_grading(
            image_id=image_id,
            disease_type="DR",
            scale_id=scale_id,
            original_grade="Moderate",
        )
        
        assert grading_id1 == grading_id2


class TestUpdateAllGrades:
    """Tests for update_all_grades function."""

    @pytest.mark.asyncio
    async def test_update_all_grades_returns_counts(self, db_connection):
        """Test that update_all_grades returns tuple of (updated_count, total_checked_count)."""
        updated, total = await update_all_grades()
        
        assert isinstance(updated, int)
        assert isinstance(total, int)
        assert updated >= 0
        assert total >= 0

    @pytest.mark.asyncio
    async def test_update_all_grades_updates_grades_when_mapping_added(self, db_connection):
        """Test that update_all_grades updates existing grades when new mappings are added."""
        # Setup: create dataset, image, scales
        dataset_id = generate_dataset_uuid(dataset_name="TestDataset5")
        dataset = Dataset(dataset_id=dataset_id, dataset_name="TestDataset5")
        await upsert_dataset(dataset)
        
        image_id = await insert_test_image(db_connection, dataset_id, "img005")
        
        source_scale_id = await register_grading_scale(
            scale_name="CustomScale",
            disease_type="DR",
        )
        
        # Store grading without mapping
        grading_id = await store_disease_grading(
            image_id=image_id,
            disease_type="DR",
            scale_id=source_scale_id,
            original_grade="TestGrade",
        )
        
        # Verify no scaled_grade initially
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT scaled_grade FROM disease_grading WHERE grading_id = %s",
                (grading_id,)
            )
            row = await cur.fetchone()
            assert row[0] is None
        
        # Now add a mapping
        target_scale_id = await register_grading_scale(
            scale_name="ICDR",
            disease_type="DR",
        )
        await create_mapping(
            source_scale_id=source_scale_id,
            target_scale_id=target_scale_id,
            source_value="TestGrade",
            target_value=2,
        )
        
        # Update all grades
        updated, total = await update_all_grades()
        
        # Verify scaled_grade is now set
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT scaled_grade FROM disease_grading WHERE grading_id = %s",
                (grading_id,)
            )
            row = await cur.fetchone()
            assert row[0] == 2
        
        assert updated >= 1

    @pytest.mark.asyncio
    async def test_update_all_grades_handles_empty_database(self, db_connection):
        """Test that update_all_grades handles database gracefully when no updates are needed."""
        # Note: This test may run after other tests have inserted data,
        # so we just verify it completes successfully and returns valid counts
        updated, total = await update_all_grades()
        
        assert isinstance(updated, int)
        assert isinstance(total, int)
        assert updated >= 0
        assert total >= updated  # total should be >= updated


class TestCreateMapping:
    """Tests for create_mapping function."""

    @pytest.mark.asyncio
    async def test_create_mapping_returns_uuid(self, db_connection):
        """Test that create_mapping returns a UUID."""
        source_scale_id = await register_grading_scale(
            scale_name="Source",
            disease_type="DR",
        )
        target_scale_id = await register_grading_scale(
            scale_name="Target",
            disease_type="DR",
        )
        
        mapping_id = await create_mapping(
            source_scale_id=source_scale_id,
            target_scale_id=target_scale_id,
            source_value="Mild",
            target_value=1,
        )
        
        assert isinstance(mapping_id, UUID)

    @pytest.mark.asyncio
    async def test_create_mapping_stores_mapping_correctly(self, db_connection):
        """Test that create_mapping stores mapping with correct values."""
        source_scale_id = await register_grading_scale(
            scale_name="Source",
            disease_type="DR",
        )
        target_scale_id = await register_grading_scale(
            scale_name="Target",
            disease_type="DR",
        )
        
        mapping_id = await create_mapping(
            source_scale_id=source_scale_id,
            target_scale_id=target_scale_id,
            source_value="Severe",
            target_value=3,
            mapping_confidence="exact",
        )
        
        # Verify mapping was stored
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT source_value, target_value, mapping_confidence FROM grading_scale_mappings WHERE mapping_id = %s",
                (mapping_id,)
            )
            row = await cur.fetchone()
            assert row[0] == "Severe"
            assert row[1] == 3
            assert row[2] == "exact"

    @pytest.mark.asyncio
    async def test_create_mapping_is_idempotent(self, db_connection):
        """Test that create_mapping is idempotent (can be called multiple times)."""
        source_scale_id = await register_grading_scale(
            scale_name="Source",
            disease_type="DR",
        )
        target_scale_id = await register_grading_scale(
            scale_name="Target",
            disease_type="DR",
        )
        
        mapping_id1 = await create_mapping(
            source_scale_id=source_scale_id,
            target_scale_id=target_scale_id,
            source_value="Moderate",
            target_value=2,
        )
        
        mapping_id2 = await create_mapping(
            source_scale_id=source_scale_id,
            target_scale_id=target_scale_id,
            source_value="Moderate",
            target_value=2,
        )
        
        assert mapping_id1 == mapping_id2

    @pytest.mark.asyncio
    async def test_create_mapping_with_different_confidence_levels(self, db_connection):
        """Test that create_mapping supports different confidence levels."""
        source_scale_id = await register_grading_scale(
            scale_name="Source",
            disease_type="DR",
        )
        target_scale_id = await register_grading_scale(
            scale_name="Target",
            disease_type="DR",
        )
        
        for confidence in ["exact", "approximate", "manual_review_required"]:
            mapping_id = await create_mapping(
                source_scale_id=source_scale_id,
                target_scale_id=target_scale_id,
                source_value=f"Grade_{confidence}",
                target_value=1,
                mapping_confidence=confidence,
            )
            
            async with db_connection.cursor() as cur:
                await cur.execute(
                    "SELECT mapping_confidence FROM grading_scale_mappings WHERE mapping_id = %s",
                    (mapping_id,)
                )
                row = await cur.fetchone()
                assert row[0] == confidence

    @pytest.mark.asyncio
    async def test_create_mapping_with_none_target_value(self, db_connection):
        """Test that create_mapping can store None as target_value."""
        source_scale_id = await register_grading_scale(
            scale_name="Source",
            disease_type="DR",
        )
        target_scale_id = await register_grading_scale(
            scale_name="Target",
            disease_type="DR",
        )
        
        mapping_id = await create_mapping(
            source_scale_id=source_scale_id,
            target_scale_id=target_scale_id,
            source_value="Unmappable",
            target_value=None,
            mapping_confidence="manual_review_required",
        )
        
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT target_value FROM grading_scale_mappings WHERE mapping_id = %s",
                (mapping_id,)
            )
            row = await cur.fetchone()
            assert row[0] is None
