"""
Tests for image-related database operations.

Tests based on docstring specifications only.
"""

import pytest
from datetime import datetime
from uuid import UUID

from chaksudb.db.queries.images import (
    upsert_image_group,
    upsert_image,
    bulk_upsert_images,
    upsert_patient_image,
    bulk_upsert_patient_images,
)
from chaksudb.db.queries.datasets import upsert_dataset
from chaksudb.db.models import Dataset, ImageGroup, Image, PatientImage


@pytest.mark.asyncio
async def test_upsert_image_group_creates_new_record(db_connection, test_uuids):
    """Test that upsert_image_group creates a new image group record.
    
    Based on docstring: 'Upsert an image group record.'
    """
    # First create a dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["oct"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Create an image group
    group_id = UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")
    image_group = ImageGroup(
        group_id=group_id,
        dataset_id=test_uuids["dataset_1"],
        group_type="oct_volume",
        created_at=datetime.now(),
    )
    
    await upsert_image_group(image_group)
    
    # Verify image group was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT dataset_id, group_type FROM image_groups WHERE group_id = %s",
            (group_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == test_uuids["dataset_1"]
    assert result[1] == "oct_volume"


@pytest.mark.asyncio
async def test_upsert_image_creates_new_record(db_connection, test_uuids):
    """Test that upsert_image creates a new image record.
    
    Based on docstring: 'Upsert an image record.'
    """
    # First create a dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Create an image
    image = Image(
        image_id=test_uuids["image_1"],
        dataset_id=test_uuids["dataset_1"],
        original_image_id="IMG_001",
        storage_provider="local",
        file_path="/data/images/img001.jpg",
        file_format="jpg",
        modality="fundus",
        resolution_width=1024,
        resolution_height=768,
        eye_laterality="left",
        created_at=datetime.now(),
    )
    
    await upsert_image(image)
    
    # Verify image was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT original_image_id, file_path, modality FROM images WHERE image_id = %s",
            (test_uuids["image_1"],)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "IMG_001"
    assert result[1] == "/data/images/img001.jpg"
    assert result[2] == "fundus"


@pytest.mark.asyncio
async def test_upsert_image_updates_existing_record(db_connection, test_uuids):
    """Test that upsert_image updates an existing image record.
    
    Based on docstring: 'Upsert an image record.' (upsert implies insert or update)
    """
    # First create a dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Create an image
    image_v1 = Image(
        image_id=test_uuids["image_1"],
        dataset_id=test_uuids["dataset_1"],
        original_image_id="IMG_001",
        storage_provider="local",
        file_path="/data/images/img001_old.jpg",
        file_format="jpg",
        modality="fundus",
        resolution_width=1024,
        resolution_height=768,
        created_at=datetime.now(),
    )
    await upsert_image(image_v1)
    
    # Update the image
    image_v2 = Image(
        image_id=test_uuids["image_1"],
        dataset_id=test_uuids["dataset_1"],
        original_image_id="IMG_001_UPDATED",
        storage_provider="local",
        file_path="/data/images/img001_new.jpg",
        file_format="png",
        modality="fundus",
        resolution_width=2048,
        resolution_height=1536,
        created_at=datetime.now(),
    )
    await upsert_image(image_v2)
    
    # Verify image was updated
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT original_image_id, file_path, file_format, resolution_width FROM images WHERE image_id = %s",
            (test_uuids["image_1"],)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "IMG_001_UPDATED"
    assert result[1] == "/data/images/img001_new.jpg"
    assert result[2] == "png"
    assert result[3] == 2048


@pytest.mark.asyncio
async def test_bulk_upsert_images_creates_multiple_records(db_connection, test_uuids):
    """Test that bulk_upsert_images creates multiple image records.
    
    Based on docstring: 'Bulk upsert image records.'
    """
    # First create a dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Create multiple images
    images = [
        Image(
            image_id=test_uuids["image_1"],
            dataset_id=test_uuids["dataset_1"],
            original_image_id="IMG_001",
            storage_provider="local",
            file_path="/data/images/img001.jpg",
            file_format="jpg",
            modality="fundus",
            created_at=datetime.now(),
        ),
        Image(
            image_id=test_uuids["image_2"],
            dataset_id=test_uuids["dataset_1"],
            original_image_id="IMG_002",
            storage_provider="local",
            file_path="/data/images/img002.jpg",
            file_format="jpg",
            modality="fundus",
            created_at=datetime.now(),
        ),
    ]
    
    rows_inserted = await bulk_upsert_images(images)
    
    # Should return count of rows inserted
    assert rows_inserted == 2
    
    # Verify images were created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT COUNT(*) FROM images WHERE image_id IN (%s, %s)",
            (test_uuids["image_1"], test_uuids["image_2"])
        )
        count = await cur.fetchone()
        
    assert count[0] == 2


@pytest.mark.asyncio
async def test_bulk_upsert_images_with_custom_batch_size(db_connection, test_uuids):
    """Test that bulk_upsert_images respects custom batch_size parameter.
    
    Based on docstring signature showing batch_size parameter with default 1000.
    """
    # First create a dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Create multiple images with unique IDs (starting from IMG_100 to avoid conflicts)
    images = [
        Image(
            image_id=UUID(f"00000000-0000-0000-0000-{str(i+100).zfill(12)}"),
            dataset_id=test_uuids["dataset_1"],
            original_image_id=f"IMG_{i+100:03d}",
            storage_provider="local",
            file_path=f"/data/images/img{i+100:03d}.jpg",
            file_format="jpg",
            modality="fundus",
            created_at=datetime.now(),
        )
        for i in range(5)
    ]
    
    # Use small batch size to test batching
    rows_inserted = await bulk_upsert_images(images, batch_size=2)
    
    assert rows_inserted == 5


@pytest.mark.asyncio
async def test_upsert_patient_image_creates_new_record(db_connection, test_uuids):
    """Test that upsert_patient_image creates a new patient-image relationship record.
    
    Based on docstring: 'Upsert a patient-image relationship record.'
    """
    # First create a dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Create a patient
    from chaksudb.db.queries.patients import upsert_patient
    from chaksudb.db.models import Patient
    
    patient = Patient(
        patient_id=test_uuids["patient_1"],
        dataset_id=test_uuids["dataset_1"],
        original_patient_id="PAT_001",
        created_at=datetime.now(),
    )
    await upsert_patient(patient)
    
    # Create an image
    image = Image(
        image_id=test_uuids["image_1"],
        dataset_id=test_uuids["dataset_1"],
        original_image_id="IMG_001",
        storage_provider="local",
        file_path="/data/images/img001.jpg",
        file_format="jpg",
        modality="fundus",
        created_at=datetime.now(),
    )
    await upsert_image(image)
    
    # Create a patient-image relationship
    relationship_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    patient_image = PatientImage(
        relationship_id=relationship_id,
        patient_id=test_uuids["patient_1"],
        image_id=test_uuids["image_1"],
        exam_date=None,
        created_at=datetime.now(),
    )
    
    await upsert_patient_image(patient_image)
    
    # Verify patient-image relationship was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT patient_id, image_id FROM patient_images WHERE relationship_id = %s",
            (relationship_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == test_uuids["patient_1"]
    assert result[1] == test_uuids["image_1"]


@pytest.mark.asyncio
async def test_bulk_upsert_patient_images_creates_multiple_records(db_connection, test_uuids):
    """Test that bulk_upsert_patient_images creates multiple patient-image relationship records.
    
    Based on docstring: 'Bulk upsert patient-image relationship records.'
    """
    # First create a dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Create a patient
    from chaksudb.db.queries.patients import upsert_patient
    from chaksudb.db.models import Patient
    
    patient = Patient(
        patient_id=test_uuids["patient_1"],
        dataset_id=test_uuids["dataset_1"],
        original_patient_id="PAT_001",
        created_at=datetime.now(),
    )
    await upsert_patient(patient)
    
    # Create multiple images
    images = [
        Image(
            image_id=test_uuids["image_1"],
            dataset_id=test_uuids["dataset_1"],
            original_image_id="IMG_001",
            storage_provider="local",
            file_path="/data/images/img001.jpg",
            file_format="jpg",
            modality="fundus",
            created_at=datetime.now(),
        ),
        Image(
            image_id=test_uuids["image_2"],
            dataset_id=test_uuids["dataset_1"],
            original_image_id="IMG_002",
            storage_provider="local",
            file_path="/data/images/img002.jpg",
            file_format="jpg",
            modality="fundus",
            created_at=datetime.now(),
        ),
    ]
    await bulk_upsert_images(images)
    
    # Create multiple patient-image relationships
    patient_images = [
        PatientImage(
            relationship_id=UUID("cccccccc-cccc-cccc-cccc-000000000001"),
            patient_id=test_uuids["patient_1"],
            image_id=test_uuids["image_1"],
            exam_date=None,
            created_at=datetime.now(),
        ),
        PatientImage(
            relationship_id=UUID("cccccccc-cccc-cccc-cccc-000000000002"),
            patient_id=test_uuids["patient_1"],
            image_id=test_uuids["image_2"],
            exam_date=None,
            created_at=datetime.now(),
        ),
    ]
    
    rows_inserted = await bulk_upsert_patient_images(patient_images)
    
    # Should return count of rows inserted
    assert rows_inserted == 2
    
    # Verify patient-image relationships were created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT COUNT(*) FROM patient_images WHERE patient_id = %s",
            (test_uuids["patient_1"],)
        )
        count = await cur.fetchone()
        
    assert count[0] == 2
