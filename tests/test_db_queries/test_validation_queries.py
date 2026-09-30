"""
Tests for foreign key validation and utility query functions.

Tests based on docstring specifications only.
"""

import pytest
from datetime import datetime
from uuid import UUID

from chaksudb.db.queries.validation import (
    validate_dataset_exists,
    validate_image_exists,
    validate_patient_exists,
)
from chaksudb.db.queries.datasets import upsert_dataset
from chaksudb.db.queries.images import upsert_image
from chaksudb.db.models import Dataset, Image


@pytest.mark.asyncio
async def test_validate_dataset_exists_returns_true_when_dataset_exists(db_connection, test_uuids):
    """Test that validate_dataset_exists returns True when dataset exists.
    
    Based on docstring: 'Check if a dataset exists.'
    Args: dataset_id (UUID)
    Returns: bool
    """
    # Create a dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Validate dataset exists
    exists = await validate_dataset_exists(test_uuids["dataset_1"])
    
    assert exists is True


@pytest.mark.asyncio
async def test_validate_dataset_exists_returns_false_when_dataset_not_exists(db_connection, test_uuids):
    """Test that validate_dataset_exists returns False when dataset does not exist.
    
    Based on docstring: 'Check if a dataset exists.'
    Returns: bool
    """
    non_existent_id = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    
    # Validate dataset does not exist
    exists = await validate_dataset_exists(non_existent_id)
    
    assert exists is False


@pytest.mark.asyncio
async def test_validate_image_exists_returns_true_when_image_exists(db_connection, test_uuids):
    """Test that validate_image_exists returns True when image exists.
    
    Based on docstring: 'Check if an image exists.'
    Args: image_id (UUID)
    Returns: bool
    """
    # Create a dataset and image
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
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
    
    # Validate image exists
    exists = await validate_image_exists(test_uuids["image_1"])
    
    assert exists is True


@pytest.mark.asyncio
async def test_validate_image_exists_returns_false_when_image_not_exists(db_connection, test_uuids):
    """Test that validate_image_exists returns False when image does not exist.
    
    Based on docstring: 'Check if an image exists.'
    Returns: bool
    """
    non_existent_id = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    
    # Validate image does not exist
    exists = await validate_image_exists(non_existent_id)
    
    assert exists is False


@pytest.mark.asyncio
async def test_validate_patient_exists_returns_true_when_patient_exists(db_connection, test_uuids):
    """Test that validate_patient_exists returns True when patient exists.
    
    Based on docstring: 'Check if a patient exists.'
    Args: patient_id (UUID)
    Returns: bool
    """
    # Create a dataset and patient
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
    
    # Validate patient exists
    exists = await validate_patient_exists(test_uuids["patient_1"])
    
    assert exists is True


@pytest.mark.asyncio
async def test_validate_patient_exists_returns_false_when_patient_not_exists(db_connection, test_uuids):
    """Test that validate_patient_exists returns False when patient does not exist.
    
    Based on docstring: 'Check if a patient exists.'
    Returns: bool
    """
    non_existent_id = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    
    # Validate patient does not exist
    exists = await validate_patient_exists(non_existent_id)
    
    assert exists is False


@pytest.mark.asyncio
async def test_validation_functions_are_independent(db_connection, test_uuids):
    """Test that validation functions check only their specific entity.
    
    Based on docstrings: Each function checks a specific entity type.
    """
    # Create only a dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Dataset should exist
    dataset_exists = await validate_dataset_exists(test_uuids["dataset_1"])
    assert dataset_exists is True
    
    # Image and patient should not exist (using same UUID for testing)
    image_exists = await validate_image_exists(test_uuids["dataset_1"])
    assert image_exists is False
    
    patient_exists = await validate_patient_exists(test_uuids["dataset_1"])
    assert patient_exists is False


@pytest.mark.asyncio
async def test_validation_functions_with_multiple_records(db_connection, test_uuids):
    """Test that validation functions work correctly with multiple records.
    
    Based on docstrings: Functions check for specific UUID existence.
    """
    # Create multiple datasets
    dataset1 = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset 1",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset1)
    
    dataset2 = Dataset(
        dataset_id=test_uuids["dataset_2"],
        dataset_name="Test Dataset 2",
        source_url="https://example.com",
        license="MIT",
        modality_types=["oct"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset2)
    
    # Both datasets should exist
    exists1 = await validate_dataset_exists(test_uuids["dataset_1"])
    exists2 = await validate_dataset_exists(test_uuids["dataset_2"])
    
    assert exists1 is True
    assert exists2 is True
    
    # Non-existent dataset should not exist
    non_existent = await validate_dataset_exists(UUID("ffffffff-ffff-ffff-ffff-ffffffffffff"))
    assert non_existent is False
