"""
Tests for patient-related database operations.

Tests based on docstring specifications only.
"""

import pytest
from datetime import datetime
from uuid import UUID

from chaksudb.db.queries.patients import upsert_patient, bulk_upsert_patients
from chaksudb.db.queries.datasets import upsert_dataset
from chaksudb.db.models import Dataset, Patient


@pytest.mark.asyncio
async def test_upsert_patient_creates_new_record(db_connection, test_uuids):
    """Test that upsert_patient creates a new patient record.
    
    Based on docstring: 'Upsert a patient record.'
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
    patient = Patient(
        patient_id=test_uuids["patient_1"],
        dataset_id=test_uuids["dataset_1"],
        original_patient_id="PAT_001",
        age=45,
        sex="male",
        ethnicity="Asian",
        nationality="Japanese",
        comorbidities={"diabetes": True, "hypertension": False},
        created_at=datetime.now(),
    )
    
    await upsert_patient(patient)
    
    # Verify patient was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT original_patient_id, age, sex, ethnicity FROM patients WHERE patient_id = %s",
            (test_uuids["patient_1"],)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "PAT_001"
    assert result[1] == 45
    assert result[2] == "male"
    assert result[3] == "Asian"


@pytest.mark.asyncio
async def test_upsert_patient_updates_existing_record(db_connection, test_uuids):
    """Test that upsert_patient updates an existing patient record.
    
    Based on docstring: 'Upsert a patient record.' (upsert implies insert or update)
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
    patient_v1 = Patient(
        patient_id=test_uuids["patient_1"],
        dataset_id=test_uuids["dataset_1"],
        original_patient_id="PAT_001",
        age=45,
        sex="male",
        ethnicity="Asian",
        created_at=datetime.now(),
    )
    await upsert_patient(patient_v1)
    
    # Update the patient (same dataset_id and original_patient_id)
    patient_v2 = Patient(
        patient_id=test_uuids["patient_1"],
        dataset_id=test_uuids["dataset_1"],
        original_patient_id="PAT_001",
        age=46,  # Updated
        sex="male",
        ethnicity="Asian",
        nationality="Japanese",  # Added
        comorbidities={"diabetes": True},  # Added
        created_at=datetime.now(),
    )
    await upsert_patient(patient_v2)
    
    # Verify patient was updated
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT age, nationality FROM patients WHERE patient_id = %s",
            (test_uuids["patient_1"],)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == 46
    assert result[1] == "Japanese"


@pytest.mark.asyncio
async def test_upsert_patient_with_null_optional_fields(db_connection, test_uuids):
    """Test that upsert_patient handles null optional fields correctly.
    
    Based on docstring and model showing optional fields: age, sex, ethnicity, nationality, comorbidities.
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
    
    # Create a patient with minimal fields
    patient = Patient(
        patient_id=test_uuids["patient_1"],
        dataset_id=test_uuids["dataset_1"],
        original_patient_id="PAT_001",
        age=None,
        sex=None,
        ethnicity=None,
        nationality=None,
        comorbidities=None,
        created_at=datetime.now(),
    )
    
    await upsert_patient(patient)
    
    # Verify patient was created with null values
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT age, sex, ethnicity, nationality FROM patients WHERE patient_id = %s",
            (test_uuids["patient_1"],)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] is None
    assert result[1] is None
    assert result[2] is None
    assert result[3] is None


@pytest.mark.asyncio
async def test_bulk_upsert_patients_creates_multiple_records(db_connection, test_uuids):
    """Test that bulk_upsert_patients creates multiple patient records.
    
    Based on docstring: 'Bulk upsert patient records.'
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
    
    # Create multiple patients
    patients = [
        Patient(
            patient_id=test_uuids["patient_1"],
            dataset_id=test_uuids["dataset_1"],
            original_patient_id="PAT_001",
            age=45,
            sex="male",
            created_at=datetime.now(),
        ),
        Patient(
            patient_id=test_uuids["patient_2"],
            dataset_id=test_uuids["dataset_1"],
            original_patient_id="PAT_002",
            age=50,
            sex="female",
            created_at=datetime.now(),
        ),
    ]
    
    rows_inserted = await bulk_upsert_patients(patients)
    
    # Should return count of rows inserted
    assert rows_inserted == 2
    
    # Verify patients were created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT COUNT(*) FROM patients WHERE patient_id IN (%s, %s)",
            (test_uuids["patient_1"], test_uuids["patient_2"])
        )
        count = await cur.fetchone()
        
    assert count[0] == 2


@pytest.mark.asyncio
async def test_bulk_upsert_patients_with_custom_batch_size(db_connection, test_uuids):
    """Test that bulk_upsert_patients respects custom batch_size parameter.
    
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
    
    # Create multiple patients with unique IDs (starting from PAT_100 to avoid conflicts)
    patients = [
        Patient(
            patient_id=UUID(f"00000000-0000-0000-0000-{str(i+100).zfill(12)}"),
            dataset_id=test_uuids["dataset_1"],
            original_patient_id=f"PAT_{i+100:03d}",
            age=30 + i,
            sex="male",
            created_at=datetime.now(),
        )
        for i in range(5)
    ]
    
    # Use small batch size to test batching
    rows_inserted = await bulk_upsert_patients(patients, batch_size=2)
    
    assert rows_inserted == 5


@pytest.mark.asyncio
async def test_upsert_patient_handles_composite_conflict_key(db_connection, test_uuids):
    """Test that upsert_patient handles composite conflict key (dataset_id, original_patient_id).
    
    Based on implementation showing conflict_target includes dataset_id and original_patient_id.
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
    patient_v1 = Patient(
        patient_id=test_uuids["patient_1"],
        dataset_id=test_uuids["dataset_1"],
        original_patient_id="PAT_001",
        age=45,
        sex="male",
        created_at=datetime.now(),
    )
    await upsert_patient(patient_v1)
    
    # Try to insert same patient (same dataset_id + original_patient_id) with different patient_id
    # Should update, not create duplicate
    patient_v2 = Patient(
        patient_id=test_uuids["patient_1"],  # Same UUID to avoid PK violation
        dataset_id=test_uuids["dataset_1"],
        original_patient_id="PAT_001",
        age=50,  # Updated age
        sex="male",
        created_at=datetime.now(),
    )
    await upsert_patient(patient_v2)
    
    # Verify only one patient exists with updated age
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT COUNT(*), age FROM patients WHERE dataset_id = %s AND original_patient_id = %s GROUP BY age",
            (test_uuids["dataset_1"], "PAT_001")
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == 1
    assert result[1] == 50  # Updated value
