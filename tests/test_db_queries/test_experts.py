"""
Tests for expert-related database operations.

Tests based on docstring specifications only.
"""

import pytest
from datetime import datetime
from uuid import UUID

from chaksudb.db.queries.experts import upsert_expert, upsert_expert_annotation
from chaksudb.db.queries.datasets import upsert_dataset
from chaksudb.db.models import Dataset, Expert, ExpertAnnotation


@pytest.mark.asyncio
async def test_upsert_expert_creates_new_record_with_dataset(db_connection, test_uuids):
    """Test that upsert_expert creates a new expert record associated with a dataset.
    
    Based on docstring: 'Upsert an expert record.'
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
    
    # Create an expert associated with dataset
    expert = Expert(
        expert_id=test_uuids["expert_1"],
        expert_name="Dr. Smith",
        expertise_area="Ophthalmology",
        dataset_id=test_uuids["dataset_1"],
        model_id=None,
        created_at=datetime.now(),
    )
    
    await upsert_expert(expert)
    
    # Verify expert was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT expert_name, expertise_area, dataset_id FROM experts WHERE expert_id = %s",
            (test_uuids["expert_1"],)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "Dr. Smith"
    assert result[1] == "Ophthalmology"
    assert result[2] == test_uuids["dataset_1"]


@pytest.mark.asyncio
async def test_upsert_expert_creates_new_record_with_model(db_connection, test_uuids):
    """Test that upsert_expert creates a new expert record associated with a model.
    
    Based on docstring: 'Upsert an expert record.' and model showing either dataset_id or model_id.
    """
    # First create a model using the models query function
    from chaksudb.db.queries.models import upsert_model
    from chaksudb.db.models import Model
    
    model = Model(
        model_id=test_uuids["expert_1"],
        model_name="ResNet50",
        model_description="Deep learning model for DR detection",
        model_url=None,
    )
    await upsert_model(model)
    
    # Create an expert associated with model
    expert = Expert(
        expert_id=test_uuids["expert_2"],
        expert_name="ResNet50_v1",
        expertise_area="Automated DR Detection",
        dataset_id=None,
        model_id=test_uuids["expert_1"],
        created_at=datetime.now(),
    )
    
    await upsert_expert(expert)
    
    # Verify expert was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT expert_name, model_id FROM experts WHERE expert_id = %s",
            (test_uuids["expert_2"],)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "ResNet50_v1"
    assert result[1] == test_uuids["expert_1"]


@pytest.mark.asyncio
async def test_upsert_expert_updates_existing_record(db_connection, test_uuids):
    """Test that upsert_expert updates an existing expert record.
    
    Based on docstring: 'Upsert an expert record.' (upsert implies insert or update)
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
    
    # Create an expert
    expert_v1 = Expert(
        expert_id=test_uuids["expert_1"],
        expert_name="Dr. Smith",
        expertise_area="Ophthalmology",
        dataset_id=test_uuids["dataset_1"],
        model_id=None,
        created_at=datetime.now(),
    )
    await upsert_expert(expert_v1)
    
    # Update the expert
    expert_v2 = Expert(
        expert_id=test_uuids["expert_1"],
        expert_name="Dr. John Smith",  # Updated
        expertise_area="Retinal Diseases",  # Updated
        dataset_id=test_uuids["dataset_1"],
        model_id=None,
        created_at=datetime.now(),
    )
    await upsert_expert(expert_v2)
    
    # Verify expert was updated
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT expert_name, expertise_area FROM experts WHERE expert_id = %s",
            (test_uuids["expert_1"],)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "Dr. John Smith"
    assert result[1] == "Retinal Diseases"


@pytest.mark.asyncio
async def test_upsert_expert_annotation_creates_new_record(db_connection, test_uuids):
    """Test that upsert_expert_annotation creates a new expert annotation record.
    
    Based on docstring: 'Upsert an expert annotation record.'
    """
    # First create a dataset and expert
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    expert = Expert(
        expert_id=test_uuids["expert_1"],
        expert_name="Dr. Smith",
        expertise_area="Ophthalmology",
        dataset_id=test_uuids["dataset_1"],
        model_id=None,
        created_at=datetime.now(),
    )
    await upsert_expert(expert)
    
    # Create an expert annotation
    annotation_id = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
    expert_annotation = ExpertAnnotation(
        expert_annotation_id=annotation_id,
        expert_id=test_uuids["expert_1"],
        annotation_task="grading",
        raw_data_id=None,
        annotation_value={"grade": 2, "disease": "diabetic_retinopathy"},
        confidence_level="high",
        annotation_timestamp=datetime.now(),
        created_at=datetime.now(),
    )
    
    await upsert_expert_annotation(expert_annotation)
    
    # Verify expert annotation was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT expert_id, annotation_task, confidence_level FROM expert_annotations WHERE expert_annotation_id = %s",
            (annotation_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == test_uuids["expert_1"]
    assert result[1] == "grading"
    assert result[2] == "high"


@pytest.mark.asyncio
async def test_upsert_expert_annotation_updates_existing_record(db_connection, test_uuids):
    """Test that upsert_expert_annotation updates an existing expert annotation record.
    
    Based on docstring: 'Upsert an expert annotation record.' (upsert implies insert or update)
    """
    # First create a dataset and expert
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    expert = Expert(
        expert_id=test_uuids["expert_1"],
        expert_name="Dr. Smith",
        expertise_area="Ophthalmology",
        dataset_id=test_uuids["dataset_1"],
        model_id=None,
        created_at=datetime.now(),
    )
    await upsert_expert(expert)
    
    # Create an expert annotation
    annotation_id = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
    expert_annotation_v1 = ExpertAnnotation(
        expert_annotation_id=annotation_id,
        expert_id=test_uuids["expert_1"],
        annotation_task="grading",
        raw_data_id=None,
        annotation_value={"grade": 2},
        confidence_level="medium",
        annotation_timestamp=datetime.now(),
        created_at=datetime.now(),
    )
    await upsert_expert_annotation(expert_annotation_v1)
    
    # Update the expert annotation
    expert_annotation_v2 = ExpertAnnotation(
        expert_annotation_id=annotation_id,
        expert_id=test_uuids["expert_1"],
        annotation_task="grading",
        raw_data_id=None,
        annotation_value={"grade": 3, "revised": True},  # Updated
        confidence_level="high",  # Updated
        annotation_timestamp=datetime.now(),
        created_at=datetime.now(),
    )
    await upsert_expert_annotation(expert_annotation_v2)
    
    # Verify expert annotation was updated
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT confidence_level FROM expert_annotations WHERE expert_annotation_id = %s",
            (annotation_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "high"


@pytest.mark.asyncio
async def test_upsert_expert_annotation_with_null_optional_fields(db_connection, test_uuids):
    """Test that upsert_expert_annotation handles null optional fields correctly.
    
    Based on docstring and model showing optional fields: raw_data_id, annotation_value, confidence_level, annotation_timestamp.
    """
    # First create a dataset and expert
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    expert = Expert(
        expert_id=test_uuids["expert_1"],
        expert_name="Dr. Smith",
        expertise_area="Ophthalmology",
        dataset_id=test_uuids["dataset_1"],
        model_id=None,
        created_at=datetime.now(),
    )
    await upsert_expert(expert)
    
    # Create an expert annotation with minimal fields
    annotation_id = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
    expert_annotation = ExpertAnnotation(
        expert_annotation_id=annotation_id,
        expert_id=test_uuids["expert_1"],
        annotation_task="grading",
        raw_data_id=None,
        annotation_value=None,
        confidence_level=None,
        annotation_timestamp=None,
        created_at=datetime.now(),
    )
    
    await upsert_expert_annotation(expert_annotation)
    
    # Verify expert annotation was created with null values
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT raw_data_id, confidence_level FROM expert_annotations WHERE expert_annotation_id = %s",
            (annotation_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] is None
    assert result[1] is None
