"""
Tests for model-related database operations.

Tests based on docstring specifications only.
"""

import pytest
from uuid import UUID

from chaksudb.db.queries.models import upsert_model
from chaksudb.db.models import Model


@pytest.mark.asyncio
async def test_upsert_model_creates_new_record(db_connection, test_uuids):
    """Test that upsert_model creates a new model record.
    
    Based on docstring: 'Upsert a model record.'
    """
    model = Model(
        model_id=test_uuids["expert_1"],
        model_name="ResNet50",
        model_description="Deep learning model for diabetic retinopathy detection",
        model_url="https://github.com/example/resnet50-dr",
    )
    
    await upsert_model(model)
    
    # Verify model was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT model_name, model_description, model_url FROM models WHERE model_id = %s",
            (test_uuids["expert_1"],)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "ResNet50"
    assert result[1] == "Deep learning model for diabetic retinopathy detection"
    assert result[2] == "https://github.com/example/resnet50-dr"


@pytest.mark.asyncio
async def test_upsert_model_updates_existing_record(db_connection, test_uuids):
    """Test that upsert_model updates an existing model record.
    
    Based on docstring: 'Upsert a model record.' (upsert implies insert or update)
    """
    # Create a model
    model_v1 = Model(
        model_id=test_uuids["expert_1"],
        model_name="ResNet50",
        model_description="Initial description",
        model_url="https://github.com/example/resnet50-v1",
    )
    await upsert_model(model_v1)
    
    # Update the model
    model_v2 = Model(
        model_id=test_uuids["expert_1"],
        model_name="ResNet50-v2",  # Updated
        model_description="Updated description with better performance",  # Updated
        model_url="https://github.com/example/resnet50-v2",  # Updated
    )
    await upsert_model(model_v2)
    
    # Verify model was updated
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT model_name, model_description, model_url FROM models WHERE model_id = %s",
            (test_uuids["expert_1"],)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "ResNet50-v2"
    assert result[1] == "Updated description with better performance"
    assert result[2] == "https://github.com/example/resnet50-v2"


@pytest.mark.asyncio
async def test_upsert_model_with_null_optional_fields(db_connection, test_uuids):
    """Test that upsert_model handles null optional fields correctly.
    
    Based on docstring and model showing optional fields: model_description, model_url.
    """
    model = Model(
        model_id=test_uuids["expert_1"],
        model_name="SimpleCNN",
        model_description=None,
        model_url=None,
    )
    
    await upsert_model(model)
    
    # Verify model was created with null values
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT model_name, model_description, model_url FROM models WHERE model_id = %s",
            (test_uuids["expert_1"],)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "SimpleCNN"
    assert result[1] is None
    assert result[2] is None
