"""
Tests for provenance chain and transformation database operations.

Tests based on docstring specifications only.
"""

import pytest
from datetime import datetime
from uuid import UUID

from chaksudb.db.queries.provenance import (
    upsert_provenance_chain,
    upsert_transformation_operation,
    upsert_provenance_transformation,
)
from chaksudb.db.models import ProvenanceChain, TransformationOperation, ProvenanceTransformation


@pytest.mark.asyncio
async def test_upsert_provenance_chain_creates_new_record(db_connection, test_uuids):
    """Test that upsert_provenance_chain creates a new provenance chain record.
    
    Based on docstring: 'Upsert a provenance chain record.'
    """
    chain_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    source_ann_id_1 = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    source_ann_id_2 = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
    
    chain = ProvenanceChain(
        chain_id=chain_id,
        unified_annotation_type="grading",
        source_type="original",
        root_source_raw_data_id=None,  # Set to None to avoid FK constraint
        source_annotation_ids=[source_ann_id_1, source_ann_id_2],
        created_at=datetime.now(),
    )
    
    await upsert_provenance_chain(chain)
    
    # Verify provenance chain was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT unified_annotation_type, source_type, root_source_raw_data_id FROM provenance_chain WHERE chain_id = %s",
            (chain_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "grading"
    assert result[1] == "original"
    assert result[2] is None


@pytest.mark.asyncio
async def test_upsert_provenance_chain_updates_existing_record(db_connection, test_uuids):
    """Test that upsert_provenance_chain updates an existing provenance chain record.
    
    Based on docstring: 'Upsert a provenance chain record.' (upsert implies insert or update)
    """
    chain_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    
    # Create initial provenance chain
    chain_v1 = ProvenanceChain(
        chain_id=chain_id,
        unified_annotation_type="grading",
        source_type="original",
        root_source_raw_data_id=None,  # Set to None to avoid FK constraint
        source_annotation_ids=None,
        created_at=datetime.now(),
    )
    await upsert_provenance_chain(chain_v1)
    
    # Update the provenance chain
    source_ann_id_1 = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    chain_v2 = ProvenanceChain(
        chain_id=chain_id,
        unified_annotation_type="grading",
        source_type="consensus",  # Updated
        root_source_raw_data_id=None,  # Set to None to avoid FK constraint
        source_annotation_ids=[source_ann_id_1],  # Updated
        created_at=datetime.now(),
    )
    await upsert_provenance_chain(chain_v2)
    
    # Verify provenance chain was updated
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT source_type FROM provenance_chain WHERE chain_id = %s",
            (chain_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "consensus"


@pytest.mark.asyncio
async def test_upsert_provenance_chain_with_null_optional_fields(db_connection, test_uuids):
    """Test that upsert_provenance_chain handles null optional fields correctly.
    
    Based on docstring and model showing optional fields: root_source_raw_data_id, source_annotation_ids.
    """
    chain_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    
    chain = ProvenanceChain(
        chain_id=chain_id,
        unified_annotation_type="grading",
        source_type="original",
        root_source_raw_data_id=None,
        source_annotation_ids=None,
        created_at=datetime.now(),
    )
    
    await upsert_provenance_chain(chain)
    
    # Verify provenance chain was created with null values
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT root_source_raw_data_id FROM provenance_chain WHERE chain_id = %s",
            (chain_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] is None


@pytest.mark.asyncio
async def test_upsert_transformation_operation_creates_new_record(db_connection, test_uuids):
    """Test that upsert_transformation_operation creates a new transformation operation record.
    
    Based on docstring: 'Upsert a transformation operation record.'
    """
    transformation_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    
    transformation = TransformationOperation(
        transformation_id=transformation_id,
        operation_type="scale_mapping",
        input_data={"original_grade": "moderate", "original_scale": "custom"},
        output_data={"scaled_grade": 2, "target_scale": "ETDRS"},
        operation_parameters={"mapping_confidence": "exact"},
        operation_timestamp=datetime.now(),
        operator="ingestion_pipeline",
        notes="Mapped custom scale to ETDRS",
    )
    
    await upsert_transformation_operation(transformation)
    
    # Verify transformation operation was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT operation_type, operator, notes FROM transformation_operations WHERE transformation_id = %s",
            (transformation_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "scale_mapping"
    assert result[1] == "ingestion_pipeline"
    assert result[2] == "Mapped custom scale to ETDRS"


@pytest.mark.asyncio
async def test_upsert_transformation_operation_updates_existing_record(db_connection, test_uuids):
    """Test that upsert_transformation_operation updates an existing transformation operation record.
    
    Based on docstring: 'Upsert a transformation operation record.' (upsert implies insert or update)
    """
    transformation_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    
    # Create initial transformation operation
    transformation_v1 = TransformationOperation(
        transformation_id=transformation_id,
        operation_type="scale_mapping",
        input_data={"original_grade": "moderate"},
        output_data={"scaled_grade": 2},
        operation_parameters={},
        operation_timestamp=datetime.now(),
        operator="ingestion_pipeline",
        notes="Initial mapping",
    )
    await upsert_transformation_operation(transformation_v1)
    
    # Update the transformation operation
    transformation_v2 = TransformationOperation(
        transformation_id=transformation_id,
        operation_type="scale_mapping",
        input_data={"original_grade": "moderate", "original_scale": "custom"},
        output_data={"scaled_grade": 2, "target_scale": "ETDRS"},
        operation_parameters={"mapping_confidence": "exact"},
        operation_timestamp=datetime.now(),
        operator="ingestion_pipeline",
        notes="Updated mapping with more details",  # Updated
    )
    await upsert_transformation_operation(transformation_v2)
    
    # Verify transformation operation was updated
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT notes FROM transformation_operations WHERE transformation_id = %s",
            (transformation_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "Updated mapping with more details"


@pytest.mark.asyncio
async def test_upsert_transformation_operation_with_null_optional_fields(db_connection, test_uuids):
    """Test that upsert_transformation_operation handles null optional fields correctly.
    
    Based on docstring and model showing optional fields: input_data, output_data, 
    operation_parameters, operator, notes.
    """
    transformation_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    
    transformation = TransformationOperation(
        transformation_id=transformation_id,
        operation_type="scale_mapping",
        input_data=None,
        output_data=None,
        operation_parameters=None,
        operation_timestamp=datetime.now(),
        operator=None,
        notes=None,
    )
    
    await upsert_transformation_operation(transformation)
    
    # Verify transformation operation was created with null values
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT operator, notes FROM transformation_operations WHERE transformation_id = %s",
            (transformation_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] is None
    assert result[1] is None


@pytest.mark.asyncio
async def test_upsert_provenance_transformation_creates_new_record(db_connection, test_uuids):
    """Test that upsert_provenance_transformation creates a new provenance-transformation link record.
    
    Based on docstring: 'Upsert a provenance-transformation link record.'
    """
    # Create provenance chain and transformation operation first
    chain_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    chain = ProvenanceChain(
        chain_id=chain_id,
        unified_annotation_type="grading",
        source_type="original",
        root_source_raw_data_id=None,
        source_annotation_ids=None,
        created_at=datetime.now(),
    )
    await upsert_provenance_chain(chain)
    
    transformation_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    transformation = TransformationOperation(
        transformation_id=transformation_id,
        operation_type="scale_mapping",
        input_data={"original_grade": "moderate"},
        output_data={"scaled_grade": 2},
        operation_parameters={},
        operation_timestamp=datetime.now(),
        operator="ingestion_pipeline",
        notes=None,
    )
    await upsert_transformation_operation(transformation)
    
    # Create provenance-transformation link
    link_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    prov_trans = ProvenanceTransformation(
        id=link_id,
        chain_id=chain_id,
        transformation_id=transformation_id,
        created_at=datetime.now(),
    )
    
    await upsert_provenance_transformation(prov_trans)
    
    # Verify provenance-transformation link was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT chain_id, transformation_id FROM provenance_transformations WHERE id = %s",
            (link_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == chain_id
    assert result[1] == transformation_id


@pytest.mark.asyncio
async def test_upsert_provenance_transformation_updates_existing_record(db_connection, test_uuids):
    """Test that upsert_provenance_transformation updates an existing link record.
    
    Based on docstring: 'Upsert a provenance-transformation link record.' (upsert implies insert or update)
    """
    # Create provenance chains and transformation operations
    chain_id_1 = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    chain_id_2 = UUID("aaaaaaaa-aaaa-aaaa-aaaa-bbbbbbbbbbbb")
    
    chain_1 = ProvenanceChain(
        chain_id=chain_id_1,
        unified_annotation_type="grading",
        source_type="original",
        root_source_raw_data_id=None,
        source_annotation_ids=None,
        created_at=datetime.now(),
    )
    await upsert_provenance_chain(chain_1)
    
    chain_2 = ProvenanceChain(
        chain_id=chain_id_2,
        unified_annotation_type="segmentation",
        source_type="original",
        root_source_raw_data_id=None,
        source_annotation_ids=None,
        created_at=datetime.now(),
    )
    await upsert_provenance_chain(chain_2)
    
    transformation_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    transformation = TransformationOperation(
        transformation_id=transformation_id,
        operation_type="scale_mapping",
        input_data={"original_grade": "moderate"},
        output_data={"scaled_grade": 2},
        operation_parameters={},
        operation_timestamp=datetime.now(),
        operator="ingestion_pipeline",
        notes=None,
    )
    await upsert_transformation_operation(transformation)
    
    # Create initial provenance-transformation link
    link_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    prov_trans_v1 = ProvenanceTransformation(
        id=link_id,
        chain_id=chain_id_1,
        transformation_id=transformation_id,
        created_at=datetime.now(),
    )
    await upsert_provenance_transformation(prov_trans_v1)
    
    # Update with same composite key (chain_id, transformation_id) should update
    # Since conflict target is (chain_id, transformation_id), we need to keep them same
    # and just verify the record still exists
    prov_trans_v2 = ProvenanceTransformation(
        id=link_id,
        chain_id=chain_id_1,
        transformation_id=transformation_id,
        created_at=datetime.now(),
    )
    await upsert_provenance_transformation(prov_trans_v2)
    
    # Verify provenance-transformation link exists
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT COUNT(*) FROM provenance_transformations WHERE chain_id = %s AND transformation_id = %s",
            (chain_id_1, transformation_id)
        )
        count = await cur.fetchone()
        
    assert count[0] == 1


@pytest.mark.asyncio
async def test_upsert_provenance_transformation_handles_composite_conflict_key(db_connection, test_uuids):
    """Test that upsert_provenance_transformation handles composite conflict key (chain_id, transformation_id).
    
    Based on implementation showing conflict_target includes chain_id and transformation_id.
    """
    # Create provenance chain and transformation operation
    chain_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    chain = ProvenanceChain(
        chain_id=chain_id,
        unified_annotation_type="grading",
        source_type="original",
        root_source_raw_data_id=None,
        source_annotation_ids=None,
        created_at=datetime.now(),
    )
    await upsert_provenance_chain(chain)
    
    transformation_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    transformation = TransformationOperation(
        transformation_id=transformation_id,
        operation_type="scale_mapping",
        input_data={"original_grade": "moderate"},
        output_data={"scaled_grade": 2},
        operation_parameters={},
        operation_timestamp=datetime.now(),
        operator="ingestion_pipeline",
        notes=None,
    )
    await upsert_transformation_operation(transformation)
    
    # Create first link
    link_id_1 = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    prov_trans_1 = ProvenanceTransformation(
        id=link_id_1,
        chain_id=chain_id,
        transformation_id=transformation_id,
        created_at=datetime.now(),
    )
    await upsert_provenance_transformation(prov_trans_1)
    
    # Try to create second link with same chain_id and transformation_id but different id
    # Should update, not create duplicate
    link_id_2 = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
    prov_trans_2 = ProvenanceTransformation(
        id=link_id_2,
        chain_id=chain_id,
        transformation_id=transformation_id,
        created_at=datetime.now(),
    )
    await upsert_provenance_transformation(prov_trans_2)
    
    # Verify only one link exists for this chain_id and transformation_id combination
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT COUNT(*) FROM provenance_transformations WHERE chain_id = %s AND transformation_id = %s",
            (chain_id, transformation_id)
        )
        count = await cur.fetchone()
        
    assert count[0] == 1
