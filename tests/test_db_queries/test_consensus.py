"""
Tests for consensus annotation database operations.

Tests based on docstring specifications only.
"""

import pytest
from datetime import datetime
from uuid import UUID

from chaksudb.db.queries.consensus import upsert_consensus_annotation
from chaksudb.db.queries.datasets import upsert_dataset
from chaksudb.db.queries.images import upsert_image
from chaksudb.db.models import Dataset, Image, ConsensusAnnotation


@pytest.mark.asyncio
async def test_upsert_consensus_annotation_creates_new_record(db_connection, test_uuids):
    """Test that upsert_consensus_annotation creates a new consensus annotation record.
    
    Based on docstring: 'Upsert a consensus annotation record.'
    """
    # Setup dataset and image
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
    
    # Create consensus annotation
    consensus_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    expert_ann_id_1 = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    expert_ann_id_2 = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    
    consensus = ConsensusAnnotation(
        consensus_id=consensus_id,
        image_id=test_uuids["image_1"],
        annotation_task="grading",
        consensus_method="majority_vote",
        expert_annotation_ids=[expert_ann_id_1, expert_ann_id_2],
        consensus_value={"grade": 2, "confidence": 0.9},
        agreement_score=0.85,
        disagreement_details={"expert_1": 2, "expert_2": 2},
        adjudicator_id=None,
        created_at=datetime.now(),
    )
    
    await upsert_consensus_annotation(consensus)
    
    # Verify consensus annotation was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT image_id, annotation_task, consensus_method, agreement_score FROM consensus_annotations WHERE consensus_id = %s",
            (consensus_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == test_uuids["image_1"]
    assert result[1] == "grading"
    assert result[2] == "majority_vote"
    assert result[3] == 0.85


@pytest.mark.asyncio
async def test_upsert_consensus_annotation_updates_existing_record(db_connection, test_uuids):
    """Test that upsert_consensus_annotation updates an existing consensus annotation record.
    
    Based on docstring: 'Upsert a consensus annotation record.' (upsert implies insert or update)
    """
    # Setup dataset and image
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
    
    consensus_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    expert_ann_id_1 = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    expert_ann_id_2 = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    
    # Create initial consensus annotation
    consensus_v1 = ConsensusAnnotation(
        consensus_id=consensus_id,
        image_id=test_uuids["image_1"],
        annotation_task="grading",
        consensus_method="majority_vote",
        expert_annotation_ids=[expert_ann_id_1, expert_ann_id_2],
        consensus_value={"grade": 2},
        agreement_score=0.75,
        disagreement_details=None,
        adjudicator_id=None,
        created_at=datetime.now(),
    )
    await upsert_consensus_annotation(consensus_v1)
    
    # Create an adjudicator expert
    from chaksudb.db.queries.experts import upsert_expert
    from chaksudb.db.models import Expert
    
    adjudicator = Expert(
        expert_id=test_uuids["expert_1"],
        expert_name="Senior Ophthalmologist",
        expertise_area="Retinal Diseases",
        dataset_id=test_uuids["dataset_1"],
        model_id=None,
        created_at=datetime.now(),
    )
    await upsert_expert(adjudicator)
    
    # Update the consensus annotation
    consensus_v2 = ConsensusAnnotation(
        consensus_id=consensus_id,
        image_id=test_uuids["image_1"],
        annotation_task="grading",
        consensus_method="adjudicated",  # Updated
        expert_annotation_ids=[expert_ann_id_1, expert_ann_id_2],
        consensus_value={"grade": 3, "revised": True},  # Updated
        agreement_score=0.95,  # Updated
        disagreement_details={"resolution": "adjudicated"},
        adjudicator_id=test_uuids["expert_1"],  # Updated
        created_at=datetime.now(),
    )
    await upsert_consensus_annotation(consensus_v2)
    
    # Verify consensus annotation was updated
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT consensus_method, agreement_score, adjudicator_id FROM consensus_annotations WHERE consensus_id = %s",
            (consensus_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "adjudicated"
    assert result[1] == 0.95
    assert result[2] == test_uuids["expert_1"]


@pytest.mark.asyncio
async def test_upsert_consensus_annotation_with_null_optional_fields(db_connection, test_uuids):
    """Test that upsert_consensus_annotation handles null optional fields correctly.
    
    Based on docstring and model showing optional fields: expert_annotation_ids, consensus_value, 
    agreement_score, disagreement_details, adjudicator_id.
    """
    # Setup dataset and image
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
    
    # Create consensus annotation with minimal fields
    consensus_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    consensus = ConsensusAnnotation(
        consensus_id=consensus_id,
        image_id=test_uuids["image_1"],
        annotation_task="grading",
        consensus_method="majority_vote",
        expert_annotation_ids=None,
        consensus_value=None,
        agreement_score=None,
        disagreement_details=None,
        adjudicator_id=None,
        created_at=datetime.now(),
    )
    
    await upsert_consensus_annotation(consensus)
    
    # Verify consensus annotation was created with null values
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT agreement_score, adjudicator_id FROM consensus_annotations WHERE consensus_id = %s",
            (consensus_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] is None
    assert result[1] is None


@pytest.mark.asyncio
async def test_upsert_consensus_annotation_with_different_consensus_methods(db_connection, test_uuids):
    """Test that upsert_consensus_annotation supports different consensus methods.
    
    Based on model showing consensus_method validation with values: 
    majority_vote, mean, median, staple, adjudicated, senior_review.
    """
    # Setup dataset and image
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
    
    # Test different consensus methods
    consensus_methods = ["majority_vote", "mean", "median", "staple", "adjudicated", "senior_review"]
    
    for i, method in enumerate(consensus_methods):
        consensus_id = UUID(f"aaaaaaaa-aaaa-aaaa-aaaa-{str(i).zfill(12)}")
        consensus = ConsensusAnnotation(
            consensus_id=consensus_id,
            image_id=test_uuids["image_1"],
            annotation_task="grading",
            consensus_method=method,
            expert_annotation_ids=None,
            consensus_value=None,
            agreement_score=None,
            disagreement_details=None,
            adjudicator_id=None,
            created_at=datetime.now(),
        )
        
        await upsert_consensus_annotation(consensus)
        
        # Verify consensus annotation was created with the correct method
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT consensus_method FROM consensus_annotations WHERE consensus_id = %s",
                (consensus_id,)
            )
            result = await cur.fetchone()
            
        assert result is not None
        assert result[0] == method
