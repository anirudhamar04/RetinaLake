"""
Tests for dataset-related database operations.

Tests based on docstring specifications only.
"""

import pytest
from datetime import datetime
from uuid import UUID

from chaksudb.db.queries.datasets import (
    upsert_dataset,
    bulk_upsert_datasets,
    upsert_dataset_split,
    upsert_image_split,
)
from chaksudb.db.models import Dataset, DatasetSplit, ImageSplit


@pytest.mark.asyncio
async def test_upsert_dataset_creates_new_record(db_connection, test_uuids):
    """Test that upsert_dataset creates a new dataset record.
    
    Based on docstring: 'Upsert a dataset record.'
    """
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    
    # Should not raise any exception
    await upsert_dataset(dataset)
    
    # Verify dataset was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT dataset_name, source_url, license FROM datasets WHERE dataset_id = %s",
            (test_uuids["dataset_1"],)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "Test Dataset"
    assert result[1] == "https://example.com"
    assert result[2] == "MIT"


@pytest.mark.asyncio
async def test_upsert_dataset_updates_existing_record(db_connection, test_uuids):
    """Test that upsert_dataset updates an existing dataset record.
    
    Based on docstring: 'Upsert a dataset record.' (upsert implies insert or update)
    """
    dataset_v1 = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Original Name",
        source_url="https://example.com/v1",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    
    await upsert_dataset(dataset_v1)
    
    # Update the dataset
    dataset_v2 = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Updated Name",
        source_url="https://example.com/v2",
        license="Apache 2.0",
        modality_types=["oct"],
        created_at=datetime.now(),
    )
    
    await upsert_dataset(dataset_v2)
    
    # Verify dataset was updated
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT dataset_name, source_url, license FROM datasets WHERE dataset_id = %s",
            (test_uuids["dataset_1"],)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "Updated Name"
    assert result[1] == "https://example.com/v2"
    assert result[2] == "Apache 2.0"


@pytest.mark.asyncio
async def test_bulk_upsert_datasets_creates_multiple_records(db_connection, test_uuids):
    """Test that bulk_upsert_datasets creates multiple dataset records.
    
    Based on docstring: 'Bulk upsert dataset records.'
    """
    datasets = [
        Dataset(
            dataset_id=test_uuids["dataset_1"],
            dataset_name="Dataset 1",
            source_url="https://example.com/1",
            license="MIT",
            modality_types=["fundus"],
            created_at=datetime.now(),
        ),
        Dataset(
            dataset_id=test_uuids["dataset_2"],
            dataset_name="Dataset 2",
            source_url="https://example.com/2",
            license="Apache 2.0",
            modality_types=["oct"],
            created_at=datetime.now(),
        ),
    ]
    
    rows_inserted = await bulk_upsert_datasets(datasets)
    
    # Should return count of rows inserted
    assert rows_inserted == 2
    
    # Verify datasets were created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT COUNT(*) FROM datasets WHERE dataset_id IN (%s, %s)",
            (test_uuids["dataset_1"], test_uuids["dataset_2"])
        )
        count = await cur.fetchone()
        
    assert count[0] == 2


@pytest.mark.asyncio
async def test_bulk_upsert_datasets_with_custom_batch_size(db_connection, test_uuids):
    """Test that bulk_upsert_datasets respects custom batch_size parameter.
    
    Based on docstring signature showing batch_size parameter with default 1000.
    """
    datasets = [
        Dataset(
            dataset_id=test_uuids["dataset_1"],
            dataset_name=f"Dataset {i}",
            source_url=f"https://example.com/{i}",
            license="MIT",
            modality_types=["fundus"],
            created_at=datetime.now(),
        )
        for i in range(5)
    ]
    
    # Use small batch size to test batching
    rows_inserted = await bulk_upsert_datasets(datasets, batch_size=2)
    
    assert rows_inserted == 5


@pytest.mark.asyncio
async def test_upsert_dataset_split_creates_new_record(db_connection, test_uuids):
    """Test that upsert_dataset_split creates a new dataset split record.
    
    Based on docstring: 'Upsert a dataset split record.'
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
    
    # Create a dataset split
    split_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    split = DatasetSplit(
        split_id=split_id,
        dataset_id=test_uuids["dataset_1"],
        split_name="train",
        split_type="explicit",
        task_type="grading",
        image_count=100,
        created_at=datetime.now(),
    )
    
    await upsert_dataset_split(split)
    
    # Verify split was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT split_name, split_type, image_count FROM dataset_splits WHERE split_id = %s",
            (split_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "train"
    assert result[1] == "explicit"
    assert result[2] == 100


@pytest.mark.asyncio
async def test_upsert_image_split_creates_new_record(db_connection, test_uuids):
    """Test that upsert_image_split creates a new image split assignment record.
    
    Based on docstring: 'Upsert an image split assignment record.'
    """
    # First create a dataset, image, and split
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
    from chaksudb.db.queries.images import upsert_image
    from chaksudb.db.models import Image
    
    image = Image(
        image_id=test_uuids["image_1"],
        dataset_id=test_uuids["dataset_1"],
        storage_provider="local",
        file_path="/test/path.jpg",
        file_format="jpg",
        modality="fundus",
        created_at=datetime.now(),
    )
    await upsert_image(image)
    
    # Create a split
    split_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    split = DatasetSplit(
        split_id=split_id,
        dataset_id=test_uuids["dataset_1"],
        split_name="train",
        split_type="explicit",
        task_type="grading",
        image_count=100,
        created_at=datetime.now(),
    )
    await upsert_dataset_split(split)
    
    # Create an image split assignment
    assignment_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    image_split = ImageSplit(
        assignment_id=assignment_id,
        image_id=test_uuids["image_1"],
        split_id=split_id,
        task_type="grading",
        is_primary=True,
        created_at=datetime.now(),
    )
    
    await upsert_image_split(image_split)
    
    # Verify image split was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT image_id, split_id, task_type, is_primary FROM image_splits WHERE assignment_id = %s",
            (assignment_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == test_uuids["image_1"]
    assert result[1] == split_id
    assert result[2] == "grading"
    assert result[3] is True


@pytest.mark.asyncio
async def test_upsert_image_split_handles_composite_conflict_key(db_connection, test_uuids):
    """Test that upsert_image_split handles composite conflict key (image_id, split_id, task_type).
    
    Based on implementation showing conflict_target includes image_id, split_id, and task_type.
    """
    # Setup dataset, image, and split
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
    from chaksudb.db.queries.images import upsert_image
    from chaksudb.db.models import Image
    
    image = Image(
        image_id=test_uuids["image_1"],
        dataset_id=test_uuids["dataset_1"],
        storage_provider="local",
        file_path="/test/path.jpg",
        file_format="jpg",
        modality="fundus",
        created_at=datetime.now(),
    )
    await upsert_image(image)
    
    split_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    split = DatasetSplit(
        split_id=split_id,
        dataset_id=test_uuids["dataset_1"],
        split_name="train",
        split_type="explicit",
        task_type="grading",
        image_count=100,
        created_at=datetime.now(),
    )
    await upsert_dataset_split(split)
    
    # First insert
    assignment_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    image_split_v1 = ImageSplit(
        assignment_id=assignment_id,
        image_id=test_uuids["image_1"],
        split_id=split_id,
        task_type="grading",
        is_primary=True,
        created_at=datetime.now(),
    )
    await upsert_image_split(image_split_v1)
    
    # Second insert with same image_id, split_id, task_type but different assignment_id
    # Should update, not fail
    image_split_v2 = ImageSplit(
        assignment_id=assignment_id,
        image_id=test_uuids["image_1"],
        split_id=split_id,
        task_type="grading",
        is_primary=False,  # Changed
        created_at=datetime.now(),
    )
    await upsert_image_split(image_split_v2)
    
    # Verify only one record exists with updated value
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT COUNT(*), is_primary FROM image_splits WHERE image_id = %s AND split_id = %s AND task_type = %s GROUP BY is_primary",
            (test_uuids["image_1"], split_id, "grading")
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == 1
    assert result[1] is False  # Updated value
