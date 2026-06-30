"""
Tests for raw annotation file database operations.

Tests based on docstring specifications only.
"""

import pytest
from datetime import datetime
from uuid import UUID

from chaksudb.db.queries.raw_annotations import (
    upsert_raw_annotation_file,
    bulk_upsert_raw_annotation_files,
)
from chaksudb.db.queries.datasets import upsert_dataset
from chaksudb.db.models import Dataset, RawAnnotationFile


@pytest.mark.asyncio
async def test_upsert_raw_annotation_file_creates_new_record(db_connection, test_uuids):
    """Test that upsert_raw_annotation_file creates a new raw annotation file record.
    
    Based on docstring: 'Upsert a raw annotation file record.'
    """
    # Setup dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Create raw annotation file
    raw_file_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    raw_file = RawAnnotationFile(
        raw_file_id=raw_file_id,
        dataset_id=test_uuids["dataset_1"],
        storage_provider="local",
        bucket=None,
        object_key=None,
        version_id=None,
        file_path="/data/annotations/labels.csv",
        file_type="csv",
        file_name="labels.csv",
        file_hash="abc123def456",
        file_size=1024,
        encoding="utf-8",
        parsed_status="not_parsed",
        parse_errors=None,
        created_at=datetime.now(),
        updated_at=None,
    )
    
    await upsert_raw_annotation_file(raw_file)
    
    # Verify raw annotation file was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT file_path, file_type, file_hash, parsed_status FROM raw_annotation_files WHERE raw_file_id = %s",
            (raw_file_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "/data/annotations/labels.csv"
    assert result[1] == "csv"
    assert result[2] == "abc123def456"
    assert result[3] == "not_parsed"


@pytest.mark.asyncio
async def test_upsert_raw_annotation_file_updates_existing_record(db_connection, test_uuids):
    """Test that upsert_raw_annotation_file updates an existing raw annotation file record.
    
    Based on docstring: 'Upsert a raw annotation file record.' (upsert implies insert or update)
    """
    # Setup dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    raw_file_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    
    # Create initial raw annotation file
    raw_file_v1 = RawAnnotationFile(
        raw_file_id=raw_file_id,
        dataset_id=test_uuids["dataset_1"],
        storage_provider="local",
        file_path="/data/annotations/labels.csv",
        file_type="csv",
        file_name="labels.csv",
        file_hash="abc123def456",
        file_size=1024,
        encoding="utf-8",
        parsed_status="not_parsed",
        parse_errors=None,
        created_at=datetime.now(),
    )
    await upsert_raw_annotation_file(raw_file_v1)
    
    # Update the raw annotation file
    raw_file_v2 = RawAnnotationFile(
        raw_file_id=raw_file_id,
        dataset_id=test_uuids["dataset_1"],
        storage_provider="local",
        file_path="/data/annotations/labels.csv",
        file_type="csv",
        file_name="labels.csv",
        file_hash="abc123def456",
        file_size=1024,
        encoding="utf-8",
        parsed_status="parsed",  # Updated
        parse_errors=None,
        created_at=datetime.now(),
        updated_at=datetime.now(),
    )
    await upsert_raw_annotation_file(raw_file_v2)
    
    # Verify raw annotation file was updated
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT parsed_status FROM raw_annotation_files WHERE raw_file_id = %s",
            (raw_file_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "parsed"


@pytest.mark.asyncio
async def test_upsert_raw_annotation_file_with_parse_errors(db_connection, test_uuids):
    """Test that upsert_raw_annotation_file handles parse errors correctly.
    
    Based on model showing parsed_status and parse_errors fields.
    """
    # Setup dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Create raw annotation file with parse errors
    raw_file_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    raw_file = RawAnnotationFile(
        raw_file_id=raw_file_id,
        dataset_id=test_uuids["dataset_1"],
        storage_provider="local",
        file_path="/data/annotations/malformed.csv",
        file_type="csv",
        file_name="malformed.csv",
        file_hash="def456ghi789",
        file_size=512,
        encoding="utf-8",
        parsed_status="error",
        parse_errors="Invalid CSV format: missing columns",
        created_at=datetime.now(),
    )
    
    await upsert_raw_annotation_file(raw_file)
    
    # Verify raw annotation file was created with errors
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT parsed_status, parse_errors FROM raw_annotation_files WHERE raw_file_id = %s",
            (raw_file_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "error"
    assert result[1] == "Invalid CSV format: missing columns"


@pytest.mark.asyncio
async def test_bulk_upsert_raw_annotation_files_creates_multiple_records(db_connection, test_uuids):
    """Test that bulk_upsert_raw_annotation_files creates multiple raw annotation file records.
    
    Based on docstring: 'Bulk upsert raw annotation file records.'
    """
    # Setup dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Create multiple raw annotation files
    raw_files = [
        RawAnnotationFile(
            raw_file_id=UUID(f"aaaaaaaa-aaaa-aaaa-aaaa-{str(i).zfill(12)}"),
            dataset_id=test_uuids["dataset_1"],
            storage_provider="local",
            file_path=f"/data/annotations/labels{i}.csv",
            file_type="csv",
            file_name=f"labels{i}.csv",
            file_hash=f"hash{i}",
            file_size=1024 * i,
            encoding="utf-8",
            parsed_status="not_parsed",
            created_at=datetime.now(),
        )
        for i in range(1, 4)
    ]
    
    rows_inserted = await bulk_upsert_raw_annotation_files(raw_files)
    
    # Should return count of rows inserted
    assert rows_inserted == 3
    
    # Verify raw annotation files were created (check for the specific hashes we created)
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT COUNT(*) FROM raw_annotation_files WHERE dataset_id = %s AND file_hash IN ('hash1', 'hash2', 'hash3')",
            (test_uuids["dataset_1"],)
        )
        count = await cur.fetchone()
        
    assert count[0] == 3


@pytest.mark.asyncio
async def test_bulk_upsert_raw_annotation_files_with_custom_batch_size(db_connection, test_uuids):
    """Test that bulk_upsert_raw_annotation_files respects custom batch_size parameter.
    
    Based on docstring signature showing batch_size parameter with default 1000.
    """
    # Setup dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Create multiple raw annotation files with unique hashes (starting from 100 to avoid conflicts)
    raw_files = [
        RawAnnotationFile(
            raw_file_id=UUID(f"00000000-0000-0000-0000-{str(i+100).zfill(12)}"),
            dataset_id=test_uuids["dataset_1"],
            storage_provider="local",
            file_path=f"/data/annotations/file{i+100}.csv",
            file_type="csv",
            file_name=f"file{i+100}.csv",
            file_hash=f"hash{i+100}",
            file_size=1024,
            encoding="utf-8",
            parsed_status="not_parsed",
            created_at=datetime.now(),
        )
        for i in range(5)
    ]
    
    # Use small batch size to test batching
    rows_inserted = await bulk_upsert_raw_annotation_files(raw_files, batch_size=2)
    
    assert rows_inserted == 5


@pytest.mark.asyncio
async def test_upsert_raw_annotation_file_idempotency(db_connection, test_uuids):
    """Test that upsert_raw_annotation_file is idempotent (re-inserting doesn't create duplicates).
    
    Based on docstring and requirement that ingestion must be idempotent.
    """
    # Setup dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    raw_file_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    
    # Create raw annotation file twice with same ID
    raw_file = RawAnnotationFile(
        raw_file_id=raw_file_id,
        dataset_id=test_uuids["dataset_1"],
        storage_provider="local",
        file_path="/data/annotations/labels.csv",
        file_type="csv",
        file_name="labels.csv",
        file_hash="abc123def456",
        file_size=1024,
        encoding="utf-8",
        parsed_status="not_parsed",
        created_at=datetime.now(),
    )
    
    await upsert_raw_annotation_file(raw_file)
    await upsert_raw_annotation_file(raw_file)  # Insert again
    
    # Verify only one record exists
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT COUNT(*) FROM raw_annotation_files WHERE raw_file_id = %s",
            (raw_file_id,)
        )
        count = await cur.fetchone()
        
    assert count[0] == 1
