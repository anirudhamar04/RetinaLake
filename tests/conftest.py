"""
Shared pytest fixtures for ChaksuDB tests.

Provides database connection fixtures, test data fixtures, and helper utilities.
"""

import asyncio
import pytest
import psycopg
from pathlib import Path
from typing import AsyncGenerator
from uuid import UUID

from chaksudb.config.config import get_db_async_connection_string
from chaksudb.db import connection as db_conn


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_db_url() -> str:
    """
    Get the database connection URL from environment.
    
    Uses existing database connection from chaksudb.config.config.
    """
    return get_db_async_connection_string()


@pytest.fixture(scope="session")
async def test_db_schema(test_db_url: str):
    """
    Apply database schema once per test session.
    
    Reads schema/schema.sql and applies it to the test database.
    """
    schema_path = Path(__file__).parent.parent / "schema" / "schema.sql"
    schema_sql = schema_path.read_text()
    
    # Connect directly to apply schema
    async with await psycopg.AsyncConnection.connect(test_db_url) as conn:
        # Drop all tables first (clean slate)
        await conn.execute("""
            DO $$ DECLARE
                r RECORD;
            BEGIN
                FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
                    EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
                END LOOP;
            END $$;
        """)
        
        # Apply schema
        await conn.execute(schema_sql)
        await conn.commit()
        
    yield
    
    # Cleanup after all tests
    async with await psycopg.AsyncConnection.connect(test_db_url) as conn:
        await conn.execute("""
            DO $$ DECLARE
                r RECORD;
            BEGIN
                FOR r IN (SELECT tablename FROM pg_tables WHERE schemaname = 'public') LOOP
                    EXECUTE 'DROP TABLE IF EXISTS ' || quote_ident(r.tablename) || ' CASCADE';
                END LOOP;
            END $$;
        """)
        await conn.commit()


@pytest.fixture(scope="function")
async def clean_pool():
    """
    Ensure the connection pool is clean before and after each test.
    
    This fixture closes any existing pool and sets it to None before the test runs,
    ensuring each test starts with a clean slate for pool management tests.
    """
    # Close and reset pool before test
    await db_conn.close_pool()
    
    yield
    
    # Close and reset pool after test
    await db_conn.close_pool()


@pytest.fixture(scope="function")
async def db_connection(test_db_schema) -> AsyncGenerator[psycopg.AsyncConnection, None]:
    """
    Provide a database connection for tests with transaction rollback.
    
    Each test runs in a transaction that is rolled back after the test completes,
    ensuring test isolation.
    
    Note: Due to the query functions using their own connections from the pool,
    tests should use unique, non-overlapping test data to avoid conflicts.
    """
    async with db_conn.get_connection() as conn:
        # Start a transaction
        async with conn.transaction():
            # Create a savepoint
            async with conn.transaction():
                yield conn
                # Rollback happens automatically when exiting the context


@pytest.fixture(scope="function")
def test_uuids() -> dict[str, UUID]:
    """
    Provide deterministic UUIDs for testing.
    
    Returns a dictionary of UUID objects for use in tests.
    """
    return {
        "dataset_1": UUID("11111111-1111-1111-1111-111111111111"),
        "dataset_2": UUID("22222222-2222-2222-2222-222222222222"),
        "patient_1": UUID("33333333-3333-3333-3333-333333333333"),
        "patient_2": UUID("44444444-4444-4444-4444-444444444444"),
        "image_1": UUID("55555555-5555-5555-5555-555555555555"),
        "image_2": UUID("66666666-6666-6666-6666-666666666666"),
        "expert_1": UUID("77777777-7777-7777-7777-777777777777"),
        "expert_2": UUID("88888888-8888-8888-8888-888888888888"),
        "annotation_1": UUID("99999999-9999-9999-9999-999999999999"),
        "annotation_2": UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa"),
    }


@pytest.fixture(scope="function")
async def test_dataset_in_db(test_uuids, test_db_schema):
    """
    Create a test dataset in the database for integration tests.
    
    This fixture ensures that tests requiring foreign key references to datasets
    have a valid dataset record to reference.
    """
    from chaksudb.db.models import Dataset
    from chaksudb.db.queries.datasets import upsert_dataset
    
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="TEST_DATASET",
        source_url="https://test.example.com",
        license="CC-BY-4.0",
        modality_types=["fundus"],
    )
    await upsert_dataset(dataset)
    return test_uuids["dataset_1"]


@pytest.fixture(scope="function")
async def test_image_in_db(test_dataset_in_db, request):
    """
    Create a test image in the database for integration tests.
    
    This fixture ensures that tests requiring foreign key references to images
    have a valid image record to reference. Each test gets a unique image to
    avoid duplicate key violations.
    """
    from chaksudb.db.models import Image
    from chaksudb.db.queries.images import upsert_image
    from chaksudb.ingest.framework.gen_uuid import generate_image_uuid
    import hashlib
    
    # Generate unique image ID based on test name to avoid collisions
    test_name = request.node.name
    hash_suffix = hashlib.md5(test_name.encode()).hexdigest()[:8]
    original_image_id = f"test_image_{hash_suffix}"
    
    image_id = generate_image_uuid(
        dataset_id=test_dataset_in_db,
        original_image_id=original_image_id
    )
    
    image = Image(
        image_id=image_id,
        dataset_id=test_dataset_in_db,
        original_image_id=original_image_id,
        storage_provider="local",
        file_path=f"/test/path/{original_image_id}.jpg",
        modality="fundus",
    )
    await upsert_image(image)
    return image_id


@pytest.fixture(scope="function")
async def test_images_in_db(test_dataset_in_db):
    """
    Create multiple test images in the database for bulk operation tests.
    
    Returns a list of 10 image UUIDs that can be used for testing bulk operations
    like split assignment, patient linking, etc.
    """
    from chaksudb.db.models import Image
    from chaksudb.db.queries.images import upsert_image
    from chaksudb.ingest.framework.gen_uuid import generate_image_uuid
    
    image_ids = []
    for i in range(10):
        original_image_id = f"test_image_{i:03d}"
        image_id = generate_image_uuid(
            dataset_id=test_dataset_in_db,
            original_image_id=original_image_id
        )
        image = Image(
            image_id=image_id,
            dataset_id=test_dataset_in_db,
            original_image_id=original_image_id,
            storage_provider="local",
            file_path=f"/test/path/{original_image_id}.jpg",
            modality="fundus",
        )
        await upsert_image(image)
        image_ids.append(image_id)
    
    return image_ids