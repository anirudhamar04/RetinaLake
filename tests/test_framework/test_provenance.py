"""
Tests for chaksudb/ingest/framework/provenance.py

Tests provenance chain management functions based on their docstrings.
"""

import pytest
from uuid import UUID
from datetime import datetime

from chaksudb.ingest.framework.provenance import (
    create_provenance_chain,
    link_transformation,
    create_provenance_chain_for_raw_file,
    ingest_raw_annotation_file_with_provenance,
    apply_transformation_to_chain,
    create_transformed_provenance_chain,
)
from chaksudb.db.queries import (
    upsert_dataset,
)
from chaksudb.db.models import Dataset, RawAnnotationFile
from chaksudb.ingest.framework.gen_uuid import (
    generate_dataset_uuid,
    generate_raw_file_uuid,
    generate_provenance_chain_uuid,
)
from chaksudb.ingest.framework.hashing import compute_content_hash


class TestCreateProvenanceChain:
    """Tests for create_provenance_chain function."""

    @pytest.mark.asyncio
    async def test_create_provenance_chain_returns_uuid(self, db_connection):
        """Test that create_provenance_chain returns a UUID."""
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        
        assert isinstance(chain_id, UUID)

    @pytest.mark.asyncio
    async def test_create_provenance_chain_stores_record(self, db_connection):
        """Test that create_provenance_chain creates a provenance chain record."""
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        
        # Verify chain was created
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT unified_annotation_type, source_type FROM provenance_chain WHERE chain_id = %s",
                (chain_id,)
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "grading"
            assert row[1] == "original"

    @pytest.mark.asyncio
    async def test_create_provenance_chain_with_root_source(self, db_connection):
        """Test that create_provenance_chain stores root_source_raw_data_id."""
        # Setup: create dataset and raw file
        dataset_id = generate_dataset_uuid(dataset_name="TestDataset")
        dataset = Dataset(dataset_id=dataset_id, dataset_name="TestDataset")
        await upsert_dataset(dataset)
        
        file_hash = compute_content_hash(b"test content")
        raw_file_id = generate_raw_file_uuid(dataset_id=dataset_id, file_hash=file_hash)
        
        # Create the raw file first
        from chaksudb.db.models import RawAnnotationFile
        from chaksudb.db.queries import upsert_raw_annotation_file
        raw_file = RawAnnotationFile(
            raw_file_id=raw_file_id,
            dataset_id=dataset_id,
            file_path="/test/path.csv",
            file_hash=file_hash,
            file_type="csv",
            file_name="test.csv",
        )
        await upsert_raw_annotation_file(raw_file)
        
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
            root_source_raw_data_id=raw_file_id,
        )
        
        # Verify root_source_raw_data_id is stored
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT root_source_raw_data_id FROM provenance_chain WHERE chain_id = %s",
                (chain_id,)
            )
            row = await cur.fetchone()
            assert row[0] == raw_file_id

    @pytest.mark.asyncio
    async def test_create_provenance_chain_with_source_annotation_ids(self, db_connection):
        """Test that create_provenance_chain stores source_annotation_ids."""
        source_ids = [
            UUID("11111111-1111-1111-1111-111111111111"),
            UUID("22222222-2222-2222-2222-222222222222"),
        ]
        
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="transformed",
            source_annotation_ids=source_ids,
        )
        
        # Verify source_annotation_ids are stored
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT source_annotation_ids FROM provenance_chain WHERE chain_id = %s",
                (chain_id,)
            )
            row = await cur.fetchone()
            assert row[0] == source_ids

    @pytest.mark.asyncio
    async def test_create_provenance_chain_handles_none_source_annotation_ids(self, db_connection):
        """Test that create_provenance_chain handles None source_annotation_ids."""
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
            source_annotation_ids=None,
        )
        
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT source_annotation_ids FROM provenance_chain WHERE chain_id = %s",
                (chain_id,)
            )
            row = await cur.fetchone()
            # PostgreSQL may return [] for empty array or None
            assert row[0] is None or row[0] == []

    @pytest.mark.asyncio
    async def test_create_provenance_chain_is_idempotent(self, db_connection):
        """Test that create_provenance_chain is idempotent (upsert)."""
        chain_id1 = await create_provenance_chain(
            unified_annotation_type="segmentation",
            source_type="original",
        )
        
        chain_id2 = await create_provenance_chain(
            unified_annotation_type="segmentation",
            source_type="original",
        )
        
        # Should be deterministic
        assert chain_id1 == chain_id2

    @pytest.mark.asyncio
    async def test_create_provenance_chain_different_types(self, db_connection):
        """Test that create_provenance_chain supports different annotation types."""
        annotation_types = ["grading", "segmentation", "classification", "localization", "quality", "keyword", "description"]
        
        for ann_type in annotation_types:
            chain_id = await create_provenance_chain(
                unified_annotation_type=ann_type,
                source_type="original",
            )
            
            assert isinstance(chain_id, UUID)

    @pytest.mark.asyncio
    async def test_create_provenance_chain_different_source_types(self, db_connection):
        """Test that create_provenance_chain supports different source types."""
        source_types = ["original", "transformed", "pseudo_generated", "consensus"]
        
        for source_type in source_types:
            chain_id = await create_provenance_chain(
                unified_annotation_type="grading",
                source_type=source_type,
            )
            
            assert isinstance(chain_id, UUID)


class TestLinkTransformation:
    """Tests for link_transformation function."""

    @pytest.mark.asyncio
    async def test_link_transformation_creates_link(self, db_connection):
        """Test that link_transformation creates a link between chain and transformation."""
        # Setup: create a provenance chain
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        
        # Create a transformation first
        from chaksudb.ingest.framework.transformations import log_transformation
        transformation_id = await log_transformation(
            transformation_type="test_transformation",
            input_data={"test": "data"},
        )
        
        # Link transformation (returns None per docstring)
        result = await link_transformation(
            chain_id=chain_id,
            transformation_id=transformation_id,
        )
        
        assert result is None
        
        # Verify link was created
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT chain_id, transformation_id FROM provenance_transformations WHERE chain_id = %s AND transformation_id = %s",
                (chain_id, transformation_id)
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == chain_id
            assert row[1] == transformation_id

    @pytest.mark.asyncio
    async def test_link_transformation_is_idempotent(self, db_connection):
        """Test that link_transformation is idempotent (upsert)."""
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        
        # Create a transformation first
        from chaksudb.ingest.framework.transformations import log_transformation
        transformation_id = await log_transformation(
            transformation_type="test_transformation_2",
            input_data={"test": "data2"},
        )
        
        # Link twice
        await link_transformation(chain_id=chain_id, transformation_id=transformation_id)
        await link_transformation(chain_id=chain_id, transformation_id=transformation_id)
        
        # Verify only one link exists
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM provenance_transformations WHERE chain_id = %s AND transformation_id = %s",
                (chain_id, transformation_id)
            )
            count = (await cur.fetchone())[0]
            assert count == 1


class TestCreateProvenanceChainForRawFile:
    """Tests for create_provenance_chain_for_raw_file function."""

    @pytest.mark.asyncio
    async def test_create_provenance_chain_for_raw_file_returns_uuid(self, db_connection):
        """Test that create_provenance_chain_for_raw_file returns a UUID."""
        # Setup: create dataset and raw file first
        dataset_id = generate_dataset_uuid(dataset_name="TestDataset_raw1")
        dataset = Dataset(dataset_id=dataset_id, dataset_name="TestDataset_raw1")
        await upsert_dataset(dataset)
        
        file_hash = compute_content_hash(b"test content raw1")
        raw_file_id = generate_raw_file_uuid(dataset_id=dataset_id, file_hash=file_hash)
        
        from chaksudb.db.models import RawAnnotationFile
        from chaksudb.db.queries import upsert_raw_annotation_file
        raw_file = RawAnnotationFile(
            raw_file_id=raw_file_id,
            dataset_id=dataset_id,
            file_path="/test/raw1.csv",
            file_hash=file_hash,
            file_type="csv",
            file_name="raw1.csv",
        )
        await upsert_raw_annotation_file(raw_file)
        
        chain_id = await create_provenance_chain_for_raw_file(
            raw_file_id=raw_file_id,
            unified_annotation_type="grading",
        )
        
        assert isinstance(chain_id, UUID)

    @pytest.mark.asyncio
    async def test_create_provenance_chain_for_raw_file_sets_source_type_original(self, db_connection):
        """Test that create_provenance_chain_for_raw_file sets source_type to 'original'."""
        # Setup: create dataset and raw file first
        dataset_id = generate_dataset_uuid(dataset_name="TestDataset_raw2")
        dataset = Dataset(dataset_id=dataset_id, dataset_name="TestDataset_raw2")
        await upsert_dataset(dataset)
        
        file_hash = compute_content_hash(b"test content raw2")
        raw_file_id = generate_raw_file_uuid(dataset_id=dataset_id, file_hash=file_hash)
        
        from chaksudb.db.models import RawAnnotationFile
        from chaksudb.db.queries import upsert_raw_annotation_file
        raw_file = RawAnnotationFile(
            raw_file_id=raw_file_id,
            dataset_id=dataset_id,
            file_path="/test/raw2.csv",
            file_hash=file_hash,
            file_type="csv",
            file_name="raw2.csv",
        )
        await upsert_raw_annotation_file(raw_file)
        
        chain_id = await create_provenance_chain_for_raw_file(
            raw_file_id=raw_file_id,
            unified_annotation_type="grading",
        )
        
        # Verify source_type is 'original'
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT source_type, root_source_raw_data_id FROM provenance_chain WHERE chain_id = %s",
                (chain_id,)
            )
            row = await cur.fetchone()
            assert row[0] == "original"
            assert row[1] == raw_file_id

    @pytest.mark.asyncio
    async def test_create_provenance_chain_for_raw_file_links_to_raw_file(self, db_connection):
        """Test that create_provenance_chain_for_raw_file links chain to raw file."""
        # Setup: create dataset and raw file first
        dataset_id = generate_dataset_uuid(dataset_name="TestDataset_raw3")
        dataset = Dataset(dataset_id=dataset_id, dataset_name="TestDataset_raw3")
        await upsert_dataset(dataset)
        
        file_hash = compute_content_hash(b"test content raw3")
        raw_file_id = generate_raw_file_uuid(dataset_id=dataset_id, file_hash=file_hash)
        
        from chaksudb.db.models import RawAnnotationFile
        from chaksudb.db.queries import upsert_raw_annotation_file
        raw_file = RawAnnotationFile(
            raw_file_id=raw_file_id,
            dataset_id=dataset_id,
            file_path="/test/raw3.csv",
            file_hash=file_hash,
            file_type="csv",
            file_name="raw3.csv",
        )
        await upsert_raw_annotation_file(raw_file)
        
        chain_id = await create_provenance_chain_for_raw_file(
            raw_file_id=raw_file_id,
            unified_annotation_type="segmentation",
        )
        
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT root_source_raw_data_id FROM provenance_chain WHERE chain_id = %s",
                (chain_id,)
            )
            row = await cur.fetchone()
            assert row[0] == raw_file_id


class TestIngestRawAnnotationFileWithProvenance:
    """Tests for ingest_raw_annotation_file_with_provenance function."""

    @pytest.mark.asyncio
    async def test_ingest_raw_annotation_file_with_provenance_returns_tuple(self, db_connection):
        """Test that ingest_raw_annotation_file_with_provenance returns (raw_file_id, chain_id) tuple."""
        # Setup
        dataset_id = generate_dataset_uuid(dataset_name="TestDataset")
        dataset = Dataset(dataset_id=dataset_id, dataset_name="TestDataset")
        await upsert_dataset(dataset)
        
        file_hash = compute_content_hash(b"test annotation data")
        raw_file_id = generate_raw_file_uuid(dataset_id=dataset_id, file_hash=file_hash)
        
        raw_file = RawAnnotationFile(
            raw_file_id=raw_file_id,
            dataset_id=dataset_id,
            file_path="/path/to/annotations.csv",
            file_hash=file_hash,
            file_type="csv",
            file_name="annotations.csv",
        )
        
        result = await ingest_raw_annotation_file_with_provenance(
            raw_file=raw_file,
            unified_annotation_type="grading",
        )
        
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], UUID)
        assert isinstance(result[1], UUID)

    @pytest.mark.asyncio
    async def test_ingest_raw_annotation_file_with_provenance_stores_raw_file(self, db_connection):
        """Test that ingest_raw_annotation_file_with_provenance stores the raw file."""
        # Setup
        dataset_id = generate_dataset_uuid(dataset_name="TestDataset")
        dataset = Dataset(dataset_id=dataset_id, dataset_name="TestDataset")
        await upsert_dataset(dataset)
        
        file_hash = compute_content_hash(b"test annotation data")
        raw_file_id = generate_raw_file_uuid(dataset_id=dataset_id, file_hash=file_hash)
        
        raw_file = RawAnnotationFile(
            raw_file_id=raw_file_id,
            dataset_id=dataset_id,
            file_path="/path/to/annotations.csv",
            file_hash=file_hash,
            file_type="csv",
            file_name="annotations.csv",
        )
        
        returned_file_id, chain_id = await ingest_raw_annotation_file_with_provenance(
            raw_file=raw_file,
            unified_annotation_type="grading",
        )
        
        # Verify raw file was stored
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT file_name, file_type FROM raw_annotation_files WHERE raw_file_id = %s",
                (returned_file_id,)
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "annotations.csv"
            assert row[1] == "csv"

    @pytest.mark.asyncio
    async def test_ingest_raw_annotation_file_with_provenance_creates_chain(self, db_connection):
        """Test that ingest_raw_annotation_file_with_provenance creates initial provenance chain."""
        # Setup
        dataset_id = generate_dataset_uuid(dataset_name="TestDataset")
        dataset = Dataset(dataset_id=dataset_id, dataset_name="TestDataset")
        await upsert_dataset(dataset)
        
        file_hash = compute_content_hash(b"test annotation data")
        raw_file_id = generate_raw_file_uuid(dataset_id=dataset_id, file_hash=file_hash)
        
        raw_file = RawAnnotationFile(
            raw_file_id=raw_file_id,
            dataset_id=dataset_id,
            file_path="/path/to/annotations.csv",
            file_hash=file_hash,
            file_type="csv",
            file_name="annotations.csv",
        )
        
        returned_file_id, chain_id = await ingest_raw_annotation_file_with_provenance(
            raw_file=raw_file,
            unified_annotation_type="grading",
        )
        
        # Verify provenance chain was created
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT unified_annotation_type, source_type, root_source_raw_data_id FROM provenance_chain WHERE chain_id = %s",
                (chain_id,)
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "grading"
            assert row[1] == "original"
            assert row[2] == raw_file_id


class TestApplyTransformationToChain:
    """Tests for apply_transformation_to_chain function."""

    @pytest.mark.asyncio
    async def test_apply_transformation_to_chain_returns_uuid(self, db_connection):
        """Test that apply_transformation_to_chain returns transformation UUID."""
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        
        transformation_id = await apply_transformation_to_chain(
            chain_id=chain_id,
            transformation_type="scale_grade",
            input_data={"original_grade": "Mild", "scale": "ICDR"},
            output_data={"scaled_grade": 1, "scale": "ETDRS"},
        )
        
        assert isinstance(transformation_id, UUID)

    @pytest.mark.asyncio
    async def test_apply_transformation_to_chain_logs_and_links(self, db_connection):
        """Test that apply_transformation_to_chain logs transformation and links it to chain."""
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        
        transformation_id = await apply_transformation_to_chain(
            chain_id=chain_id,
            transformation_type="scale_grade",
            input_data={"original_grade": "Mild"},
            output_data={"scaled_grade": 1},
            parameters={"mapping_method": "exact"},
            operator="test_script",
            notes="Test transformation",
        )
        
        # Verify transformation was logged
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT operation_type FROM transformation_operations WHERE transformation_id = %s",
                (transformation_id,)
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "scale_grade"
        
        # Verify link was created
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT chain_id FROM provenance_transformations WHERE transformation_id = %s",
                (transformation_id,)
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == chain_id


class TestCreateTransformedProvenanceChain:
    """Tests for create_transformed_provenance_chain function."""

    @pytest.mark.asyncio
    async def test_create_transformed_provenance_chain_returns_uuid(self, db_connection):
        """Test that create_transformed_provenance_chain returns a UUID."""
        source_ids = [
            UUID("11111111-1111-1111-1111-111111111111"),
            UUID("22222222-2222-2222-2222-222222222222"),
        ]
        
        chain_id = await create_transformed_provenance_chain(
            unified_annotation_type="grading",
            source_annotation_ids=source_ids,
        )
        
        assert isinstance(chain_id, UUID)

    @pytest.mark.asyncio
    async def test_create_transformed_provenance_chain_sets_source_type_transformed(self, db_connection):
        """Test that create_transformed_provenance_chain sets source_type to 'transformed'."""
        source_ids = [UUID("33333333-3333-3333-3333-333333333333")]
        
        chain_id = await create_transformed_provenance_chain(
            unified_annotation_type="grading",
            source_annotation_ids=source_ids,
        )
        
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT source_type FROM provenance_chain WHERE chain_id = %s",
                (chain_id,)
            )
            row = await cur.fetchone()
            assert row[0] == "transformed"

    @pytest.mark.asyncio
    async def test_create_transformed_provenance_chain_stores_source_annotation_ids(self, db_connection):
        """Test that create_transformed_provenance_chain stores source_annotation_ids."""
        source_ids = [
            UUID("44444444-4444-4444-4444-444444444444"),
            UUID("55555555-5555-5555-5555-555555555555"),
        ]
        
        chain_id = await create_transformed_provenance_chain(
            unified_annotation_type="grading",
            source_annotation_ids=source_ids,
        )
        
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT source_annotation_ids FROM provenance_chain WHERE chain_id = %s",
                (chain_id,)
            )
            row = await cur.fetchone()
            assert row[0] == source_ids

    @pytest.mark.asyncio
    async def test_create_transformed_provenance_chain_links_transformation_if_provided(self, db_connection):
        """Test that create_transformed_provenance_chain links transformation if transformation_id is provided."""
        # First create a transformation
        original_chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        
        transformation_id = await apply_transformation_to_chain(
            chain_id=original_chain_id,
            transformation_type="scale_grade",
        )
        
        # Now create transformed chain with transformation link
        source_ids = [UUID("66666666-6666-6666-6666-666666666666")]
        
        new_chain_id = await create_transformed_provenance_chain(
            unified_annotation_type="grading",
            source_annotation_ids=source_ids,
            transformation_id=transformation_id,
        )
        
        # Verify transformation was linked
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT transformation_id FROM provenance_transformations WHERE chain_id = %s",
                (new_chain_id,)
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == transformation_id

    @pytest.mark.asyncio
    async def test_create_transformed_provenance_chain_preserves_root_source(self, db_connection):
        """Test that create_transformed_provenance_chain can preserve root_source_raw_data_id."""
        # Setup: create dataset and raw file first
        dataset_id = generate_dataset_uuid(dataset_name="TestDataset_raw4")
        dataset = Dataset(dataset_id=dataset_id, dataset_name="TestDataset_raw4")
        await upsert_dataset(dataset)
        
        file_hash = compute_content_hash(b"test content raw4")
        root_raw_file_id = generate_raw_file_uuid(dataset_id=dataset_id, file_hash=file_hash)
        
        from chaksudb.db.models import RawAnnotationFile
        from chaksudb.db.queries import upsert_raw_annotation_file
        raw_file = RawAnnotationFile(
            raw_file_id=root_raw_file_id,
            dataset_id=dataset_id,
            file_path="/test/raw4.csv",
            file_hash=file_hash,
            file_type="csv",
            file_name="raw4.csv",
        )
        await upsert_raw_annotation_file(raw_file)
        
        source_ids = [UUID("88888888-8888-8888-8888-888888888888")]
        
        chain_id = await create_transformed_provenance_chain(
            unified_annotation_type="grading",
            source_annotation_ids=source_ids,
            root_source_raw_data_id=root_raw_file_id,
        )
        
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT root_source_raw_data_id FROM provenance_chain WHERE chain_id = %s",
                (chain_id,)
            )
            row = await cur.fetchone()
            assert row[0] == root_raw_file_id
