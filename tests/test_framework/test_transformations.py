"""
Tests for chaksudb/ingest/framework/transformations.py

Tests transformation operation logging functions based on their docstrings.
"""

import pytest
from uuid import UUID
from datetime import datetime

from chaksudb.ingest.framework.transformations import (
    log_transformation,
    log_and_link_transformation,
)
from chaksudb.ingest.framework.provenance import create_provenance_chain


class TestLogTransformation:
    """Tests for log_transformation function."""

    @pytest.mark.asyncio
    async def test_log_transformation_returns_uuid(self, db_connection):
        """Test that log_transformation returns a UUID."""
        transformation_id = await log_transformation(
            transformation_type="scale_grade",
        )
        
        assert isinstance(transformation_id, UUID)

    @pytest.mark.asyncio
    async def test_log_transformation_stores_operation(self, db_connection):
        """Test that log_transformation stores transformation operation in database."""
        transformation_id = await log_transformation(
            transformation_type="scale_grade",
            input_data={"original_grade": "Mild", "scale": "ICDR"},
            output_data={"scaled_grade": 1, "scale": "ETDRS"},
        )
        
        # Verify transformation was stored
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT operation_type FROM transformation_operations WHERE transformation_id = %s",
                (transformation_id,)
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "scale_grade"

    @pytest.mark.asyncio
    async def test_log_transformation_stores_input_data(self, db_connection):
        """Test that log_transformation stores input_data as JSONB."""
        input_data = {"original_grade": "Moderate", "scale": "ICDR", "value": 2}
        
        transformation_id = await log_transformation(
            transformation_type="scale_grade",
            input_data=input_data,
        )
        
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT input_data FROM transformation_operations WHERE transformation_id = %s",
                (transformation_id,)
            )
            row = await cur.fetchone()
            assert row[0] == input_data

    @pytest.mark.asyncio
    async def test_log_transformation_stores_output_data(self, db_connection):
        """Test that log_transformation stores output_data as JSONB."""
        output_data = {"scaled_grade": 2, "scale": "ETDRS", "confidence": 0.95}
        
        transformation_id = await log_transformation(
            transformation_type="scale_grade",
            output_data=output_data,
        )
        
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT output_data FROM transformation_operations WHERE transformation_id = %s",
                (transformation_id,)
            )
            row = await cur.fetchone()
            assert row[0] == output_data

    @pytest.mark.asyncio
    async def test_log_transformation_stores_parameters(self, db_connection):
        """Test that log_transformation stores operation parameters as JSONB."""
        parameters = {"mapping_method": "exact", "confidence": "high", "threshold": 0.5}
        
        transformation_id = await log_transformation(
            transformation_type="normalize_mask",
            parameters=parameters,
        )
        
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT operation_parameters FROM transformation_operations WHERE transformation_id = %s",
                (transformation_id,)
            )
            row = await cur.fetchone()
            assert row[0] == parameters

    @pytest.mark.asyncio
    async def test_log_transformation_stores_operator(self, db_connection):
        """Test that log_transformation stores operator identifier."""
        transformation_id = await log_transformation(
            transformation_type="convert_format",
            operator="ingestion_script_v1.0",
        )
        
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT operator FROM transformation_operations WHERE transformation_id = %s",
                (transformation_id,)
            )
            row = await cur.fetchone()
            assert row[0] == "ingestion_script_v1.0"

    @pytest.mark.asyncio
    async def test_log_transformation_stores_notes(self, db_connection):
        """Test that log_transformation stores notes."""
        notes = "Converted ICDR grade to ETDRS scale using exact mapping"
        
        transformation_id = await log_transformation(
            transformation_type="scale_grade",
            notes=notes,
        )
        
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT notes FROM transformation_operations WHERE transformation_id = %s",
                (transformation_id,)
            )
            row = await cur.fetchone()
            assert row[0] == notes

    @pytest.mark.asyncio
    async def test_log_transformation_is_deterministic(self, db_connection):
        """Test that log_transformation generates deterministic UUID for same inputs."""
        input_data = {"grade": "Mild"}
        parameters = {"method": "exact"}
        
        transformation_id1 = await log_transformation(
            transformation_type="scale_grade",
            input_data=input_data,
            parameters=parameters,
        )
        
        transformation_id2 = await log_transformation(
            transformation_type="scale_grade",
            input_data=input_data,
            parameters=parameters,
        )
        
        # Should be deterministic (same inputs = same UUID)
        assert transformation_id1 == transformation_id2

    @pytest.mark.asyncio
    async def test_log_transformation_is_idempotent(self, db_connection):
        """Test that log_transformation is idempotent (upsert)."""
        transformation_id = await log_transformation(
            transformation_type="scale_grade",
            input_data={"grade": "Test"},
            output_data={"scaled": 1},
        )
        
        # Call again with same inputs
        transformation_id2 = await log_transformation(
            transformation_type="scale_grade",
            input_data={"grade": "Test"},
            output_data={"scaled": 1},
        )
        
        assert transformation_id == transformation_id2
        
        # Verify only one record exists
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM transformation_operations WHERE transformation_id = %s",
                (transformation_id,)
            )
            count = (await cur.fetchone())[0]
            assert count == 1

    @pytest.mark.asyncio
    async def test_log_transformation_handles_none_values(self, db_connection):
        """Test that log_transformation handles None values for optional fields."""
        transformation_id = await log_transformation(
            transformation_type="test_operation",
            input_data=None,
            output_data=None,
            parameters=None,
            operator=None,
            notes=None,
        )
        
        assert isinstance(transformation_id, UUID)
        
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT input_data, output_data, operation_parameters, operator, notes FROM transformation_operations WHERE transformation_id = %s",
                (transformation_id,)
            )
            row = await cur.fetchone()
            assert all(v is None for v in row)

    @pytest.mark.asyncio
    async def test_log_transformation_handles_complex_nested_data(self, db_connection):
        """Test that log_transformation handles complex nested JSONB data."""
        complex_data = {
            "outer": {
                "inner": {
                    "values": [1, 2, 3],
                    "metadata": {"key": "value"}
                }
            },
            "list": [{"item": 1}, {"item": 2}]
        }
        
        transformation_id = await log_transformation(
            transformation_type="complex_operation",
            input_data=complex_data,
        )
        
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT input_data FROM transformation_operations WHERE transformation_id = %s",
                (transformation_id,)
            )
            row = await cur.fetchone()
            assert row[0] == complex_data

    @pytest.mark.asyncio
    async def test_log_transformation_different_types(self, db_connection):
        """Test that log_transformation supports different transformation types."""
        transformation_types = [
            "scale_grade",
            "normalize_mask",
            "convert_format",
            "consensus_aggregation",
            "quality_check",
            "data_augmentation",
        ]
        
        for trans_type in transformation_types:
            transformation_id = await log_transformation(
                transformation_type=trans_type,
            )
            
            assert isinstance(transformation_id, UUID)


class TestLogAndLinkTransformation:
    """Tests for log_and_link_transformation function."""

    @pytest.mark.asyncio
    async def test_log_and_link_transformation_returns_uuid(self, db_connection):
        """Test that log_and_link_transformation returns a UUID."""
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        
        transformation_id = await log_and_link_transformation(
            chain_id=chain_id,
            transformation_type="scale_grade",
        )
        
        assert isinstance(transformation_id, UUID)

    @pytest.mark.asyncio
    async def test_log_and_link_transformation_logs_transformation(self, db_connection):
        """Test that log_and_link_transformation logs the transformation operation."""
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        
        transformation_id = await log_and_link_transformation(
            chain_id=chain_id,
            transformation_type="scale_grade",
            input_data={"grade": "Mild"},
            output_data={"scaled": 1},
        )
        
        # Verify transformation was logged
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT operation_type, input_data, output_data FROM transformation_operations WHERE transformation_id = %s",
                (transformation_id,)
            )
            row = await cur.fetchone()
            assert row is not None
            assert row[0] == "scale_grade"
            assert row[1] == {"grade": "Mild"}
            assert row[2] == {"scaled": 1}

    @pytest.mark.asyncio
    async def test_log_and_link_transformation_links_to_chain(self, db_connection):
        """Test that log_and_link_transformation automatically links transformation to provenance chain."""
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        
        transformation_id = await log_and_link_transformation(
            chain_id=chain_id,
            transformation_type="scale_grade",
        )
        
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
    async def test_log_and_link_transformation_with_all_parameters(self, db_connection):
        """Test that log_and_link_transformation stores all parameters correctly."""
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        
        input_data = {"original_grade": "Severe", "scale": "ICDR"}
        output_data = {"scaled_grade": 3, "scale": "ETDRS"}
        parameters = {"mapping_method": "exact", "confidence": "high"}
        operator = "ingestion_script_v2.0"
        notes = "Test transformation with all parameters"
        
        transformation_id = await log_and_link_transformation(
            chain_id=chain_id,
            transformation_type="scale_grade",
            input_data=input_data,
            output_data=output_data,
            parameters=parameters,
            operator=operator,
            notes=notes,
        )
        
        # Verify all fields were stored
        async with db_connection.cursor() as cur:
            await cur.execute(
                """SELECT operation_type, input_data, output_data, operation_parameters, operator, notes 
                   FROM transformation_operations WHERE transformation_id = %s""",
                (transformation_id,)
            )
            row = await cur.fetchone()
            assert row[0] == "scale_grade"
            assert row[1] == input_data
            assert row[2] == output_data
            assert row[3] == parameters
            assert row[4] == operator
            assert row[5] == notes

    @pytest.mark.asyncio
    async def test_log_and_link_transformation_is_idempotent(self, db_connection):
        """Test that log_and_link_transformation is idempotent."""
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        
        input_data = {"test": "data"}
        
        transformation_id1 = await log_and_link_transformation(
            chain_id=chain_id,
            transformation_type="test_op",
            input_data=input_data,
        )
        
        transformation_id2 = await log_and_link_transformation(
            chain_id=chain_id,
            transformation_type="test_op",
            input_data=input_data,
        )
        
        # Should generate same transformation ID
        assert transformation_id1 == transformation_id2
        
        # Verify only one link exists
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM provenance_transformations WHERE chain_id = %s AND transformation_id = %s",
                (chain_id, transformation_id1)
            )
            count = (await cur.fetchone())[0]
            assert count == 1

    @pytest.mark.asyncio
    async def test_log_and_link_transformation_multiple_chains(self, db_connection):
        """Test that log_and_link_transformation can link same transformation to multiple chains."""
        # Create two chains
        chain_id1 = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        chain_id2 = await create_provenance_chain(
            unified_annotation_type="segmentation",
            source_type="original",
        )
        
        input_data = {"shared": "data"}
        
        # Link same transformation to both chains
        transformation_id1 = await log_and_link_transformation(
            chain_id=chain_id1,
            transformation_type="shared_op",
            input_data=input_data,
        )
        
        transformation_id2 = await log_and_link_transformation(
            chain_id=chain_id2,
            transformation_type="shared_op",
            input_data=input_data,
        )
        
        # Should be same transformation
        assert transformation_id1 == transformation_id2
        
        # Verify links to both chains exist
        async with db_connection.cursor() as cur:
            await cur.execute(
                "SELECT COUNT(*) FROM provenance_transformations WHERE transformation_id = %s",
                (transformation_id1,)
            )
            count = (await cur.fetchone())[0]
            assert count == 2

    @pytest.mark.asyncio
    async def test_log_and_link_transformation_maintains_traceability(self, db_connection):
        """Test that log_and_link_transformation maintains complete traceability."""
        chain_id = await create_provenance_chain(
            unified_annotation_type="grading",
            source_type="original",
        )
        
        transformation_id = await log_and_link_transformation(
            chain_id=chain_id,
            transformation_type="scale_grade",
            input_data={"original": "value"},
            output_data={"transformed": "result"},
            operator="test_operator",
        )
        
        # Verify full traceability: transformation exists and is linked
        async with db_connection.cursor() as cur:
            # Check transformation exists
            await cur.execute(
                "SELECT COUNT(*) FROM transformation_operations WHERE transformation_id = %s",
                (transformation_id,)
            )
            trans_count = (await cur.fetchone())[0]
            assert trans_count == 1
            
            # Check link exists
            await cur.execute(
                "SELECT COUNT(*) FROM provenance_transformations WHERE chain_id = %s AND transformation_id = %s",
                (chain_id, transformation_id)
            )
            link_count = (await cur.fetchone())[0]
            assert link_count == 1
            
            # Check chain exists
            await cur.execute(
                "SELECT COUNT(*) FROM provenance_chain WHERE chain_id = %s",
                (chain_id,)
            )
            chain_count = (await cur.fetchone())[0]
            assert chain_count == 1
