"""
Transformation operation logging for tracking data transformations.

This module provides utilities for logging transformation operations that are
applied to annotations or data. Transformations are stored with their input data,
output data, operation parameters, and metadata for full auditability.

All transformations should be linked to provenance chains to maintain complete
traceability of data lineage.

Note: All database operations (queries, connections, models) are handled by
the internal.db module. This module only contains business logic and delegates
all database operations to internal.db.queries.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional

# All models and queries are imported from chaksudb.db - no database code here
from chaksudb.db.models import TransformationOperation
from chaksudb.db.queries import upsert_transformation_operation
from chaksudb.ingest.framework.gen_uuid import generate_transformation_uuid
from chaksudb.ingest.framework.hashing import compute_jsonb_hash

logger = logging.getLogger(__name__)


async def log_transformation(
    transformation_type: str,
    input_data: Optional[Dict[str, Any]] = None,
    output_data: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    operator: Optional[str] = None,
    notes: Optional[str] = None,
) -> uuid.UUID:
    """
    Log a transformation operation and store it in the database.

    This function creates a record of a transformation operation, storing
    the input data, output data, operation parameters, and metadata. The
    transformation is stored with a deterministic UUID based on the operation
    type and hashes of the input data and parameters.

    Args:
        transformation_type: Type of transformation operation (e.g., 'scale_grade',
            'normalize_mask', 'convert_format', 'consensus_aggregation')
        input_data: Optional dictionary containing input data for the transformation
            (stored as JSONB)
        output_data: Optional dictionary containing output data from the transformation
            (stored as JSONB)
        parameters: Optional dictionary containing operation parameters
            (stored as JSONB)
        operator: Optional identifier for who/what performed the transformation
            (e.g., user ID, model name, script name)
        notes: Optional free-text notes about the transformation

    Returns:
        transformation_id UUID of the logged transformation

    Example:
        ```python
        transformation_id = await log_transformation(
            transformation_type="scale_grade",
            input_data={"original_grade": "Mild", "scale": "ICDR"},
            output_data={"scaled_grade": 1, "scale": "ETDRS"},
            parameters={"mapping_method": "exact", "confidence": "high"},
            operator="ingestion_script_v1.0",
            notes="Converted ICDR grade to ETDRS scale"
        )
        ```
    """
    # Compute hashes for deterministic UUID generation
    input_hash = None
    if input_data:
        try:
            input_hash = compute_jsonb_hash(input_data)
        except Exception as e:
            logger.warning(
                f"Failed to compute hash for input_data: {e}. "
                "UUID will be generated without input hash."
            )

    params_hash = None
    if parameters:
        try:
            params_hash = compute_jsonb_hash(parameters)
        except Exception as e:
            logger.warning(
                f"Failed to compute hash for parameters: {e}. "
                "UUID will be generated without parameters hash."
            )

    # Generate deterministic UUID for the transformation
    transformation_id = generate_transformation_uuid(
        operation_type=transformation_type,
        input_data_hash=input_hash,
        operation_parameters_hash=params_hash,
    )

    # Create transformation operation model
    transformation = TransformationOperation(
        transformation_id=transformation_id,
        operation_type=transformation_type,
        input_data=input_data,
        output_data=output_data,
        operation_parameters=parameters,
        operation_timestamp=datetime.now(),
        operator=operator,
        notes=notes,
    )

    # Store in database (idempotent upsert)
    await upsert_transformation_operation(transformation)

    logger.debug(
        f"Logged transformation {transformation_id} of type {transformation_type}"
    )

    return transformation_id


async def log_and_link_transformation(
    chain_id: uuid.UUID,
    transformation_type: str,
    input_data: Optional[Dict[str, Any]] = None,
    output_data: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    operator: Optional[str] = None,
    notes: Optional[str] = None,
) -> uuid.UUID:
    """
    Log a transformation operation and automatically link it to a provenance chain.

    This is the recommended function for logging transformations as it ensures
    that every transformation is automatically linked to its provenance chain,
    maintaining complete traceability.

    Args:
        chain_id: UUID of the provenance chain to link this transformation to
        transformation_type: Type of transformation operation (e.g., 'scale_grade',
            'normalize_mask', 'convert_format', 'consensus_aggregation')
        input_data: Optional dictionary containing input data for the transformation
            (stored as JSONB)
        output_data: Optional dictionary containing output data from the transformation
            (stored as JSONB)
        parameters: Optional dictionary containing operation parameters
            (stored as JSONB)
        operator: Optional identifier for who/what performed the transformation
            (e.g., user ID, model name, script name)
        notes: Optional free-text notes about the transformation

    Returns:
        transformation_id UUID of the logged transformation

    Example:
        ```python
        from chaksudb.ingest.framework.transformations import log_and_link_transformation

        # Log a transformation and automatically link it to the provenance chain
        transformation_id = await log_and_link_transformation(
            chain_id=provenance_chain_id,
            transformation_type="scale_grade",
            input_data={"original_grade": "Mild", "scale": "ICDR"},
            output_data={"scaled_grade": 1, "scale": "ETDRS"},
            parameters={"mapping_method": "exact", "confidence": "high"},
            operator="ingestion_script_v1.0",
            notes="Converted ICDR grade to ETDRS scale"
        )
        ```
    """
    # Import here to avoid circular imports
    from chaksudb.ingest.framework.provenance import link_transformation

    # Log the transformation
    transformation_id = await log_transformation(
        transformation_type=transformation_type,
        input_data=input_data,
        output_data=output_data,
        parameters=parameters,
        operator=operator,
        notes=notes,
    )

    # Automatically link to provenance chain
    await link_transformation(chain_id=chain_id, transformation_id=transformation_id)

    logger.debug(
        f"Logged and linked transformation {transformation_id} of type {transformation_type} "
        f"to provenance chain {chain_id}"
    )

    return transformation_id
