"""
Provenance chain management for tracking annotation sources and transformations.

This module provides utilities for creating and managing provenance chains that
track the lineage of annotations from their original sources through any transformations.
Provenance chains link annotations to their root sources (raw annotation files)
and track any transformations applied to them.

Note: All database operations (queries, connections, models) are handled by
the internal.db module. This module only contains business logic and delegates
all database operations to internal.db.queries.
"""

import logging
import uuid
from datetime import datetime
from typing import Any, Dict, Optional, Tuple

# All models and queries are imported from chaksudb.db - no database code here
from chaksudb.db.models import ProvenanceChain, ProvenanceTransformation, RawAnnotationFile
from chaksudb.db.queries import (
    fetch_grade_conversions_for_audit,
    upsert_provenance_chain,
    upsert_provenance_transformation,
    upsert_raw_annotation_file,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_provenance_chain_uuid,
    generate_provenance_transformation_uuid,
)

logger = logging.getLogger(__name__)


async def create_provenance_chain(
    unified_annotation_type: str,
    source_type: str,
    root_source_raw_data_id: Optional[uuid.UUID] = None,
    source_annotation_ids: Optional[list[uuid.UUID]] = None,
) -> uuid.UUID:
    """
    Create a provenance chain record and store it in the database.

    A provenance chain tracks the lineage of an annotation from its original
    source through any transformations. It links annotations to their root
    sources (raw annotation files) and tracks the source type.

    Args:
        unified_annotation_type: Unified annotation type ('grading', 'segmentation',
            'classification', 'localization', 'quality', 'keyword', 'description')
        source_type: Source type ('original', 'transformed', 'pseudo_generated', 'consensus')
        root_source_raw_data_id: Optional UUID of the root raw annotation file
            that this annotation originated from
        source_annotation_ids: Optional list of source annotation UUIDs that
            this annotation is derived from

    Returns:
        chain_id UUID of the created provenance chain

    Raises:
        ValueError: If unified_annotation_type or source_type is invalid
    """
    # Normalize source_annotation_ids to empty list if None
    annotation_ids = source_annotation_ids or []

    # Generate deterministic UUID for the provenance chain
    chain_id = generate_provenance_chain_uuid(
        unified_annotation_type=unified_annotation_type,
        source_type=source_type,
        root_source_raw_data_id=root_source_raw_data_id,
        source_annotation_ids=annotation_ids,
    )

    # Create provenance chain model
    chain = ProvenanceChain(
        chain_id=chain_id,
        unified_annotation_type=unified_annotation_type,
        source_type=source_type,
        root_source_raw_data_id=root_source_raw_data_id,
        source_annotation_ids=annotation_ids if annotation_ids else None,
        created_at=datetime.now(),
    )

    # Store in database (idempotent upsert)
    await upsert_provenance_chain(chain)

    logger.debug(
        f"Created provenance chain {chain_id} for {unified_annotation_type} "
        f"from {source_type} source"
    )

    return chain_id


async def link_transformation(
    chain_id: uuid.UUID,
    transformation_id: uuid.UUID,
) -> None:
    """
    Link a transformation operation to a provenance chain.

    This function creates a link between a provenance chain and a transformation
    operation, recording that the transformation was applied to annotations
    in this chain.

    Args:
        chain_id: UUID of the provenance chain
        transformation_id: UUID of the transformation operation

    Raises:
        ValueError: If chain_id or transformation_id is invalid
    """
    # Generate deterministic UUID for the link
    link_id = generate_provenance_transformation_uuid(
        chain_id=chain_id,
        transformation_id=transformation_id,
    )

    # Create provenance transformation link model
    prov_trans = ProvenanceTransformation(
        id=link_id,
        chain_id=chain_id,
        transformation_id=transformation_id,
        created_at=datetime.now(),
    )

    # Store in database (idempotent upsert)
    await upsert_provenance_transformation(prov_trans)

    logger.debug(
        f"Linked transformation {transformation_id} to provenance chain {chain_id}"
    )


async def create_provenance_chain_for_raw_file(
    raw_file_id: uuid.UUID,
    unified_annotation_type: str,
) -> uuid.UUID:
    """
    Create a provenance chain for a raw annotation file (root source).

    This function creates an initial provenance chain that marks a raw annotation
    file as the root source for annotations of a specific type. This is the
    starting point of the provenance chain when raw files are first ingested.

    Args:
        raw_file_id: UUID of the raw annotation file (root source)
        unified_annotation_type: Unified annotation type that this raw file contains
            ('grading', 'segmentation', 'classification', 'localization',
            'quality', 'keyword', 'description')

    Returns:
        chain_id UUID of the created provenance chain

    Raises:
        ValueError: If unified_annotation_type is invalid

    Example:
        ```python
        # After ingesting a raw CSV file containing grading annotations
        chain_id = await create_provenance_chain_for_raw_file(
            raw_file_id=raw_file_id,
            unified_annotation_type="grading"
        )
        ```
    """
    return await create_provenance_chain(
        unified_annotation_type=unified_annotation_type,
        source_type="original",
        root_source_raw_data_id=raw_file_id,
        source_annotation_ids=None,
    )


async def ingest_raw_annotation_file_with_provenance(
    raw_file: RawAnnotationFile,
    unified_annotation_type: str,
) -> Tuple[uuid.UUID, uuid.UUID]:
    """
    Ingest a raw annotation file and create its initial provenance chain.

    This is a convenience function that:
    1. Stores the raw annotation file in the database
    2. Creates an initial provenance chain marking this file as the root source

    This ensures that every raw file has a provenance chain from the moment
    it's ingested, providing complete traceability.

    Args:
        raw_file: RawAnnotationFile model to ingest
        unified_annotation_type: Unified annotation type that this raw file contains
            ('grading', 'segmentation', 'classification', 'localization',
            'quality', 'keyword', 'description')

    Returns:
        Tuple of (raw_file_id, chain_id) UUIDs

    Raises:
        ValueError: If unified_annotation_type is invalid

    Example:
        ```python
        from chaksudb.ingest.framework.hashing import compute_file_hash
        from chaksudb.ingest.framework.gen_uuid import generate_raw_file_uuid
        from pathlib import Path

        file_path = Path("./data/dataset/annotations.csv")
        file_hash = compute_file_hash(file_path)
        raw_file_id = generate_raw_file_uuid(dataset_id, file_hash)

        raw_file = RawAnnotationFile(
            raw_file_id=raw_file_id,
            dataset_id=dataset_id,
            file_path=str(file_path),
            file_hash=file_hash,
            file_type="csv",
            file_name=file_path.name,
            # ... other fields
        )

        raw_file_id, chain_id = await ingest_raw_annotation_file_with_provenance(
            raw_file=raw_file,
            unified_annotation_type="grading"
        )
        ```
    """
    # Store the raw annotation file
    await upsert_raw_annotation_file(raw_file)

    logger.debug(f"Ingested raw annotation file {raw_file.raw_file_id}")

    # Create initial provenance chain for this raw file
    chain_id = await create_provenance_chain_for_raw_file(
        raw_file_id=raw_file.raw_file_id,
        unified_annotation_type=unified_annotation_type,
    )

    logger.debug(
        f"Ingested raw file {raw_file.raw_file_id} with provenance chain {chain_id} "
        f"for {unified_annotation_type} annotations"
    )

    return raw_file.raw_file_id, chain_id


async def apply_transformation_to_chain(
    chain_id: uuid.UUID,
    transformation_type: str,
    input_data: Optional[Dict[str, Any]] = None,
    output_data: Optional[Dict[str, Any]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    operator: Optional[str] = None,
    notes: Optional[str] = None,
) -> uuid.UUID:
    """
    Apply a transformation to a provenance chain (logs and links automatically).

    This is a convenience function that ensures transformations are always
    properly tracked in the provenance system. It logs the transformation
    and automatically links it to the specified provenance chain.

    Args:
        chain_id: UUID of the provenance chain to apply the transformation to
        transformation_type: Type of transformation operation
        input_data: Optional input data dictionary
        output_data: Optional output data dictionary
        parameters: Optional operation parameters dictionary
        operator: Optional operator identifier
        notes: Optional notes about the transformation

    Returns:
        transformation_id UUID of the logged transformation

    Example:
        ```python
        # Apply a grade scaling transformation to a provenance chain
        transformation_id = await apply_transformation_to_chain(
            chain_id=provenance_chain_id,
            transformation_type="scale_grade",
            input_data={"original_grade": "Mild", "scale": "ICDR"},
            output_data={"scaled_grade": 1, "scale": "ETDRS"},
            parameters={"mapping_method": "exact"},
            operator="ingestion_script",
            notes="Converted ICDR to ETDRS scale"
        )
        ```
    """
    # Import here to avoid circular imports
    from chaksudb.ingest.framework.transformations import log_and_link_transformation

    return await log_and_link_transformation(
        chain_id=chain_id,
        transformation_type=transformation_type,
        input_data=input_data,
        output_data=output_data,
        parameters=parameters,
        operator=operator,
        notes=notes,
    )


async def create_transformed_provenance_chain(
    unified_annotation_type: str,
    source_annotation_ids: list[uuid.UUID],
    root_source_raw_data_id: Optional[uuid.UUID] = None,
    transformation_id: Optional[uuid.UUID] = None,
) -> uuid.UUID:
    """
    Create a new provenance chain for transformed annotations.

    This function creates a provenance chain for annotations that have been
    transformed from other annotations. It automatically marks the source type
    as 'transformed' and optionally links a transformation operation.

    Args:
        unified_annotation_type: Unified annotation type
        source_annotation_ids: List of source annotation UUIDs that were transformed
        root_source_raw_data_id: Optional root raw file UUID (preserved from original)
        transformation_id: Optional transformation UUID to link immediately

    Returns:
        chain_id UUID of the created provenance chain

    Example:
        ```python
        # Create a provenance chain for transformed annotations
        new_chain_id = await create_transformed_provenance_chain(
            unified_annotation_type="grading",
            source_annotation_ids=[original_annotation_id1, original_annotation_id2],
            root_source_raw_data_id=original_raw_file_id,
            transformation_id=transformation_id
        )
        ```
    """
    chain_id = await create_provenance_chain(
        unified_annotation_type=unified_annotation_type,
        source_type="transformed",
        root_source_raw_data_id=root_source_raw_data_id,
        source_annotation_ids=source_annotation_ids,
    )

    # If transformation_id is provided, link it immediately
    if transformation_id:
        await link_transformation(chain_id=chain_id, transformation_id=transformation_id)
        logger.debug(
            f"Created transformed provenance chain {chain_id} and linked transformation {transformation_id}"
        )
    else:
        logger.debug(f"Created transformed provenance chain {chain_id}")

    return chain_id


async def reconcile_grade_conversions() -> int:
    """Ensure every converted disease_grading row has its audit transformation recorded.

    Completeness backstop for the LISTEN/NOTIFY path: grade-conversion NOTIFY events are
    dropped while no listener is connected, so this sweep scans all converted gradings and
    (idempotently) records any missing ``grade_scale_conversion`` transformation + chain
    link. It reuses the exact same code path as the live listener
    (``record_grade_conversion``) with identical, deterministically-keyed payloads, so
    rows the listener already wrote are no-ops.

    Returns the number of grading rows processed.
    """
    # Imported here to avoid a circular import (listener imports transformations,
    # which imports provenance.link_transformation).
    from chaksudb.ingest.framework.provenance_listener import record_grade_conversion

    events = await fetch_grade_conversions_for_audit()
    recorded = 0
    for event in events:
        if await record_grade_conversion(event, operator="reconcile_grade_conversions"):
            recorded += 1

    logger.info(
        "reconcile_grade_conversions: processed %d converted grading rows", recorded
    )
    return recorded
