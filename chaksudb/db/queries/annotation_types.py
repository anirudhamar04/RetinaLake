"""
Annotation type-specific database operations.

Includes: AnnotationType, SegmentationAnnotation, LocalizationAnnotation,
ClassificationAnnotation, QualityAnnotation, ClinicalDescription,
KeywordVocabulary, KeywordAnnotation
"""

import uuid
from typing import Optional, Sequence

import psycopg
from psycopg.rows import dict_row

from chaksudb.db.connection import get_connection
from chaksudb.db.models import (
    AnnotationType,
    ClassificationAnnotation,
    ClinicalDescription,
    KeywordAnnotation,
    KeywordVocabulary,
    LocalizationAnnotation,
    QualityAnnotation,
    SegmentationAnnotation,
)
from chaksudb.db.queries.helpers import _prepare_upsert_query, _serialize_jsonb


async def upsert_annotation_type(annotation_type: AnnotationType) -> None:
    """Upsert an annotation type record."""
    async with get_connection() as conn:
        columns = [
            "annotation_type_id",
            "annotation_type",
            "annotation_description",
        ]
        values = (
            annotation_type.annotation_type_id,
            annotation_type.annotation_type,
            annotation_type.annotation_description,
        )
        query = _prepare_upsert_query(
            "annotation_type",
            columns,
            conflict_target=["annotation_type_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def upsert_segmentation_annotation(segmentation: SegmentationAnnotation) -> None:
    """Upsert a segmentation annotation record."""
    async with get_connection() as conn:
        columns = [
            "segmentation_id",
            "image_id",
            "annotation_type_id",
            "lesion_subtype",
            "mask_file_path",
            "group_id",
            "unified_format",
            "original_format",
            "original_file_path",
            "raw_data_id",
            "coordinate_system",
            "expert_annotation_id",
            "consensus_id",
            "annotation_method",
            "confidence_score",
            "provenance_chain_id",
            "created_at",
        ]
        values = (
            segmentation.segmentation_id,
            segmentation.image_id,
            segmentation.annotation_type_id,
            segmentation.lesion_subtype,
            segmentation.mask_file_path,
            segmentation.group_id,
            segmentation.unified_format,
            segmentation.original_format,
            segmentation.original_file_path,
            segmentation.raw_data_id,
            segmentation.coordinate_system,
            segmentation.expert_annotation_id,
            segmentation.consensus_id,
            segmentation.annotation_method,
            segmentation.confidence_score,
            segmentation.provenance_chain_id,
            segmentation.created_at,
        )
        query = _prepare_upsert_query(
            "segmentation_annotations",
            columns,
            conflict_target=["segmentation_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def upsert_localization_annotation(localization: LocalizationAnnotation) -> None:
    """Upsert a localization annotation record."""
    async with get_connection() as conn:
        columns = [
            "localization_id",
            "image_id",
            "localization_type",
            "target_structure",
            "coordinates",
            "lesion_subtype",
            "raw_data_id",
            "expert_annotation_id",
            "consensus_id",
            "annotation_method",
            "provenance_chain_id",
            "created_at",
        ]
        values = (
            localization.localization_id,
            localization.image_id,
            localization.localization_type,
            localization.target_structure,
            _serialize_jsonb(localization.coordinates),
            localization.lesion_subtype,
            localization.raw_data_id,
            localization.expert_annotation_id,
            localization.consensus_id,
            localization.annotation_method,
            localization.provenance_chain_id,
            localization.created_at,
        )
        query = _prepare_upsert_query(
            "localization_annotations",
            columns,
            conflict_target=["localization_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def bulk_upsert_localization_annotations(
    localizations: Sequence[LocalizationAnnotation], batch_size: int = 1000
) -> int:
    """Bulk upsert localization annotation records."""
    async with get_connection() as conn:
        columns = [
            "localization_id",
            "image_id",
            "localization_type",
            "target_structure",
            "coordinates",
            "lesion_subtype",
            "raw_data_id",
            "expert_annotation_id",
            "consensus_id",
            "annotation_method",
            "provenance_chain_id",
            "created_at",
        ]
        values_list = [
            (
                l.localization_id,
                l.image_id,
                l.localization_type,
                l.target_structure,
                _serialize_jsonb(l.coordinates),
                l.lesion_subtype,
                l.raw_data_id,
                l.expert_annotation_id,
                l.consensus_id,
                l.annotation_method,
                l.provenance_chain_id,
                l.created_at,
            )
            for l in localizations
        ]
        
        query = _prepare_upsert_query(
            "localization_annotations",
            columns,
            conflict_target=["localization_id"],
        )
        
        total_rows = 0
        for i in range(0, len(values_list), batch_size):
            batch = values_list[i : i + batch_size]
            async with conn.cursor() as cur:
                await cur.executemany(query, batch)
                total_rows += len(batch)
        
        return total_rows


async def upsert_classification_annotation(classification: ClassificationAnnotation) -> None:
    """Upsert a classification annotation record."""
    async with get_connection() as conn:
        columns = [
            "classification_id",
            "image_id",
            "task_type",
            "task_name",
            "class_name",
            "concept",
            "is_multilabel",
            "class_index",
            "class_label",
            "sub_key",
            "class_value",
            "raw_data_id",
            "expert_annotation_id",
            "consensus_id",
            "annotation_method",
            "confidence_score",
            "provenance_chain_id",
            "created_at",
        ]
        values = (
            classification.classification_id,
            classification.image_id,
            classification.task_type,
            classification.task_name,
            classification.class_name,
            classification.concept,
            classification.is_multilabel,
            classification.class_index,
            classification.class_label,
            classification.sub_key,
            _serialize_jsonb(classification.class_value),
            classification.raw_data_id,
            classification.expert_annotation_id,
            classification.consensus_id,
            classification.annotation_method,
            classification.confidence_score,
            classification.provenance_chain_id,
            classification.created_at,
        )
        query = _prepare_upsert_query(
            "classification_annotations",
            columns,
            conflict_target=["classification_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def bulk_upsert_classification_annotations(
    classifications: list[ClassificationAnnotation], batch_size: int = 1000
) -> int:
    """Bulk upsert classification annotation records."""
    async with get_connection() as conn:
        columns = [
            "classification_id",
            "image_id",
            "task_type",
            "task_name",
            "class_name",
            "concept",
            "is_multilabel",
            "class_index",
            "class_label",
            "sub_key",
            "class_value",
            "raw_data_id",
            "expert_annotation_id",
            "consensus_id",
            "annotation_method",
            "confidence_score",
            "provenance_chain_id",
            "created_at",
        ]
        values_list = [
            (
                c.classification_id,
                c.image_id,
                c.task_type,
                c.task_name,
                c.class_name,
                c.concept,
                c.is_multilabel,
                c.class_index,
                c.class_label,
                c.sub_key,
                _serialize_jsonb(c.class_value),
                c.raw_data_id,
                c.expert_annotation_id,
                c.consensus_id,
                c.annotation_method,
                c.confidence_score,
                c.provenance_chain_id,
                c.created_at,
            )
            for c in classifications
        ]
        
        query = _prepare_upsert_query(
            "classification_annotations",
            columns,
            conflict_target=["classification_id"],
        )
        
        total_rows = 0
        for i in range(0, len(values_list), batch_size):
            batch = values_list[i : i + batch_size]
            async with conn.cursor() as cur:
                await cur.executemany(query, batch)
                total_rows += len(batch)
        
        return total_rows


async def upsert_quality_annotation(quality: QualityAnnotation) -> None:
    """Upsert a quality annotation record."""
    async with get_connection() as conn:
        columns = [
            "quality_id",
            "image_id",
            "quality_type",
            "quality_score",
            "quality_label",
            "scale_description",
            "raw_data_id",
            "expert_annotation_id",
            "provenance_chain_id",
            "created_at",
        ]
        values = (
            quality.quality_id,
            quality.image_id,
            quality.quality_type,
            quality.quality_score,
            quality.quality_label,
            quality.scale_description,
            quality.raw_data_id,
            quality.expert_annotation_id,
            quality.provenance_chain_id,
            quality.created_at,
        )
        query = _prepare_upsert_query(
            "quality_annotations",
            columns,
            conflict_target=["quality_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def bulk_upsert_quality_annotations(
    quality_annotations: list[QualityAnnotation], batch_size: int = 1000
) -> int:
    """Bulk upsert quality annotation records."""
    async with get_connection() as conn:
        columns = [
            "quality_id",
            "image_id",
            "quality_type",
            "quality_score",
            "quality_label",
            "scale_description",
            "raw_data_id",
            "expert_annotation_id",
            "provenance_chain_id",
            "created_at",
        ]
        values_list = [
            (
                q.quality_id,
                q.image_id,
                q.quality_type,
                q.quality_score,
                q.quality_label,
                q.scale_description,
                q.raw_data_id,
                q.expert_annotation_id,
                q.provenance_chain_id,
                q.created_at,
            )
            for q in quality_annotations
        ]
        
        query = _prepare_upsert_query(
            "quality_annotations",
            columns,
            conflict_target=["quality_id"],
        )
        
        total_rows = 0
        for i in range(0, len(values_list), batch_size):
            batch = values_list[i : i + batch_size]
            async with conn.cursor() as cur:
                await cur.executemany(query, batch)
                total_rows += len(batch)
        
        return total_rows


async def upsert_clinical_description(description: ClinicalDescription) -> None:
    """Upsert a clinical description record."""
    async with get_connection() as conn:
        columns = [
            "description_id",
            "image_id",
            "description_text",
            "description_type",
            "raw_data_id",
            "expert_id",
            "word_count",
            "created_at",
        ]
        values = (
            description.description_id,
            description.image_id,
            description.description_text,
            description.description_type,
            description.raw_data_id,
            description.expert_id,
            description.word_count,
            description.created_at,
        )
        query = _prepare_upsert_query(
            "clinical_descriptions",
            columns,
            conflict_target=["description_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def find_keyword_vocabulary_by_id(keyword_id: uuid.UUID) -> Optional[KeywordVocabulary]:
    """
    Find a keyword vocabulary entry by its UUID.

    Args:
        keyword_id: UUID of the keyword

    Returns:
        KeywordVocabulary if found, None otherwise
    """
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(
                """
                SELECT keyword_id, keyword_term, keyword_source, category, 
                       dataset_id, created_at
                FROM keyword_vocabulary
                WHERE keyword_id = %s
                """,
                (keyword_id,),
            )
            row = await cur.fetchone()
            if row:
                return KeywordVocabulary(
                    keyword_id=row["keyword_id"],
                    keyword_term=row["keyword_term"],
                    keyword_source=row["keyword_source"],
                    category=row["category"],
                    dataset_id=row["dataset_id"],
                    created_at=row["created_at"],
                )
            return None


async def upsert_keyword_vocabulary(keyword: KeywordVocabulary) -> None:
    """Upsert a keyword vocabulary record."""
    async with get_connection() as conn:
        columns = [
            "keyword_id",
            "keyword_term",
            "keyword_source",
            "category",
            "dataset_id",
            "created_at",
        ]
        values = (
            keyword.keyword_id,
            keyword.keyword_term,
            keyword.keyword_source,
            keyword.category,
            keyword.dataset_id,
            keyword.created_at,
        )
        query = _prepare_upsert_query(
            "keyword_vocabulary",
            columns,
            conflict_target=["keyword_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def upsert_keyword_annotation(keyword_annotation: KeywordAnnotation) -> None:
    """Upsert a keyword annotation record."""
    async with get_connection() as conn:
        columns = [
            "keyword_annotation_id",
            "image_id",
            "keyword_id",
            "keyword_text",
            "raw_data_id",
            "expert_id",
            "annotation_method",
            "provenance_chain_id",
            "created_at",
        ]
        values = (
            keyword_annotation.keyword_annotation_id,
            keyword_annotation.image_id,
            keyword_annotation.keyword_id,
            keyword_annotation.keyword_text,
            keyword_annotation.raw_data_id,
            keyword_annotation.expert_id,
            keyword_annotation.annotation_method,
            keyword_annotation.provenance_chain_id,
            keyword_annotation.created_at,
        )
        query = _prepare_upsert_query(
            "keyword_annotations",
            columns,
            conflict_target=["keyword_annotation_id"],
        )
        async with conn.cursor() as cur:
            await cur.execute(query, values)


async def list_classification_tasks(
    dataset_names: Optional[Sequence[str]] = None,
) -> list[dict]:
    """Discover the classification tasks present in the DB.

    Returns one row per (task_name, task_type) with its concept, multilabel flag, the
    multi-label sub-keys, and the observed labels. Lets callers build an ExportSpec
    without knowing the class vocabulary up front (classification_class_names becomes
    optional). Optionally scoped to a set of dataset names.
    """
    sql = """
        SELECT c.task_name,
               c.task_type,
               bool_or(c.is_multilabel) AS is_multilabel,
               array_agg(DISTINCT c.concept) FILTER (WHERE c.concept IS NOT NULL) AS concepts,
               array_agg(DISTINCT c.sub_key) FILTER (WHERE c.sub_key IS NOT NULL) AS sub_keys,
               array_agg(DISTINCT c.class_label) AS labels
        FROM classification_annotations c
    """
    params: list = []
    if dataset_names:
        sql += (
            " JOIN images i ON c.image_id = i.image_id"
            " JOIN datasets d ON i.dataset_id = d.dataset_id"
            " WHERE d.dataset_name = ANY(%s)"
        )
        params.append(list(dataset_names))
    sql += " GROUP BY c.task_name, c.task_type ORDER BY c.task_name"

    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            await cur.execute(sql, params)
            return await cur.fetchall()


async def get_or_create_quality_type(
    quality_type: str, description: str | None = None, category: str | None = None
) -> None:
    """Register a quality_type in the reference table if absent (idempotent).

    Lets ingest introduce new quality dimensions without a schema migration; the FK on
    quality_annotations.quality_type then accepts it.
    """
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "INSERT INTO quality_types (quality_type, description, category) "
                "VALUES (%s, %s, %s) ON CONFLICT (quality_type) DO NOTHING",
                (quality_type, description, category),
            )


async def list_quality_types(dataset_names: Optional[Sequence[str]] = None) -> list[str]:
    """Return the quality_type values actually present (optionally scoped to datasets).

    Used by the export so it pivots only the quality types a dataset has, instead of a
    column per registered type.
    """
    sql = "SELECT DISTINCT q.quality_type FROM quality_annotations q"
    params: list = []
    if dataset_names:
        sql += (
            " JOIN images i ON q.image_id = i.image_id"
            " JOIN datasets d ON i.dataset_id = d.dataset_id"
            " WHERE d.dataset_name = ANY(%s)"
        )
        params.append(list(dataset_names))
    sql += " ORDER BY q.quality_type"
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(sql, params)
            return [r[0] for r in await cur.fetchall()]
