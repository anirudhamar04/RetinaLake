"""
Data validation utilities for ingestion.

Provides foreign key validation, orphan detection, and count validation
to ensure data integrity during and after ingestion.
"""

import logging
import uuid
from dataclasses import dataclass
from typing import Optional

import psycopg
from psycopg.rows import dict_row

from chaksudb.db.connection import get_connection

logger = logging.getLogger(__name__)


# ============================================
# Validation Result Models
# ============================================


@dataclass
class ForeignKeyViolation:
    """Represents a foreign key validation failure."""

    source_table: str
    source_column: str
    source_id: uuid.UUID
    target_table: str
    missing_foreign_key: uuid.UUID
    message: str


@dataclass
class OrphanRecord:
    """Represents an orphaned record."""

    table: str
    record_id: uuid.UUID
    record_type: str
    message: str


@dataclass
class CountMismatch:
    """Represents a count validation failure."""

    table: str
    expected_count: int
    actual_count: int
    message: str


@dataclass
class ValidationReport:
    """Complete validation report."""

    dataset_id: Optional[uuid.UUID] = None
    foreign_key_violations: list[ForeignKeyViolation] = None
    orphan_records: list[OrphanRecord] = None
    count_mismatches: list[CountMismatch] = None
    is_valid: bool = True

    def __post_init__(self):
        if self.foreign_key_violations is None:
            self.foreign_key_violations = []
        if self.orphan_records is None:
            self.orphan_records = []
        if self.count_mismatches is None:
            self.count_mismatches = []

        # Update is_valid based on violations
        self.is_valid = (
            len(self.foreign_key_violations) == 0
            and len(self.orphan_records) == 0
            and len(self.count_mismatches) == 0
        )

    def summary(self) -> str:
        """Generate a human-readable summary of the validation report."""
        lines = ["=" * 60]
        lines.append("VALIDATION REPORT")
        lines.append("=" * 60)

        if self.dataset_id:
            lines.append(f"Dataset ID: {self.dataset_id}")
            lines.append("")

        lines.append(f"Status: {'✓ PASSED' if self.is_valid else '✗ FAILED'}")
        lines.append("")

        lines.append(f"Foreign Key Violations: {len(self.foreign_key_violations)}")
        lines.append(f"Orphan Records: {len(self.orphan_records)}")
        lines.append(f"Count Mismatches: {len(self.count_mismatches)}")

        if not self.is_valid:
            lines.append("")
            lines.append("=" * 60)
            lines.append("DETAILS")
            lines.append("=" * 60)

            if self.foreign_key_violations:
                lines.append("")
                lines.append(f"Foreign Key Violations ({len(self.foreign_key_violations)}):")
                lines.append("-" * 60)
                for violation in self.foreign_key_violations[:10]:  # Show first 10
                    lines.append(f"  • {violation.message}")
                if len(self.foreign_key_violations) > 10:
                    lines.append(f"  ... and {len(self.foreign_key_violations) - 10} more")

            if self.orphan_records:
                lines.append("")
                lines.append(f"Orphan Records ({len(self.orphan_records)}):")
                lines.append("-" * 60)
                for orphan in self.orphan_records[:10]:  # Show first 10
                    lines.append(f"  • {orphan.message}")
                if len(self.orphan_records) > 10:
                    lines.append(f"  ... and {len(self.orphan_records) - 10} more")

            if self.count_mismatches:
                lines.append("")
                lines.append(f"Count Mismatches ({len(self.count_mismatches)}):")
                lines.append("-" * 60)
                for mismatch in self.count_mismatches:
                    lines.append(f"  • {mismatch.message}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================
# Foreign Key Validation
# ============================================


async def validate_foreign_key(
    table: str,
    column: str,
    value: uuid.UUID,
) -> bool:
    """
    Check if a foreign key value exists in the referenced table.

    Args:
        table: Target table name (e.g., 'datasets', 'images')
        column: Primary key column name (e.g., 'dataset_id', 'image_id')
        value: UUID value to check

    Returns:
        True if the foreign key exists, False otherwise

    Example:
        exists = await validate_foreign_key('datasets', 'dataset_id', dataset_id)
        if not exists:
            logger.error(f"Dataset {dataset_id} does not exist")
    """
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            query = f"SELECT 1 FROM {table} WHERE {column} = %s"
            await cur.execute(query, (value,))
            result = await cur.fetchone()
            return result is not None


async def validate_all_foreign_keys(
    dataset_id: Optional[uuid.UUID] = None,
    skip_global_checks: bool = False,
) -> list[ForeignKeyViolation]:
    """
    Validate all foreign key relationships in the database.

    Checks all major foreign key relationships and returns a list of violations.
    If dataset_id is provided, only validates records related to that dataset.

    Args:
        dataset_id: Optional dataset UUID to scope validation
        skip_global_checks: If True, skip FK checks that cannot be dataset-scoped
            (annotation → image and patient_images). Use this when running per-dataset
            validation in a loop after already running a single global check.

    Returns:
        List of ForeignKeyViolation objects

    Example:
        violations = await validate_all_foreign_keys(dataset_id)
        for violation in violations:
            logger.error(violation.message)
    """
    violations = []
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Build dataset filter params
            params = [dataset_id] if dataset_id else []

            # Check images.dataset_id -> datasets.dataset_id
            await cur.execute(
                f"""
                SELECT i.image_id, i.dataset_id
                FROM images i
                LEFT JOIN datasets d ON i.dataset_id = d.dataset_id
                {'WHERE i.dataset_id = %s' if dataset_id else ''}
                {'AND' if dataset_id else 'WHERE'} d.dataset_id IS NULL
                """,
                params,
            )
            for row in await cur.fetchall():
                violations.append(
                    ForeignKeyViolation(
                        source_table="images",
                        source_column="dataset_id",
                        source_id=row["image_id"],
                        target_table="datasets",
                        missing_foreign_key=row["dataset_id"],
                        message=f"Image {row['image_id']} references non-existent dataset {row['dataset_id']}",
                    )
                )

            # Check images.group_id -> image_groups.group_id
            await cur.execute(
                f"""
                SELECT i.image_id, i.group_id
                FROM images i
                LEFT JOIN image_groups g ON i.group_id = g.group_id
                WHERE i.group_id IS NOT NULL
                {f'AND i.dataset_id = %s' if dataset_id else ''}
                AND g.group_id IS NULL
                """,
                params if dataset_id else [],
            )
            for row in await cur.fetchall():
                violations.append(
                    ForeignKeyViolation(
                        source_table="images",
                        source_column="group_id",
                        source_id=row["image_id"],
                        target_table="image_groups",
                        missing_foreign_key=row["group_id"],
                        message=f"Image {row['image_id']} references non-existent image group {row['group_id']}",
                    )
                )

            # Check patients.dataset_id -> datasets.dataset_id
            await cur.execute(
                f"""
                SELECT p.patient_id, p.dataset_id
                FROM patients p
                LEFT JOIN datasets d ON p.dataset_id = d.dataset_id
                {'WHERE p.dataset_id = %s' if dataset_id else ''}
                {'AND' if dataset_id else 'WHERE'} d.dataset_id IS NULL
                """,
                params,
            )
            for row in await cur.fetchall():
                violations.append(
                    ForeignKeyViolation(
                        source_table="patients",
                        source_column="dataset_id",
                        source_id=row["patient_id"],
                        target_table="datasets",
                        missing_foreign_key=row["dataset_id"],
                        message=f"Patient {row['patient_id']} references non-existent dataset {row['dataset_id']}",
                    )
                )

            if not skip_global_checks:
                # Check patient_images relationships (always global: patient_images has no
                # dataset_id column, and when the join target is missing its columns are NULL
                # so any dataset filter via p.dataset_id / i.dataset_id would always be FALSE)
                await cur.execute(
                    """
                    SELECT pi.relationship_id, pi.patient_id, pi.image_id
                    FROM patient_images pi
                    LEFT JOIN patients p ON pi.patient_id = p.patient_id
                    LEFT JOIN images i ON pi.image_id = i.image_id
                    WHERE p.patient_id IS NULL OR i.image_id IS NULL
                    """,
                )
                for row in await cur.fetchall():
                    violations.append(
                        ForeignKeyViolation(
                            source_table="patient_images",
                            source_column="patient_id/image_id",
                            source_id=row["relationship_id"],
                            target_table="patients/images",
                            missing_foreign_key=row["patient_id"] or row["image_id"],
                            message=f"Patient-Image relationship {row['relationship_id']} references non-existent "
                            f"patient {row['patient_id']} or image {row['image_id']}",
                        )
                    )

                # Check disease_grading.image_id -> images.image_id (always global: when the
                # image is missing there is no dataset_id to scope by, so a dataset filter
                # via the images subquery would always be FALSE)
                await cur.execute(
                    """
                    SELECT dg.grading_id, dg.image_id
                    FROM disease_grading dg
                    LEFT JOIN images i ON dg.image_id = i.image_id
                    WHERE i.image_id IS NULL
                    """,
                )
                for row in await cur.fetchall():
                    violations.append(
                        ForeignKeyViolation(
                            source_table="disease_grading",
                            source_column="image_id",
                            source_id=row["grading_id"],
                            target_table="images",
                            missing_foreign_key=row["image_id"],
                            message=f"Disease grading {row['grading_id']} references non-existent image {row['image_id']}",
                        )
                    )

            # Check disease_grading.scale_id -> grading_scales.scale_id
            # Scope via INNER JOIN to images so we only check gradings for this dataset
            await cur.execute(
                f"""
                SELECT dg.grading_id, dg.scale_id
                FROM disease_grading dg
                {'JOIN images i ON dg.image_id = i.image_id' if dataset_id else ''}
                LEFT JOIN grading_scales gs ON dg.scale_id = gs.scale_id
                WHERE gs.scale_id IS NULL
                {'AND i.dataset_id = %s' if dataset_id else ''}
                """,
                params if dataset_id else [],
            )
            for row in await cur.fetchall():
                violations.append(
                    ForeignKeyViolation(
                        source_table="disease_grading",
                        source_column="scale_id",
                        source_id=row["grading_id"],
                        target_table="grading_scales",
                        missing_foreign_key=row["scale_id"],
                        message=f"Disease grading {row['grading_id']} references non-existent grading scale {row['scale_id']}",
                    )
                )

            if not skip_global_checks:
                # Check segmentation_annotations.image_id -> images.image_id (always global)
                await cur.execute(
                    """
                    SELECT sa.segmentation_id, sa.image_id
                    FROM segmentation_annotations sa
                    LEFT JOIN images i ON sa.image_id = i.image_id
                    WHERE i.image_id IS NULL
                    """,
                )
                for row in await cur.fetchall():
                    violations.append(
                        ForeignKeyViolation(
                            source_table="segmentation_annotations",
                            source_column="image_id",
                            source_id=row["segmentation_id"],
                            target_table="images",
                            missing_foreign_key=row["image_id"],
                            message=f"Segmentation annotation {row['segmentation_id']} references non-existent image {row['image_id']}",
                        )
                    )

                # Check classification_annotations.image_id -> images.image_id (always global)
                await cur.execute(
                    """
                    SELECT ca.classification_id, ca.image_id
                    FROM classification_annotations ca
                    LEFT JOIN images i ON ca.image_id = i.image_id
                    WHERE i.image_id IS NULL
                    """,
                )
                for row in await cur.fetchall():
                    violations.append(
                        ForeignKeyViolation(
                            source_table="classification_annotations",
                            source_column="image_id",
                            source_id=row["classification_id"],
                            target_table="images",
                            missing_foreign_key=row["image_id"],
                            message=f"Classification annotation {row['classification_id']} references non-existent image {row['image_id']}",
                        )
                    )

                # Check localization_annotations.image_id -> images.image_id (always global)
                await cur.execute(
                    """
                    SELECT la.localization_id, la.image_id
                    FROM localization_annotations la
                    LEFT JOIN images i ON la.image_id = i.image_id
                    WHERE i.image_id IS NULL
                    """,
                )
                for row in await cur.fetchall():
                    violations.append(
                        ForeignKeyViolation(
                            source_table="localization_annotations",
                            source_column="image_id",
                            source_id=row["localization_id"],
                            target_table="images",
                            missing_foreign_key=row["image_id"],
                            message=f"Localization annotation {row['localization_id']} references non-existent image {row['image_id']}",
                        )
                    )

                # Check quality_annotations.image_id -> images.image_id (always global)
                await cur.execute(
                    """
                    SELECT qa.quality_id, qa.image_id
                    FROM quality_annotations qa
                    LEFT JOIN images i ON qa.image_id = i.image_id
                    WHERE i.image_id IS NULL
                    """,
                )
                for row in await cur.fetchall():
                    violations.append(
                        ForeignKeyViolation(
                            source_table="quality_annotations",
                            source_column="image_id",
                            source_id=row["quality_id"],
                            target_table="images",
                            missing_foreign_key=row["image_id"],
                            message=f"Quality annotation {row['quality_id']} references non-existent image {row['image_id']}",
                        )
                    )

    logger.info(f"Found {len(violations)} foreign key violations")
    return violations


# ============================================
# Orphan Detection
# ============================================


async def find_orphan_images(
    dataset_id: Optional[uuid.UUID] = None,
) -> list[OrphanRecord]:
    """
    Find images that have no annotations of any kind.

    An orphan image is one that exists in the database but has no:
    - Disease grading
    - Segmentation annotations
    - Classification annotations
    - Localization annotations
    - Quality annotations
    - Clinical descriptions
    - Keyword annotations

    Args:
        dataset_id: Optional dataset UUID to scope search

    Returns:
        List of OrphanRecord objects for images with no annotations

    Example:
        orphans = await find_orphan_images(dataset_id)
        logger.warning(f"Found {len(orphans)} images without any annotations")
    """
    orphans = []
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Build query with proper WHERE clause
            query = """
                SELECT i.image_id, i.original_image_id, i.dataset_id
                FROM images i
                WHERE 1=1
            """
            params = []
            
            if dataset_id:
                query += " AND i.dataset_id = %s"
                params = [dataset_id]
            
            query += """
                AND NOT EXISTS (SELECT 1 FROM disease_grading WHERE image_id = i.image_id)
                AND NOT EXISTS (SELECT 1 FROM segmentation_annotations WHERE image_id = i.image_id)
                AND NOT EXISTS (SELECT 1 FROM classification_annotations WHERE image_id = i.image_id)
                AND NOT EXISTS (SELECT 1 FROM localization_annotations WHERE image_id = i.image_id)
                AND NOT EXISTS (SELECT 1 FROM quality_annotations WHERE image_id = i.image_id)
                AND NOT EXISTS (SELECT 1 FROM clinical_descriptions WHERE image_id = i.image_id)
                AND NOT EXISTS (SELECT 1 FROM keyword_annotations WHERE image_id = i.image_id)
            """

            await cur.execute(query, params)
            for row in await cur.fetchall():
                orphans.append(
                    OrphanRecord(
                        table="images",
                        record_id=row["image_id"],
                        record_type="image",
                        message=f"Image {row['image_id']} (original_id: {row['original_image_id']}) has no annotations",
                    )
                )

    logger.info(f"Found {len(orphans)} orphan images")
    return orphans


async def find_orphan_patients(
    dataset_id: Optional[uuid.UUID] = None,
) -> list[OrphanRecord]:
    """
    Find patients that are not linked to any images.

    Args:
        dataset_id: Optional dataset UUID to scope search

    Returns:
        List of OrphanRecord objects for patients with no images

    Example:
        orphans = await find_orphan_patients(dataset_id)
        logger.warning(f"Found {len(orphans)} patients without images")
    """
    orphans = []
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Build query with proper WHERE clause
            query = """
                SELECT p.patient_id, p.original_patient_id, p.dataset_id
                FROM patients p
                WHERE 1=1
            """
            params = []
            
            if dataset_id:
                query += " AND p.dataset_id = %s"
                params = [dataset_id]
            
            query += """
                AND NOT EXISTS (SELECT 1 FROM patient_images WHERE patient_id = p.patient_id)
            """

            await cur.execute(query, params)
            for row in await cur.fetchall():
                orphans.append(
                    OrphanRecord(
                        table="patients",
                        record_id=row["patient_id"],
                        record_type="patient",
                        message=f"Patient {row['patient_id']} (original_id: {row['original_patient_id']}) has no linked images",
                    )
                )

    logger.info(f"Found {len(orphans)} orphan patients")
    return orphans


async def find_orphan_raw_files(
    dataset_id: Optional[uuid.UUID] = None,
) -> list[OrphanRecord]:
    """
    Find raw annotation files that are not referenced by any annotations.

    Args:
        dataset_id: Optional dataset UUID to scope search

    Returns:
        List of OrphanRecord objects for unreferenced raw files

    Example:
        orphans = await find_orphan_raw_files(dataset_id)
        logger.warning(f"Found {len(orphans)} raw annotation files not referenced by any annotations")
    """
    orphans = []
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Build query with proper WHERE clause
            query = """
                SELECT rf.raw_file_id, rf.file_name, rf.dataset_id
                FROM raw_annotation_files rf
                WHERE 1=1
            """
            params = []
            
            if dataset_id:
                query += " AND rf.dataset_id = %s"
                params = [dataset_id]
            
            query += """
                AND NOT EXISTS (SELECT 1 FROM disease_grading WHERE raw_data_id = rf.raw_file_id)
                AND NOT EXISTS (SELECT 1 FROM segmentation_annotations WHERE raw_data_id = rf.raw_file_id)
                AND NOT EXISTS (SELECT 1 FROM classification_annotations WHERE raw_data_id = rf.raw_file_id)
                AND NOT EXISTS (SELECT 1 FROM localization_annotations WHERE raw_data_id = rf.raw_file_id)
                AND NOT EXISTS (SELECT 1 FROM quality_annotations WHERE raw_data_id = rf.raw_file_id)
                AND NOT EXISTS (SELECT 1 FROM clinical_descriptions WHERE raw_data_id = rf.raw_file_id)
                AND NOT EXISTS (SELECT 1 FROM keyword_annotations WHERE raw_data_id = rf.raw_file_id)
                AND NOT EXISTS (SELECT 1 FROM expert_annotations WHERE raw_data_id = rf.raw_file_id)
                AND NOT EXISTS (SELECT 1 FROM provenance_chain WHERE root_source_raw_data_id = rf.raw_file_id)
            """

            await cur.execute(query, params)
            for row in await cur.fetchall():
                orphans.append(
                    OrphanRecord(
                        table="raw_annotation_files",
                        record_id=row["raw_file_id"],
                        record_type="raw_file",
                        message=f"Raw file {row['raw_file_id']} ({row['file_name']}) is not referenced by any annotations",
                    )
                )

    logger.info(f"Found {len(orphans)} orphan raw annotation files")
    return orphans


async def find_orphan_transformation_records() -> list[OrphanRecord]:
    """
    Find transformation_operations rows not linked to any provenance chain.

    These are audit rows with no traceable lineage — exactly what the old
    ``gen_random_uuid()`` grade-conversion triggers produced. A clean audit graph has
    zero of these. Not dataset-scoped (transformation_operations has no dataset_id).

    Returns:
        List of OrphanRecord objects, one per unlinked transformation operation
    """
    from chaksudb.db.queries import find_orphan_transformations

    orphans = [
        OrphanRecord(
            table="transformation_operations",
            record_id=t.transformation_id,
            record_type="transformation_operation",
            message=(
                f"Transformation {t.transformation_id} ({t.operation_type}) is not linked "
                f"to any provenance chain"
            ),
        )
        for t in await find_orphan_transformations()
    ]
    logger.info(f"Found {len(orphans)} orphan transformation operations")
    return orphans


async def find_all_orphans(
    dataset_id: Optional[uuid.UUID] = None,
) -> list[OrphanRecord]:
    """
    Find all orphaned records across all tables.

    Args:
        dataset_id: Optional dataset UUID to scope search

    Returns:
        Combined list of all orphaned records

    Example:
        orphans = await find_all_orphans(dataset_id)
        for orphan in orphans:
            logger.warning(orphan.message)
    """
    orphans = []
    orphans.extend(await find_orphan_images(dataset_id))
    orphans.extend(await find_orphan_patients(dataset_id))
    orphans.extend(await find_orphan_raw_files(dataset_id))
    # Orphan transformation rows are global (no dataset scope); only include them in an
    # unscoped sweep to avoid attributing them to a single dataset's report.
    if dataset_id is None:
        orphans.extend(await find_orphan_transformation_records())
    return orphans


# ============================================
# Count Validation
# ============================================


async def validate_counts(
    dataset_id: uuid.UUID,
    expected_counts: dict[str, int],
) -> list[CountMismatch]:
    """
    Validate that actual counts match expected counts.

    Args:
        dataset_id: Dataset UUID to validate
        expected_counts: Dictionary of table -> expected count
            Example: {
                'images': 1000,
                'disease_grading': 1000,
                'patients': 500,
            }

    Returns:
        List of CountMismatch objects for any mismatches

    Example:
        expected = {
            'images': 1000,
            'disease_grading': 1000,
            'segmentation_annotations': 500,
        }
        mismatches = await validate_counts(dataset_id, expected)
        for mismatch in mismatches:
            logger.error(mismatch.message)
    """
    mismatches = []
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            for table, expected_count in expected_counts.items():
                # Determine the query based on table structure
                if table in [
                    "datasets",
                    "patients",
                    "images",
                    "image_groups",
                    "raw_annotation_files",
                    "keyword_vocabulary",
                ]:
                    # Tables with direct dataset_id column
                    query = f"SELECT COUNT(*) as count FROM {table} WHERE dataset_id = %s"
                elif table == "patient_images":
                    # Join through patients or images
                    query = """
                        SELECT COUNT(*) as count 
                        FROM patient_images pi
                        JOIN images i ON pi.image_id = i.image_id
                        WHERE i.dataset_id = %s
                    """
                elif table == "dataset_splits":
                    query = f"SELECT COUNT(*) as count FROM {table} WHERE dataset_id = %s"
                elif table in [
                    "disease_grading",
                    "segmentation_annotations",
                    "classification_annotations",
                    "localization_annotations",
                    "quality_annotations",
                    "clinical_descriptions",
                    "keyword_annotations",
                    "consensus_annotations",
                ]:
                    # Annotation tables - join through images
                    query = f"""
                        SELECT COUNT(*) as count 
                        FROM {table} a
                        JOIN images i ON a.image_id = i.image_id
                        WHERE i.dataset_id = %s
                    """
                else:
                    logger.warning(f"Unknown table {table} - skipping count validation")
                    continue

                await cur.execute(query, (dataset_id,))
                row = await cur.fetchone()
                actual_count = row["count"] if row else 0

                if actual_count != expected_count:
                    mismatches.append(
                        CountMismatch(
                            table=table,
                            expected_count=expected_count,
                            actual_count=actual_count,
                            message=f"Count mismatch for {table}: expected {expected_count}, got {actual_count}",
                        )
                    )

    logger.info(f"Found {len(mismatches)} count mismatches")
    return mismatches


async def get_dataset_stats(dataset_id: uuid.UUID) -> dict[str, int]:
    """
    Get comprehensive statistics for a dataset.

    Args:
        dataset_id: Dataset UUID

    Returns:
        Dictionary of table -> count for all major tables

    Example:
        stats = await get_dataset_stats(dataset_id)
        print(f"Images: {stats['images']}")
        print(f"Disease gradings: {stats['disease_grading']}")
    """
    stats = {}
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            # Direct dataset tables
            for table in ["patients", "images", "image_groups", "raw_annotation_files"]:
                await cur.execute(
                    f"SELECT COUNT(*) as count FROM {table} WHERE dataset_id = %s",
                    (dataset_id,),
                )
                row = await cur.fetchone()
                stats[table] = row["count"] if row else 0

            # Patient-image relationships
            await cur.execute(
                """
                SELECT COUNT(*) as count 
                FROM patient_images pi
                JOIN images i ON pi.image_id = i.image_id
                WHERE i.dataset_id = %s
                """,
                (dataset_id,),
            )
            row = await cur.fetchone()
            stats["patient_images"] = row["count"] if row else 0

            # Annotation tables
            for table in [
                "disease_grading",
                "segmentation_annotations",
                "classification_annotations",
                "localization_annotations",
                "quality_annotations",
                "clinical_descriptions",
                "keyword_annotations",
                "consensus_annotations",
            ]:
                await cur.execute(
                    f"""
                    SELECT COUNT(*) as count 
                    FROM {table} a
                    JOIN images i ON a.image_id = i.image_id
                    WHERE i.dataset_id = %s
                    """,
                    (dataset_id,),
                )
                row = await cur.fetchone()
                stats[table] = row["count"] if row else 0

    return stats


# ============================================
# Full Validation
# ============================================


async def validate_dataset(
    dataset_id: uuid.UUID,
    expected_counts: Optional[dict[str, int]] = None,
    check_orphans: bool = True,
    check_foreign_keys: bool = True,
    skip_global_fk_checks: bool = False,
) -> ValidationReport:
    """
    Perform comprehensive validation of a dataset.

    This is the main validation function that runs all checks and returns
    a complete validation report.

    Args:
        dataset_id: Dataset UUID to validate
        expected_counts: Optional dictionary of expected counts per table
        check_orphans: Whether to check for orphaned records
        check_foreign_keys: Whether to check foreign key relationships

    Returns:
        ValidationReport with all validation results

    Example:
        report = await validate_dataset(
            dataset_id,
            expected_counts={'images': 1000, 'disease_grading': 1000},
            check_orphans=True,
            check_foreign_keys=True
        )
        print(report.summary())
        if not report.is_valid:
            logger.error("Validation failed!")
            for violation in report.foreign_key_violations:
                logger.error(violation.message)
    """
    logger.info(f"Starting validation for dataset {dataset_id}")

    report = ValidationReport(dataset_id=dataset_id)

    # Foreign key validation
    if check_foreign_keys:
        logger.info("Checking foreign key relationships...")
        report.foreign_key_violations = await validate_all_foreign_keys(
            dataset_id, skip_global_checks=skip_global_fk_checks
        )

    # Orphan detection
    if check_orphans:
        logger.info("Checking for orphaned records...")
        report.orphan_records = await find_all_orphans(dataset_id)

    # Count validation
    if expected_counts:
        logger.info("Validating counts...")
        report.count_mismatches = await validate_counts(dataset_id, expected_counts)

    # Update is_valid
    report.__post_init__()

    logger.info(f"Validation complete. Valid: {report.is_valid}")
    return report
