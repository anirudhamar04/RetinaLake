"""
ChaksuDB Internal Package

Core functionality for dataset ingestion, database operations, and storage management.

Main submodules:
- db: Database models, queries, and connection management
- config: Configuration settings and constants
- storage: Storage locators and path utilities
- common: Shared utilities (progress tracking, etc.)
- ingest.framework: Ingestion framework utilities
"""

# Configuration (most commonly used)
from chaksudb.config.config import (
    constants,
    db_config,
    storage_config,
    get_data_root,
    get_storage_root,
)

# Database connection
from chaksudb.db import (
    get_connection,
    init_pool,
    close_pool,
)

# Most commonly used models
from chaksudb.db import (
    Dataset,
    Image,
    Patient,
    DiseaseGrading,
    GradingScale,
    GradingScaleMapping,
    RawAnnotationFile,
)

# Most commonly used queries
from chaksudb.db import (
    upsert_dataset,
    upsert_image,
    upsert_patient,
    upsert_disease_grading,
    upsert_grading_scale,
    upsert_grading_scale_mapping,
    bulk_upsert_images,
)

# Storage
from chaksudb.storage import (
    create_local_locator,
    StorageLocator,
)

# Common utilities
from chaksudb.common import (
    ProgressTracker,
    OperationStatistics,
)

__all__ = [
    # Config
    "constants",
    "db_config",
    "storage_config",
    "get_data_root",
    "get_storage_root",
    # Database
    "get_connection",
    "init_pool",
    "close_pool",
    # Models
    "Dataset",
    "Image",
    "Patient",
    "DiseaseGrading",
    "GradingScale",
    "GradingScaleMapping",
    "RawAnnotationFile",
    # Queries
    "upsert_dataset",
    "upsert_image",
    "upsert_patient",
    "upsert_disease_grading",
    "upsert_grading_scale",
    "upsert_grading_scale_mapping",
    "bulk_upsert_images",
    # Storage
    "create_local_locator",
    "StorageLocator",
    # Common
    "ProgressTracker",
    "OperationStatistics",
]

# Note: For more specialized imports, use specific submodules:
# - from chaksudb.db.models import SegmentationAnnotation, LocalizationAnnotation, ...
# - from chaksudb.db.queries import upsert_segmentation_annotation, ...
# - from chaksudb.ingest.framework import gen_uuid, hashing, validation, ...
# - from chaksudb.ingest.framework.task_processors import process_disease_grade, ...