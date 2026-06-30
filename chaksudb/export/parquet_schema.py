"""
Parquet Schema Builder: Map PostgreSQL types to PyArrow schema.

Builds PyArrow schemas from ExportSpec by determining which fields will be
present in the output and mapping PostgreSQL types to appropriate PyArrow types.

Also supports building schema from query metadata (cursor description) so Parquet
columns exactly match what the database returns.
"""

import logging
from typing import Any, Sequence

import pyarrow as pa
from pyarrow import types as pat

from chaksudb.export.query_builder import QueryBuilder
from chaksudb.export.spec import ExportSpec

logger = logging.getLogger(__name__)

# PostgreSQL type OID -> PyArrow type (from pg_type)
# Covers types used in export queries: UUID, text, int, float, bool, jsonb, arrays
_PG_OID_TO_PYARROW: dict[int, pa.DataType] = {
    16: pa.bool_(),  # bool
    20: pa.int64(),  # int8
    21: pa.int64(),  # int2
    23: pa.int64(),  # int4
    25: pa.string(),  # text
    700: pa.float32(),  # float4
    701: pa.float64(),  # float8
    1042: pa.string(),  # char
    1043: pa.string(),  # varchar
    1082: pa.string(),  # date
    1114: pa.timestamp("us"),  # timestamp without time zone
    1184: pa.timestamp("us", tz="UTC"),  # timestamptz
    2950: pa.string(),  # uuid
    3802: pa.string(),  # jsonb (PG 9.4+)
    1000: pa.list_(pa.bool_()),  # bool[]
    1007: pa.list_(pa.int64()),  # int4[]
    1009: pa.list_(pa.string()),  # text[]
    199: pa.string(),  # json (array) - store as string
}


def pg_oid_to_pyarrow(oid: int) -> pa.DataType:
    """
    Map a PostgreSQL type OID to a PyArrow data type.

    Args:
        oid: PostgreSQL type OID (e.g. from cursor.description column type_code).

    Returns:
        PyArrow DataType. Unknown OIDs default to pa.string().
    """
    return _PG_OID_TO_PYARROW.get(oid, pa.string())


def build_parquet_schema_from_query_description(
    description: Sequence[Any],
) -> pa.Schema:
    """
    Build a PyArrow schema from a cursor description (query result metadata).

    Uses the actual column names and type OIDs from the executed query so
    Parquet columns match exactly what the database returns.

    Args:
        description: cursor.description from psycopg (list of Column-like objects
            or 7-tuples). Each element must have name (index 0 or .name) and
            type_code (index 1 or .type_code, PostgreSQL OID).

    Returns:
        PyArrow Schema with one field per column, in query order.
    """
    fields: list[pa.Field] = []
    for col in description:
        name = getattr(col, "name", None) or (col[0] if len(col) > 0 else None)
        type_code = getattr(col, "type_code", None) or (col[1] if len(col) > 1 else None)
        if name is None:
            continue
        pa_type = pg_oid_to_pyarrow(type_code) if type_code is not None else pa.string()
        fields.append(pa.field(name, pa_type, nullable=True))
    schema = pa.schema(fields)
    logger.debug(f"Built Parquet schema from query description: {len(fields)} fields")
    return schema


def build_parquet_schema(spec: ExportSpec) -> pa.Schema:
    """
    Build a PyArrow schema for the export based on ExportSpec.

    Determines which fields will be present in the output by building a query plan
    and collecting output fields from modules. Maps PostgreSQL types to PyArrow types:
    - UUID → string
    - JSONB → struct or string (configurable)
    - Arrays → list types
    - Timestamps → timestamp types
    - Integers → int64
    - Floats → float64
    - Strings → string
    - Booleans → bool

    Args:
        spec: The ExportSpec defining the export query

    Returns:
        PyArrow Schema with appropriate types for all fields that will be present

    Example:
        >>> spec = ExportSpec(
        ...     dataset_names=["EYEPACS"],
        ...     annotation_tasks=["grading"],
        ...     disease_types=["DR"]
        ... )
        >>> schema = build_parquet_schema(spec)
        >>> print(schema)
        pyarrow.Schema
        image_id: string
        dataset_id: string
        ...
    """
    # Build query plan to determine which modules are used
    builder = QueryBuilder()
    plan = builder.build_query(spec)

    # Collect all output fields from modules
    # We need to instantiate the modules to get their output fields
    from chaksudb.export.modules.classification import ClassificationModule
    from chaksudb.export.modules.clinical import ClinicalModule
    from chaksudb.export.modules.dataset import DatasetModule
    from chaksudb.export.modules.grading import GradingModule
    from chaksudb.export.modules.image import ImageModule
    from chaksudb.export.modules.keywords import KeywordsModule
    from chaksudb.export.modules.localization import LocalizationModule
    from chaksudb.export.modules.quality import QualityModule
    from chaksudb.export.modules.segmentation import SegmentationModule
    from chaksudb.export.modules.split import SplitModule

    # Always include core modules
    image_module = ImageModule()
    dataset_module = DatasetModule()

    # Collect fields from core modules
    output_fields: dict[str, Any] = {}
    for field in image_module.get_output_fields():
        output_fields[field] = None  # Type will be determined below
    for field in dataset_module.get_output_fields():
        output_fields[field] = None

    # Conditionally include split module
    if spec.split_names or spec.split_task_type:
        split_module = SplitModule()
        for field in split_module.get_output_fields():
            output_fields[field] = None

    # Include annotation modules based on annotation_tasks
    if spec.annotation_tasks:
        module_map = {
            "grading": GradingModule,
            "segmentation": SegmentationModule,
            "classification": ClassificationModule,
            "localization": LocalizationModule,
            "quality": QualityModule,
            "keyword": KeywordsModule,
            "description": ClinicalModule,
        }
        for task in spec.annotation_tasks:
            module_class = module_map.get(task)
            if module_class:
                module = module_class()
                for field in module.get_output_fields():
                    output_fields[field] = None

    # Build PyArrow schema from field names
    # Map field names to PyArrow types based on their semantics
    schema_fields: list[pa.Field] = []

    for field_name in sorted(output_fields.keys()):
        pa_type = _map_field_to_pyarrow_type(field_name, spec)
        schema_fields.append(pa.field(field_name, pa_type, nullable=True))

    schema = pa.schema(schema_fields)
    logger.debug(f"Built Parquet schema with {len(schema_fields)} fields")
    return schema


def _map_field_to_pyarrow_type(field_name: str, spec: ExportSpec) -> pa.DataType:
    """
    Map a field name to its PyArrow type based on naming conventions and spec.

    Uses field naming conventions to infer types:
    - Fields ending in _id → UUID → string
    - Fields ending in _label (but not _class_label) → int64 or float64 (from spec.classification_label_type)
    - Fields ending in _class_label → string
    - Fields ending in _labels (multi-label JSON fallback) → string
    - Fields with "grade" → integer
    - Fields with "score" → float
    - Fields with "count" → integer
    - Fields ending in _annotations or _masks → JSONB array → string
    - Fields named "keywords" → array of strings
    - Fields with "text" or "description" → string
    - Fields with "type", "name", "provider", "modality", "laterality" → string
    - Fields with "path" or "key" → string

    Args:
        field_name: Name of the field
        spec: ExportSpec for context (e.g., to check classification_label_type)

    Returns:
        PyArrow DataType for the field
    """
    # UUID fields (end with _id)
    if field_name.endswith("_id"):
        return pa.string()

    # Classification label fields (new pivoted columns)
    # Check _class_label first (more specific pattern)
    if field_name.endswith("_class_label"):
        return pa.string()
    
    # Check _labels (multi-label JSON fallback)
    if field_name.endswith("_labels"):
        return pa.string()
    
    # Check _label (classification numeric label)
    if field_name.endswith("_label"):
        # Use spec.classification_label_type if available
        if hasattr(spec, 'classification_label_type') and spec.classification_label_type == "float":
            return pa.float64()
        return pa.int64()

    # Integer fields
    if "grade" in field_name and "original" not in field_name:
        # Scaled grades are integers
        return pa.int64()
    if "count" in field_name:
        return pa.int64()

    # Float fields
    if "score" in field_name:
        return pa.float64()

    # JSONB array fields (aggregated annotations)
    if field_name in [
        "segmentation_masks",
        "localization_annotations",
    ]:
        # For JSONB arrays, we'll store them as string for now
        # Users can parse them later, or we could use struct types
        # For simplicity and compatibility, use string
        return pa.string()  # JSONB stored as JSON string

    # Array fields (PostgreSQL arrays)
    if field_name == "keywords":
        # Array of strings
        return pa.list_(pa.string())

    # String fields (default for most other fields)
    # Includes: file_path, object_key, storage_provider, modality, eye_laterality,
    # split_name, task_type, original_grade, scale_name, annotation_source,
    # quality_type, quality_label, description_text, description_type, etc.
    return pa.string()


def _infer_type_from_sample_value(value: Any) -> pa.DataType:
    """
    Infer PyArrow type from a sample value.

    This is a fallback method that can be used when we have actual data
    to determine types more accurately.

    Args:
        value: Sample value from a row

    Returns:
        PyArrow DataType inferred from the value
    """
    if value is None:
        # Can't infer from None, return string as default
        return pa.string()

    if isinstance(value, bool):
        return pa.bool_()
    if isinstance(value, int):
        return pa.int64()
    if isinstance(value, float):
        return pa.float64()
    if isinstance(value, str):
        return pa.string()
    if isinstance(value, list):
        if not value:
            # Empty list, default to list of strings
            return pa.list_(pa.string())
        # Infer from first element
        element_type = _infer_type_from_sample_value(value[0])
        return pa.list_(element_type)
    if isinstance(value, dict):
        # Dictionary/JSONB - store as string
        return pa.string()

    # Default to string for unknown types
    return pa.string()
