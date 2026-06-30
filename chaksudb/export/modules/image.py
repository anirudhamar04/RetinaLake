"""
ImageModule: Core image fields and filters.

Adds the base images table to the query and includes core image metadata fields.
Handles path transformations and filters by modality and storage provider.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule


class ImageModule(BaseModule):
    """
    Module for adding core image fields to export queries.

    This module is always included as it provides the base table (images) for all exports.
    It adds core image metadata fields and handles filtering by modality and storage provider.

    Output fields:
        - image_id: UUID of the image
        - dataset_id: UUID of the dataset this image belongs to
        - file_path: File path (may be transformed by base_path_for_paths)
        - storage_provider: Storage provider (e.g., 'local', 's3', 'gcs')
        - object_key: Object key for cloud storage
        - modality: Image modality (e.g., 'fundus', 'oct')
        - eye_laterality: Eye laterality (e.g., 'left', 'right', 'unknown', or None)
        - resolution_width: Image width in pixels (from images table)
        - resolution_height: Image height in pixels (from images table)
        - group_id: UUID of the OCT volume group (NULL for non-OCT or standalone images)
        - frame_index: Position within the OCT volume; 0 = key frame (B-scan), 1+ = volume frames
    """

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """
        Apply image module to the query plan.

        Adds:
        - Base FROM table: images i
        - Core image fields to SELECT
        - WHERE filters for modalities and storage_provider
        - Path transformation if base_path_for_paths is provided

        Args:
            plan: The QueryPlan to modify
            spec: The ExportSpec containing user requirements
        """
        # Add base FROM table
        if not plan.from_tables:
            plan.from_tables.append("images i")

        # Add core image fields to SELECT
        # Only include image_id (drop dataset_id - not needed for training)
        plan.add_select("i.image_id", group_by_expression="i.image_id")
        # Note: dataset_id is still available for joins but not exported

        # Handle file_path with optional base path transformation
        if spec.base_path_for_paths:
            # Use CONCAT to prepend base path if file_path is not null
            param_placeholder = plan.add_param("base_path", spec.base_path_for_paths)
            file_path_expr = (
                f"CASE WHEN i.file_path IS NOT NULL "
                f"THEN CONCAT(CAST({param_placeholder} AS TEXT), i.file_path) "
                f"ELSE i.file_path END"
            )
            plan.add_select(f"{file_path_expr} AS file_path", group_by_expression=file_path_expr)
        else:
            plan.add_select("i.file_path", group_by_expression="i.file_path")

        plan.add_select("i.storage_provider", group_by_expression="i.storage_provider")
        plan.add_select("i.object_key", group_by_expression="i.object_key")
        plan.add_select("i.modality", group_by_expression="i.modality")
        plan.add_select("i.eye_laterality", group_by_expression="i.eye_laterality")
        plan.add_select("i.resolution_width", group_by_expression="i.resolution_width")
        plan.add_select("i.resolution_height", group_by_expression="i.resolution_height")
        plan.add_select("i.group_id", group_by_expression="i.group_id")
        plan.add_select("i.frame_index", group_by_expression="i.frame_index")

        # Add filters for modalities
        if spec.modalities:
            self.add_in_clause(plan, "i.modality", "modalities", spec.modalities)

        # Filter to key frames only (frame_index = 0) when requested
        if spec.oct_key_frames_only:
            self.add_parameterized_where(
                plan,
                "i.frame_index = %s",
                "oct_key_frame_index",
                0,
            )

        # Add filter for storage_provider
        if spec.storage_provider:
            self.add_parameterized_where(
                plan,
                "i.storage_provider = %s",
                "storage_provider",
                spec.storage_provider,
            )

    def get_output_fields(self) -> list[str]:
        """
        Get the list of output field names this module adds.

        Returns:
            List of field names: image_id, file_path, storage_provider,
            object_key, modality, eye_laterality, resolution_width,
            resolution_height
        """
        return [
            "image_id",
            "file_path",
            "storage_provider",
            "object_key",
            "modality",
            "eye_laterality",
            "resolution_width",
            "resolution_height",
            "group_id",
            "frame_index",
        ]
