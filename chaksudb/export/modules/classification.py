"""
ClassificationModule: Classification annotations.

Adds classification_annotations table JOIN and produces flat, training-ready columns
using the scalar class_index and class_label columns.

All task types use the same extraction pattern:
  - binary/multi_class: CASE WHEN c.class_name = :name THEN c.class_index END
  - multi_label:        CASE WHEN c.class_name = :name AND c.sub_key = :key THEN c.class_index END

No JSONB extraction is needed.
"""

from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from chaksudb.export.query_builder import QueryPlan
    from chaksudb.export.spec import ExportSpec

from chaksudb.export.modules.base import BaseModule


class ClassificationModule(BaseModule):
    """
    Module for adding classification annotations to export queries.

    Creates flat, training-ready columns for each class_name:
    - Binary:      {class_name}_label (int), {class_name}_class_label (str)
    - Multi-class: {class_name}_label (int), {class_name}_class_label (str)
    - Multi-label: {class_name}_{sub_key} (int) for each sub-key,
                   or {class_name}_labels (JSON string) as fallback
    """

    def apply(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """Apply classification module to the query plan."""
        join_type = "INNER" if spec.require_annotations_mode == "all" else "LEFT"

        if spec.annotation_source == "prefer_consensus":
            self._add_prefer_consensus_subquery(plan, join_type, spec)
        else:
            self._add_simple_classification_join(plan, join_type, spec)

        if spec.annotation_source == "expert_only":
            plan.add_where("c.consensus_id IS NULL")
        elif spec.annotation_source == "consensus_only":
            plan.add_where("c.consensus_id IS NOT NULL")

        if spec.classification_filter:
            self._add_classification_filters(plan, spec)

        # Cross-cutting concept filter: keep images positive for a concept regardless of
        # how the dataset stored it (binary / multi_label / multi_class).
        if spec.classification_positive_for:
            self._add_positive_for_filter(plan, spec)

        plan.add_group_by("i.image_id")
        self._add_classification_fields(plan, spec)
        self._add_concept_columns(plan, spec)

    def _add_simple_classification_join(
        self, plan: "QueryPlan", join_type: str, spec: "ExportSpec"
    ) -> None:
        """Add simple classification JOIN, scoped to requested class_names."""
        if spec.classification_class_names:
            param = plan.add_param("classification_class_names_join", spec.classification_class_names)
            plan.add_join(
                f"{join_type} JOIN classification_annotations c "
                f"ON i.image_id = c.image_id AND c.class_name = ANY({param})"
            )
        else:
            plan.add_join(
                f"{join_type} JOIN classification_annotations c ON i.image_id = c.image_id"
            )

    def _add_prefer_consensus_subquery(
        self, plan: "QueryPlan", join_type: str, spec: "ExportSpec"
    ) -> None:
        """Add classification JOIN with prefer_consensus using window function subquery."""
        where_conditions = []

        if spec.classification_class_names:
            param_placeholder = plan.add_param(
                "classification_class_names_subq", spec.classification_class_names
            )
            where_conditions.append(f"c_inner.class_name = ANY({param_placeholder})")

        if spec.classification_filter:
            filters = dict(spec.classification_filter)

            if "class_name" in filters:
                class_name_val = filters.pop("class_name")
                if "class_names" not in filters:
                    filters["class_names"] = []
                if isinstance(class_name_val, str):
                    filters["class_names"].append(class_name_val)
                elif isinstance(class_name_val, list):
                    filters["class_names"].extend(class_name_val)

            if "class_names" in filters and filters["class_names"]:
                param_placeholder = plan.add_param(
                    "classification_filter_class_names_subq", filters["class_names"]
                )
                where_conditions.append(f"c_inner.class_name = ANY({param_placeholder})")

            if "task_type" in filters and filters["task_type"]:
                param_placeholder = plan.add_param(
                    "classification_task_type_subq", filters["task_type"]
                )
                where_conditions.append(f"c_inner.task_type = {param_placeholder}")

        where_clause = ""
        if where_conditions:
            where_clause = "WHERE " + " AND ".join(where_conditions)

        # Partition includes sub_key so each multi-label sub-key is ranked independently
        subquery = (
            f"{join_type} JOIN ("
            "  SELECT c_inner.*, "
            "    ROW_NUMBER() OVER ("
            "      PARTITION BY c_inner.image_id, c_inner.task_type, c_inner.class_name, c_inner.sub_key "
            "      ORDER BY (c_inner.consensus_id IS NOT NULL) DESC, "
            "               c_inner.created_at DESC, c_inner.classification_id DESC"
            "    ) AS rn "
            "  FROM classification_annotations c_inner"
            f"  {where_clause}"
            ") c ON i.image_id = c.image_id AND c.rn = 1"
        )

        plan.add_join(subquery)

    def _add_concept_columns(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """Add per-concept binary presence columns ({concept}_present, 0/1).

        Unifies storage shapes: a concept is present if a binary/multi_label row for it is
        positive (class_index=1) OR a multi_class row's winning class maps to the concept
        (the concept is only set on the winning row). This is the cross-dataset
        "glaucoma binary classification" view.
        """
        if not spec.classification_concepts:
            return
        for concept in spec.classification_concepts:
            col = concept.replace("-", "_").replace(" ", "_")
            cp = plan.add_param(f"concept_{col}", concept)
            plan.add_select(
                f"COALESCE(MAX(CASE WHEN c.concept = {cp} AND "
                f"(c.task_type = 'multi_class' OR c.class_index = 1) "
                f"THEN 1 ELSE 0 END), 0) AS {col}_present"
            )

    def _add_positive_for_filter(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """Keep only images positive for at least one requested concept (any storage shape)."""
        ph = plan.add_param("positive_concepts", spec.classification_positive_for)
        plan.add_where(
            "EXISTS (SELECT 1 FROM classification_annotations cpf "
            "WHERE cpf.image_id = i.image_id "
            f"AND cpf.concept = ANY({ph}) "
            "AND (cpf.task_type = 'multi_class' OR cpf.class_index = 1))"
        )

    def _add_classification_fields(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """Add pivoted classification columns using scalar class_index/class_label."""
        # class_names is optional now; concept columns (above) are the primary interface.
        if not spec.classification_class_names:
            return
        label_cast = "float" if spec.classification_label_type == "float" else "int"
        task_types = spec.classification_task_types or {}

        for class_name in spec.classification_class_names:
            col_name = class_name.replace("-", "_").replace(" ", "_")
            task_type = task_types.get(class_name)

            # Infer task_type from multi_label_keys if not explicitly declared
            if task_type is None and spec.multi_label_keys and class_name in spec.multi_label_keys:
                task_type = "multi_label"

            if task_type == "multi_label":
                self._add_multi_label_columns(plan, spec, class_name, col_name, label_cast)
            else:
                # binary and multi_class use the same extraction: class_index and class_label
                self._add_scalar_columns(plan, class_name, col_name, label_cast, task_type)

    def _add_scalar_columns(
        self, plan: "QueryPlan", class_name: str, col_name: str,
        label_cast: str, task_type: Optional[str] = None,
    ) -> None:
        """Add scalar label columns for binary or multi_class (same pattern for both)."""
        cn_param = plan.add_param(f"cls_{col_name}", class_name)

        # Build the task_type filter clause
        if task_type:
            tt_param = plan.add_param(f"cls_tt_{col_name}", task_type)
            condition = f"c.class_name = {cn_param} AND c.task_type = {tt_param}"
        else:
            condition = f"c.class_name = {cn_param}"

        plan.add_select(
            f"MAX(CASE WHEN {condition} "
            f"THEN c.class_index::{label_cast} END) AS {col_name}_label"
        )

        # Reuse same class_name param name with a label suffix
        cn_param2 = plan.add_param(f"cls_lbl_{col_name}", class_name)
        if task_type:
            tt_param2 = plan.add_param(f"cls_tt_lbl_{col_name}", task_type)
            condition2 = f"c.class_name = {cn_param2} AND c.task_type = {tt_param2}"
        else:
            condition2 = f"c.class_name = {cn_param2}"

        plan.add_select(
            f"MAX(CASE WHEN {condition2} "
            f"THEN c.class_label END) AS {col_name}_class_label"
        )

    def _add_multi_label_columns(
        self, plan: "QueryPlan", spec: "ExportSpec", class_name: str,
        col_name: str, label_cast: str,
    ) -> None:
        """Add multi-label columns using sub_key (one column per sub-key)."""
        if spec.multi_label_keys and class_name in spec.multi_label_keys:
            for key in spec.multi_label_keys[class_name]:
                key_col = key.replace("-", "_").replace(" ", "_")
                cn_param = plan.add_param(f"cls_ml_{col_name}_{key_col}", class_name)
                sk_param = plan.add_param(f"cls_sk_{col_name}_{key_col}", key)
                plan.add_select(
                    f"MAX(CASE WHEN c.class_name = {cn_param} "
                    f"AND c.sub_key = {sk_param} "
                    f"THEN c.class_index::{label_cast} END) AS {col_name}_{key_col}"
                )
        else:
            # Fallback: aggregate all sub-keys as JSON string
            cn_param = plan.add_param(f"cls_ml_fb_{col_name}", class_name)
            plan.add_select(
                f"jsonb_object_agg(c.sub_key, c.class_index) "
                f"FILTER (WHERE c.class_name = {cn_param} AND c.sub_key IS NOT NULL) "
                f"AS {col_name}_labels"
            )

    def _add_classification_filters(self, plan: "QueryPlan", spec: "ExportSpec") -> None:
        """Add classification_filter conditions to WHERE clause."""
        if not spec.classification_filter:
            return

        filters = dict(spec.classification_filter)
        conditions = []

        if "class_name" in filters:
            class_name_val = filters.pop("class_name")
            if "class_names" not in filters:
                filters["class_names"] = []
            if isinstance(class_name_val, str):
                filters["class_names"].append(class_name_val)
            elif isinstance(class_name_val, list):
                filters["class_names"].extend(class_name_val)

        if "class_names" in filters and filters["class_names"]:
            param_placeholder = plan.add_param(
                "classification_class_names", filters["class_names"]
            )
            conditions.append(f"c.class_name = ANY({param_placeholder})")

        if "class_values" in filters and filters["class_values"]:
            value_conditions = []
            for idx, val in enumerate(filters["class_values"]):
                param_placeholder = plan.add_param(
                    f"classification_class_value_{idx}", val
                )
                value_conditions.append(f"c.class_index = {param_placeholder}")

            if value_conditions:
                conditions.append(f"({' OR '.join(value_conditions)})")

        if "task_type" in filters and filters["task_type"]:
            param_placeholder = plan.add_param(
                "classification_task_type", filters["task_type"]
            )
            conditions.append(f"c.task_type = {param_placeholder}")

        if conditions:
            combined = " AND ".join(conditions)
            plan.add_where(f"(c.classification_id IS NULL OR ({combined}))")

    def get_output_fields(self) -> list[str]:
        """Fields are dynamic based on spec -- discovered from cursor.description."""
        return []

    def get_primary_id_column(self) -> str | None:
        """Primary ID column for require_annotations_mode='any' HAVING clause."""
        return "c.classification_id"
