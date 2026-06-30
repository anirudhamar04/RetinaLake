#!/usr/bin/env -S uv run
"""
Example: Classification Export with Flat Columns

Demonstrates classification export with flat, training-ready columns.

Run from repo root: uv run examples/export_classification_pivoted.py

Shows:
  1. Binary classification (glaucoma) - pivoted to int labels
  2. Multi-class classification (disease category) - pivoted to int labels
  3. Multi-label classification with explicit keys - flattened columns
  4. Multi-label classification without keys - JSON string fallback
  5. Float labels for soft-label training
  6. Combined with grading - complete training dataset
"""

from pathlib import Path

from chaksudb.export import ExportSpec, export

OUT_DIR = Path("examples/export_output/classification_pivoted")


# -----------------------------------------------------------------------------
# Example 1: Binary classification - glaucoma (int labels)
# -----------------------------------------------------------------------------
def export_binary_classification_int() -> Path:
    """
    Export glaucoma classification as flat int columns.
    
    Output columns:
    - glaucoma_label: int (0 or 1)
    - glaucoma_class_label: str ('positive' or 'negative')
    """
    spec = ExportSpec(
        annotation_tasks=["classification"],
        classification_class_names=["glaucoma"],
        classification_task_types={"glaucoma": "binary"},
        classification_label_type="int",  # default
        require_annotations=True,
    )
    path = OUT_DIR / "glaucoma_binary_int.parquet"
    export(spec, parquet_path=path)
    print(f"Binary classification (int): {path}")
    return path


# -----------------------------------------------------------------------------
# Example 2: Multi-class classification - disease category
# -----------------------------------------------------------------------------
def export_multiclass_classification() -> Path:
    """
    Export disease category classification as flat int columns.
    
    Output columns:
    - disease_category_label: int (0-N, class index)
    - disease_category_class_label: str (class name like 'normal', 'mild', 'moderate', 'severe')
    """
    spec = ExportSpec(
        annotation_tasks=["classification"],
        classification_class_names=["disease_category"],
        classification_task_types={"disease_category": "multi_class"},
        classification_label_type="int",
        require_annotations=True,
    )
    path = OUT_DIR / "disease_category_multiclass.parquet"
    export(spec, parquet_path=path)
    print(f"Multi-class classification: {path}")
    return path


# -----------------------------------------------------------------------------
# Example 3: Multi-label classification with explicit keys
# -----------------------------------------------------------------------------
def export_multilabel_with_keys() -> Path:
    """
    Export multi-label disease indicators with explicit sublabel keys.
    
    Output columns (one per sublabel):
    - disease_indicators_normal: int (0 or 1)
    - disease_indicators_diabetes: int (0 or 1)
    - disease_indicators_glaucoma: int (0 or 1)
    - disease_indicators_cataract: int (0 or 1)
    - disease_indicators_amd: int (0 or 1)
    - disease_indicators_hypertension: int (0 or 1)
    - disease_indicators_myopia: int (0 or 1)
    - disease_indicators_other: int (0 or 1)
    """
    spec = ExportSpec(
        annotation_tasks=["classification"],
        classification_class_names=["disease_indicators"],
        classification_task_types={"disease_indicators": "multi_label"},
        multi_label_keys={
            "disease_indicators": [
                "normal",
                "diabetes",
                "glaucoma",
                "cataract",
                "amd",
                "hypertension",
                "myopia",
                "other",
            ]
        },
        classification_label_type="int",
        require_annotations=True,
    )
    path = OUT_DIR / "disease_indicators_multilabel_flat.parquet"
    export(spec, parquet_path=path)
    print(f"Multi-label classification (flat): {path}")
    return path


# -----------------------------------------------------------------------------
# Example 4: Multi-label without keys - JSON string fallback
# -----------------------------------------------------------------------------
def export_multilabel_json_fallback() -> Path:
    """
    Export multi-label classification without explicit keys.
    
    Output columns:
    - disease_indicators_labels: str (JSON string like '{"normal": 0, "diabetes": 1, ...}')
    """
    spec = ExportSpec(
        annotation_tasks=["classification"],
        classification_class_names=["disease_indicators"],
        classification_task_types={"disease_indicators": "multi_label"},
        # No multi_label_keys provided - falls back to JSON string
        require_annotations=True,
    )
    path = OUT_DIR / "disease_indicators_multilabel_json.parquet"
    export(spec, parquet_path=path)
    print(f"Multi-label classification (JSON fallback): {path}")
    return path


# -----------------------------------------------------------------------------
# Example 5: Float labels for soft-label training
# -----------------------------------------------------------------------------
def export_binary_classification_float() -> Path:
    """
    Export glaucoma classification as float columns for soft labels.
    
    Output columns:
    - glaucoma_label: float (0.0 or 1.0, can be used with BCELoss)
    - glaucoma_class_label: str ('positive' or 'negative')
    """
    spec = ExportSpec(
        annotation_tasks=["classification"],
        classification_class_names=["glaucoma"],
        classification_task_types={"glaucoma": "binary"},
        classification_label_type="float",  # float instead of int
        require_annotations=True,
    )
    path = OUT_DIR / "glaucoma_binary_float.parquet"
    export(spec, parquet_path=path)
    print(f"Binary classification (float): {path}")
    return path


# -----------------------------------------------------------------------------
# Example 6: Multiple classifications in one export
# -----------------------------------------------------------------------------
def export_multiple_classifications() -> Path:
    """
    Export multiple classification tasks in one parquet file.
    
    Output columns:
    - glaucoma_label: int
    - glaucoma_class_label: str
    - disease_category_label: int
    - disease_category_class_label: str
    """
    spec = ExportSpec(
        annotation_tasks=["classification"],
        classification_class_names=["glaucoma", "disease_category"],
        classification_task_types={"glaucoma": "binary", "disease_category": "multi_class"},
        classification_label_type="int",
        require_annotations=True,
    )
    path = OUT_DIR / "multiple_classifications.parquet"
    export(spec, parquet_path=path)
    print(f"Multiple classifications: {path}")
    return path


# -----------------------------------------------------------------------------
# Example 7: Combined with grading - complete training dataset
# -----------------------------------------------------------------------------
def export_combined_grading_classification() -> Path:
    """
    Export grading + classification in one training-ready dataset.
    
    Output columns:
    - dr_grade: int (DR severity: 0-4)
    - dr_original_grade: str
    - dr_scale_name: str
    - glaucoma_label: int (0 or 1)
    - glaucoma_class_label: str
    - ... plus image metadata
    """
    spec = ExportSpec(
        annotation_tasks=["grading", "classification"],
        disease_types=["DR"],
        classification_class_names=["glaucoma"],
        classification_task_types={"glaucoma": "binary"},
        classification_label_type="int",
        require_annotations=True,
    )
    path = OUT_DIR / "combined_grading_classification.parquet"
    export(spec, parquet_path=path)
    print(f"Combined grading + classification: {path}")
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    
    print("Classification Export Examples")
    print("=" * 60)
    
    # Run all examples
    export_binary_classification_int()
    export_multiclass_classification()
    export_multilabel_with_keys()
    export_multilabel_json_fallback()
    export_binary_classification_float()
    export_multiple_classifications()
    export_combined_grading_classification()
    
    print("\nAll exports complete!")
    print(f"\nOutput directory: {OUT_DIR}")
    print("\nYou can inspect the parquet files with:")
    print("  python -c 'import pyarrow.parquet as pq; print(pq.read_table(\"path/to/file.parquet\").to_pandas())'")


if __name__ == "__main__":
    main()
