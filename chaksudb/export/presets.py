"""
Export Presets: One-line factory functions for common ML task configurations.

Each preset returns a fully configured ExportSpec ready for use with
``export()``.
"""

from typing import Optional

from chaksudb.export.spec import ExportSpec


def dr_classification(
    datasets: Optional[list[str]] = None,
    split: Optional[str] = None,
) -> ExportSpec:
    """5-class DR severity classification preset."""
    return ExportSpec(
        dataset_names=datasets,
        split_names=[split] if split else None,
        annotation_tasks=["grading"],
        disease_types=["DR"],
        modalities=["fundus"],
        require_annotations_mode="all",
    )


def glaucoma_detection(
    datasets: Optional[list[str]] = None,
    split: Optional[str] = None,
) -> ExportSpec:
    """Binary glaucoma detection from classification annotations."""
    return ExportSpec(
        dataset_names=datasets,
        split_names=[split] if split else None,
        annotation_tasks=["classification"],
        classification_class_names=["glaucoma"],
        classification_task_types={"glaucoma": "binary"},
        modalities=["fundus"],
        require_annotations_mode="all",
    )


def lesion_segmentation(
    lesion_types: Optional[list[str]] = None,
    datasets: Optional[list[str]] = None,
    split: Optional[str] = None,
) -> ExportSpec:
    """Lesion segmentation preset (default: MA, HE, EX, SE)."""
    return ExportSpec(
        dataset_names=datasets,
        split_names=[split] if split else None,
        annotation_tasks=["segmentation"],
        segmentation_types=["lesion"],
        lesion_subtypes=lesion_types,
        modalities=["fundus"],
        require_annotations_mode="all",
    )


def optic_disc_segmentation(
    datasets: Optional[list[str]] = None,
    split: Optional[str] = None,
) -> ExportSpec:
    """OD/cup segmentation for CDR estimation."""
    return ExportSpec(
        dataset_names=datasets,
        split_names=[split] if split else None,
        annotation_tasks=["segmentation"],
        segmentation_types=["optic_disc", "optic_cup"],
        modalities=["fundus"],
        require_annotations_mode="all",
    )


def lesion_detection_coco(
    datasets: Optional[list[str]] = None,
    split: Optional[str] = None,
    category_map: Optional[dict[str, int]] = None,
) -> ExportSpec:
    """COCO-format lesion detection preset."""
    return ExportSpec(
        dataset_names=datasets,
        split_names=[split] if split else None,
        annotation_tasks=["localization"],
        localization_types=["bounding_box"],
        modalities=["fundus"],
        require_annotations_mode="all",
        detection_format="coco",
        detection_category_map=category_map or {"lesions": 1},
    )


def fundus_captioning(
    datasets: Optional[list[str]] = None,
    split: Optional[str] = None,
) -> ExportSpec:
    """Image captioning preset with dictionary-augmented captions."""
    return ExportSpec(
        dataset_names=datasets,
        split_names=[split] if split else None,
        caption_mode="all",
        modalities=["fundus"],
    )


def quality_assessment(
    datasets: Optional[list[str]] = None,
    split: Optional[str] = None,
) -> ExportSpec:
    """Image quality assessment preset."""
    return ExportSpec(
        dataset_names=datasets,
        split_names=[split] if split else None,
        annotation_tasks=["quality"],
        modalities=["fundus"],
        require_annotations_mode="all",
    )


def multi_label_disease(
    class_names: Optional[list[str]] = None,
    datasets: Optional[list[str]] = None,
    split: Optional[str] = None,
) -> ExportSpec:
    """Multi-label disease detection (ODIR-style)."""
    names = class_names or ["disease_indicators"]
    task_types = {n: "multi_label" for n in names}
    return ExportSpec(
        dataset_names=datasets,
        split_names=[split] if split else None,
        annotation_tasks=["classification"],
        classification_class_names=names,
        classification_task_types=task_types,
        modalities=["fundus"],
        require_annotations_mode="all",
    )


def landmark_detection(
    datasets: Optional[list[str]] = None,
    split: Optional[str] = None,
) -> ExportSpec:
    """Fovea + OD center keypoint detection preset."""
    return ExportSpec(
        dataset_names=datasets,
        split_names=[split] if split else None,
        annotation_tasks=["localization"],
        localization_types=["keypoint", "center_point"],
        modalities=["fundus"],
        require_annotations_mode="all",
    )


def multi_task(
    tasks: Optional[list[str]] = None,
    datasets: Optional[list[str]] = None,
    split: Optional[str] = None,
) -> ExportSpec:
    """Multi-task learning preset combining grading+segmentation+localization."""
    ann_tasks = tasks or ["grading", "segmentation", "localization"]
    kwargs: dict = dict(
        dataset_names=datasets,
        split_names=[split] if split else None,
        annotation_tasks=ann_tasks,
        modalities=["fundus"],
        require_annotations_mode="none",
    )
    # grading requires disease_types
    if "grading" in ann_tasks:
        kwargs["disease_types"] = ["DR"]
    return ExportSpec(**kwargs)
