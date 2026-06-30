#!/usr/bin/env -S uv run
"""
Object detection export in COCO format.

Localization annotations (bounding boxes) can be written as a COCO JSON sidecar alongside the
Parquet, ready for detectron2 / mmdetection / pycocotools.

    uv run python examples/export_coco_detection.py

Writes two files: a Parquet table and a COCO JSON. `detection_category_map` fixes the
target_structure -> category_id mapping (omit it to auto-assign).
"""

from pathlib import Path

from chaksudb.export import ExportSpec, export
from chaksudb.export import presets

OUT_DIR = Path("examples/export_output")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Option A: build the spec explicitly.
    spec = ExportSpec(
        dataset_names=["IDRID", "DDR"],
        annotation_tasks=["localization"],
        localization_types=["bounding_box"],
        require_annotations_mode="all",
        detection_format="coco",
        detection_category_map={"microaneurysm": 1, "hemorrhage": 2, "hard_exudate": 3},
    )
    parquet_path = OUT_DIR / "lesions_detection.parquet"
    coco_path = OUT_DIR / "lesions_detection_coco.json"
    export(spec, parquet_path=parquet_path, coco_path=coco_path)
    print(f"COCO export: {parquet_path} + {coco_path}")

    # Option B: the same thing via the preset.
    spec = presets.lesion_detection_coco(datasets=["IDRID"])
    export(
        spec,
        parquet_path=OUT_DIR / "preset_detection.parquet",
        coco_path=OUT_DIR / "preset_detection_coco.json",
    )
    print("lesion_detection_coco preset -> preset_detection.parquet + preset_detection_coco.json")


if __name__ == "__main__":
    main()
