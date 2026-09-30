#!/usr/bin/env -S uv run
"""
Use the built-in presets for common tasks.

`chaksudb.export.presets` provides one-line factory functions that return a ready-to-use
ExportSpec for the most common training setups, so you don't have to remember every field.

    uv run python examples/export_with_presets.py

Available presets include: dr_classification, glaucoma_detection, lesion_segmentation,
optic_disc_segmentation, lesion_detection_coco, fundus_captioning, quality_assessment,
multi_label_disease, landmark_detection, multi_task.
"""

from pathlib import Path

from chaksudb.export import export
from chaksudb.export import presets

OUT_DIR = Path("examples/export_output")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # DR classification across whichever datasets you have ingested.
    spec = presets.dr_classification()
    path = export(spec, parquet_path=OUT_DIR / "preset_dr_classification.parquet")
    print(f"dr_classification preset -> {path}")

    # Glaucoma detection, scoped to a couple of datasets.
    spec = presets.glaucoma_detection(datasets=["REFUGE", "AIROGS"])
    path = export(spec, parquet_path=OUT_DIR / "preset_glaucoma.parquet")
    print(f"glaucoma_detection preset -> {path}")

    # Lesion segmentation.
    spec = presets.lesion_segmentation(datasets=["IDRID"])
    path = export(spec, parquet_path=OUT_DIR / "preset_lesion_seg.parquet")
    print(f"lesion_segmentation preset -> {path}")


if __name__ == "__main__":
    main()
