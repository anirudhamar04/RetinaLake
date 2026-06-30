#!/usr/bin/env -S uv run
"""
Filter by disease grade with `grade_filter`.

`grade_filter` keeps only images whose normalized grade matches. Pass a list of allowed grades
or a {"min": .., "max": ..} range, keyed by disease type (must be a subset of `disease_types`).

    uv run python examples/dr_grade_filtering.py

Here: only DR grade 0 (no diabetic retinopathy) images. Use {"DR": {"min": 1, "max": 3}} for a
mild-to-severe range instead.
"""

from pathlib import Path

from chaksudb.export import ExportSpec, export

OUT_DIR = Path("examples/export_output")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = ExportSpec(
        split_names=["train", "val", "test"],
        modalities=["fundus"],
        annotation_tasks=["grading"],
        require_annotations_mode="all",
        disease_types=["DR"],
        grade_filter={"DR": [0]},          # keep only DR grade 0
    )
    path = export(spec, parquet_path=OUT_DIR / "dr_grade0.parquet")
    print(f"DR grade-0 export: {path}")


if __name__ == "__main__":
    main()
