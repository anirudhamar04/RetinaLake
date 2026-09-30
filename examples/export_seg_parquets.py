#!/usr/bin/env -S uv run
"""
Export 3x3 segmentation parquets:  {od_oc, av, lesions} x {train, val, test}

Output layout:
  data/od_oc_train.parquet
  data/od_oc_val.parquet
  data/od_oc_test.parquet
  data/av_train.parquet
  data/av_val.parquet
  data/av_test.parquet
  data/lesions_train.parquet
  data/lesions_val.parquet
  data/lesions_test.parquet

AV covers all datasets uniformly via the single "av" color RGB mask
(R=arteries, G=overlap, B=veins) — AV-DRIVE, Fundus-AVSeg, HRF-v1/v2,
LES-AV, and RITE all store it the same way.  OD/OC are exported together in
one row.  Lesions uses consensus_only annotations.
"""

from pathlib import Path

from chaksudb.export import ExportSpec, export

OUT = Path("data")
OUT.mkdir(parents=True, exist_ok=True)

TASKS = {
    "od_oc": dict(
        annotation_tasks=["segmentation"],
        segmentation_types=["optic_disc", "optic_cup"],
        require_annotations_mode="all",
        modalities=["fundus"]
    ),
    "av": dict(
        annotation_tasks=["segmentation"],
        # Single uniform color RGB mask across all AV datasets (incl. RITE):
        # R=arteries, G=overlap, B=veins. Load the mask as RGB.
        segmentation_types=["av"],
        require_annotations_mode="all",
        modalities=["fundus"],
    ),
    "lesions": dict(
        annotation_tasks=["segmentation"],
        segmentation_types=["lesions"],
        require_annotations_mode="all",
        annotation_source="prefer_consensus",
        modalities=["fundus"],
    ),
}

SPLITS = ["train", "val", "test"]


def main() -> None:
    for task_name, spec_kwargs in TASKS.items():
        for split in SPLITS:
            out_path = OUT / f"{task_name}_{split}.parquet"
            spec = ExportSpec(
                split_names=[split],
                **spec_kwargs,
            )
            print(f"exporting {task_name}/{split} → {out_path} ...", flush=True)
            export(spec, parquet_path=out_path)
            size = out_path.stat().st_size // 1024 if out_path.exists() else 0
            print(f"  done  ({size} KB)")

    print(f"\nall 9 parquets written to {OUT.resolve()}")


if __name__ == "__main__":
    main()
