"""
Export "is this image normal?" together with fundus ROI + IQA quality, per image.

Three orthogonal signals, one flat row:
  - health_status      'normal' | 'abnormal' | None  (derived across grading + disease
                       classification — works uniformly however a dataset recorded disease)
  - fundus_roi_*       cx / cy / radius / method      (AutoMorph M0 fundus circle fit)
  - quality_score/label IQA p_good (0-1) + good/usable/bad  (AutoMorph M1 EyePACS ensemble)

    uv run python examples/export_normal_roi_iqa.py            # label every image
    uv run python examples/export_normal_roi_iqa.py normal     # only normal images
    uv run python examples/export_normal_roi_iqa.py abnormal   # only abnormal images
"""

import asyncio
import sys

from chaksudb.db.connection import init_pool
from chaksudb.export.api import export
from chaksudb.export.spec import ExportSpec


async def main() -> None:
    which = sys.argv[1] if len(sys.argv) > 1 else None   # None | 'normal' | 'abnormal'
    await init_pool()

    spec = ExportSpec(
        dataset_names=["FIVES", "AIROGS", "IDRID", "MESSIDOR", "ODIR-5K"],

        # --- normal / abnormal ---
        include_health_status=True,            # emit the health_status column
        health_status_filter=which,            # None keeps all images; else filter

        # --- IQA quality (AutoMorph M1) ---
        # emits quality_score (p_good 0-1) + quality_label (good/usable/bad) as columns;
        # uncomment a threshold to also drop low-quality images at export time.
        quality_types=["overall"],
        # iqa_min_quality_score=0.7,
        # iqa_quality_labels=["good", "usable"],

        # --- fundus ROI (AutoMorph M0) ---
        include_fundus_roi=True,               # fundus_roi_cx/cy/radius/method columns
    )

    suffix = which or "all"
    path = export(spec, parquet_path=f"examples/export_output/normal_roi_iqa_{suffix}.parquet")
    print(f"Exported {suffix} images with ROI + IQA -> {path}")


if __name__ == "__main__":
    asyncio.run(main())
