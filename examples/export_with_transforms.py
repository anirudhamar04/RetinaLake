#!/usr/bin/env -S uv run
"""
Spatial + photometric transforms on a PyTorch dataset.

Spatial transforms (resize, flip, ROI mask) update the image *and* its masks/boxes/keypoints
together; photometric transforms (CLAHE, normalize) touch only the image. This keeps
segmentation masks aligned with the image under augmentation.

    uv run python examples/export_with_transforms.py

`torch="dataset"` requires a `parquet_path` — rows are written once, then read per sample.
"""

from pathlib import Path

from chaksudb.export import ExportSpec, export
from chaksudb.export.transforms import (
    SpatialCompose,
    PhotometricCompose,
    Resize,
    RandomHorizontalFlip,
    FundusROIMask,
    CLAHE,
    Normalize,
)

OUT_DIR = Path("examples/export_output")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    spec = ExportSpec(
        dataset_names=["IDRID"],
        annotation_tasks=["segmentation"],
        require_annotations_mode="all",
        include_fundus_roi=True,        # carries the fundus circle for FundusROIMask
    )

    spatial = SpatialCompose([
        Resize((512, 512)),
        RandomHorizontalFlip(p=0.5),
        FundusROIMask(),                # zero everything outside the fundus circle (image + masks)
    ])
    photometric = PhotometricCompose([
        CLAHE(clip_limit=2.0),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = export(
        spec,
        parquet_path=OUT_DIR / "transforms_demo.parquet",
        torch="dataset",
        spatial=spatial,
        transform=photometric,
    )
    print(f"Dataset with transforms: len={len(dataset)}")
    sample = dataset[0]
    print(f"First sample keys: {list(sample.keys()) if isinstance(sample, dict) else type(sample)}")


if __name__ == "__main__":
    main()
