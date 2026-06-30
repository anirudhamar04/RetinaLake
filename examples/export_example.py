#!/usr/bin/env -S uv run
"""
Example: all export types using the simplified export API.

Run from repo root:  uv run examples/export_example.py

Shows:
  A. DR grading images with segmentation or localization + full metadata (Parquet)
  B. All images with all metadata (Parquet)
  1. Export to Parquet only
  2. Export to PyTorch Dataset (query-backed)
  3. Export to PyTorch DataLoader (query-backed, with optional transform)
  4. Export to Parquet then DataLoader from file (combined)
  5. Export to Dataset with transform
"""

from pathlib import Path

from chaksudb.export import ExportSpec, export

OUT_DIR = Path("examples/export_output")


# -----------------------------------------------------------------------------
# Example A: All DR grading images that have segmentation or localization
#            with full metadata (grading, segmentation, localization, image, dataset)
# -----------------------------------------------------------------------------
def export_dr_grading_with_segmentation_or_localization() -> Path:
    spec = ExportSpec(
        annotation_tasks=["grading"],
        disease_types=["DR"],
        require_annotations=True,  # only images that have annotations for requested tasks
        annotation_source="prefer_consensus",
        include_original_grade=True,
        include_scaled_grade=True,
    )
    path = OUT_DIR / "dr_grading_segmentation_localization.parquet"
    export(spec, parquet_path=path)
    return path


# -----------------------------------------------------------------------------
# Example B: All images with all metadata (no filters; include all annotation types)
# -----------------------------------------------------------------------------
def export_all_images_all_metadata() -> Path:
    spec = ExportSpec(
        annotation_tasks=[
            "grading",
            "segmentation",
            "localization",
            "classification",
            "quality",
            "keyword",
            "description",
        ],
        classification_class_names=["glaucoma", "disease_category"],  # optional: exact task pivots
        include_original_grade=True,
        include_scaled_grade=True,
        require_annotations=False,  # include every image; annotation columns null where absent
    )
    path = OUT_DIR / "all_images_all_metadata.parquet"
    export(spec, parquet_path=path)
    return path


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------------------
    # A. DR grading + segmentation/localization (full metadata)
    # -------------------------------------------------------------------------
    p = export_dr_grading_with_segmentation_or_localization()
    print(f"A. DR grading + segmentation/localization: {p}")

    # -------------------------------------------------------------------------
    # B. All images, all metadata
    # -------------------------------------------------------------------------
    p = export_all_images_all_metadata()
    print(f"B. All images, all metadata: {p}")

    # -------------------------------------------------------------------------
    # Generic spec for remaining demos (customize as needed)
    # -------------------------------------------------------------------------
    spec = ExportSpec(
        dataset_names=["EYEPACS"],  # or any dataset you have; omit for all
        annotation_tasks=["grading"],
        disease_types=["DR"],
        require_annotations=True,
    )

    # -------------------------------------------------------------------------
    # 1. Export to Parquet only (synchronous)
    # -------------------------------------------------------------------------
    parquet_path = OUT_DIR / "export.parquet"
    result = export(spec, parquet_path=parquet_path)
    print(f"1. Parquet export: wrote to {result}")

    # -------------------------------------------------------------------------
    # 2. PyTorch Dataset (file-backed; torch=... requires a parquet_path)
    # -------------------------------------------------------------------------
    dataset = export(spec, parquet_path=OUT_DIR / "ds.parquet", torch="dataset")
    print(f"2. Dataset (file-backed): len={len(dataset)}")

    # -------------------------------------------------------------------------
    # 3. PyTorch DataLoader (query-backed, with spatial + photometric)
    # -------------------------------------------------------------------------
    from chaksudb.export.transforms import Resize

    def photo_transform(image):
        """Image-only photometric transform: PIL -> Tensor."""
        try:
            import torchvision.transforms as T
        except ImportError as e:
            raise ImportError(
                "torchvision is required for photo_transform; "
                "install it with: pip install torchvision"
            ) from e
        return T.ToTensor()(image)

    dataloader = export(
        spec,
        parquet_path=OUT_DIR / "dl.parquet",
        torch="dataloader",
        spatial=[Resize(224)],
        transform=photo_transform,
        batch_size=32,
        shuffle=True,
        num_workers=0,
    )
    print(f"3. DataLoader (file-backed + spatial + transform): batch_size={dataloader.batch_size}")

    # -------------------------------------------------------------------------
    # 4. Parquet + DataLoader from file (write once, then load from file)
    # -------------------------------------------------------------------------
    parquet_path_2 = OUT_DIR / "export_then_dl.parquet"
    dl_from_file = export(
        spec,
        parquet_path=parquet_path_2,
        torch="dataloader",
        spatial=[Resize(224)],
        transform=photo_transform,
        batch_size=16,
        num_workers=0,
    )
    print(f"4. Parquet + DataLoader from file: {parquet_path_2} -> DataLoader")

    # -------------------------------------------------------------------------
    # 5. Dataset with spatial transform (e.g. for custom batching)
    # -------------------------------------------------------------------------
    dataset_with_transform = export(
        spec,
        parquet_path=OUT_DIR / "ds_transform.parquet",
        torch="dataset",
        spatial=[Resize(224)],
        transform=photo_transform,
    )
    print(f"5. Dataset with transform: len={len(dataset_with_transform)}")

    print("\nDone. Outputs under examples/export_output/")
    print("  A. dr_grading_segmentation_localization.parquet  — DR + seg/localization + full metadata")
    print("  B. all_images_all_metadata.parquet               — All images, all annotation columns")


if __name__ == "__main__":
    main()
