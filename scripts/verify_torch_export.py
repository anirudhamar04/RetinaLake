#!/usr/bin/env -S uv run
"""
Verify sanity of all PyTorch export combinations from export_example.py.

Run from repo root:  uv run examples/verify_torch_export.py

Covers:
  Query-backed (from spec):
    1. spec → Dataset (no transform)
    2. spec + transform → Dataset
    3. spec → DataLoader (no transform)
    4. spec + transform → DataLoader

  Parquet-backed (for each existing parquet file, same 4):
    5. parquet → Dataset (no transform)
    6. parquet + transform → Dataset
    7. parquet → DataLoader (no transform)  [variable-size images padded to batch max]
    8. parquet + transform → DataLoader

  Per-task specs (grading, segmentation, classification, keyword):
    For each task: Dataset (no transform), Dataset + transform, DataLoader + transform.
"""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image

from chaksudb.export import ExportSpec, export
from chaksudb.export.torch_dataset import ParquetDataset, create_dataloader

OUT_DIR = Path("examples/export_output")

# Parquet files produced by export_example.py (skip if missing)
PARQUET_FILES = [
    "export.parquet",
    "export_then_dl.parquet",
    "dr_grading_segmentation_localization.parquet",
    "all_images_all_metadata.parquet",
]

# Annotation tasks to test with dedicated specs
TASKS_TO_VERIFY = ["grading", "segmentation", "classification", "keyword"]

# Keys to show in sample output per task (in addition to dataset_name, image_id)
TASK_SAMPLE_KEYS = {
    "grading": ["dr_grade", "dr_original_grade", "dr_scale_name"],
    "segmentation": ["segmentation_masks"],
    "classification": ["classification_annotations"],
    "keyword": ["keywords"],
}


def get_spatial():
    """Spatial transforms: resize to 224x224."""
    from chaksudb.export.transforms import Resize
    return [Resize(224)]


def get_transform():
    """Image-only photometric transform: PIL -> Tensor."""
    try:
        import torchvision.transforms as T
    except ImportError:
        def transform(image):
            return image
        return transform

    _to_tensor = T.ToTensor()

    def transform(image):
        return _to_tensor(image)

    return transform


def get_eyepacs_spec():
    """Same spec as export_example.py for demos 2-5."""
    return ExportSpec(
        dataset_names=["EYEPACS"],
        annotation_tasks=["grading"],
        disease_types=["DR"],
        require_annotations=True,
    )


def get_spec_for_task(task: str):
    """Return a minimal ExportSpec for a single annotation task (no dataset filter)."""
    if task == "grading":
        return ExportSpec(
            annotation_tasks=["grading"],
            disease_types=["DR"],
            require_annotations=True,
        )
    if task == "segmentation":
        return ExportSpec(
            annotation_tasks=["segmentation"],
            require_annotations=True,
        )
    if task == "classification":
        return ExportSpec(
            annotation_tasks=["classification"],
            require_annotations=True,
        )
    if task == "keyword":
        return ExportSpec(
            annotation_tasks=["keyword"],
            require_annotations=True,
        )
    raise ValueError(f"Unknown task: {task}")


def check_tensor(name: str, t: torch.Tensor, expected_ndim: int | None = None) -> bool:
    """Print shape, dtype, min/max, finite; return True if sane."""
    ok = True
    print(f"    {name}: shape={tuple(t.shape)}, dtype={t.dtype}, min={t.min().item():.4f}, max={t.max().item():.4f}")
    if not torch.isfinite(t).all():
        print(f"    WARNING: {name} contains NaN or Inf")
        ok = False
    if expected_ndim is not None and t.ndim != expected_ndim:
        print(f"    WARNING: expected ndim={expected_ndim}, got {t.ndim}")
        ok = False
    return ok


def check_sample(
    label: str,
    image,
    ann: dict,
    has_transform: bool,
    extra_keys: list[str] | None = None,
) -> bool:
    """Check one (image, annotations) sample. image may be PIL or tensor."""
    ok = True
    if torch.is_tensor(image):
        expected = (3, 224, 224) if has_transform else None
        if expected and tuple(image.shape) != expected:
            print(f"    WARNING: image shape {tuple(image.shape)} != {expected}")
            ok = False
        check_tensor("image", image, expected_ndim=3)
    else:
        size = getattr(image, "size", "?")
        print(f"    image: PIL size={size}")
    print(f"    annotation keys: {sorted(ann.keys())[:12]}{'...' if len(ann) > 12 else ''}")
    keys_to_show = list(extra_keys) if extra_keys else ["dr_grade"]
    keys_to_show += ["dataset_name", "image_id"]
    for k in keys_to_show:
        if k in ann:
            v = ann[k]
            if hasattr(v, "__len__") and not isinstance(v, (str, bytes)):
                print(f"      {k}: <{type(v).__name__} len={len(v)}>")
            else:
                print(f"      {k}: {v}")
    return ok


def check_batch(batch_images: torch.Tensor, batch_anns: dict) -> bool:
    """Check one batch from DataLoader."""
    ok = True
    B = batch_images.size(0)
    check_tensor("batch_images", batch_images, expected_ndim=4)
    if batch_images.size(1) != 3:
        print("    WARNING: expected 3 channels, got", batch_images.size(1))
        ok = False
    for k in list(batch_anns.keys())[:3]:
        v = batch_anns[k]
        if hasattr(v, "__len__"):
            print(f"    batch_ann[{k}]: len={len(v)}")
        else:
            print(f"    batch_ann[{k}]: {type(v).__name__}")
    return ok


def _overlay_segmentation_masks(
    base_image,
    masks: list[dict],
    alpha: float = 0.45,
) -> "Image.Image":
    """Overlay segmentation masks on a base image with distinct colours and a legend.

    Args:
        base_image: PIL Image (RGB) or torch Tensor (C,H,W).
        masks: List of mask dicts, each with ``annotation_type`` and ``mask_file_path``.
        alpha: Opacity of the mask overlay (0 = transparent, 1 = opaque).

    Returns:
        PIL Image with coloured mask overlays and a small legend in the top-left corner.
    """
    from PIL import Image as PILImage, ImageDraw, ImageFont
    import numpy as np

    # Distinct colours for up to 10 annotation types
    PALETTE = [
        (255, 0, 0),      # red
        (0, 200, 0),      # green
        (0, 100, 255),    # blue
        (255, 200, 0),    # yellow
        (255, 0, 200),    # magenta
        (0, 220, 220),    # cyan
        (255, 128, 0),    # orange
        (128, 0, 255),    # purple
        (0, 255, 128),    # spring green
        (200, 200, 200),  # grey
    ]

    # Convert tensor to PIL if needed
    if torch.is_tensor(base_image):
        arr = base_image.permute(1, 2, 0).mul(255).clamp(0, 255).byte().numpy()
        base_pil = PILImage.fromarray(arr, "RGB")
    else:
        base_pil = base_image.convert("RGB")

    overlay = base_pil.copy()
    draw_overlay = PILImage.new("RGBA", base_pil.size, (0, 0, 0, 0))

    legend_entries: list[tuple[tuple[int, int, int], str]] = []

    for idx, mask_info in enumerate(masks):
        mask_path = mask_info.get("mask_file_path")
        ann_type = mask_info.get("annotation_type", f"mask_{idx}")
        if not mask_path:
            continue
        try:
            mask_img = PILImage.open(mask_path).convert("L")
        except Exception:
            continue

        # Resize mask to match base image if sizes differ
        if mask_img.size != base_pil.size:
            mask_img = mask_img.resize(base_pil.size, PILImage.NEAREST)

        colour = PALETTE[idx % len(PALETTE)]
        legend_entries.append((colour, ann_type))

        mask_arr = np.array(mask_img)
        # Treat any non-zero pixel as foreground
        fg = mask_arr > 0

        colour_layer = PILImage.new("RGBA", base_pil.size, (*colour, int(255 * alpha)))
        mask_binary = PILImage.fromarray((fg * 255).astype(np.uint8), "L")
        draw_overlay.paste(colour_layer, mask=mask_binary)

    # Composite overlay onto base
    base_rgba = base_pil.convert("RGBA")
    composited = PILImage.alpha_composite(base_rgba, draw_overlay).convert("RGB")

    # Draw legend
    if legend_entries:
        draw = ImageDraw.Draw(composited)
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
        except Exception:
            font = ImageFont.load_default()
        x, y = 8, 8
        for colour, label in legend_entries:
            draw.rectangle([x, y, x + 14, y + 14], fill=colour, outline=(0, 0, 0))
            draw.text((x + 20, y), label, fill=(255, 255, 255), font=font,
                       stroke_width=1, stroke_fill=(0, 0, 0))
            y += 20

    return composited


def save_sample_images(dataset, count: int, save_dir: Path, has_transform: bool) -> None:
    """Save first ``count`` samples as PNGs into ``save_dir``.

    For segmentation samples (those with ``segmentation_masks`` in annotations),
    an additional ``sample_N_seg.png`` is saved with coloured mask overlays and a legend.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    n = min(count, len(dataset))
    for i in range(n):
        image, ann = dataset[i]
        path = save_dir / f"sample_{i}.png"

        # Save the plain image
        if torch.is_tensor(image):
            try:
                from torchvision.utils import save_image

                save_image(image, path)
            except ImportError:
                pass
        else:
            try:
                image.convert("RGB").save(path)
            except Exception:
                pass
        if path.exists():
            print(f"    Saved {path}")

        # Save segmentation overlay if masks are present
        mask_meta = ann.get("_mask_meta", [])
        if mask_meta and isinstance(mask_meta, list) and any(m.get("mask_file_path") for m in mask_meta):
            seg_path = save_dir / f"sample_{i}_seg.png"
            try:
                seg_img = _overlay_segmentation_masks(image, mask_meta)
                seg_img.save(seg_path)
                if seg_path.exists():
                    print(f"    Saved {seg_path}  (mask overlay: {len(mask_meta)} masks)")
            except Exception as e:
                print(f"    WARNING: could not save mask overlay: {e}")


def run_dataset_checks(
    dataset,
    label: str,
    has_transform: bool,
    save_dir: Path,
    extra_keys: list[str] | None = None,
) -> None:
    """Run sanity checks on a Dataset: len, sample[0], save sample images."""
    n = len(dataset)
    print(f"  len = {n}")
    if n == 0:
        print("  SKIP (empty)")
        return
    print("  sample[0]:")
    image, ann = dataset[0]
    check_sample(label, image, ann, has_transform, extra_keys=extra_keys)
    save_sample_images(dataset, 4, save_dir, has_transform)


def run_dataloader_batch_check(dataloader, label: str) -> None:
    """Run sanity check on one batch from a DataLoader."""
    try:
        batch_images, batch_anns = next(iter(dataloader))
        print(f"  one batch: {batch_images.shape}, annotations: {list(batch_anns.keys())[:5]}...")
        check_batch(batch_images, batch_anns)
        if "_original_height" in batch_anns:
            print(f"    (padded to batch max — original sizes: "
                  f"h={batch_anns['_original_height'][:3]}..., "
                  f"w={batch_anns['_original_width'][:3]}...)")
    except StopIteration:
        print("  SKIP (empty DataLoader)")
    except Exception as e:
        print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# Query-backed: 4 combinations
# ---------------------------------------------------------------------------
def verify_query_backed() -> None:
    spec = get_eyepacs_spec()
    spatial = get_spatial()
    transform = get_transform()
    base = OUT_DIR / "query_backed"

    print("\n" + "=" * 60)
    print("QUERY-BACKED (spec only)")
    print("=" * 60)

    # 1. spec → Dataset (no transform)
    print("\n1. spec → Dataset (no transform)")
    try:
        dataset = export(spec, parquet_path=base / "qa_dataset.parquet", torch="dataset")
        run_dataset_checks(
            dataset, "query_dataset", has_transform=False,
            save_dir=base / "no_transform",
        )
    except Exception as e:
        print(f"  ERROR: {e}")

    # 2. spec + spatial + transform → Dataset
    print("\n2. spec + spatial + transform → Dataset")
    try:
        dataset = export(spec, parquet_path=base / "qa_dataset_tf.parquet", torch="dataset", spatial=spatial, transform=transform)
        run_dataset_checks(
            dataset, "query_dataset_tf", has_transform=True,
            save_dir=base / "with_transform",
        )
    except Exception as e:
        print(f"  ERROR: {e}")

    # 3. spec → DataLoader (no transform)
    print("\n3. spec → DataLoader (no transform)")
    try:
        dl = export(spec, torch="dataloader", batch_size=8, shuffle=False, num_workers=0)
        run_dataloader_batch_check(dl, "query_dl")
    except Exception as e:
        print(f"  ERROR: {e}")

    # 4. spec + spatial + transform → DataLoader
    print("\n4. spec + spatial + transform → DataLoader")
    try:
        dl = export(
            spec,
            torch="dataloader",
            spatial=spatial,
            transform=transform,
            batch_size=8,
            shuffle=False,
            num_workers=0,
        )
        run_dataloader_batch_check(dl, "query_dl_tf")
    except Exception as e:
        print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# Parquet-backed: for each file, 4 combinations
# ---------------------------------------------------------------------------
def verify_parquet_backed() -> None:
    spatial = get_spatial()
    transform = get_transform()
    parquet_dir = OUT_DIR / "parquet"

    for parquet_name in PARQUET_FILES:
        path = parquet_dir / parquet_name
        if not path.exists():
            path = OUT_DIR / parquet_name
        if not path.exists():
            print(f"\n--- Parquet: {parquet_name} --- SKIP (file not found)")
            continue

        print("\n" + "=" * 60)
        print(f"PARQUET-BACKED: {parquet_name}")
        print("=" * 60)
        folder_name = parquet_name.replace(".parquet", "").replace("-", "_")
        base = OUT_DIR / "parquet_backed" / folder_name

        # 5. parquet → Dataset (no transform)
        print("\n5. parquet → Dataset (no transform)")
        try:
            dataset = ParquetDataset(parquet_path=path)
            run_dataset_checks(
                dataset, "parquet_ds", has_transform=False,
                save_dir=base / "no_transform",
            )
        except Exception as e:
            print(f"  ERROR: {e}")

        # 6. parquet + spatial + transform → Dataset
        print("\n6. parquet + spatial + transform → Dataset")
        try:
            dataset = ParquetDataset(parquet_path=path, spatial=spatial, transform=transform)
            run_dataset_checks(
                dataset, "parquet_ds_tf", has_transform=True,
                save_dir=base / "with_transform",
            )
        except Exception as e:
            print(f"  ERROR: {e}")

        # 7. parquet → DataLoader (no transform)
        print("\n7. parquet → DataLoader (no transform)")
        try:
            dl = create_dataloader(parquet_path=path, batch_size=8, shuffle=False, num_workers=0)
            run_dataloader_batch_check(dl, "parquet_dl")
        except Exception as e:
            print(f"  ERROR: {e}")

        # 8. parquet + spatial + transform → DataLoader
        print("\n8. parquet + spatial + transform → DataLoader")
        try:
            dl = create_dataloader(
                parquet_path=path,
                spatial=spatial,
                transform=transform,
                batch_size=8,
                shuffle=False,
                num_workers=0,
            )
            run_dataloader_batch_check(dl, "parquet_dl_tf")
        except Exception as e:
            print(f"  ERROR: {e}")


# ---------------------------------------------------------------------------
# Per-task specs: grading, segmentation, classification, keyword
# ---------------------------------------------------------------------------
def verify_per_task_specs() -> None:
    """Run verification for each annotation task with a dedicated spec."""
    spatial = get_spatial()
    transform = get_transform()
    for task in TASKS_TO_VERIFY:
        print("\n" + "=" * 60)
        print(f"PER-TASK SPEC: {task}")
        print("=" * 60)
        try:
            spec = get_spec_for_task(task)
        except ValueError as e:
            print(f"  SKIP: {e}")
            continue
        extra_keys = TASK_SAMPLE_KEYS.get(task)
        base = OUT_DIR / "per_task" / task

        print(f"\n  {task} → Dataset (no transform)")
        try:
            dataset = export(spec, parquet_path=base / "qa.parquet", torch="dataset")
            run_dataset_checks(
                dataset,
                f"task_{task}",
                has_transform=False,
                save_dir=base / "no_transform",
                extra_keys=extra_keys,
            )
        except Exception as e:
            print(f"  ERROR: {e}")

        print(f"\n  {task} + spatial + transform → Dataset")
        try:
            dataset = export(spec, parquet_path=base / "qa_tf.parquet", torch="dataset", spatial=spatial, transform=transform)
            run_dataset_checks(
                dataset,
                f"task_{task}_tf",
                has_transform=True,
                save_dir=base / "with_transform",
                extra_keys=extra_keys,
            )
        except Exception as e:
            print(f"  ERROR: {e}")

        print(f"\n  {task} + spatial + transform → DataLoader (one batch)")
        try:
            dl = export(
                spec,
                torch="dataloader",
                spatial=spatial,
                transform=transform,
                batch_size=8,
                shuffle=False,
                num_workers=0,
            )
            run_dataloader_batch_check(dl, f"task_{task}_dl_tf")
        except Exception as e:
            print(f"  ERROR: {e}")


def main() -> None:
    print("Verifying all PyTorch export combinations (spec + parquet, with/without transform, dataset/dataloader)")
    print(f"Output directory: {OUT_DIR.resolve()}")
    verify_query_backed()
    verify_parquet_backed()
    print("\n" + "=" * 60)
    print("Per-task specs: grading, segmentation, classification, keyword")
    print("=" * 60)
    verify_per_task_specs()
    print("\n" + "=" * 60)
    print(f"Done. Sample images saved to:")
    print(f"  {OUT_DIR}/query_backed/       — query-backed dataset samples")
    print(f"  {OUT_DIR}/parquet_backed/     — parquet-backed dataset samples")
    print(f"  {OUT_DIR}/per_task/<task>/    — per-task samples (grading, segmentation, ...)")
    print("=" * 60)


if __name__ == "__main__":
    main()
