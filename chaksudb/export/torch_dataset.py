"""
PyTorch Dataset: Query-backed and Parquet-backed datasets for training.

Provides PyTorch Dataset implementations that can load data directly from
database queries or from pre-exported Parquet files. Images are automatically
loaded using the storage module for easy training workflows.

The spatial transform pipeline (``spatial`` parameter) applies geometric
transforms to the image, all masks, and all coordinate annotations together.
The photometric pipeline (``transform`` parameter) applies image-only
transforms after the spatial step.
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional, Union

import torch
from PIL import Image as PILImage
from torch.utils.data import Dataset

from chaksudb.db import get_connection
from chaksudb.export.path_resolution import resolve_local_path, resolve_paths_in_row
from chaksudb.export.query_builder import QueryBuilder
from chaksudb.export.spec import ExportSpec
from chaksudb.export.transforms.base import SpatialSample
from chaksudb.export.transforms.collate import default_collate, get_collate_fn
from chaksudb.export.transforms.compose import PhotometricCompose, SpatialCompose
from chaksudb.storage import StorageLocator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _pil_to_tensor(img: PILImage.Image) -> torch.Tensor:
    """Convert a PIL Image to a float32 tensor in (C, H, W) format, values in [0, 1]."""
    try:
        import torchvision.transforms as T
        return T.ToTensor()(img)
    except (ImportError, RuntimeError):
        pass
    try:
        import numpy as np
        arr = np.array(img, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[np.newaxis, ...]
        else:
            arr = arr.transpose((2, 0, 1))
        return torch.from_numpy(arr.copy())
    except (ImportError, RuntimeError):
        pass
    w, h = img.size
    raw = list(img.getdata())
    if raw and isinstance(raw[0], (tuple, list)):
        pixels = [v for px in raw for v in px]
    else:
        pixels = raw
    t = torch.tensor(pixels, dtype=torch.float32).reshape(h, w, -1).permute(2, 0, 1) / 255.0
    if t.shape[0] == 1:
        t = t.expand(3, h, w)
    return t


def _resolve_image_path(locator: "StorageLocator") -> Path:
    """Resolve image path from storage locator.

    Only local storage is supported.  HTTP and cloud providers raise
    ValueError because PIL.Image.open cannot open a URL or object key
    directly.
    """
    if locator.storage_provider == "local":
        if not locator.file_path:
            raise ValueError(f"Local storage locator missing file_path: {locator}")
        return resolve_local_path(locator.file_path)
    raise ValueError(
        f"_resolve_image_path: storage_provider={locator.storage_provider!r} is not "
        f"supported. PILImage.open requires a local filesystem path. "
        f"Download the image first or implement a storage-backed loader for "
        f"StorageLocator(object_key={locator.object_key!r})."
    )


def _load_masks(masks_raw: Any) -> tuple[list[PILImage.Image], list[dict[str, Any]]]:
    """Load mask images from segmentation_masks annotation list.

    Returns (loaded_mask_images, mask_meta_dicts).
    """
    if masks_raw is None:
        return [], []

    if isinstance(masks_raw, str):
        try:
            masks_raw = json.loads(masks_raw)
        except json.JSONDecodeError:
            return [], []

    if not isinstance(masks_raw, list):
        return [], []

    loaded: list[PILImage.Image] = []
    meta: list[dict[str, Any]] = []
    for item in masks_raw:
        if not isinstance(item, dict):
            continue
        raw_path = item.get("mask_file_path")
        if not raw_path:
            continue
        try:
            resolved_path = resolve_local_path(raw_path)
            # color_mask = AV RGB (R=arteries, G=overlap, B=veins); everything else is grayscale
            mode = "RGB" if item.get("unified_format") == "color_mask" else "L"
            mask_img = PILImage.open(resolved_path).convert(mode)
            loaded.append(mask_img)
            meta.append(item)
        except Exception as e:
            logger.warning("Could not load mask %s: %s", raw_path, e)

    return loaded, meta


def _extract_coords(
    localization_annotations: Any,
) -> tuple[list[dict], list[dict], list[dict]]:
    """Partition localization_annotations into (bboxes, keypoints, circles)."""
    bboxes: list[dict] = []
    keypoints: list[dict] = []
    circles: list[dict] = []

    if localization_annotations is None:
        return bboxes, keypoints, circles

    if isinstance(localization_annotations, str):
        try:
            localization_annotations = json.loads(localization_annotations)
        except json.JSONDecodeError:
            return bboxes, keypoints, circles

    if not isinstance(localization_annotations, list):
        return bboxes, keypoints, circles

    for ann in localization_annotations:
        if not isinstance(ann, dict):
            continue
        coords = ann.get("coordinates", {})
        if not isinstance(coords, dict):
            continue
        loc_type = ann.get("localization_type", "")
        entry = dict(coords)
        entry["target_structure"] = ann.get("target_structure", "")
        entry["lesion_subtype"] = ann.get("lesion_subtype")

        if loc_type == "bounding_box":
            bboxes.append(entry)
        elif loc_type == "keypoint":
            keypoints.append(entry)
        elif loc_type == "center_point":
            circles.append(entry)

    return bboxes, keypoints, circles


def _maybe_add_fundus_roi_circle(circles: list[dict], row: dict[str, Any]) -> None:
    """Append the flat ``fundus_roi_*`` columns as a center_point circle when present.

    ``FundusROIMask`` reads its circle from ``sample.circles``. That list is built from the
    ``localization_annotations`` JSONB column, which is only populated when the spec includes
    the ``'localization'`` task (and ``center_point`` survives any ``localization_types``
    filter). The far more common way to request the ROI is ``include_fundus_roi=True``, which
    adds flat ``fundus_roi_cx/cy/radius`` columns instead — so without this bridge, the mask
    transform would silently fall back to the full image.

    Coordinates are in ORIGINAL image space; the spatial pipeline rescales them via
    ``transform_circle`` as later geometric transforms run. No-op when the flat columns are
    absent, or when a ``fundus_roi`` circle is already present (from the localization task) so
    we never feed FundusROIMask two competing circles.
    """
    if any(c.get("target_structure") == "fundus_roi" for c in circles):
        return
    cx, cy, r = row.get("fundus_roi_cx"), row.get("fundus_roi_cy"), row.get("fundus_roi_radius")
    if cx is None or cy is None or r is None:
        return
    circles.append({
        "target_structure": "fundus_roi",
        "center_x": float(cx),
        "center_y": float(cy),
        "radius": float(r),
        "method": row.get("fundus_roi_method"),
    })


def _build_spatial_sample(
    image: PILImage.Image,
    row: dict[str, Any],
) -> SpatialSample:
    """Build a SpatialSample from a loaded image and row dict."""
    masks, mask_meta = _load_masks(row.get("segmentation_masks"))
    bboxes, keypoints, circles = _extract_coords(row.get("localization_annotations"))
    _maybe_add_fundus_roi_circle(circles, row)

    non_spatial_keys = {
        "file_path", "object_key", "storage_provider", "bucket", "version_id",
        "segmentation_masks", "localization_annotations",
    }
    annotations = {k: v for k, v in row.items() if k not in non_spatial_keys}

    return SpatialSample(
        image=image,
        masks=masks,
        mask_meta=mask_meta,
        bboxes=bboxes,
        keypoints=keypoints,
        circles=circles,
        original_width=int(row.get("resolution_width", 0) or image.size[0]),
        original_height=int(row.get("resolution_height", 0) or image.size[1]),
        annotations=annotations,
    )


def _flatten_sample(
    sample: SpatialSample,
) -> tuple[PILImage.Image, dict[str, Any]]:
    """Flatten SpatialSample back to (image, annotations_dict)."""
    annotations = dict(sample.annotations)
    annotations["_loaded_masks"] = sample.masks
    annotations["_mask_meta"] = sample.mask_meta
    annotations["_bboxes"] = sample.bboxes
    annotations["_keypoints"] = sample.keypoints
    annotations["_circles"] = sample.circles
    return sample.image, annotations


def _fmt_classification(
    image: Union[PILImage.Image, "torch.Tensor"],
    row: dict[str, Any],
    class_names: list[str],
) -> tuple:
    """Return (image, int) for one class or (image, dict[str, int]) for multiple."""
    if len(class_names) == 1:
        val = row.get(f"{class_names[0]}_label")
        return image, int(val) if val is not None else 0
    return image, {name: int(row.get(f"{name}_label") or 0) for name in class_names}


def _fmt_grading(
    image: Union[PILImage.Image, "torch.Tensor"],
    row: dict[str, Any],
    disease_types: list[str],
) -> tuple:
    """Return (image, int) for one disease or (image, dict[str, int]) for multiple."""
    if len(disease_types) == 1:
        val = row.get(f"{disease_types[0].lower()}_grade")
        return image, int(val) if val is not None else 0
    grades = {}
    for d in disease_types:
        val = row.get(f"{d.lower()}_grade")
        if val is not None:
            grades[d] = int(val)
    return image, grades


def _fmt_segmentation(
    image: Union[PILImage.Image, "torch.Tensor"],
    row: dict[str, Any],
) -> tuple:
    """Return (image, annotation_dict) for segmentation training.

    The dict exposes both views:
      - ``segmentation``: {structure_name: PIL.Image} for direct per-structure access
      - ``_structure_index``: {structure_name: channel} fixed channel assignment
      - ``_loaded_masks`` / ``_mask_meta``: the raw lists the collate functions consume,
        so the documented ``output_format="segmentation"`` path still produces a ``masks``
        tensor when batched (previously these were dropped and the model got no targets).
    """
    masks: list[PILImage.Image] = row.get("_loaded_masks") or []
    meta: list[dict] = row.get("_mask_meta") or []
    structure_counts: dict[str, int] = {}
    structured: dict[str, PILImage.Image] = {}
    structure_index: dict[str, int] = {}
    for channel, (mask, md) in enumerate(zip(masks, meta)):
        structure = md.get("target_structure") or md.get("annotation_type") or "unknown"
        count = structure_counts.get(structure, 0)
        key = structure if count == 0 else f"{structure}_{count}"
        structure_counts[structure] = count + 1
        structured[key] = mask
        structure_index[key] = channel
    return image, {
        "segmentation": structured,
        "_structure_index": structure_index,
        "_loaded_masks": masks,
        "_mask_meta": meta,
    }


def _fmt_detection(
    image: Union[PILImage.Image, "torch.Tensor"],
    row: dict[str, Any],
) -> tuple:
    """Return (image, {"boxes": [...], "labels": [...], "keypoints": [...]})."""
    bboxes: list[dict] = row.get("_bboxes") or []
    keypoints: list[dict] = row.get("_keypoints") or []
    boxes = [
        [float(b.get("x_min", 0)), float(b.get("y_min", 0)),
         float(b.get("x_max", 0)), float(b.get("y_max", 0))]
        for b in bboxes
    ]
    labels = [b.get("target_structure") or "unknown" for b in bboxes]
    kps = [
        {"point": [float(k.get("x", 0)), float(k.get("y", 0))],
         "label": k.get("target_structure") or "unknown"}
        for k in keypoints
    ]
    return image, {"boxes": boxes, "labels": labels, "keypoints": kps}


def _fmt_vision_language(
    image: Union[PILImage.Image, "torch.Tensor"],
    row: dict[str, Any],
) -> tuple:
    """Return (image, caption_str).

    Uses the synthesized ``caption`` column written by parquet_export when
    ``caption_mode`` is set on the spec.  Falls back to ``caption_clinical_text``
    then an empty string if neither is present.
    """
    caption = (
        row.get("caption")
        or row.get("caption_clinical_text")
        or ""
    )
    return image, str(caption)


def _fmt_ssl_image_only(
    image: Union[PILImage.Image, "torch.Tensor"],
) -> Union[PILImage.Image, "torch.Tensor"]:
    """Return the image with no label — standard for SSL / self-supervised pre-training."""
    return image


def _apply_output_format(
    image: Union[PILImage.Image, "torch.Tensor"],
    row: dict[str, Any],
    output_format: str,
    class_names: Optional[list[str]],
    disease_types: Optional[list[str]],
) -> tuple:
    """Dispatch to the appropriate format helper based on output_format."""
    if output_format == "classification":
        return _fmt_classification(image, row, class_names or [])
    if output_format == "grading":
        return _fmt_grading(image, row, disease_types or [])
    if output_format == "segmentation":
        return _fmt_segmentation(image, row)
    if output_format == "detection":
        return _fmt_detection(image, row)
    if output_format == "vision_language":
        return _fmt_vision_language(image, row)
    if output_format == "ssl":
        return _fmt_ssl_image_only(image)
    return image, row


def _getitem_common(
    image: PILImage.Image,
    row: dict[str, Any],
    spatial: Optional[list] = None,
    transform: Any = None,
) -> tuple[Union[PILImage.Image, torch.Tensor], dict[str, Any]]:
    """Shared __getitem__ logic for both QueryDataset and ParquetDataset."""
    sample = _build_spatial_sample(image, row)

    if spatial:
        compose = SpatialCompose(spatial)
        sample = compose(sample)

    if transform is not None:
        if isinstance(transform, list):
            compose_photo = PhotometricCompose(transform)
            sample.image = compose_photo(sample.image)
        elif callable(transform):
            sample.image = transform(sample.image)

    return _flatten_sample(sample)


# ===================================================================
# QueryDataset
# ===================================================================


class QueryDataset(Dataset):
    """PyTorch Dataset backed by database queries.

    Supports the spatial + photometric transform pipeline.
    """

    def __init__(
        self,
        spec: ExportSpec,
        spatial: Optional[list] = None,
        transform: Any = None,
        cache_rows: bool = False,
    ):
        self.spec = spec
        self.spatial = spatial
        self.transform = transform
        self.cache_rows = cache_rows

        builder = QueryBuilder()
        plan = builder.build_query(spec)
        self._query_sql = plan.render_sql()
        self._query_params = plan.params

        self._rows: Optional[list[dict[str, Any]]] = None
        self._total_count: Optional[int] = None
        self._items_loaded: int = 0

        logger.debug(
            "Initialized QueryDataset: cache_rows=%s, spec=%s",
            cache_rows, spec.dataset_names,
        )

    def __len__(self) -> int:
        if self._total_count is not None:
            return self._total_count

        count_sql = f"SELECT COUNT(*) FROM ({self._query_sql}) _cnt"

        import asyncio
        import concurrent.futures

        async def _count():
            async with get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(count_sql, self._query_params)
                    result = await cur.fetchone()
                    return int(result[0]) if result else 0

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._total_count = asyncio.run(_count())
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, _count())
                self._total_count = fut.result()
        logger.debug("Dataset length: %d rows", self._total_count)
        return self._total_count

    def __getitem__(self, idx: int) -> tuple[Union[PILImage.Image, torch.Tensor], dict[str, Any]]:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        if self.cache_rows:
            if self._rows is None:
                self._load_all_rows()
            row = self._rows[idx]
        else:
            row = self._fetch_row(idx)

        row = resolve_paths_in_row(row)

        locator = StorageLocator(
            storage_provider=row.get("storage_provider", "local"),
            file_path=row.get("file_path"),
            bucket=row.get("bucket"),
            object_key=row.get("object_key"),
            version_id=row.get("version_id"),
        )
        image_path = _resolve_image_path(locator)
        logger.debug("loading image [%d] %s", idx, image_path)

        try:
            image = PILImage.open(image_path).convert("RGB")
        except Exception as e:
            logger.error("failed to load image [%d] %s: %s", idx, image_path, e)
            raise ValueError(f"Failed to load image from {image_path}: {e}") from e

        self._items_loaded += 1
        if self._items_loaded % 500 == 0:
            logger.info("QueryDataset: loaded %d images so far (last: %s)", self._items_loaded, image_path)

        image, out = _getitem_common(image, row, self.spatial, self.transform)
        if self.spec.output_format:
            return _apply_output_format(
                image, out, self.spec.output_format,
                self.spec.classification_class_names,
                self.spec.disease_types,
            )
        return image, out

    # --- private helpers (unchanged from original) ---

    def _load_all_rows(self) -> None:
        logger.debug("Loading all rows into memory cache...")
        import asyncio
        import concurrent.futures

        async def _load():
            async with get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(self._query_sql, self._query_params)
                    rows = await cur.fetchall()
                    if cur.description:
                        column_names = [desc.name for desc in cur.description]
                        return [dict(zip(column_names, row)) for row in rows]
                    else:
                        return [dict(row) for row in rows]

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            self._rows = asyncio.run(_load())
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, _load())
                self._rows = fut.result()
        logger.debug("Cached %d rows in memory", len(self._rows))

    def _fetch_row(self, idx: int) -> dict[str, Any]:
        fetch_sql = f"{self._query_sql}\nLIMIT 1 OFFSET {idx}"
        import asyncio
        import concurrent.futures

        async def _fetch():
            async with get_connection() as conn:
                async with conn.cursor() as cur:
                    await cur.execute(fetch_sql, self._query_params)
                    row = await cur.fetchone()
                    if row is None:
                        raise IndexError(f"Row {idx} not found")
                    if cur.description:
                        column_names = [desc.name for desc in cur.description]
                        return dict(zip(column_names, row))
                    else:
                        return dict(row)

        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(_fetch())
        else:
            with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(asyncio.run, _fetch())
                return fut.result()


# ===================================================================
# ParquetDataset
# ===================================================================


class ParquetDataset(Dataset):
    """PyTorch Dataset backed by a Parquet file.

    Supports the spatial + photometric transform pipeline.
    """

    def __init__(
        self,
        parquet_path: Path,
        spatial: Optional[list] = None,
        transform: Any = None,
        output_format: Optional[str] = None,
        class_names: Optional[list[str]] = None,
        disease_types: Optional[list[str]] = None,
    ):
        self.parquet_path = Path(parquet_path)
        self.spatial = spatial
        self.transform = transform
        self._output_format = output_format
        self._class_names = class_names
        self._disease_types = disease_types

        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Parquet file not found: {parquet_path}")

        self._table: Optional[Any] = None
        self._items_loaded: int = 0
        logger.info("ParquetDataset initialized: %s", parquet_path)

    def _load_table(self) -> None:
        if self._table is not None:
            return
        logger.info("reading parquet file into memory: %s", self.parquet_path)
        try:
            import pyarrow.parquet as pq
            self._table = pq.read_table(self.parquet_path)
            logger.info("parquet loaded: %d rows, %d columns", len(self._table), len(self._table.schema))
        except Exception as e:
            logger.error("failed to read parquet file %s: %s", self.parquet_path, e)
            raise ValueError(f"Failed to load Parquet file: {e}") from e

    def __len__(self) -> int:
        self._load_table()
        return len(self._table)  # type: ignore

    def __getitem__(self, idx: int) -> tuple[Union[PILImage.Image, torch.Tensor], dict[str, Any]]:
        if idx < 0:
            idx = len(self) + idx
        if idx < 0 or idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")

        self._load_table()
        row = self._table.slice(idx, 1).to_pylist()[0]  # type: ignore
        row = resolve_paths_in_row(row)

        locator = StorageLocator(
            storage_provider=row.get("storage_provider", "local"),
            file_path=row.get("file_path"),
            bucket=row.get("bucket"),
            object_key=row.get("object_key"),
            version_id=row.get("version_id"),
        )
        image_path = _resolve_image_path(locator)
        logger.debug("loading image [%d] %s", idx, image_path)

        try:
            image = PILImage.open(image_path).convert("RGB")
        except Exception as e:
            logger.error("failed to load image [%d] %s: %s", idx, image_path, e)
            raise ValueError(f"Failed to load image from {image_path}: {e}") from e

        self._items_loaded += 1
        if self._items_loaded % 500 == 0:
            logger.info("ParquetDataset: loaded %d images so far (last: %s)", self._items_loaded, image_path)

        image, out = _getitem_common(image, row, self.spatial, self.transform)
        if self._output_format:
            return _apply_output_format(
                image, out, self._output_format,
                self._class_names,
                self._disease_types,
            )
        return image, out


# ===================================================================
# DataLoader factory (kept for backward compat but wired to new collate)
# ===================================================================


def create_dataloader(
    spec: Optional[ExportSpec] = None,
    parquet_path: Optional[Path] = None,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 0,
    spatial: Optional[list] = None,
    transform: Any = None,
    collate_fn: Any = "default",
    cache_rows: bool = False,
    **dataloader_kwargs: Any,
) -> "torch.utils.data.DataLoader":
    """Create a PyTorch DataLoader from an ExportSpec or Parquet file.

    Args:
        spec: ExportSpec for query-backed dataset.
        parquet_path: Path to Parquet file.
        batch_size: Samples per batch (default 32).
        shuffle: Shuffle dataset (default False).
        num_workers: Worker processes (default 0).
        spatial: List of BaseSpatialTransform instances for the spatial pipeline.
        transform: Photometric transforms (list or single callable, image-only).
        collate_fn: ``"default"`` | ``"padded"`` | ``"packed"`` | callable.
        cache_rows: Cache rows in memory (query-backed only).
        **dataloader_kwargs: Extra args forwarded to DataLoader.
    """
    from torch.utils.data import DataLoader

    if spec is None and parquet_path is None:
        raise ValueError("Either spec or parquet_path must be provided")
    if spec is not None and parquet_path is not None:
        raise ValueError("Cannot provide both spec and parquet_path")

    if parquet_path is not None:
        dataset: Dataset = ParquetDataset(
            parquet_path=parquet_path, spatial=spatial, transform=transform,
        )
    else:
        dataset = QueryDataset(
            spec=spec, spatial=spatial, transform=transform, cache_rows=cache_rows,
        )

    resolved_collate = get_collate_fn(collate_fn)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=resolved_collate,
        **dataloader_kwargs,
    )

    logger.debug(
        "Created DataLoader: batch_size=%d, shuffle=%s, num_workers=%d, dataset_size=%d",
        batch_size, shuffle, num_workers, len(dataset),
    )
    return dataloader
