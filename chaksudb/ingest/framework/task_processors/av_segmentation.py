"""
Shared utility for processing artery/vein (AV) segmentation masks.

Produces exactly two segmentation annotations per image, both sharing a group_id:

    av      (unified_format="color_mask")
            RGB PNG — R channel = arteries, G channel = overlap, B channel = veins.
            For binary-only sources (e.g. LES-AV), G=0 since there is no overlap layer.

    vessels (unified_format="binary_mask")
            Grayscale PNG — all non-background pixels merged into a single binary mask.
            For binary sources, derived as arteries | veins (or from an explicit vessel file).

Export usage:
    # Binary vessel mask
    ExportSpec(annotation_tasks=["segmentation"], segmentation_types=["vessels"])

    # AV color mask — load as RGB; R=arteries, G=overlap, B=veins
    ExportSpec(annotation_tasks=["segmentation"], segmentation_types=["av"])
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import numpy as np
from PIL import Image as PILImage

from chaksudb.db.models import SegmentationAnnotation
from chaksudb.ingest.framework.gen_uuid import generate_image_group_uuid, generate_segmentation_uuid
from chaksudb.ingest.framework.raw_file_helpers import register_individual_file
from chaksudb.ingest.framework.task_processors.segmentation_processor import (
    get_or_create_annotation_type,
    process_segmentation_from_binary_mask,
    _build_locator_file_path,
)
from chaksudb.storage.paths import generate_storage_path, get_storage_root

logger = logging.getLogger(__name__)

_AV_DESCRIPTION = (
    "Color-coded AV mask: R=arteries, G=overlap (crossings), B=veins. "
    "Load as RGB; each channel is a binary (0/255) vessel-class map."
)
_VESSELS_DESCRIPTION = "Binary vessel mask — union of all AV classes (AV-derived)."


async def _save_av_color_mask(
    rgb: np.ndarray,
    dataset_name: str,
    segmentation_id: UUID,
) -> str:
    """Save an RGB numpy array as a color AV mask PNG; return the DB-safe path string."""
    output_path = generate_storage_path(
        dataset_name=dataset_name,
        subdirectory="masks/av",
        filename=f"{str(segmentation_id)[:8]}.png",
    )
    pil_img = PILImage.fromarray(rgb, mode="RGB")
    await asyncio.to_thread(pil_img.save, str(output_path))
    return _build_locator_file_path(output_path, root=get_storage_root())


async def _make_av_annotation(
    rgb: np.ndarray,
    source_path: Path,
    image_id: UUID,
    dataset_id: UUID,
    dataset_name: str,
    group_id: UUID,
    raw_file_id: UUID,
    chain_id: UUID,
    annotation_method: str,
) -> SegmentationAnnotation:
    """Create and store the 'av' color-mask SegmentationAnnotation."""
    annotation_type_id = await get_or_create_annotation_type("av", _AV_DESCRIPTION)
    segmentation_id = generate_segmentation_uuid(
        image_id=image_id,
        annotation_type_id=annotation_type_id,
        raw_data_id=raw_file_id,
    )
    mask_file_path = await _save_av_color_mask(rgb, dataset_name, segmentation_id)
    original_file_path = _build_locator_file_path(source_path, root=None)

    return SegmentationAnnotation(
        segmentation_id=segmentation_id,
        image_id=image_id,
        annotation_type_id=annotation_type_id,
        mask_file_path=mask_file_path,
        unified_format="color_mask",
        original_format=source_path.suffix.lstrip("."),
        original_file_path=original_file_path,
        group_id=group_id,
        raw_data_id=raw_file_id,
        provenance_chain_id=chain_id,
        annotation_method=annotation_method,
    )


async def _make_vessels_annotation(
    binary: np.ndarray,
    source_path: Path,
    image_id: UUID,
    dataset_id: UUID,
    dataset_name: str,
    group_id: UUID,
    raw_file_id: UUID,
    chain_id: UUID,
    annotation_method: str,
) -> SegmentationAnnotation:
    """Create and store the 'vessels' binary SegmentationAnnotation via a temp file."""
    fd, tmp_str = tempfile.mkstemp(suffix=".png")
    os.close(fd)
    tmp = Path(tmp_str)
    try:
        await asyncio.to_thread(PILImage.fromarray(binary, mode="L").save, str(tmp))
        return await process_segmentation_from_binary_mask(
            mask_path=tmp,
            annotation_type="vessels",
            annotation_description=_VESSELS_DESCRIPTION,
            image_id=image_id,
            group_id=group_id,
            raw_data_id=raw_file_id,
            provenance_chain_id=chain_id,
            annotation_method=annotation_method,
            dataset_name=dataset_name,
            dataset_id=dataset_id,
            original_source_path=source_path,
        )
    finally:
        tmp.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

async def process_av_color_mask(
    av_mask_path: Path,
    image_id: UUID,
    dataset_id: UUID,
    dataset_name: str,
    group_identifier: str,
    annotation_method: str = "manual",
) -> List[SegmentationAnnotation]:
    """
    Process a color-coded RGB AV mask into two segmentation annotations.

    Standard color convention: red=arteries, blue=veins, green=overlap, white=uncertain.
    The stored 'av' mask preserves this as R/G/B channels. The stored 'vessels' mask
    is a binary union of all non-background pixels.

    Args:
        av_mask_path: Path to RGB PNG AV mask.
        image_id: UUID of the parent image.
        dataset_id: Dataset UUID.
        dataset_name: Dataset name for mask storage path.
        group_identifier: Unique string per image (e.g. image stem) for group_id derivation.
        annotation_method: "manual" (default) or "semi_automatic".

    Returns:
        List of two SegmentationAnnotation: [av_annotation, vessels_annotation].
    """
    raw_file_id, chain_id = await register_individual_file(
        file_path=av_mask_path,
        dataset_id=dataset_id,
        unified_annotation_type="segmentation",
        auto_detect_type=False,
    )

    rgb = await asyncio.to_thread(
        lambda: np.array(PILImage.open(av_mask_path).convert("RGB"))
    )
    vessels_binary = (rgb.any(axis=2).astype(np.uint8) * 255)

    group_id = generate_image_group_uuid(
        dataset_id=dataset_id, group_type="av_annotation", group_identifier=group_identifier
    )

    av_ann, vessels_ann = await asyncio.gather(
        _make_av_annotation(
            rgb=rgb,
            source_path=av_mask_path,
            image_id=image_id,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            group_id=group_id,
            raw_file_id=raw_file_id,
            chain_id=chain_id,
            annotation_method=annotation_method,
        ),
        _make_vessels_annotation(
            binary=vessels_binary,
            source_path=av_mask_path,
            image_id=image_id,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            group_id=group_id,
            raw_file_id=raw_file_id,
            chain_id=chain_id,
            annotation_method=annotation_method,
        ),
    )
    return [av_ann, vessels_ann]


async def process_av_binary_masks(
    artery_mask_path: Path,
    vein_mask_path: Path,
    image_id: UUID,
    dataset_id: UUID,
    dataset_name: str,
    group_identifier: str,
    vessel_mask_path: Optional[Path] = None,
    annotation_method: str = "manual",
) -> List[SegmentationAnnotation]:
    """
    Process separate binary artery and vein mask files into two segmentation annotations.

    Used for datasets (e.g. LES-AV) that provide arteries/veins as distinct binary PNGs
    rather than a color-coded RGB mask. Composes them into the same 'av' RGB convention
    (R=arteries, G=0 — no overlap layer available, B=veins) for a consistent export API.

    Args:
        artery_mask_path: Binary PNG for arteries (uint8 or bool).
        vein_mask_path: Binary PNG for veins (uint8 or bool).
        image_id: UUID of the parent image.
        dataset_id: Dataset UUID.
        dataset_name: Dataset name for mask storage path.
        group_identifier: Unique string per image for group_id derivation.
        vessel_mask_path: Explicit binary vessel mask; derived as arteries | veins if None.
        annotation_method: "manual" (default) or "semi_automatic".

    Returns:
        List of two SegmentationAnnotation: [av_annotation, vessels_annotation].
    """
    def _load(path: Path) -> np.ndarray:
        arr = np.array(PILImage.open(path))
        if arr.dtype == bool:
            arr = arr.astype(np.uint8) * 255
        if arr.ndim == 3:
            arr = arr.any(axis=2).astype(np.uint8) * 255
        return arr.astype(np.uint8)

    arteries, veins = await asyncio.gather(
        asyncio.to_thread(_load, artery_mask_path),
        asyncio.to_thread(_load, vein_mask_path),
    )

    # Compose: R=arteries, G=0 (no overlap), B=veins
    h, w = arteries.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :, 0] = arteries
    rgb[:, :, 2] = veins

    if vessel_mask_path is not None:
        vessels_binary = await asyncio.to_thread(_load, vessel_mask_path)
    else:
        vessels_binary = np.clip(
            arteries.astype(np.uint16) + veins.astype(np.uint16), 0, 255
        ).astype(np.uint8)

    raw_file_id, chain_id = await register_individual_file(
        file_path=artery_mask_path,
        dataset_id=dataset_id,
        unified_annotation_type="segmentation",
        auto_detect_type=False,
    )

    group_id = generate_image_group_uuid(
        dataset_id=dataset_id, group_type="av_annotation", group_identifier=group_identifier
    )

    av_ann, vessels_ann = await asyncio.gather(
        _make_av_annotation(
            rgb=rgb,
            source_path=artery_mask_path,
            image_id=image_id,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            group_id=group_id,
            raw_file_id=raw_file_id,
            chain_id=chain_id,
            annotation_method=annotation_method,
        ),
        _make_vessels_annotation(
            binary=vessels_binary,
            source_path=artery_mask_path,
            image_id=image_id,
            dataset_id=dataset_id,
            dataset_name=dataset_name,
            group_id=group_id,
            raw_file_id=raw_file_id,
            chain_id=chain_id,
            annotation_method=annotation_method,
        ),
    )
    return [av_ann, vessels_ann]
