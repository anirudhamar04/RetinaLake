"""
Run ROI detection and IQA scoring on all images in the database, storing results
back into the DB. Both the fundus ROI circle and the quality decision come from
AutoMorph (``external/automorph/``):

  * ROI   — AutoMorph M0 fundus mask fit (``M0_Preprocess/fundus_prep.get_mask``),
            which returns the fundus circle center + radius in original-image space.
  * IQA   — AutoMorph M1 EyePACS quality ensemble (8 EfficientNet-b4 models). The
            mean softmax over {good, usable, bad} gives p_good; the gradability
            decision follows AutoMorph's merge rule (good, or usable with p_bad<0.25,
            is gradable; everything else is bad).

Stored as:
  quality_annotations  — quality_type='overall', scale_description='AutoMorph EyePACS QA'
  localization_annotations — localization_type='center_point', target_structure='fundus_roi'

Both are idempotent: re-running upserts the same primary key and updates values.

By default, images that already have results in either the DB or in results.json
are skipped. Use --force to re-run inference on all images regardless.

Results are saved/merged into results.json before any DB upsert, so a failed
upsert can be retried with --from-file without rerunning inference.

Usage:
    uv run python chaksudb/ingest/scripts/run_roi_iqa.py
    uv run python chaksudb/ingest/scripts/run_roi_iqa.py --dataset MESSIDOR
    uv run python chaksudb/ingest/scripts/run_roi_iqa.py --batch-size 64 --no-roi
    uv run python chaksudb/ingest/scripts/run_roi_iqa.py --force
    uv run python chaksudb/ingest/scripts/run_roi_iqa.py --save-file custom.json
    uv run python chaksudb/ingest/scripts/run_roi_iqa.py --from-file results.json
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from PIL import Image as PILImage
from psycopg.rows import dict_row
from torch.utils.data import DataLoader, Dataset as TorchDataset
from tqdm import tqdm

# AutoMorph M0 fundus preprocessing (ROI mask fit) lives outside the package tree.
_REPO_ROOT = Path(__file__).resolve().parents[3]
_AUTOMORPH_M0 = _REPO_ROOT / "external" / "automorph" / "M0_Preprocess"
if str(_AUTOMORPH_M0) not in sys.path:
    sys.path.insert(0, str(_AUTOMORPH_M0))
import fundus_prep as prep  # noqa: E402  (path injected above)

from chaksudb.db import close_pool, get_connection
from chaksudb.db.models import LocalizationAnnotation, QualityAnnotation
from chaksudb.db.queries.annotation_types import (
    bulk_upsert_localization_annotations,
    bulk_upsert_quality_annotations,
)
from chaksudb.export.path_resolution import resolve_local_path
from chaksudb.ingest.framework.gen_uuid import (
    generate_localization_uuid,
    generate_quality_uuid,
)
from chaksudb.ingest.framework.provenance import create_provenance_chain
from chaksudb.ingest.framework.transformations import log_and_link_transformation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_IQA_IMAGE_SIZE = 512
_SCALE_DESCRIPTION = "AutoMorph EyePACS QA (ensemble p_good)"

# AutoMorph M1 EyePACS quality ensemble: 8 EfficientNet-b4 checkpoints.
_QA_CKPT_DIR = (
    _REPO_ROOT
    / "external" / "automorph"
    / "M1_Retinal_Image_quality_EyePACS"
    / "Retinal_quality" / "EyePACS_quality" / "efficientnet"
)
# AutoMorph quality classes: 0=good, 1=usable, 2=bad
_QA_CLASS_NAMES = {0: "good", 1: "usable", 2: "bad"}
# Gradability rule from AutoMorph's merge_quality_assessment: a 'usable' prediction
# only counts as gradable when the mean bad-probability is below this threshold.
_QA_BAD_THRESHOLD = 0.25

# Reference image for LAB (Reinhard) illumination normalization applied BEFORE ROI
# detection. On badly-illuminated images Otsu can latch onto the bright optic disc instead
# of the fundus boundary; normalizing the per-channel mean/std to a well-exposed reference
# makes the fundus-vs-background contrast consistent so the circle fits the whole disc.
# Override with --roi-reference (or the ROI_REFERENCE_IMAGE env var); detection silently
# skips normalization if the reference is missing/unset. A well-exposed fundus image works
# best as the reference.
_DEFAULT_ROI_REFERENCE = (
    Path(os.environ["ROI_REFERENCE_IMAGE"])
    if os.environ.get("ROI_REFERENCE_IMAGE")
    else None
)


def lab_color_transfer(source: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Reinhard LAB color transfer: shift source's per-channel mean/std to match the
    reference. Both inputs are uint8 RGB; returns uint8 RGB. Geometry is untouched, so ROI
    coordinates computed on the result are valid on the original image.
    """
    src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB).astype(np.float32)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB).astype(np.float32)

    result = np.zeros_like(src_lab)
    for c in range(3):
        src_mean, src_std = src_lab[..., c].mean(), src_lab[..., c].std() + 1e-6
        ref_mean, ref_std = ref_lab[..., c].mean(), ref_lab[..., c].std() + 1e-6
        result[..., c] = (src_lab[..., c] - src_mean) * (ref_std / src_std) + ref_mean

    result = np.clip(result, 0, 255).astype(np.uint8)
    return cv2.cvtColor(result, cv2.COLOR_LAB2RGB)


def _load_roi_reference(path: Optional[Path]) -> Optional[np.ndarray]:
    """Load the ROI reference image as uint8 RGB, or None (with a warning) if unavailable."""
    if path is None:
        return None
    if not Path(path).exists():
        logger.warning(
            "ROI reference image not found at %s — proceeding without LAB normalization.",
            path,
        )
        return None
    ref_bgr = cv2.imread(str(path))
    if ref_bgr is None:
        logger.warning("Could not read ROI reference image %s — skipping normalization.", path)
        return None
    logger.info("Using ROI illumination reference: %s", path)
    return cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2RGB)


# ---------------------------------------------------------------------------
# IQA model
# ---------------------------------------------------------------------------

def _get_device() -> torch.device:
    """Return CUDA device if functional, otherwise CPU.

    torch.cuda.is_available() returns True even when the installed PyTorch
    wheels don't include a kernel image for the current GPU (e.g. Blackwell
    sm_120 with a PyTorch build that only ships up to sm_90).  A small smoke
    test catches that case before we commit to running the whole pipeline on
    a device that will just raise CUDA errors for every batch.
    """
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device="cuda")
            return torch.device("cuda")
        except Exception as e:
            logger.warning("CUDA not usable (%s); falling back to CPU", e)
    return torch.device("cpu")


def _build_qa_model() -> nn.Module:
    """Build one AutoMorph EyePACS-quality EfficientNet-b4 with the 3-class head.

    Uses ``from_name`` (not ``from_pretrained``) — the ImageNet weights are
    immediately overwritten by the AutoMorph checkpoint, so there's no need to
    download them.
    """
    model = EfficientNet.from_name("efficientnet-b4")
    model._fc = nn.Sequential(
        nn.Linear(1792, 256),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(256, 64),
        nn.ReLU(),
        nn.Dropout(p=0.5),
        nn.Linear(64, 3),
    )
    return model


class _IQAModel:
    """AutoMorph M1 EyePACS quality ensemble (8 EfficientNet-b4 models).

    The mean softmax over {good, usable, bad} gives ``p_good`` (the stored score).
    The label follows AutoMorph's gradability decision (merge_quality_assessment):
    a 'good' prediction is good; a 'usable' prediction is gradable only when the
    mean bad-probability is below ``_QA_BAD_THRESHOLD``, otherwise it's bad.
    """

    def __init__(self) -> None:
        ckpts = sorted(_QA_CKPT_DIR.glob("*_seed_*/best_loss_checkpoint.pth"))
        if not ckpts:
            raise FileNotFoundError(
                f"AutoMorph EyePACS quality checkpoints not found under {_QA_CKPT_DIR}. "
                "Expected 8 best_loss_checkpoint.pth files."
            )
        self.device = _get_device()
        logger.info("Loading %d AutoMorph EyePACS QA models on %s", len(ckpts), self.device)
        self.models: list[nn.Module] = []
        for cp in ckpts:
            model = _build_qa_model()
            model.load_state_dict(torch.load(cp, map_location=self.device))
            model.eval().to(self.device)
            self.models.append(model)

    def predict_batch(self, tensors: torch.Tensor) -> list[tuple[float, str]]:
        """Return (p_good, label) for each image in the batch."""
        imgs = tensors.to(self.device, dtype=torch.float32)
        probs_sum: Optional[torch.Tensor] = None
        with torch.no_grad():
            for model in self.models:
                probs = torch.softmax(model(imgs), dim=1)
                probs_sum = probs if probs_sum is None else probs_sum + probs
        mean_probs = (probs_sum / len(self.models)).cpu().numpy()  # (N, 3)

        results: list[tuple[float, str]] = []
        for row in mean_probs:
            p_good, _p_usable, p_bad = float(row[0]), float(row[1]), float(row[2])
            pred = int(row.argmax())
            if pred == 0:
                label = "good"
            elif pred == 1 and p_bad < _QA_BAD_THRESHOLD:
                label = "usable"
            else:
                label = "bad"
            results.append((p_good, label))
        return results


# ---------------------------------------------------------------------------
# ROI extractor
# ---------------------------------------------------------------------------

class _ROIExtractor:
    """AutoMorph M0 fundus-circle detector.

    When a reference image is provided, each source image is illumination-normalized via
    Reinhard LAB transfer before detection (purely for detection — the transfer preserves
    geometry, so the returned circle is valid on the original image). Detection itself uses
    AutoMorph's ``fundus_prep.get_mask``, which fits the fundus boundary and returns the
    circle center + radius in original-image coordinates.
    """

    def __init__(self, reference_rgb: Optional[np.ndarray] = None) -> None:
        self._reference_rgb = reference_rgb

    def get_roi(self, img_rgb: np.ndarray) -> tuple[Optional[tuple[int, int]], Optional[int], str]:
        """Return ((cx, cy), radius, method) or (None, None, 'failed').

        ``img_rgb`` is uint8 RGB. LAB normalization (if enabled) is applied first, then
        AutoMorph's mask fit. The transfer preserves geometry, so the circle is valid on
        the original image.
        """
        detect_rgb = self._normalize(img_rgb)
        try:
            _mask, _bbox, center, radius = prep.get_mask(detect_rgb)
            # get_mask returns center as [row, col] == [y, x].
            cy, cx = center
            return (int(cx), int(cy)), int(radius), "automorph"
        except Exception as exc:
            logger.debug("AutoMorph ROI fit failed (%s).", exc)
            return None, None, "failed"

    def _normalize(self, img_rgb: np.ndarray) -> np.ndarray:
        """Apply LAB color transfer to the reference if one is set, else pass through."""
        if self._reference_rgb is None:
            return img_rgb
        try:
            return lab_color_transfer(img_rgb, self._reference_rgb)
        except Exception as exc:  # never let normalization break detection
            logger.debug("LAB normalization failed (%s); using original image.", exc)
            return img_rgb


# ---------------------------------------------------------------------------
# PyTorch dataset for IQA batching
# ---------------------------------------------------------------------------

def _crop_fundus(img_rgb: np.ndarray) -> np.ndarray:
    """AutoMorph M0 crop: mask the fundus, remove the border, square-pad with black.

    Mirrors ``EyeQ_process_main`` so the quality ensemble sees the same input it was
    trained/evaluated on. No LAB normalization here — that's ROI-only.
    """
    r_img, _borders, _mask, _label, _r, _cw, _ch = prep.process_without_gb(
        img_rgb, img_rgb, [], [], []
    )
    return r_img


def _qa_preprocess(img_rgb: np.ndarray) -> torch.Tensor:
    """AutoMorph M1 preprocessing: resize to 512, normalize by the mean/std of the
    non-zero (fundus) pixels, return a CHW float tensor.
    """
    pil = PILImage.fromarray(img_rgb).resize((_IQA_IMAGE_SIZE, _IQA_IMAGE_SIZE))
    arr = np.array(pil).astype(np.float32)
    fg = arr[arr > 0]
    mean = float(fg.mean()) if fg.size else 0.0
    std = float(fg.std()) if fg.size else 1.0
    if std == 0.0:
        std = 1.0
    arr = (arr - mean) / std
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=2)
    arr = arr.transpose(2, 0, 1)
    return torch.from_numpy(arr).float()


class _ImageDataset(TorchDataset):
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int):
        row = self.rows[idx]
        try:
            path = resolve_local_path(row["file_path"])
            img_rgb = prep.imread(str(path))  # uint8 RGB
            cropped = _crop_fundus(img_rgb)
            tensor = _qa_preprocess(cropped)
            return tensor, str(row["image_id"]), str(row["file_path"]), ""
        except Exception as exc:
            dummy = torch.zeros(3, _IQA_IMAGE_SIZE, _IQA_IMAGE_SIZE)
            return dummy, str(row["image_id"]), str(row["file_path"]), str(exc)


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

async def _fetch_images(dataset_names: Optional[list[str]]) -> list[dict]:
    async with get_connection() as conn:
        async with conn.cursor(row_factory=dict_row) as cur:
            if dataset_names:
                await cur.execute(
                    """
                    SELECT i.image_id, i.file_path, i.storage_provider
                    FROM images i
                    JOIN datasets d ON i.dataset_id = d.dataset_id
                    WHERE i.storage_provider = 'local'
                      AND i.file_path IS NOT NULL
                      AND d.dataset_name = ANY(%s)
                    ORDER BY i.image_id
                    """,
                    (dataset_names,),
                )
            else:
                await cur.execute(
                    """
                    SELECT image_id, file_path, storage_provider
                    FROM images
                    WHERE storage_provider = 'local' AND file_path IS NOT NULL
                    ORDER BY image_id
                    """
                )
            return await cur.fetchall()


# ---------------------------------------------------------------------------
# Build annotation models
# ---------------------------------------------------------------------------

def _make_quality_ann(
    image_id: uuid.UUID,
    p_good: float,
    label: str,
    provenance_chain_id: Optional[uuid.UUID] = None,
) -> QualityAnnotation:
    # UUID stable per image/quality_type (no score in UUID → upsert updates on re-run)
    quality_id = generate_quality_uuid(image_id=image_id, quality_type="pseudo_quality")
    return QualityAnnotation(
        quality_id=quality_id,
        image_id=image_id,
        quality_type="overall",
        quality_score=round(p_good, 6),
        quality_label=label,
        scale_description=_SCALE_DESCRIPTION,
        provenance_chain_id=provenance_chain_id,
    )


def _make_roi_ann(
    image_id: uuid.UUID,
    cx: int, cy: int, radius: int, method: str,
    provenance_chain_id: Optional[uuid.UUID] = None,
) -> LocalizationAnnotation:
    coordinates = {
        "center_x": float(cx),
        "center_y": float(cy),
        "radius": float(radius),
        "method": method,
    }
    # UUID stable per image/type/target — upsert updates coordinates on re-run
    loc_id = generate_localization_uuid(
        image_id=image_id,
        localization_type="center_point",
        target_structure="fundus_roi",
    )
    return LocalizationAnnotation(
        localization_id=loc_id,
        image_id=image_id,
        localization_type="center_point",
        target_structure="fundus_roi",
        coordinates=coordinates,
        annotation_method="pseudo",
        provenance_chain_id=provenance_chain_id,
    )


async def _ensure_pseudo_chain(unified_annotation_type: str, transformation_type: str, notes: str) -> uuid.UUID:
    """Create (idempotently) a ``pseudo_generated`` provenance chain for a pseudo-annotation
    kind and log the operation-level transformation that produced it.

    The chain carries no ``root_source_raw_data_id`` (pseudo annotations have no raw source),
    so its UUID is stable across runs. The transformation is logged once per kind
    (operation-level), not per image — per-image specifics live in the annotation rows.
    """
    chain_id = await create_provenance_chain(
        unified_annotation_type=unified_annotation_type,
        source_type="pseudo_generated",
    )
    await log_and_link_transformation(
        chain_id=chain_id,
        transformation_type=transformation_type,
        parameters={"annotation_method": "pseudo"},
        operator="run_roi_iqa",
        notes=notes,
    )
    return chain_id


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

_DEFAULT_RESULTS_FILE = Path("results.json")


def _default_save_path() -> Path:
    return _DEFAULT_RESULTS_FILE


def _save_checkpoint(
    path: Path,
    quality_anns: list[QualityAnnotation],
    loc_anns: list[LocalizationAnnotation],
) -> None:
    existing_quality: list = []
    existing_loc: list = []
    if path.exists():
        try:
            existing = json.loads(path.read_text())
            existing_quality = existing.get("quality_annotations", [])
            existing_loc = existing.get("localization_annotations", [])
        except Exception as exc:
            logger.warning("Could not read existing checkpoint %s for merge: %s", path, exc)

    all_quality = existing_quality + [a.model_dump(mode="json") for a in quality_anns]
    all_loc = existing_loc + [a.model_dump(mode="json") for a in loc_anns]

    data = {
        "quality_annotations": all_quality,
        "localization_annotations": all_loc,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, default=str))
    logger.info(
        "Checkpoint saved: +%d quality +%d ROI → %s (total in file: %d quality, %d ROI)",
        len(quality_anns), len(loc_anns), path, len(all_quality), len(all_loc),
    )


def _load_checkpoint(path: Path) -> tuple[list[QualityAnnotation], list[LocalizationAnnotation]]:
    data = json.loads(path.read_text())
    quality_anns = [QualityAnnotation(**a) for a in data.get("quality_annotations", [])]
    loc_anns = [LocalizationAnnotation(**a) for a in data.get("localization_annotations", [])]
    logger.info(
        "Loaded checkpoint: %d quality + %d ROI annotations from %s",
        len(quality_anns), len(loc_anns), path,
    )
    return quality_anns, loc_anns


def _existing_ids_from_file(path: Path) -> tuple[set[str], set[str]]:
    """Return (iqa_image_ids, roi_image_ids) already stored in the results file."""
    if not path.exists():
        return set(), set()
    try:
        data = json.loads(path.read_text())
        iqa_ids = {str(a["image_id"]) for a in data.get("quality_annotations", [])}
        roi_ids = {str(a["image_id"]) for a in data.get("localization_annotations", [])}
        return iqa_ids, roi_ids
    except Exception as exc:
        logger.warning("Could not read existing IDs from %s: %s", path, exc)
        return set(), set()


async def _existing_iqa_ids_from_db() -> set[str]:
    """Return image_id strings that already have QuickQual IQA annotations in DB."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT image_id::text FROM quality_annotations WHERE scale_description = %s",
                (_SCALE_DESCRIPTION,),
            )
            return {r[0] for r in await cur.fetchall()}


async def _existing_roi_ids_from_db() -> set[str]:
    """Return image_id strings that already have fundus ROI annotations in DB."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute(
                "SELECT image_id::text FROM localization_annotations WHERE target_structure = 'fundus_roi'"
            )
            return {r[0] for r in await cur.fetchall()}


# ---------------------------------------------------------------------------
# Main processing
# ---------------------------------------------------------------------------

async def run(
    dataset_names: Optional[list[str]] = None,
    batch_size: int = 32,
    run_iqa: bool = True,
    run_roi: bool = True,
    num_workers: int = 4,
    save_file: Optional[Path] = None,
    from_file: Optional[Path] = None,
    force: bool = False,
    roi_reference: Optional[Path] = _DEFAULT_ROI_REFERENCE,
) -> None:
    """Run IQA + ROI over the selected images and upsert results.

    Pool lifecycle: this function never opens or closes the connection pool — it only
    uses it (lazily via ``get_connection``). The caller owns the pool: the standalone
    CLI closes it in ``main``; ``setup_full_database`` closes it in its own ``finally``.
    """
    # --- Retry path: load checkpoint and go straight to upsert ---
    if from_file is not None:
        quality_anns, loc_anns = _load_checkpoint(from_file)
        if quality_anns:
            logger.info("Upserting %d quality annotations from checkpoint…", len(quality_anns))
            await bulk_upsert_quality_annotations(quality_anns)
            logger.info("IQA upsert done.")
        if loc_anns:
            logger.info("Upserting %d localization annotations from checkpoint…", len(loc_anns))
            await bulk_upsert_localization_annotations(loc_anns)
            logger.info("ROI upsert done.")
        return

    # --- Inference path ---
    label = ", ".join(dataset_names) if dataset_names else "all"
    logger.info("Fetching images from DB (datasets=%s)…", label)
    rows = await _fetch_images(dataset_names)
    logger.info("Found %d images in DB", len(rows))

    if not rows:
        logger.warning("No images found.")
        return

    # --- Skip already-processed images (unless --force) ---
    results_path = save_file or _default_save_path()
    if not force:
        file_iqa_ids, file_roi_ids = _existing_ids_from_file(results_path)
        db_iqa_ids: set[str] = set()
        db_roi_ids: set[str] = set()
        if run_iqa:
            db_iqa_ids = await _existing_iqa_ids_from_db()
        if run_roi:
            db_roi_ids = await _existing_roi_ids_from_db()

        done_iqa = file_iqa_ids | db_iqa_ids
        done_roi = file_roi_ids | db_roi_ids

        iqa_rows = [r for r in rows if str(r["image_id"]) not in done_iqa] if run_iqa else []
        roi_rows = [r for r in rows if str(r["image_id"]) not in done_roi] if run_roi else []

        logger.info(
            "Skipping already-done: %d/%d IQA, %d/%d ROI",
            len(rows) - len(iqa_rows), len(rows),
            len(rows) - len(roi_rows), len(rows),
        )
    else:
        iqa_rows = list(rows) if run_iqa else []
        roi_rows = list(rows) if run_roi else []

    if not iqa_rows and not roi_rows:
        logger.info("All images already have results. Nothing to do. Use --force to re-run.")
        return

    iqa_model = _IQAModel() if (run_iqa and iqa_rows) else None
    roi_extractor = (
        _ROIExtractor(reference_rgb=_load_roi_reference(roi_reference))
        if (run_roi and roi_rows)
        else None
    )

    quality_anns: list[QualityAnnotation] = []
    loc_anns: list[LocalizationAnnotation] = []

    # Pseudo-annotation provenance chains (created lazily, once per kind, when we have
    # work to do). Keeps the IQA/ROI pseudo-generation step in the audit trail.
    iqa_chain_id: Optional[uuid.UUID] = None
    roi_chain_id: Optional[uuid.UUID] = None

    # --- IQA: batched via DataLoader ---
    if run_iqa and iqa_model is not None and iqa_rows:
        iqa_chain_id = await _ensure_pseudo_chain(
            unified_annotation_type="quality",
            transformation_type="iqa_automorph_eyepacs",
            notes="AutoMorph M1 EyePACS quality ensemble (8x EfficientNet-b4) → p_good quality.",
        )
        logger.info("Running IQA scoring on %d images…", len(iqa_rows))
        ds = _ImageDataset(iqa_rows)
        loader = DataLoader(ds, batch_size=batch_size, num_workers=num_workers, pin_memory=True)

        for tensors, image_ids, file_paths, errors in tqdm(loader, desc="IQA"):
            valid_idx = [i for i, e in enumerate(errors) if not e]
            if not valid_idx:
                continue
            valid_tensors = tensors[valid_idx]
            try:
                preds = iqa_model.predict_batch(valid_tensors)
            except Exception as exc:
                logger.warning("IQA batch failed: %s", exc)
                continue
            for local_i, pred in zip(valid_idx, preds):
                img_id = uuid.UUID(image_ids[local_i])
                p_good, label = pred
                quality_anns.append(_make_quality_ann(img_id, p_good, label, iqa_chain_id))

        logger.info("IQA scoring complete: %d annotations collected.", len(quality_anns))

    # --- ROI: one image at a time (CPU-bound, no batching benefit) ---
    if run_roi and roi_extractor is not None and roi_rows:
        roi_chain_id = await _ensure_pseudo_chain(
            unified_annotation_type="localization",
            transformation_type="roi_automorph_fit",
            notes="AutoMorph M0 fundus mask fit (LAB-normalized) → fundus_roi center_point.",
        )
        logger.info("Running ROI detection on %d images…", len(roi_rows))
        failed = 0
        for row in tqdm(roi_rows, desc="ROI"):
            img_id = uuid.UUID(str(row["image_id"]))
            try:
                path = resolve_local_path(row["file_path"])
                img_rgb = prep.imread(str(path))  # uint8 RGB
                center, radius, method = roi_extractor.get_roi(img_rgb)
                if center is None or radius is None:
                    failed += 1
                    logger.debug("ROI detection failed for %s", row["file_path"])
                    continue
                cx, cy = center
                loc_anns.append(_make_roi_ann(img_id, cx, cy, radius, method, roi_chain_id))
            except Exception as exc:
                failed += 1
                logger.warning("ROI error for %s: %s", row["file_path"], exc)

        logger.info(
            "ROI detection complete: %d succeeded, %d failed.",
            len(loc_anns), failed,
        )

    # --- Save checkpoint before any upsert (merges with existing results file) ---
    _save_checkpoint(results_path, quality_anns, loc_anns)

    # --- Upsert ---
    if quality_anns:
        logger.info("Upserting %d quality annotations…", len(quality_anns))
        await bulk_upsert_quality_annotations(quality_anns)
        logger.info("IQA upsert done.")

    if loc_anns:
        logger.info("Upserting %d localization annotations…", len(loc_anns))
        await bulk_upsert_localization_annotations(loc_anns)
        logger.info("ROI upsert done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run IQA + ROI detection on all DB images")
    parser.add_argument(
        "--datasets", nargs="+", default=None, metavar="NAME",
        help="Only process these datasets (space-separated names, e.g. --datasets MESSIDOR DRIVE)"
    )
    parser.add_argument(
        "--dataset", default=None, metavar="NAME",
        help="Only process this single dataset (alias for --datasets with one name)"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="IQA batch size (default 32)")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers (default 4)")
    parser.add_argument("--no-iqa", action="store_true", help="Skip IQA scoring")
    parser.add_argument("--no-roi", action="store_true", help="Skip ROI detection")
    parser.add_argument(
        "--save-file", type=Path, default=None, metavar="PATH",
        help="Path to save the checkpoint JSON (default: roi_iqa_results_<timestamp>.json)"
    )
    parser.add_argument(
        "--from-file", type=Path, default=None, metavar="PATH",
        help="Skip inference; load annotations from this checkpoint file and upsert to DB"
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run inference on all images even if results exist in DB or results.json"
    )
    parser.add_argument(
        "--roi-reference", type=Path, default=_DEFAULT_ROI_REFERENCE, metavar="PATH",
        help="Reference image for LAB illumination normalization before ROI detection "
             f"(default: {_DEFAULT_ROI_REFERENCE}). Helps the circle fit the fundus instead "
             "of the optic disc on dark images."
    )
    parser.add_argument(
        "--no-roi-reference", action="store_true",
        help="Disable LAB illumination normalization before ROI detection."
    )
    args = parser.parse_args()

    if args.from_file and not args.from_file.exists():
        parser.error(f"--from-file path does not exist: {args.from_file}")

    # Merge --dataset and --datasets into one list
    dataset_names = args.datasets or []
    if args.dataset:
        dataset_names = list({*dataset_names, args.dataset})
    dataset_names = dataset_names or None

    async def _run_and_close() -> None:
        # This CLI owns the pool lifecycle: run(), then always close. When run() is
        # called as a library function (e.g. from setup_full_database), the caller
        # owns and closes the pool instead — run() never closes it itself.
        try:
            await run(
                dataset_names=dataset_names,
                batch_size=args.batch_size,
                run_iqa=not args.no_iqa,
                run_roi=not args.no_roi,
                num_workers=args.num_workers,
                save_file=args.save_file,
                from_file=args.from_file,
                force=args.force,
                roi_reference=None if args.no_roi_reference else args.roi_reference,
            )
        finally:
            await close_pool()

    asyncio.run(_run_and_close())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
