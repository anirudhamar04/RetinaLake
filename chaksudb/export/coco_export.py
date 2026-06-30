"""
COCO JSON Export: Write localization annotations in COCO detection format.

Produces a sidecar JSON file compatible with detectron2, mmdet, and
pycocotools alongside the Parquet export.
"""

import json
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def export_coco_json(
    rows: list[dict[str, Any]],
    output_path: Path,
    category_map: Optional[dict[str, int]] = None,
) -> Path:
    """Write a COCO-format JSON from exported rows.

    Args:
        rows: List of row dicts that contain ``image_id``, ``file_path``,
            ``resolution_width``, ``resolution_height``, and
            ``localization_annotations`` (JSONB list).
        output_path: Path for the output JSON file.
        category_map: Mapping of target_structure → category_id.
            If ``None``, categories are auto-assigned from data.

    Returns:
        The path to the written JSON file.
    """
    images: list[dict] = []
    annotations: list[dict] = []
    seen_categories: dict[str, int] = dict(category_map) if category_map else {}
    next_cat_id = max(seen_categories.values(), default=0) + 1
    ann_id = 1

    for img_idx, row in enumerate(rows):
        image_id = img_idx + 1
        width = int(row.get("resolution_width") or 0)
        height = int(row.get("resolution_height") or 0)
        file_name = row.get("file_path", "")

        images.append({
            "id": image_id,
            "file_name": file_name,
            "width": width,
            "height": height,
        })

        loc_anns = row.get("localization_annotations")
        if not loc_anns:
            continue
        if isinstance(loc_anns, str):
            loc_anns = json.loads(loc_anns)

        for ann in loc_anns:
            if not isinstance(ann, dict):
                continue
            coords = ann.get("coordinates", {})
            if not isinstance(coords, dict):
                continue

            target = ann.get("target_structure", "unknown")
            if target not in seen_categories:
                seen_categories[target] = next_cat_id
                next_cat_id += 1
            cat_id = ann.get("category_id", seen_categories[target])

            # Convert to COCO [x, y, w, h]
            loc_type = ann.get("localization_type", "")
            if loc_type == "bounding_box":
                x = float(coords.get("x", coords.get("x_min", 0)))
                y = float(coords.get("y", coords.get("y_min", 0)))
                w = float(coords.get("w", coords.get("width", 0)))
                h = float(coords.get("h", coords.get("height", 0)))
                bbox = [x, y, w, h]
                area = w * h
            elif loc_type in ("keypoint", "center_point"):
                # Represent as a 1x1 bbox for compatibility
                cx = float(coords.get("x", 0))
                cy = float(coords.get("y", 0))
                bbox = [cx, cy, 1.0, 1.0]
                area = 1.0
            else:
                continue

            annotations.append({
                "id": ann_id,
                "image_id": image_id,
                "category_id": cat_id,
                "bbox": bbox,
                "area": area,
                "iscrowd": 0,
            })
            ann_id += 1

    categories = [
        {"id": cid, "name": name}
        for name, cid in sorted(seen_categories.items(), key=lambda x: x[1])
    ]

    coco = {
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(coco, f, indent=2)

    logger.info(
        "COCO JSON written: %d images, %d annotations, %d categories -> %s",
        len(images), len(annotations), len(categories), output_path,
    )
    return output_path
