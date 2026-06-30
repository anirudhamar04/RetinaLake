"""Tests for COCO JSON export."""

import json
import pytest
from pathlib import Path

from chaksudb.export.coco_export import export_coco_json


@pytest.fixture
def sample_rows():
    return [
        {
            "image_id": "img-001",
            "file_path": "/data/img1.jpg",
            "resolution_width": 800,
            "resolution_height": 600,
            "localization_annotations": [
                {
                    "localization_type": "bounding_box",
                    "target_structure": "lesions",
                    "coordinates": {"x": 10, "y": 20, "w": 50, "h": 40},
                    "lesion_subtype": "microaneurysm",
                },
                {
                    "localization_type": "bounding_box",
                    "target_structure": "optic_disc",
                    "coordinates": {"x": 100, "y": 200, "w": 80, "h": 80},
                    "lesion_subtype": None,
                },
            ],
        },
        {
            "image_id": "img-002",
            "file_path": "/data/img2.jpg",
            "resolution_width": 1024,
            "resolution_height": 768,
            "localization_annotations": [
                {
                    "localization_type": "keypoint",
                    "target_structure": "fovea",
                    "coordinates": {"x": 512, "y": 384},
                    "lesion_subtype": None,
                },
            ],
        },
        {
            "image_id": "img-003",
            "file_path": "/data/img3.jpg",
            "resolution_width": 640,
            "resolution_height": 480,
            "localization_annotations": None,
        },
    ]


class TestExportCocoJson:
    def test_basic_export(self, sample_rows, tmp_path):
        out = tmp_path / "coco.json"
        result = export_coco_json(sample_rows, out)
        assert result == out
        assert out.exists()

        with open(out) as f:
            coco = json.load(f)

        assert len(coco["images"]) == 3
        # 2 bboxes + 1 keypoint = 3 annotations
        assert len(coco["annotations"]) == 3
        assert len(coco["categories"]) == 3  # lesions, optic_disc, fovea

    def test_category_map(self, sample_rows, tmp_path):
        out = tmp_path / "coco.json"
        cat_map = {"lesions": 1, "optic_disc": 2, "fovea": 3}
        export_coco_json(sample_rows, out, category_map=cat_map)

        with open(out) as f:
            coco = json.load(f)

        cats_by_id = {c["id"]: c["name"] for c in coco["categories"]}
        assert cats_by_id[1] == "lesions"
        assert cats_by_id[2] == "optic_disc"

    def test_bbox_format(self, sample_rows, tmp_path):
        out = tmp_path / "coco.json"
        export_coco_json(sample_rows, out)

        with open(out) as f:
            coco = json.load(f)

        bbox_ann = coco["annotations"][0]
        assert "bbox" in bbox_ann
        assert len(bbox_ann["bbox"]) == 4
        assert bbox_ann["bbox"] == [10, 20, 50, 40]
        assert bbox_ann["area"] == 2000  # 50 * 40

    def test_no_annotations(self, tmp_path):
        rows = [
            {
                "image_id": "img-001",
                "file_path": "/data/img1.jpg",
                "resolution_width": 100,
                "resolution_height": 100,
                "localization_annotations": None,
            }
        ]
        out = tmp_path / "coco.json"
        export_coco_json(rows, out)

        with open(out) as f:
            coco = json.load(f)

        assert len(coco["images"]) == 1
        assert len(coco["annotations"]) == 0

    def test_string_localization_annotations(self, tmp_path):
        """Test that JSON string localization_annotations are parsed."""
        rows = [
            {
                "image_id": "img-001",
                "file_path": "/data/img1.jpg",
                "resolution_width": 100,
                "resolution_height": 100,
                "localization_annotations": json.dumps([
                    {
                        "localization_type": "bounding_box",
                        "target_structure": "lesions",
                        "coordinates": {"x": 5, "y": 5, "w": 10, "h": 10},
                    }
                ]),
            }
        ]
        out = tmp_path / "coco.json"
        export_coco_json(rows, out)

        with open(out) as f:
            coco = json.load(f)

        assert len(coco["annotations"]) == 1
