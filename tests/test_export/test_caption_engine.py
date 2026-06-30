"""Tests for CaptionEngine."""

import json
import pytest

from chaksudb.export.caption_engine import CaptionEngine


@pytest.fixture
def engine():
    """Create a CaptionEngine with sample definitions."""
    definitions = {
        "no diabetic retinopathy": ["no diabetic retinopathy", "no microaneurysms"],
        "mild diabetic retinopathy": ["only few microaneurysms"],
        "moderate diabetic retinopathy": [
            "many exudates near the macula",
            "hard exudates",
            "cotton wool spots",
        ],
        "microaneurysms": ["small red dots"],
        "haemorrhages": ["dense, dark red, sharply outlined lesion"],
        "glaucoma": [
            "optic nerve abnormalities",
            "abnormal size of the optic cup",
        ],
        "normal": ["healthy", "no findings"],
        "vessel": ["retinal blood vessels"],
        "optic_disc": ["optic disc structure"],
        "lesion": ["retinal lesion"],
    }
    abbreviations = {
        "no diabetic retinopathy": "noDR",
        "mild diabetic retinopathy": "mildDR",
        "glaucoma": "G",
    }
    return CaptionEngine(definitions=definitions, abbreviations=abbreviations)


class TestFromGrading:
    def test_known_grade(self, engine):
        caption = engine.from_grading(
            "DR", 1, original_grade="mild diabetic retinopathy"
        )
        assert caption is not None
        assert "mild diabetic retinopathy" in caption
        assert "microaneurysms" in caption

    def test_unknown_grade_fallback(self, engine):
        caption = engine.from_grading("DR", 99, original_grade="unknown grade")
        assert caption is not None
        assert "unknown grade" in caption

    def test_none_grade_returns_none(self, engine):
        assert engine.from_grading("DR", None) is None


class TestFromGradingData:
    def test_known_grade_label(self, engine):
        grade_data = [
            {"disease_type": "DR", "original_grade": "mild diabetic retinopathy", "grade_label": "mild diabetic retinopathy"}
        ]
        caption = engine.from_grading_data(grade_data)
        assert caption is not None
        assert "mild diabetic retinopathy" in caption
        assert "microaneurysms" in caption

    def test_json_string_input(self, engine):
        grade_data = json.dumps([
            {"disease_type": "DR", "original_grade": "no diabetic retinopathy", "grade_label": None}
        ])
        caption = engine.from_grading_data(grade_data)
        assert caption is not None
        assert "normal fundus" in caption or "no diabetic retinopathy" in caption

    def test_multiple_diseases(self, engine):
        grade_data = [
            {"disease_type": "DR", "original_grade": "mild diabetic retinopathy", "grade_label": "mild diabetic retinopathy"},
            {"disease_type": "Glaucoma", "original_grade": "glaucoma", "grade_label": "glaucoma"},
        ]
        caption = engine.from_grading_data(grade_data)
        assert caption is not None
        assert "mild diabetic retinopathy" in caption
        assert "glaucoma" in caption

    def test_none_returns_none(self, engine):
        assert engine.from_grading_data(None) is None

    def test_empty_list_returns_none(self, engine):
        assert engine.from_grading_data([]) is None


class TestFromKeywords:
    def test_known_keywords(self, engine):
        caption = engine.from_keywords(["microaneurysms", "haemorrhages"])
        assert caption is not None
        assert "small red dots" in caption
        assert "haemorrhages" in caption

    def test_unknown_keyword(self, engine):
        caption = engine.from_keywords(["unknown_thing"])
        assert caption is not None
        assert "unknown_thing" in caption

    def test_empty_list(self, engine):
        assert engine.from_keywords([]) is None


class TestFromClassification:
    def test_known_class(self, engine):
        caption = engine.from_classification({"glaucoma": "positive"})
        assert caption is not None
        assert "optic nerve abnormalities" in caption

    def test_empty_dict(self, engine):
        assert engine.from_classification({}) is None


class TestFromClassificationData:
    def test_dict_input(self, engine):
        class_data = {"glaucoma": 1}
        caption = engine.from_classification_data(class_data)
        assert caption is not None
        assert "glaucoma" in caption
        assert "optic nerve abnormalities" in caption

    def test_json_string_input(self, engine):
        class_data = json.dumps({"glaucoma": 1, "normal": 0})
        caption = engine.from_classification_data(class_data)
        assert caption is not None
        assert "glaucoma" in caption

    def test_none_values_skipped(self, engine):
        class_data = {"glaucoma": None, "normal": 1}
        caption = engine.from_classification_data(class_data)
        assert caption is not None
        assert "glaucoma" not in caption
        assert "normal" in caption

    def test_empty_returns_none(self, engine):
        assert engine.from_classification_data({}) is None

    def test_none_returns_none(self, engine):
        assert engine.from_classification_data(None) is None


class TestFromStructures:
    def test_list_input(self, engine):
        caption = engine.from_structures(["vessel", "optic_disc"])
        assert caption is not None
        assert "retinal blood vessels" in caption
        assert "optic disc structure" in caption

    def test_comma_string_input(self, engine):
        caption = engine.from_structures("vessel,optic_disc")
        assert caption is not None
        assert "vessel" in caption

    def test_unknown_structure(self, engine):
        caption = engine.from_structures(["fovea"])
        assert caption is not None
        assert "fovea" in caption

    def test_enriched_structure(self, engine):
        caption = engine.from_structures(["lesion"])
        assert caption is not None
        assert "retinal lesion" in caption

    def test_empty_returns_none(self, engine):
        assert engine.from_structures([]) is None
        assert engine.from_structures(None) is None
        assert engine.from_structures("") is None


class TestSynthesize:
    def test_grading_mode(self, engine):
        row = {
            "caption_grade_data": [
                {"disease_type": "DR", "original_grade": "mild diabetic retinopathy", "grade_label": "mild diabetic retinopathy"}
            ]
        }
        caption = engine.synthesize(row, mode="grading")
        assert caption is not None
        assert "mild diabetic retinopathy" in caption
        assert "microaneurysms" in caption

    def test_classification_mode(self, engine):
        row = {"caption_class_data": {"glaucoma": 1}}
        caption = engine.synthesize(row, mode="classification")
        assert caption is not None
        assert "glaucoma" in caption

    def test_synthetic_mode_combines_sources(self, engine):
        row = {
            "caption_grade_data": [
                {"disease_type": "DR", "original_grade": "mild diabetic retinopathy", "grade_label": "mild diabetic retinopathy"}
            ],
            "caption_class_data": {"glaucoma": 1},
            "caption_loc_structures": "microaneurysms,haemorrhages",
            "caption_seg_structures": "vessel",
        }
        caption = engine.synthesize(row, mode="synthetic")
        assert caption is not None
        assert "mild diabetic retinopathy" in caption
        assert "glaucoma" in caption
        assert "microaneurysms" in caption
        assert "vessel" in caption

    def test_all_mode_includes_clinical_and_keywords(self, engine):
        row = {
            "caption_clinical_text": "Grade 2 DR fundus image.",
            "caption_keywords": ["microaneurysms"],
            "caption_grade_data": None,
            "caption_class_data": None,
            "caption_loc_structures": None,
            "caption_seg_structures": None,
        }
        caption = engine.synthesize(row, mode="all")
        assert caption is not None
        assert "Grade 2 DR" in caption
        assert "microaneurysms" in caption

    def test_empty_row_returns_none(self, engine):
        assert engine.synthesize({}, mode="all") is None

    def test_keyword_mode(self, engine):
        row = {"caption_keywords": ["microaneurysms", "haemorrhages"]}
        caption = engine.synthesize(row, mode="keyword")
        assert caption is not None
        assert "small red dots" in caption

    def test_clinical_mode(self, engine):
        row = {"caption_clinical_text": "Severe DR with macular edema."}
        caption = engine.synthesize(row, mode="clinical")
        assert caption is not None
        assert "Severe DR with macular edema." in caption


class TestGenerateVariants:
    def test_basic_variants(self, engine):
        base = "A fundus photograph showing mild diabetic retinopathy."
        variants = engine.generate_variants(base, n_variants=3, seed=42)
        assert len(variants) == 3
        for v in variants:
            assert "mild diabetic retinopathy" in v

    def test_empty_base(self, engine):
        assert engine.generate_variants("") == []

    def test_deterministic_with_seed(self, engine):
        base = "A fundus image showing microaneurysms."
        v1 = engine.generate_variants(base, n_variants=2, seed=123)
        v2 = engine.generate_variants(base, n_variants=2, seed=123)
        assert v1 == v2
