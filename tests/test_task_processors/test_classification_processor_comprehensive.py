"""
Comprehensive tests for classification_processor.

Tests cover:
- Binary classification (returns single-element list with class_index/class_label)
- Multi-class classification (returns single-element list with class_index/class_label)
- Multi-label classification (returns one element per sub-key)
- Scalar column population (class_index, class_label, sub_key)
- Backward-compatible format helpers
- Deterministic UUIDs
- Validation
"""

import uuid
import pytest

from chaksudb.db.queries.annotation_types import upsert_classification_annotation
from chaksudb.ingest.framework.task_processors.classification_processor import (
    format_binary_classification,
    format_multi_class_classification,
    format_multi_label_classification,
    process_classification,
    prepare_classification_for_upsert,
)


pytestmark = pytest.mark.asyncio


class TestClassificationProcessorWithRealData:
    """Test classification processor with real data from datasets."""

    @pytest.fixture
    async def test_image_id(self, test_image_in_db):
        """Create a test image ID with database record."""
        return test_image_in_db

    @pytest.fixture
    async def test_dataset_id(self, test_dataset_in_db):
        """Create a test dataset ID with database record."""
        return test_dataset_in_db

    # ---- Binary classification ----

    async def test_binary_classification_bool(self, test_image_id):
        """Test binary classification with boolean values."""
        results = await process_classification(
            class_value=True,
            task_type="binary",
            class_name="referable_dr",
            image_id=test_image_id,
        )

        assert len(results) == 1
        c = results[0]
        assert c.task_type == "binary"
        assert c.class_name == "referable_dr"
        assert c.class_index == 1
        assert c.class_label == "positive"
        assert c.sub_key is None
        assert c.class_value == {"referable_dr": True}

    async def test_binary_classification_int(self, test_image_id):
        """Test binary classification with integer values (0/1)."""
        results_1 = await process_classification(
            class_value=1,
            task_type="binary",
            class_name="has_lesion",
            image_id=test_image_id,
        )
        assert results_1[0].class_index == 1
        assert results_1[0].class_label == "positive"
        assert results_1[0].class_value == {"has_lesion": True}

        results_0 = await process_classification(
            class_value=0,
            task_type="binary",
            class_name="has_lesion",
            image_id=test_image_id,
        )
        assert results_0[0].class_index == 0
        assert results_0[0].class_label == "negative"
        assert results_0[0].class_value == {"has_lesion": False}

    async def test_binary_classification_string(self, test_image_id):
        """Test binary classification with string values."""
        for val, expected_idx in [
            ("yes", 1), ("no", 0), ("positive", 1), ("negative", 0),
            ("true", 1), ("false", 0), ("present", 1), ("absent", 0),
        ]:
            results = await process_classification(
                class_value=val,
                task_type="binary",
                class_name="gradable",
                image_id=test_image_id,
            )
            assert results[0].class_index == expected_idx

    # ---- Multi-class classification ----

    async def test_multi_class_classification_int(self, test_image_id):
        """Test multi-class classification with integer indices."""
        class_labels = {
            0: "non_glaucoma",
            1: "suspicious_glaucoma",
            2: "definite_glaucoma",
        }

        results = await process_classification(
            class_value=1,
            task_type="multi_class",
            class_name="glaucoma_severity",
            image_id=test_image_id,
            class_labels=class_labels,
        )

        assert len(results) == 1
        c = results[0]
        assert c.task_type == "multi_class"
        assert c.class_index == 1
        assert c.class_label == "suspicious_glaucoma"
        assert c.sub_key is None
        assert c.class_value == {
            "glaucoma_severity": {
                "class_index": 1,
                "class_label": "suspicious_glaucoma",
            }
        }

    async def test_multi_class_classification_string_requires_labels(self, test_image_id):
        """multi_class now REQUIRES a class_labels map so class_index is never NULL.

        (multi_class is for mutually-exclusive *categories* like disease_category — ordinal
        severity such as DR grade belongs in the grading table, not classification.)
        """
        with pytest.raises(ValueError, match="requires a class_labels map"):
            await process_classification(
                class_value="cataract",
                task_type="multi_class",
                class_name="disease_category",
                image_id=test_image_id,
            )

    async def test_multi_class_reverse_lookup(self, test_image_id):
        """Test multi-class string label with reverse index lookup."""
        class_labels = {0: "normal", 1: "cataract", 2: "glaucoma", 3: "other"}

        results = await process_classification(
            class_value="glaucoma",
            task_type="multi_class",
            class_name="disease_category",
            image_id=test_image_id,
            class_labels=class_labels,
        )

        c = results[0]
        assert c.class_index == 2
        assert c.class_label == "glaucoma"

    # ---- Multi-label classification (exploded) ----

    async def test_multi_label_classification_odir5k_style(self, test_image_id):
        """Test multi-label classification explodes into one row per sub-key."""
        odir_labels = {
            "normal": 0,
            "diabetes": 1,
            "glaucoma": 0,
            "cataract": 1,
            "amd": 0,
            "hypertension": 0,
            "myopia": 1,
            "other": 0,
        }

        results = await process_classification(
            class_value=odir_labels,
            task_type="multi_label",
            class_name="ocular_diseases",
            image_id=test_image_id,
        )

        assert len(results) == 8  # One row per sub-key

        by_key = {r.sub_key: r for r in results}
        assert by_key["diabetes"].class_index == 1
        assert by_key["diabetes"].class_label == "diabetes"
        assert by_key["diabetes"].sub_key == "diabetes"
        assert by_key["diabetes"].class_name == "ocular_diseases"

        assert by_key["glaucoma"].class_index == 0
        assert by_key["normal"].class_index == 0
        assert by_key["cataract"].class_index == 1
        assert by_key["myopia"].class_index == 1
        assert by_key["hypertension"].class_index == 0

    async def test_multi_label_classification_with_probabilities(self, test_image_id):
        """Test multi-label classification with probability scores."""
        lesion_probs = {
            "microaneurysms": 0.95,
            "hemorrhages": 0.82,
            "exudates": 0.13,
            "cotton_wool_spots": 0.05,
        }

        results = await process_classification(
            class_value=lesion_probs,
            task_type="multi_label",
            class_name="lesion_detection",
            image_id=test_image_id,
        )

        assert len(results) == 4
        by_key = {r.sub_key: r for r in results}
        # Float values get rounded to class_index
        assert by_key["microaneurysms"].class_index == 1  # round(0.95)
        assert by_key["exudates"].class_index == 0  # round(0.13)
        # Original float preserved in class_value
        assert by_key["microaneurysms"].class_value == {"microaneurysms": 0.95}

    async def test_multi_label_unique_uuids(self, test_image_id):
        """Test that each sub-key gets a unique UUID."""
        results = await process_classification(
            class_value={"a": True, "b": False, "c": True},
            task_type="multi_label",
            class_name="test",
            image_id=test_image_id,
        )

        uuids = [r.classification_id for r in results]
        assert len(set(uuids)) == 3  # All unique

    # ---- Scalar fields ----

    async def test_classification_with_confidence_score(self, test_image_id):
        """Test classification with confidence scores."""
        results = await process_classification(
            class_value=True,
            task_type="binary",
            class_name="disease_present",
            image_id=test_image_id,
            confidence_score=0.87,
        )

        assert results[0].confidence_score == 0.87

    async def test_classification_with_raw_data_id(self, test_image_id):
        """Test classification with raw_data_id."""
        raw_data_id = uuid.uuid4()
        results = await process_classification(
            class_value=1,
            task_type="binary",
            class_name="test_class",
            image_id=test_image_id,
            raw_data_id=raw_data_id,
        )

        assert results[0].raw_data_id == raw_data_id

    # ---- Validation ----

    async def test_classification_invalid_task_type(self, test_image_id):
        """Test that invalid task types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid task_type"):
            await process_classification(
                class_value=1,
                task_type="invalid_type",
                class_name="test",
                image_id=test_image_id,
            )

    async def test_meta_task_rejected(self, test_image_id):
        """Test that meta_task is no longer a valid task type."""
        with pytest.raises(ValueError, match="Invalid task_type"):
            await process_classification(
                class_value={"key": "value"},
                task_type="meta_task",
                class_name="test",
                image_id=test_image_id,
            )

    async def test_classification_invalid_annotation_method(self, test_image_id):
        """Test that invalid annotation methods raise ValueError."""
        with pytest.raises(ValueError, match="Invalid annotation_method"):
            await process_classification(
                class_value=1,
                task_type="binary",
                class_name="test",
                image_id=test_image_id,
                annotation_method="invalid_method",
            )

    async def test_classification_invalid_confidence_score(self, test_image_id):
        """Test that invalid confidence scores raise ValueError."""
        with pytest.raises(ValueError, match="confidence_score must be in"):
            await process_classification(
                class_value=1,
                task_type="binary",
                class_name="test",
                image_id=test_image_id,
                confidence_score=1.5,
            )

    # ---- Backward-compatible format helpers ----

    async def test_format_binary_classification_various_inputs(self):
        """Test format_binary_classification helper with various inputs."""
        assert format_binary_classification(True, "test") == {"test": True}
        assert format_binary_classification(False, "test") == {"test": False}
        assert format_binary_classification(1, "test") == {"test": True}
        assert format_binary_classification(0, "test") == {"test": False}
        assert format_binary_classification("yes", "test") == {"test": True}
        assert format_binary_classification("no", "test") == {"test": False}
        assert format_binary_classification("positive", "test") == {"test": True}
        assert format_binary_classification("negative", "test") == {"test": False}

    async def test_format_binary_classification_invalid(self):
        """Test that invalid binary values raise ValueError."""
        with pytest.raises(ValueError):
            format_binary_classification(2, "test")
        with pytest.raises(ValueError):
            format_binary_classification("maybe", "test")

    async def test_format_multi_class_classification(self):
        """Test format_multi_class_classification helper."""
        labels = {0: "none", 1: "mild", 2: "severe"}

        result = format_multi_class_classification(1, "severity", labels)
        assert result == {"severity": {"class_index": 1, "class_label": "mild"}}

        result = format_multi_class_classification(2, "severity")
        assert result == {"severity": {"class_index": 2}}

        result = format_multi_class_classification("severe", "severity")
        assert result == {"severity": {"class_label": "severe"}}

    async def test_format_multi_label_classification(self):
        """Test format_multi_label_classification helper."""
        result = format_multi_label_classification({"a": True, "b": False})
        assert result == {"a": True, "b": False}

        result = format_multi_label_classification({"a": 1, "b": 0})
        assert result == {"a": True, "b": False}

        result = format_multi_label_classification({"a": 0.8, "b": 0.2})
        assert result == {"a": 0.8, "b": 0.2}

    async def test_format_multi_label_invalid(self):
        """Test that invalid multi-label values raise ValueError."""
        with pytest.raises(ValueError):
            format_multi_label_classification({"a": 2})
        with pytest.raises(ValueError):
            format_multi_label_classification({"a": 1.5})

    # ---- Alias ----

    async def test_prepare_classification_for_upsert_alias(self, test_image_id):
        """Test that prepare_classification_for_upsert is an alias returning a list."""
        results = await prepare_classification_for_upsert(
            class_value=True,
            task_type="binary",
            class_name="test",
            image_id=test_image_id,
        )

        assert isinstance(results, list)
        assert len(results) == 1
        assert results[0].class_value == {"test": True}

    # ---- Deterministic UUIDs ----

    async def test_classification_deterministic_uuids(self, test_image_id):
        """Test that processing the same classification twice produces the same UUID."""
        results_1 = await process_classification(
            class_value=True,
            task_type="binary",
            class_name="test_deterministic",
            image_id=test_image_id,
        )

        results_2 = await process_classification(
            class_value=True,
            task_type="binary",
            class_name="test_deterministic",
            image_id=test_image_id,
        )

        assert results_1[0].classification_id == results_2[0].classification_id

    async def test_multi_label_deterministic_uuids(self, test_image_id):
        """Test that multi-label sub-keys produce deterministic UUIDs."""
        results_1 = await process_classification(
            class_value={"a": True, "b": False},
            task_type="multi_label",
            class_name="test",
            image_id=test_image_id,
        )

        results_2 = await process_classification(
            class_value={"a": True, "b": False},
            task_type="multi_label",
            class_name="test",
            image_id=test_image_id,
        )

        ids_1 = {r.sub_key: r.classification_id for r in results_1}
        ids_2 = {r.sub_key: r.classification_id for r in results_2}
        assert ids_1 == ids_2

    # ---- Integration ----

    async def test_process_and_upsert_integration(self, test_image_id):
        """Test full integration: process and upsert to database."""
        results = await process_classification(
            class_value={"diabetes": 1, "glaucoma": 0, "cataract": 1},
            task_type="multi_label",
            class_name="diseases",
            image_id=test_image_id,
        )

        assert len(results) == 3

        # Upsert each to database
        for r in results:
            await upsert_classification_annotation(r)

        # Verify idempotency - upsert again
        for r in results:
            await upsert_classification_annotation(r)

    async def test_odir5k_patient_scenario(self, test_image_id):
        """Test ODIR-5K style patient with multiple disease labels."""
        results = await process_classification(
            class_value={
                "normal": 0,
                "diabetes": 1,
                "glaucoma": 0,
                "cataract": 0,
                "amd": 0,
                "hypertension": 1,
                "myopia": 0,
                "other": 0,
            },
            task_type="multi_label",
            class_name="ocular_diseases",
            image_id=test_image_id,
            annotation_method="manual",
        )

        by_key = {r.sub_key: r for r in results}
        assert by_key["diabetes"].class_index == 1
        assert by_key["hypertension"].class_index == 1
        assert by_key["glaucoma"].class_index == 0

    async def test_lag_glaucoma_classification(self, test_image_id):
        """Test LAG dataset glaucoma classification (binary)."""
        results = await process_classification(
            class_value=1,
            task_type="binary",
            class_name="glaucoma",
            image_id=test_image_id,
        )

        c = results[0]
        assert c.class_index == 1
        assert c.class_label == "positive"
        assert c.class_value == {"glaucoma": True}

    # ---- Class name normalization ----

    async def test_class_name_normalization(self, test_image_id):
        """Test that class_name variants are normalized."""
        results = await process_classification(
            class_value=True,
            task_type="binary",
            class_name="diabetic_retinopathy",
            image_id=test_image_id,
        )
        assert results[0].class_name == "DR"

        results = await process_classification(
            class_value=True,
            task_type="binary",
            class_name="armd",
            image_id=test_image_id,
        )
        assert results[0].class_name == "AMD"
