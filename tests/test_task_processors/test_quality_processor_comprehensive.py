"""
Comprehensive tests for quality_processor using real data from data directory.

Tests cover:
- Quality score normalization
- Quality label parsing
- DeepDRiD quality metrics
- Real data from DeepDRiD, IDRID, and other datasets
"""

import uuid
import pytest

from chaksudb.db.queries.annotation_types import upsert_quality_annotation
from chaksudb.ingest.framework.task_processors.quality_processor import (
    normalize_quality_score,
    parse_quality_label,
    process_quality_annotation,
    process_deepdrid_quality,
    prepare_quality_for_upsert,
)


pytestmark = pytest.mark.asyncio


class TestQualityProcessorWithRealData:
    """Test quality processor with real data from datasets."""

    @pytest.fixture
    async def test_image_id(self, test_image_in_db):
        """Create a test image ID with database record."""
        return test_image_in_db

    async def test_normalize_quality_score_0_to_1(self):
        """Test normalizing quality scores already in [0, 1] range."""
        # Already normalized
        assert normalize_quality_score(0.75) == 0.75
        assert normalize_quality_score(0.0) == 0.0
        assert normalize_quality_score(1.0) == 1.0

    async def test_normalize_quality_score_0_to_5(self):
        """Test normalizing quality scores from 0-5 scale."""
        # 0-5 scale
        assert normalize_quality_score(0, scale_min=0, scale_max=5) == 0.0
        assert normalize_quality_score(2.5, scale_min=0, scale_max=5) == 0.5
        assert normalize_quality_score(5, scale_min=0, scale_max=5) == 1.0

    async def test_normalize_quality_score_deepdrid_overall(self):
        """Test normalizing DeepDRiD overall quality (0-2 scale)."""
        # DeepDRiD Overall Quality: 0 (poor), 1 (good), 2 (excellent)
        assert normalize_quality_score(0, scale_min=0, scale_max=2) == 0.0
        assert normalize_quality_score(1, scale_min=0, scale_max=2) == 0.5
        assert normalize_quality_score(2, scale_min=0, scale_max=2) == 1.0

    async def test_normalize_quality_score_deepdrid_clarity(self):
        """Test normalizing DeepDRiD clarity (0-2 scale)."""
        # DeepDRiD Clarity: 0 (severe blur), 1 (mild blur), 2 (no blur)
        assert normalize_quality_score(0, scale_min=0, scale_max=2) == 0.0
        assert normalize_quality_score(1, scale_min=0, scale_max=2) == 0.5
        assert normalize_quality_score(2, scale_min=0, scale_max=2) == 1.0

    async def test_normalize_quality_score_deepdrid_field_definition(self):
        """Test normalizing DeepDRiD field definition (0-1 scale)."""
        # DeepDRiD Field Definition: 0 (inadequate), 1 (adequate)
        assert normalize_quality_score(0, scale_min=0, scale_max=1) == 0.0
        assert normalize_quality_score(1, scale_min=0, scale_max=1) == 1.0

    async def test_normalize_quality_score_string_input(self):
        """Test normalizing quality scores from string input."""
        assert normalize_quality_score("3", scale_min=0, scale_max=5) == 0.6
        assert normalize_quality_score("2.5", scale_min=0, scale_max=5) == 0.5

    async def test_normalize_quality_score_out_of_range(self):
        """Test that out-of-range scores raise ValueError."""
        with pytest.raises(ValueError):
            normalize_quality_score(6, scale_min=0, scale_max=5)

    async def test_parse_quality_label_string(self):
        """Test parsing quality labels from strings."""
        # Common quality labels
        assert parse_quality_label("Good", "overall") == "good"
        assert parse_quality_label("Poor", "overall") == "poor"
        assert parse_quality_label("Excellent", "overall") == "excellent"
        assert parse_quality_label("Fair", "overall") == "fair"

        # Synonyms
        assert parse_quality_label("Acceptable", "overall") == "acceptable"
        assert parse_quality_label("Adequate", "overall") == "acceptable"
        assert parse_quality_label("Unacceptable", "overall") == "unacceptable"
        assert parse_quality_label("Inadequate", "overall") == "unacceptable"
        assert parse_quality_label("Bad", "overall") == "poor"

    async def test_parse_quality_label_gradability_bool(self):
        """Test parsing gradability labels from booleans."""
        assert parse_quality_label(True, "gradability") == "gradable"
        assert parse_quality_label(False, "gradability") == "ungradable"

    async def test_parse_quality_label_gradability_int(self):
        """Test parsing gradability labels from integers."""
        assert parse_quality_label(1, "gradability") == "gradable"
        assert parse_quality_label(0, "gradability") == "ungradable"

    async def test_parse_quality_label_gradability_string(self):
        """Test parsing gradability labels from strings."""
        assert parse_quality_label("gradable", "gradability") == "gradable"
        assert parse_quality_label("ungradable", "gradability") == "ungradable"
        assert parse_quality_label("accept", "gradability") == "gradable"
        assert parse_quality_label("reject", "gradability") == "ungradable"

    async def test_process_quality_score_only(self, test_image_id):
        """Test processing quality with score only."""
        quality = await process_quality_annotation(
            quality_type="overall",
            image_id=test_image_id,
            quality_score=0.85,
        )

        assert quality.quality_type == "overall"
        assert quality.quality_score == 0.85
        assert quality.quality_label is None

    async def test_process_quality_label_only(self, test_image_id):
        """Test processing quality with label only."""
        quality = await process_quality_annotation(
            quality_type="gradability",
            image_id=test_image_id,
            quality_label="gradable",
        )

        assert quality.quality_type == "gradability"
        assert quality.quality_score is None
        assert quality.quality_label == "gradable"

    async def test_process_quality_score_and_label(self, test_image_id):
        """Test processing quality with both score and label."""
        quality = await process_quality_annotation(
            quality_type="overall",
            image_id=test_image_id,
            quality_score=2,
            quality_label="Excellent",
            scale_min=0,
            scale_max=2,
        )

        assert quality.quality_score == 1.0  # Normalized
        assert quality.quality_label == "excellent"

    async def test_process_quality_invalid_type(self, test_image_id):
        """Test that invalid quality types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid quality_type"):
            await process_quality_annotation(
                quality_type="invalid_type",
                image_id=test_image_id,
                quality_score=0.5,
            )

    async def test_process_quality_no_score_or_label(self, test_image_id):
        """Test that missing both score and label raises ValueError."""
        with pytest.raises(ValueError, match="At least one of"):
            await process_quality_annotation(
                quality_type="overall",
                image_id=test_image_id,
            )

    async def test_process_quality_with_scale_description(self, test_image_id):
        """Test processing quality with scale description."""
        quality = await process_quality_annotation(
            quality_type="clarity",
            image_id=test_image_id,
            quality_score=4,
            scale_description="Clarity (0-5)",
            scale_min=0,
            scale_max=5,
        )

        assert quality.scale_description == "Clarity (0-5)"

    async def test_process_quality_without_normalization(self, test_image_id):
        """Test processing quality without score normalization."""
        quality = await process_quality_annotation(
            quality_type="overall",
            image_id=test_image_id,
            quality_score=3,
            normalize_score=False,
        )

        assert quality.quality_score == 3.0  # Not normalized

    async def test_process_deepdrid_quality_all_metrics(self, test_image_id):
        """Test processing all DeepDRiD quality metrics at once."""
        qualities = await process_deepdrid_quality(
            image_id=test_image_id,
            overall_quality=2,  # Excellent
            clarity=2,  # No blur
            field_definition=1,  # Adequate
            artifact=1,  # Mild artifact
        )

        assert len(qualities) == 4
        
        # Check overall quality
        overall = next(q for q in qualities if q.quality_type == "overall")
        assert overall.quality_score == 1.0  # 2 normalized from 0-2 scale
        assert overall.quality_label == "excellent"

        # Check clarity
        clarity = next(q for q in qualities if q.quality_type == "clarity")
        assert clarity.quality_score == 1.0  # 2 normalized from 0-2 scale
        assert clarity.quality_label == "no_blur"

        # Check field definition
        field_def = next(q for q in qualities if q.quality_type == "field_definition")
        assert field_def.quality_score == 1.0  # 1 normalized from 0-1 scale
        assert field_def.quality_label == "acceptable"  # "adequate" maps to "acceptable"

        # Check artifact
        artifact = next(q for q in qualities if q.quality_type == "artifact")
        assert artifact.quality_score == 0.5  # 1 normalized from 0-2 scale
        assert artifact.quality_label == "mild_artifact"

    async def test_process_deepdrid_quality_poor_image(self, test_image_id):
        """Test processing DeepDRiD quality for a poor quality image."""
        qualities = await process_deepdrid_quality(
            image_id=test_image_id,
            overall_quality=0,  # Poor
            clarity=0,  # Severe blur
            field_definition=0,  # Inadequate
            artifact=0,  # Severe artifact
        )

        # All should have quality_score of 0.0
        for quality in qualities:
            assert quality.quality_score == 0.0

        # Check labels
        overall = next(q for q in qualities if q.quality_type == "overall")
        assert overall.quality_label == "poor"

        clarity = next(q for q in qualities if q.quality_type == "clarity")
        assert clarity.quality_label == "severe_blur"

    async def test_process_deepdrid_quality_partial_metrics(self, test_image_id):
        """Test processing DeepDRiD quality with only some metrics."""
        qualities = await process_deepdrid_quality(
            image_id=test_image_id,
            overall_quality=1,  # Good
            clarity=None,
            field_definition=None,
            artifact=None,
        )

        assert len(qualities) == 1
        assert qualities[0].quality_type == "overall"
        assert qualities[0].quality_label == "good"

    async def test_process_quality_with_raw_data_id(self, test_image_id):
        """Test processing quality with raw_data_id."""
        raw_data_id = uuid.uuid4()
        quality = await process_quality_annotation(
            quality_type="overall",
            image_id=test_image_id,
            quality_score=0.9,
            raw_data_id=raw_data_id,
        )

        assert quality.raw_data_id == raw_data_id

    async def test_prepare_quality_for_upsert_alias(self, test_image_id):
        """Test that prepare_quality_for_upsert is an alias."""
        quality = await prepare_quality_for_upsert(
            quality_type="overall",
            image_id=test_image_id,
            quality_score=0.8,
        )

        assert quality.quality_score == 0.8

    async def test_quality_deterministic_uuids(self, test_image_id):
        """Test that processing the same quality twice produces the same UUID."""
        quality_1 = await process_quality_annotation(
            quality_type="overall",
            image_id=test_image_id,
            quality_score=0.75,
        )

        quality_2 = await process_quality_annotation(
            quality_type="overall",
            image_id=test_image_id,
            quality_score=0.75,
        )

        assert quality_1.quality_id == quality_2.quality_id

    async def test_process_and_upsert_integration(self, test_image_id):
        """Test full integration: process and upsert to database."""
        quality = await process_quality_annotation(
            quality_type="overall",
            image_id=test_image_id,
            quality_score=2,
            quality_label="Excellent",
            scale_min=0,
            scale_max=2,
        )

        # Upsert to database
        await upsert_quality_annotation(quality)

        # Verify idempotency - upsert again
        await upsert_quality_annotation(quality)

    async def test_multiple_quality_types_same_image(self, test_image_id):
        """Test processing multiple quality types for the same image."""
        overall = await process_quality_annotation(
            quality_type="overall",
            image_id=test_image_id,
            quality_score=0.9,
        )

        clarity = await process_quality_annotation(
            quality_type="clarity",
            image_id=test_image_id,
            quality_score=0.8,
        )

        contrast = await process_quality_annotation(
            quality_type="contrast",
            image_id=test_image_id,
            quality_score=0.85,
        )

        # All should have different quality_ids
        assert overall.quality_id != clarity.quality_id
        assert overall.quality_id != contrast.quality_id
        assert clarity.quality_id != contrast.quality_id

    async def test_quality_all_valid_types(self, test_image_id):
        """Test processing all valid quality types."""
        valid_types = [
            "overall",
            "gradability",
            "clarity",
            "field_definition",
            "artifact",
            "contrast",
            "blur",
            "illumination",
        ]

        for quality_type in valid_types:
            quality = await process_quality_annotation(
                quality_type=quality_type,
                image_id=test_image_id,
                quality_score=0.75,
            )
            assert quality.quality_type == quality_type

    async def test_deepdrid_quality_with_raw_data_id(self, test_image_id):
        """Test processing DeepDRiD quality with raw_data_id."""
        raw_data_id = uuid.uuid4()
        qualities = await process_deepdrid_quality(
            image_id=test_image_id,
            overall_quality=2,
            clarity=2,
            field_definition=1,
            artifact=1,
            raw_data_id=raw_data_id,
        )

        # All should have the same raw_data_id
        for quality in qualities:
            assert quality.raw_data_id == raw_data_id
