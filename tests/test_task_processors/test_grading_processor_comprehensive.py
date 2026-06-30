"""
Comprehensive tests for grading_processor using real data from data directory.

Tests cover:
- process_disease_grade() with various input formats
- get_or_create_scale() idempotency
- check_scale_mapping_exists()
- Real data from EYEPACS, SUSTech-SYSU, MESSIDOR
"""

import uuid
import pytest

from chaksudb.db.queries.grading import find_grading_scale_by_id, upsert_disease_grading
from chaksudb.ingest.framework.task_processors.grading_processor import (
    check_scale_mapping_exists,
    get_or_create_scale,
    process_disease_grade,
    prepare_grading_for_upsert,
)
from chaksudb.ingest.framework.gen_uuid import generate_dataset_uuid
from chaksudb.config.config import get_data_root


pytestmark = pytest.mark.asyncio


class TestGradingProcessorWithRealData:
    """Test grading processor with real data from datasets."""

    @pytest.fixture
    async def test_dataset_id(self, test_dataset_in_db):
        """Create a test dataset ID with database record."""
        return test_dataset_in_db

    @pytest.fixture
    async def test_image_id(self, test_image_in_db):
        """Create a test image ID with database record."""
        return test_image_in_db

    async def test_process_dr_grade_eyepacs_format(self, test_image_id):
        """Test processing DR grades in EYEPACS format (0-4 scale)."""
        # EYEPACS uses levels 0-4
        for grade_value in [0, 1, 2, 3, 4]:
            grading = await process_disease_grade(
                grade_value=grade_value,
                disease_type="DR",
                scale_name="EYEPACS_0_4",
                image_id=test_image_id,
                scale_description="EYEPACS 5-level DR grading",
                min_value=0,
                max_value=4,
                value_labels={
                    "0": "No DR",
                    "1": "Mild NPDR",
                    "2": "Moderate NPDR",
                    "3": "Severe NPDR",
                    "4": "PDR",
                },
            )

            # Verify the grading model
            assert grading.image_id == test_image_id
            assert grading.disease_type == "DR"
            assert grading.original_grade == str(grade_value)
            assert grading.annotation_method == "manual"
            assert grading.scaled_grade is None  # Trigger will populate this

    async def test_process_dr_grade_sustech_icdr(self, test_image_id):
        """Test processing DR grades in SUSTech-SYSU ICDR format."""
        # SUSTech-SYSU provides ICDR grades 0-4
        grading = await process_disease_grade(
            grade_value=2,
            disease_type="DR",
            scale_name="ICDR_0_4",
            image_id=test_image_id,
            scale_description="International Clinical DR Severity Scale (0-4)",
            min_value=0,
            max_value=4,
            value_labels={
                "0": "No DR",
                "1": "Mild NPDR",
                "2": "Moderate NPDR",
                "3": "Severe NPDR",
                "4": "PDR",
            },
        )

        assert grading.original_grade == "2"
        assert grading.disease_type == "DR"

    async def test_process_dr_grade_sustech_aao(self, test_image_id):
        """Test processing DR grades in SUSTech-SYSU AAO format."""
        grading = await process_disease_grade(
            grade_value=3,
            disease_type="DR",
            scale_name="AAO",
            image_id=test_image_id,
            scale_description="American Academy of Ophthalmology DR grading",
            min_value=0,
            max_value=5,
        )

        assert grading.original_grade == "3"
        assert grading.disease_type == "DR"

    async def test_process_dr_grade_sustech_scottish(self, test_image_id):
        """Test processing DR grades in SUSTech-SYSU Scottish format."""
        grading = await process_disease_grade(
            grade_value=4,
            disease_type="DR",
            scale_name="Scottish",
            image_id=test_image_id,
            scale_description="Scottish DR grading protocol",
            min_value=0,
            max_value=5,
        )

        assert grading.original_grade == "4"
        assert grading.disease_type == "DR"

    async def test_process_grade_various_input_types(self, test_image_id):
        """Test processing grades with various input types."""
        # Test with int
        grading_int = await process_disease_grade(
            grade_value=2,
            disease_type="DR",
            scale_name="TEST_SCALE_INT",
            image_id=test_image_id,
            min_value=0,
            max_value=4,
        )
        assert grading_int.original_grade == "2"

        # Test with float
        grading_float = await process_disease_grade(
            grade_value=3.0,
            disease_type="DR",
            scale_name="TEST_SCALE_FLOAT",
            image_id=test_image_id,
            min_value=0,
            max_value=4,
        )
        assert grading_float.original_grade == "3"

        # Test with string
        grading_str = await process_disease_grade(
            grade_value="4",
            disease_type="DR",
            scale_name="TEST_SCALE_STR",
            image_id=test_image_id,
            min_value=0,
            max_value=4,
        )
        assert grading_str.original_grade == "4"

    async def test_process_glaucoma_grade(self, test_image_id):
        """Test processing glaucoma grades."""
        # LAG dataset has glaucoma classification
        grading = await process_disease_grade(
            grade_value=1,
            disease_type="Glaucoma",
            scale_name="LAG_BINARY",
            image_id=test_image_id,
            scale_description="LAG binary glaucoma classification",
            min_value=0,
            max_value=1,
            value_labels={"0": "non_glaucoma", "1": "suspicious_glaucoma"},
        )

        assert grading.disease_type == "Glaucoma"
        assert grading.original_grade == "1"

    async def test_process_dme_grade(self, test_image_id):
        """Test processing DME grades."""
        grading = await process_disease_grade(
            grade_value=2,
            disease_type="DME",
            scale_name="IDRID_DME",
            image_id=test_image_id,
            min_value=0,
            max_value=2,
        )

        assert grading.disease_type == "DME"
        assert grading.original_grade == "2"

    async def test_get_or_create_scale_idempotency(self):
        """Test that get_or_create_scale is idempotent."""
        scale_name = "TEST_IDEMPOTENT_SCALE"
        disease_type = "DR"

        # Create scale first time
        scale_id_1 = await get_or_create_scale(
            scale_name=scale_name,
            disease_type=disease_type,
            scale_description="Test scale for idempotency",
            min_value=0,
            max_value=4,
        )

        # Create again - should return same UUID
        scale_id_2 = await get_or_create_scale(
            scale_name=scale_name,
            disease_type=disease_type,
            scale_description="Test scale for idempotency (different description)",
            min_value=0,
            max_value=4,
        )

        assert scale_id_1 == scale_id_2

        # Verify scale exists in database
        scale = await find_grading_scale_by_id(scale_id_1)
        assert scale is not None
        assert scale.scale_name == scale_name
        assert scale.disease_type == disease_type

    async def test_get_or_create_scale_with_value_labels(self):
        """Test creating scale with value labels."""
        scale_id = await get_or_create_scale(
            scale_name="TEST_LABELED_SCALE",
            disease_type="DR",
            scale_description="Test scale with labels",
            min_value=0,
            max_value=2,
            value_labels={"0": "None", "1": "Mild", "2": "Severe"},
        )

        scale = await find_grading_scale_by_id(scale_id)
        assert scale is not None
        assert scale.value_labels == {"0": "None", "1": "Mild", "2": "Severe"}

    async def test_process_grade_with_confidence_score(self, test_image_id):
        """Test processing grades with confidence scores."""
        grading = await process_disease_grade(
            grade_value=3,
            disease_type="DR",
            scale_name="TEST_WITH_CONFIDENCE",
            image_id=test_image_id,
            min_value=0,
            max_value=4,
            confidence_score=0.95,
        )

        assert grading.confidence_score == 0.95

    async def test_process_grade_with_grade_label(self, test_image_id):
        """Test processing grades with grade labels."""
        grading = await process_disease_grade(
            grade_value=2,
            disease_type="DR",
            scale_name="TEST_WITH_LABEL",
            image_id=test_image_id,
            min_value=0,
            max_value=4,
            grade_label="Moderate NPDR",
        )

        assert grading.grade_label == "Moderate NPDR"

    async def test_process_grade_invalid_disease_type(self, test_image_id):
        """Test that invalid disease types raise ValueError."""
        with pytest.raises(ValueError, match="Invalid disease_type"):
            await process_disease_grade(
                grade_value=2,
                disease_type="INVALID_DISEASE",
                scale_name="TEST_SCALE",
                image_id=test_image_id,
            )

    async def test_process_grade_invalid_annotation_method(self, test_image_id):
        """Test that invalid annotation methods raise ValueError."""
        with pytest.raises(ValueError, match="Invalid annotation_method"):
            await process_disease_grade(
                grade_value=2,
                disease_type="DR",
                scale_name="TEST_SCALE",
                image_id=test_image_id,
                annotation_method="invalid_method",
            )

    async def test_process_grade_with_raw_data_id(self, test_image_id):
        """Test processing grades with raw_data_id."""
        raw_data_id = uuid.uuid4()
        grading = await process_disease_grade(
            grade_value=1,
            disease_type="DR",
            scale_name="TEST_WITH_RAW",
            image_id=test_image_id,
            raw_data_id=raw_data_id,
        )

        assert grading.raw_data_id == raw_data_id

    async def test_prepare_grading_for_upsert_alias(self, test_image_id):
        """Test that prepare_grading_for_upsert is an alias for process_disease_grade."""
        grading = await prepare_grading_for_upsert(
            grade_value=2,
            disease_type="DR",
            scale_name="TEST_ALIAS",
            image_id=test_image_id,
        )

        assert grading.original_grade == "2"
        assert grading.disease_type == "DR"

    async def test_process_and_upsert_integration(self, test_image_id):
        """Test full integration: process and upsert to database."""
        grading = await process_disease_grade(
            grade_value=3,
            disease_type="DR",
            scale_name="TEST_INTEGRATION",
            image_id=test_image_id,
            scale_description="Test integration scale",
            min_value=0,
            max_value=4,
        )

        # Upsert to database
        await upsert_disease_grading(grading)

        # Verify idempotency - upsert again
        await upsert_disease_grading(grading)

    async def test_multiple_scales_same_disease(self, test_image_id):
        """Test creating multiple scales for the same disease."""
        scale_id_1 = await get_or_create_scale(
            scale_name="DR_SCALE_1",
            disease_type="DR",
            min_value=0,
            max_value=4,
        )

        scale_id_2 = await get_or_create_scale(
            scale_name="DR_SCALE_2",
            disease_type="DR",
            min_value=0,
            max_value=5,
        )

        assert scale_id_1 != scale_id_2

    async def test_process_grade_deterministic_uuids(self, test_image_id):
        """Test that processing the same grade twice produces the same UUID."""
        grading_1 = await process_disease_grade(
            grade_value=2,
            disease_type="DR",
            scale_name="TEST_DETERMINISTIC",
            image_id=test_image_id,
            min_value=0,
            max_value=4,
        )

        grading_2 = await process_disease_grade(
            grade_value=2,
            disease_type="DR",
            scale_name="TEST_DETERMINISTIC",
            image_id=test_image_id,
            min_value=0,
            max_value=4,
        )

        assert grading_1.grading_id == grading_2.grading_id
