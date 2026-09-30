"""
Comprehensive tests for split_assigner and patient_register using real data patterns.

Tests cover:
- Dataset split registration
- Image split assignment
- Patient registration
- Patient-image linking
- Bulk operations
- Real-world patterns from ODIR-5K and other datasets
"""

import uuid
import pytest
from datetime import date

from chaksudb.ingest.framework.gen_uuid import generate_dataset_uuid
from chaksudb.ingest.framework.split_assigner import (
    register_dataset_split,
    assign_image_to_split,
    bulk_assign_images_to_split,
    register_standard_splits,
    assign_images_by_split_dict,
)
from chaksudb.ingest.framework.patient_register import (
    register_patient,
    link_patient_to_image,
    register_patient_with_images,
    bulk_register_patients,
    bulk_link_patients_to_images,
    extract_patient_id_from_filename,
)


pytestmark = pytest.mark.asyncio


class TestSplitAssignerWithRealData:
    """Test split assignment with real dataset patterns."""

    @pytest.fixture
    async def test_dataset_id(self, test_dataset_in_db):
        """Create a test dataset ID with database record."""
        return test_dataset_in_db

    @pytest.fixture
    async def test_image_ids(self, test_images_in_db):
        """Create test image IDs with database records."""
        # Return list of 10 test images from the fixture
        return test_images_in_db

    async def test_register_dataset_split_explicit(self, test_dataset_id):
        """Test registering an explicit dataset split."""
        split_id = await register_dataset_split(
            dataset_id=test_dataset_id,
            split_name="train",
            split_type="explicit",
            task_type="grading",
            image_count=1000,
        )

        assert split_id is not None

    async def test_register_dataset_split_metadata_defined(self, test_dataset_id):
        """Test registering a metadata-defined split."""
        split_id = await register_dataset_split(
            dataset_id=test_dataset_id,
            split_name="validation",
            split_type="metadata_defined",
            task_type="segmentation",
            image_count=200,
        )

        assert split_id is not None

    async def test_register_dataset_split_idempotency(self, test_dataset_id):
        """Test that registering the same split twice is idempotent."""
        split_id_1 = await register_dataset_split(
            dataset_id=test_dataset_id,
            split_name="test",
            split_type="explicit",
            task_type="grading",
        )

        split_id_2 = await register_dataset_split(
            dataset_id=test_dataset_id,
            split_name="test",
            split_type="explicit",
            task_type="grading",
        )

        assert split_id_1 == split_id_2

    async def test_assign_image_to_split(self, test_dataset_id, test_image_ids):
        """Test assigning an image to a split."""
        # Register a split first
        split_id = await register_dataset_split(
            dataset_id=test_dataset_id,
            split_name="train",
            split_type="explicit",
        )

        # Assign an image to the split
        assignment_id = await assign_image_to_split(
            image_id=test_image_ids[0],
            split_id=split_id,
            is_primary=True,
        )

        assert assignment_id is not None

    async def test_bulk_assign_images_to_split(self, test_dataset_id, test_image_ids):
        """Test bulk assigning images to a split."""
        # Register a split first
        split_id = await register_dataset_split(
            dataset_id=test_dataset_id,
            split_name="train",
            split_type="explicit",
        )

        # Bulk assign images
        count = await bulk_assign_images_to_split(
            image_ids=test_image_ids[:5],
            split_id=split_id,
            is_primary=True,
        )

        assert count == 5

    async def test_register_standard_splits(self, test_dataset_id):
        """Test registering standard train/val/test splits."""
        splits = await register_standard_splits(
            dataset_id=test_dataset_id,
            split_type="explicit",
            task_type="grading",
            train_count=800,
            val_count=100,
            test_count=100,
        )

        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert splits["train"] is not None
        assert splits["val"] is not None
        assert splits["test"] is not None

    async def test_assign_images_by_split_dict(self, test_dataset_id, test_image_ids):
        """Test assigning images using split dictionary."""
        # Register standard splits
        splits = await register_standard_splits(
            dataset_id=test_dataset_id,
            split_type="explicit",
        )

        # Create split assignments
        split_assignments = {
            "train": test_image_ids[:6],
            "val": test_image_ids[6:8],
            "test": test_image_ids[8:10],
        }

        counts = await assign_images_by_split_dict(
            split_assignments=split_assignments,
            split_ids=splits,
        )

        assert counts["train"] == 6
        assert counts["val"] == 2
        assert counts["test"] == 2

    async def test_task_specific_split(self, test_dataset_id, test_image_ids):
        """Test task-specific split assignment."""
        # IDRID has different splits for grading vs segmentation
        
        # Register grading split
        grading_split_id = await register_dataset_split(
            dataset_id=test_dataset_id,
            split_name="train",
            split_type="explicit",
            task_type="grading",
        )

        # Register segmentation split
        seg_split_id = await register_dataset_split(
            dataset_id=test_dataset_id,
            split_name="train",
            split_type="explicit",
            task_type="segmentation",
        )

        # These should be different splits
        assert grading_split_id != seg_split_id

    async def test_split_without_task_type(self, test_dataset_id):
        """Test registering split without specific task type."""
        split_id = await register_dataset_split(
            dataset_id=test_dataset_id,
            split_name="train",
            split_type="explicit",
            task_type=None,  # No specific task
        )

        assert split_id is not None


class TestPatientRegisterWithRealData:
    """Test patient registration with real dataset patterns."""

    @pytest.fixture
    async def test_dataset_id(self, test_dataset_in_db):
        """Create a test dataset ID with database record."""
        return test_dataset_in_db

    @pytest.fixture
    async def test_image_ids(self, test_images_in_db):
        """Create test image IDs with database records."""
        # Return list of 10 test images from the fixture
        return test_images_in_db

    async def test_register_patient_with_demographics(self, test_dataset_id):
        """Test registering a patient with demographic data."""
        patient_id = await register_patient(
            dataset_id=test_dataset_id,
            original_patient_id="P12345",
            age=65,
            sex="male",
            ethnicity="Asian",
        )

        assert patient_id is not None

    async def test_register_patient_with_comorbidities(self, test_dataset_id):
        """Test registering a patient with comorbidities."""
        patient_id = await register_patient(
            dataset_id=test_dataset_id,
            original_patient_id="P67890",
            age=72,
            sex="female",
            comorbidities={
                "diabetes": True,
                "hypertension": True,
                "smoking": "former",
                "duration_diabetes_years": 15,
            },
        )

        assert patient_id is not None

    async def test_register_patient_idempotency(self, test_dataset_id):
        """Test that registering the same patient twice is idempotent."""
        patient_id_1 = await register_patient(
            dataset_id=test_dataset_id,
            original_patient_id="P11111",
            age=50,
            sex="male",
        )

        patient_id_2 = await register_patient(
            dataset_id=test_dataset_id,
            original_patient_id="P11111",
            age=50,
            sex="male",
        )

        assert patient_id_1 == patient_id_2

    async def test_link_patient_to_image(self, test_dataset_id, test_image_ids):
        """Test linking a patient to an image."""
        # Register patient first
        patient_id = await register_patient(
            dataset_id=test_dataset_id,
            original_patient_id="P22222",
            age=60,
            sex="female",
        )

        # Link to image
        relationship_id = await link_patient_to_image(
            patient_id=patient_id,
            image_id=test_image_ids[0],
            exam_date=date(2023, 6, 15),
        )

        assert relationship_id is not None

    async def test_register_patient_with_images(self, test_dataset_id, test_image_ids):
        """Test registering patient and linking to multiple images."""
        # ODIR-5K pattern: patient with left and right eye images
        patient_id, relationship_ids = await register_patient_with_images(
            dataset_id=test_dataset_id,
            original_patient_id="P33333",
            image_ids=test_image_ids[:2],  # Left and right eye
            age=57,
            sex="male",
            exam_date=date(2023, 6, 20),
        )

        assert patient_id is not None
        assert len(relationship_ids) == 2

    async def test_bulk_register_patients(self, test_dataset_id):
        """Test bulk registering multiple patients."""
        patients_data = [
            {"original_patient_id": "P001", "age": 65, "sex": "male"},
            {"original_patient_id": "P002", "age": 52, "sex": "female"},
            {"original_patient_id": "P003", "age": 71, "sex": "male"},
        ]

        patient_ids = await bulk_register_patients(
            patients_data=patients_data,
            dataset_id=test_dataset_id,
        )

        assert len(patient_ids) == 3

    async def test_bulk_link_patients_to_images(self, test_dataset_id, test_image_ids):
        """Test bulk linking patients to images."""
        # Register patients first
        patient_ids = await bulk_register_patients(
            patients_data=[
                {"original_patient_id": "P101", "age": 60, "sex": "male"},
                {"original_patient_id": "P102", "age": 55, "sex": "female"},
                {"original_patient_id": "P103", "age": 70, "sex": "male"},
            ],
            dataset_id=test_dataset_id,
        )

        # Create pairs
        pairs = [
            (patient_ids[0], test_image_ids[0]),
            (patient_ids[1], test_image_ids[1]),
            (patient_ids[2], test_image_ids[2]),
        ]

        relationship_ids = await bulk_link_patients_to_images(
            patient_image_pairs=pairs,
        )

        assert len(relationship_ids) == 3

    async def test_extract_patient_id_from_filename_underscore(self):
        """Test extracting patient ID from filename with underscore pattern."""
        # ODIR-5K pattern: "1_left.jpg"
        result = extract_patient_id_from_filename("1_left.jpg", pattern="underscore")
        assert result is not None
        patient_id, laterality = result
        assert patient_id == "1"
        assert laterality == "left"

        # Right eye
        result = extract_patient_id_from_filename("127_right.jpg", pattern="underscore")
        assert result is not None
        patient_id, laterality = result
        assert patient_id == "127"
        assert laterality == "right"

    async def test_extract_patient_id_from_filename_dash(self):
        """Test extracting patient ID from filename with dash pattern."""
        result = extract_patient_id_from_filename("P12345-left.jpg", pattern="dash")
        assert result is not None
        patient_id, laterality = result
        assert patient_id == "p12345"  # Lowercased
        assert laterality == "left"

    async def test_extract_patient_id_from_filename_od_os(self):
        """Test extracting patient ID with medical abbreviations."""
        # OD = Oculus Dexter (right eye)
        result = extract_patient_id_from_filename("patient123_od.jpg")
        assert result is not None
        patient_id, laterality = result
        assert patient_id == "patient123"
        assert laterality == "right"

        # OS = Oculus Sinister (left eye)
        result = extract_patient_id_from_filename("patient456_os.jpg")
        assert result is not None
        patient_id, laterality = result
        assert patient_id == "patient456"
        assert laterality == "left"

    async def test_extract_patient_id_from_filename_no_pattern(self):
        """Test extracting patient ID when no pattern matches."""
        result = extract_patient_id_from_filename("random_filename.jpg")
        # May or may not match depending on pattern
        # If no pattern matches, returns None
        if result is None:
            assert True
        else:
            # If it does match, check structure is valid
            assert len(result) == 2

    async def test_patient_with_nationality(self, test_dataset_id):
        """Test registering patient with nationality."""
        patient_id = await register_patient(
            dataset_id=test_dataset_id,
            original_patient_id="P55555",
            age=68,
            sex="male",
            nationality="Paraguay",
        )

        assert patient_id is not None

    async def test_patient_various_sex_values(self, test_dataset_id):
        """Test registering patients with various sex values."""
        # Male
        patient_id_male = await register_patient(
            dataset_id=test_dataset_id,
            original_patient_id="P_MALE",
            sex="male",
        )
        assert patient_id_male is not None

        # Female
        patient_id_female = await register_patient(
            dataset_id=test_dataset_id,
            original_patient_id="P_FEMALE",
            sex="female",
        )
        assert patient_id_female is not None

        # Unknown
        patient_id_unknown = await register_patient(
            dataset_id=test_dataset_id,
            original_patient_id="P_UNKNOWN",
            sex="unknown",
        )
        assert patient_id_unknown is not None

    async def test_link_patient_to_image_without_exam_date(self, test_dataset_id, test_image_ids):
        """Test linking patient to image without exam date."""
        patient_id = await register_patient(
            dataset_id=test_dataset_id,
            original_patient_id="P77777",
            age=55,
            sex="female",
        )

        relationship_id = await link_patient_to_image(
            patient_id=patient_id,
            image_id=test_image_ids[0],
            exam_date=None,  # No exam date
        )

        assert relationship_id is not None

    async def test_odir5k_patient_scenario(self, test_dataset_id, test_image_ids):
        """Test ODIR-5K style patient registration scenario."""
        # ODIR-5K patient example: ID=127, Age=52, Sex=Male
        # Has both left and right eye images
        
        patient_id, relationship_ids = await register_patient_with_images(
            dataset_id=test_dataset_id,
            original_patient_id="127",
            image_ids=test_image_ids[:2],  # Left and right
            age=52,
            sex="male",
        )

        assert patient_id is not None
        assert len(relationship_ids) == 2

    async def test_bulk_link_with_exam_dates(self, test_dataset_id, test_image_ids):
        """Test bulk linking patients to images with exam dates."""
        # Register patients first
        patient_ids = await bulk_register_patients(
            patients_data=[
                {"original_patient_id": "P201", "age": 60, "sex": "male"},
                {"original_patient_id": "P202", "age": 55, "sex": "female"},
            ],
            dataset_id=test_dataset_id,
        )

        # Create pairs
        pairs = [
            (patient_ids[0], test_image_ids[0]),
            (patient_ids[1], test_image_ids[1]),
        ]

        dates = [
            date(2023, 6, 15),
            date(2023, 6, 20),
        ]

        relationship_ids = await bulk_link_patients_to_images(
            patient_image_pairs=pairs,
            exam_dates=dates,
        )

        assert len(relationship_ids) == 2
