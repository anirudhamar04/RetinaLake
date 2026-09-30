"""
Tests for chaksudb/ingest/framework/gen_uuid.py

Tests UUID v5 generation functions for all database tables based on their docstrings.
All tests verify determinism, correct return types, and edge cases.
"""

import uuid
import pytest

from chaksudb.config.config import constants
from chaksudb.ingest.framework.gen_uuid import (
    generate_uuid_v5,
    # Core Tables
    generate_dataset_uuid,
    generate_model_uuid,
    generate_expert_uuid,
    # Patient & Image Tables
    generate_patient_uuid,
    generate_image_group_uuid,
    generate_image_uuid,
    generate_patient_image_uuid,
    # Raw Annotation Files
    generate_raw_file_uuid,
    # Expert Annotations
    generate_expert_annotation_uuid,
    # Annotation Types & Scales
    generate_annotation_type_uuid,
    generate_grading_scale_uuid,
    generate_grading_scale_mapping_uuid,
    # Consensus Annotations
    generate_consensus_uuid,
    # Provenance & Transformations
    generate_provenance_chain_uuid,
    generate_transformation_uuid,
    generate_provenance_transformation_uuid,
    # Annotation Tables
    generate_segmentation_uuid,
    generate_disease_grading_uuid,
    generate_localization_uuid,
    generate_classification_uuid,
    generate_quality_uuid,
    generate_description_uuid,
    # Keywords
    generate_keyword_uuid,
    generate_keyword_annotation_uuid,
    # Dataset Splits
    generate_dataset_split_uuid,
    generate_image_split_uuid,
)


class TestCoreUUIDGeneration:
    """Tests for core UUID v5 generation function."""

    def test_generate_uuid_v5_returns_uuid(self):
        """Test that generate_uuid_v5 returns a UUID object."""
        result = generate_uuid_v5(constants.NAMESPACE_DATASET, "test")
        assert isinstance(result, uuid.UUID)

    def test_generate_uuid_v5_is_deterministic(self):
        """Test that generate_uuid_v5 produces same UUID for same inputs."""
        namespace = constants.NAMESPACE_DATASET
        name = "test_dataset"
        
        uuid1 = generate_uuid_v5(namespace, name)
        uuid2 = generate_uuid_v5(namespace, name)
        
        assert uuid1 == uuid2

    def test_generate_uuid_v5_different_names_produce_different_uuids(self):
        """Test that different names produce different UUIDs."""
        namespace = constants.NAMESPACE_DATASET
        
        uuid1 = generate_uuid_v5(namespace, "name1")
        uuid2 = generate_uuid_v5(namespace, "name2")
        
        assert uuid1 != uuid2

    def test_generate_uuid_v5_different_namespaces_produce_different_uuids(self):
        """Test that different namespaces produce different UUIDs."""
        name = "same_name"
        
        uuid1 = generate_uuid_v5(constants.NAMESPACE_DATASET, name)
        uuid2 = generate_uuid_v5(constants.NAMESPACE_IMAGE, name)
        
        assert uuid1 != uuid2

    def test_generate_uuid_v5_with_empty_string(self):
        """Test that generate_uuid_v5 handles empty string name."""
        result = generate_uuid_v5(constants.NAMESPACE_DATASET, "")
        assert isinstance(result, uuid.UUID)

    def test_generate_uuid_v5_with_special_characters(self):
        """Test that generate_uuid_v5 handles special characters in name."""
        name = "test:name/with@special#characters"
        result = generate_uuid_v5(constants.NAMESPACE_DATASET, name)
        assert isinstance(result, uuid.UUID)


class TestCoreTables:
    """Tests for core table UUID generation functions."""

    def test_generate_dataset_uuid_returns_uuid(self):
        """Test that generate_dataset_uuid returns a UUID."""
        result = generate_dataset_uuid("my_dataset")
        assert isinstance(result, uuid.UUID)

    def test_generate_dataset_uuid_is_deterministic(self):
        """Test that generate_dataset_uuid produces same UUID for same dataset_name."""
        dataset_name = "test_dataset"
        
        uuid1 = generate_dataset_uuid(dataset_name)
        uuid2 = generate_dataset_uuid(dataset_name)
        
        assert uuid1 == uuid2

    def test_generate_dataset_uuid_different_names_produce_different_uuids(self):
        """Test that different dataset names produce different UUIDs."""
        uuid1 = generate_dataset_uuid("dataset1")
        uuid2 = generate_dataset_uuid("dataset2")
        
        assert uuid1 != uuid2

    def test_generate_model_uuid_returns_uuid(self):
        """Test that generate_model_uuid returns a UUID."""
        result = generate_model_uuid("my_model")
        assert isinstance(result, uuid.UUID)

    def test_generate_model_uuid_is_deterministic(self):
        """Test that generate_model_uuid produces same UUID for same model_name."""
        model_name = "resnet50"
        
        uuid1 = generate_model_uuid(model_name)
        uuid2 = generate_model_uuid(model_name)
        
        assert uuid1 == uuid2

    def test_generate_model_uuid_different_names_produce_different_uuids(self):
        """Test that different model names produce different UUIDs."""
        uuid1 = generate_model_uuid("model1")
        uuid2 = generate_model_uuid("model2")
        
        assert uuid1 != uuid2

    def test_generate_expert_uuid_with_dataset_id_returns_uuid(self):
        """Test that generate_expert_uuid returns UUID for dataset-based expert."""
        dataset_id = uuid.uuid4()
        result = generate_expert_uuid(dataset_id=dataset_id, model_id=None)
        assert isinstance(result, uuid.UUID)

    def test_generate_expert_uuid_with_model_id_returns_uuid(self):
        """Test that generate_expert_uuid returns UUID for model-based expert."""
        model_id = uuid.uuid4()
        result = generate_expert_uuid(dataset_id=None, model_id=model_id)
        assert isinstance(result, uuid.UUID)

    def test_generate_expert_uuid_with_dataset_is_deterministic(self):
        """Test that generate_expert_uuid is deterministic for same dataset_id."""
        dataset_id = uuid.uuid4()
        
        uuid1 = generate_expert_uuid(dataset_id=dataset_id, model_id=None)
        uuid2 = generate_expert_uuid(dataset_id=dataset_id, model_id=None)
        
        assert uuid1 == uuid2

    def test_generate_expert_uuid_with_model_is_deterministic(self):
        """Test that generate_expert_uuid is deterministic for same model_id."""
        model_id = uuid.uuid4()
        
        uuid1 = generate_expert_uuid(dataset_id=None, model_id=model_id)
        uuid2 = generate_expert_uuid(dataset_id=None, model_id=model_id)
        
        assert uuid1 == uuid2

    def test_generate_expert_uuid_with_expert_name_returns_uuid(self):
        """Test that generate_expert_uuid accepts optional expert_name parameter."""
        dataset_id = uuid.uuid4()
        result = generate_expert_uuid(
            dataset_id=dataset_id,
            model_id=None,
            expert_name="Dr. Smith"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_expert_uuid_with_expert_name_produces_different_uuid(self):
        """Test that expert_name affects UUID generation."""
        dataset_id = uuid.uuid4()
        
        uuid1 = generate_expert_uuid(dataset_id=dataset_id, model_id=None)
        uuid2 = generate_expert_uuid(
            dataset_id=dataset_id,
            model_id=None,
            expert_name="Dr. Smith"
        )
        
        assert uuid1 != uuid2

    def test_generate_expert_uuid_raises_error_when_both_none(self):
        """Test that generate_expert_uuid raises ValueError when both dataset_id and model_id are None."""
        with pytest.raises(ValueError, match="Either dataset_id or model_id must be provided"):
            generate_expert_uuid(dataset_id=None, model_id=None)

    def test_generate_expert_uuid_dataset_vs_model_produce_different_uuids(self):
        """Test that dataset-based and model-based experts produce different UUIDs."""
        dataset_id = uuid.uuid4()
        model_id = uuid.uuid4()
        
        uuid1 = generate_expert_uuid(dataset_id=dataset_id, model_id=None)
        uuid2 = generate_expert_uuid(dataset_id=None, model_id=model_id)
        
        assert uuid1 != uuid2


class TestPatientAndImageTables:
    """Tests for patient and image table UUID generation functions."""

    def test_generate_patient_uuid_returns_uuid(self):
        """Test that generate_patient_uuid returns a UUID."""
        dataset_id = uuid.uuid4()
        result = generate_patient_uuid(dataset_id, "patient_001")
        assert isinstance(result, uuid.UUID)

    def test_generate_patient_uuid_is_deterministic(self):
        """Test that generate_patient_uuid produces same UUID for same inputs."""
        dataset_id = uuid.uuid4()
        patient_id = "patient_001"
        
        uuid1 = generate_patient_uuid(dataset_id, patient_id)
        uuid2 = generate_patient_uuid(dataset_id, patient_id)
        
        assert uuid1 == uuid2

    def test_generate_patient_uuid_different_patients_produce_different_uuids(self):
        """Test that different patient IDs produce different UUIDs."""
        dataset_id = uuid.uuid4()
        
        uuid1 = generate_patient_uuid(dataset_id, "patient_001")
        uuid2 = generate_patient_uuid(dataset_id, "patient_002")
        
        assert uuid1 != uuid2

    def test_generate_patient_uuid_different_datasets_produce_different_uuids(self):
        """Test that same patient ID in different datasets produces different UUIDs."""
        patient_id = "patient_001"
        
        uuid1 = generate_patient_uuid(uuid.uuid4(), patient_id)
        uuid2 = generate_patient_uuid(uuid.uuid4(), patient_id)
        
        assert uuid1 != uuid2

    def test_generate_image_group_uuid_returns_uuid(self):
        """Test that generate_image_group_uuid returns a UUID."""
        dataset_id = uuid.uuid4()
        result = generate_image_group_uuid(dataset_id, "oct_volume", "volume_001")
        assert isinstance(result, uuid.UUID)

    def test_generate_image_group_uuid_is_deterministic(self):
        """Test that generate_image_group_uuid produces same UUID for same inputs."""
        dataset_id = uuid.uuid4()
        group_type = "oct_volume"
        group_id = "volume_001"
        
        uuid1 = generate_image_group_uuid(dataset_id, group_type, group_id)
        uuid2 = generate_image_group_uuid(dataset_id, group_type, group_id)
        
        assert uuid1 == uuid2

    def test_generate_image_group_uuid_different_types_produce_different_uuids(self):
        """Test that different group types produce different UUIDs."""
        dataset_id = uuid.uuid4()
        group_id = "group_001"
        
        uuid1 = generate_image_group_uuid(dataset_id, "oct_volume", group_id)
        uuid2 = generate_image_group_uuid(dataset_id, "video", group_id)
        
        assert uuid1 != uuid2

    def test_generate_image_uuid_returns_uuid(self):
        """Test that generate_image_uuid returns a UUID."""
        dataset_id = uuid.uuid4()
        result = generate_image_uuid(dataset_id, "image_001")
        assert isinstance(result, uuid.UUID)

    def test_generate_image_uuid_is_deterministic(self):
        """Test that generate_image_uuid produces same UUID for same inputs."""
        dataset_id = uuid.uuid4()
        image_id = "image_001"
        
        uuid1 = generate_image_uuid(dataset_id, image_id)
        uuid2 = generate_image_uuid(dataset_id, image_id)
        
        assert uuid1 == uuid2

    def test_generate_image_uuid_different_images_produce_different_uuids(self):
        """Test that different image IDs produce different UUIDs."""
        dataset_id = uuid.uuid4()
        
        uuid1 = generate_image_uuid(dataset_id, "image_001")
        uuid2 = generate_image_uuid(dataset_id, "image_002")
        
        assert uuid1 != uuid2

    def test_generate_patient_image_uuid_returns_uuid(self):
        """Test that generate_patient_image_uuid returns a UUID."""
        patient_id = uuid.uuid4()
        image_id = uuid.uuid4()
        result = generate_patient_image_uuid(patient_id, image_id)
        assert isinstance(result, uuid.UUID)

    def test_generate_patient_image_uuid_is_deterministic(self):
        """Test that generate_patient_image_uuid produces same UUID for same inputs."""
        patient_id = uuid.uuid4()
        image_id = uuid.uuid4()
        
        uuid1 = generate_patient_image_uuid(patient_id, image_id)
        uuid2 = generate_patient_image_uuid(patient_id, image_id)
        
        assert uuid1 == uuid2

    def test_generate_patient_image_uuid_different_relationships_produce_different_uuids(self):
        """Test that different patient-image relationships produce different UUIDs."""
        patient_id1 = uuid.uuid4()
        patient_id2 = uuid.uuid4()
        image_id = uuid.uuid4()
        
        uuid1 = generate_patient_image_uuid(patient_id1, image_id)
        uuid2 = generate_patient_image_uuid(patient_id2, image_id)
        
        assert uuid1 != uuid2


class TestRawAnnotationFiles:
    """Tests for raw annotation file UUID generation functions."""

    def test_generate_raw_file_uuid_returns_uuid(self):
        """Test that generate_raw_file_uuid returns a UUID."""
        dataset_id = uuid.uuid4()
        file_hash = "abc123def456"
        result = generate_raw_file_uuid(dataset_id, file_hash)
        assert isinstance(result, uuid.UUID)

    def test_generate_raw_file_uuid_is_deterministic(self):
        """Test that generate_raw_file_uuid produces same UUID for same inputs."""
        dataset_id = uuid.uuid4()
        file_hash = "abc123def456"
        
        uuid1 = generate_raw_file_uuid(dataset_id, file_hash)
        uuid2 = generate_raw_file_uuid(dataset_id, file_hash)
        
        assert uuid1 == uuid2

    def test_generate_raw_file_uuid_different_hashes_produce_different_uuids(self):
        """Test that different file hashes produce different UUIDs."""
        dataset_id = uuid.uuid4()
        
        uuid1 = generate_raw_file_uuid(dataset_id, "hash1")
        uuid2 = generate_raw_file_uuid(dataset_id, "hash2")
        
        assert uuid1 != uuid2

    def test_generate_raw_file_uuid_with_sha256_hash(self):
        """Test that generate_raw_file_uuid handles SHA256 hash strings."""
        dataset_id = uuid.uuid4()
        # Typical SHA256 hash format
        file_hash = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        result = generate_raw_file_uuid(dataset_id, file_hash)
        assert isinstance(result, uuid.UUID)


class TestExpertAnnotations:
    """Tests for expert annotation UUID generation functions."""

    def test_generate_expert_annotation_uuid_returns_uuid(self):
        """Test that generate_expert_annotation_uuid returns a UUID."""
        expert_id = uuid.uuid4()
        result = generate_expert_annotation_uuid(
            expert_id=expert_id,
            annotation_task="grading",
            raw_data_id=None
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_expert_annotation_uuid_is_deterministic(self):
        """Test that generate_expert_annotation_uuid produces same UUID for same inputs."""
        expert_id = uuid.uuid4()
        task = "grading"
        
        uuid1 = generate_expert_annotation_uuid(expert_id, task, None)
        uuid2 = generate_expert_annotation_uuid(expert_id, task, None)
        
        assert uuid1 == uuid2

    def test_generate_expert_annotation_uuid_with_raw_data_id(self):
        """Test that generate_expert_annotation_uuid accepts optional raw_data_id."""
        expert_id = uuid.uuid4()
        raw_data_id = uuid.uuid4()
        
        result = generate_expert_annotation_uuid(
            expert_id=expert_id,
            annotation_task="grading",
            raw_data_id=raw_data_id
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_expert_annotation_uuid_with_value_hash(self):
        """Test that generate_expert_annotation_uuid accepts optional annotation_value_hash."""
        expert_id = uuid.uuid4()
        
        result = generate_expert_annotation_uuid(
            expert_id=expert_id,
            annotation_task="grading",
            raw_data_id=None,
            annotation_value_hash="hash123"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_expert_annotation_uuid_different_tasks_produce_different_uuids(self):
        """Test that different annotation tasks produce different UUIDs."""
        expert_id = uuid.uuid4()
        
        uuid1 = generate_expert_annotation_uuid(expert_id, "grading", None)
        uuid2 = generate_expert_annotation_uuid(expert_id, "segmentation", None)
        
        assert uuid1 != uuid2

    def test_generate_expert_annotation_uuid_raw_data_affects_uuid(self):
        """Test that raw_data_id affects UUID generation."""
        expert_id = uuid.uuid4()
        task = "grading"
        raw_data_id = uuid.uuid4()
        
        uuid1 = generate_expert_annotation_uuid(expert_id, task, None)
        uuid2 = generate_expert_annotation_uuid(expert_id, task, raw_data_id)
        
        assert uuid1 != uuid2


class TestAnnotationTypesAndScales:
    """Tests for annotation type and grading scale UUID generation functions."""

    def test_generate_annotation_type_uuid_returns_uuid(self):
        """Test that generate_annotation_type_uuid returns a UUID."""
        result = generate_annotation_type_uuid("drusen")
        assert isinstance(result, uuid.UUID)

    def test_generate_annotation_type_uuid_is_deterministic(self):
        """Test that generate_annotation_type_uuid produces same UUID for same type."""
        annotation_type = "drusen"
        
        uuid1 = generate_annotation_type_uuid(annotation_type)
        uuid2 = generate_annotation_type_uuid(annotation_type)
        
        assert uuid1 == uuid2

    def test_generate_annotation_type_uuid_different_types_produce_different_uuids(self):
        """Test that different annotation types produce different UUIDs."""
        uuid1 = generate_annotation_type_uuid("drusen")
        uuid2 = generate_annotation_type_uuid("hemorrhage")
        
        assert uuid1 != uuid2

    def test_generate_grading_scale_uuid_returns_uuid(self):
        """Test that generate_grading_scale_uuid returns a UUID."""
        result = generate_grading_scale_uuid("ICDR", "DR")
        assert isinstance(result, uuid.UUID)

    def test_generate_grading_scale_uuid_is_deterministic(self):
        """Test that generate_grading_scale_uuid produces same UUID for same inputs."""
        scale_name = "ICDR"
        disease_type = "DR"
        
        uuid1 = generate_grading_scale_uuid(scale_name, disease_type)
        uuid2 = generate_grading_scale_uuid(scale_name, disease_type)
        
        assert uuid1 == uuid2

    def test_generate_grading_scale_uuid_different_scales_produce_different_uuids(self):
        """Test that different scale names produce different UUIDs."""
        disease_type = "DR"
        
        uuid1 = generate_grading_scale_uuid("ICDR", disease_type)
        uuid2 = generate_grading_scale_uuid("ETDRS", disease_type)
        
        assert uuid1 != uuid2

    def test_generate_grading_scale_uuid_different_diseases_produce_different_uuids(self):
        """Test that different disease types produce different UUIDs."""
        scale_name = "ICDR"
        
        uuid1 = generate_grading_scale_uuid(scale_name, "DR")
        uuid2 = generate_grading_scale_uuid(scale_name, "DME")
        
        assert uuid1 != uuid2

    def test_generate_grading_scale_mapping_uuid_returns_uuid(self):
        """Test that generate_grading_scale_mapping_uuid returns a UUID."""
        source_scale_id = uuid.uuid4()
        target_scale_id = uuid.uuid4()
        result = generate_grading_scale_mapping_uuid(
            source_scale_id,
            target_scale_id,
            "grade_2"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_grading_scale_mapping_uuid_is_deterministic(self):
        """Test that generate_grading_scale_mapping_uuid produces same UUID for same inputs."""
        source_scale_id = uuid.uuid4()
        target_scale_id = uuid.uuid4()
        source_value = "grade_2"
        
        uuid1 = generate_grading_scale_mapping_uuid(
            source_scale_id, target_scale_id, source_value
        )
        uuid2 = generate_grading_scale_mapping_uuid(
            source_scale_id, target_scale_id, source_value
        )
        
        assert uuid1 == uuid2

    def test_generate_grading_scale_mapping_uuid_different_values_produce_different_uuids(self):
        """Test that different source values produce different UUIDs."""
        source_scale_id = uuid.uuid4()
        target_scale_id = uuid.uuid4()
        
        uuid1 = generate_grading_scale_mapping_uuid(
            source_scale_id, target_scale_id, "grade_1"
        )
        uuid2 = generate_grading_scale_mapping_uuid(
            source_scale_id, target_scale_id, "grade_2"
        )
        
        assert uuid1 != uuid2


class TestConsensusAnnotations:
    """Tests for consensus annotation UUID generation functions."""

    def test_generate_consensus_uuid_returns_uuid(self):
        """Test that generate_consensus_uuid returns a UUID."""
        image_id = uuid.uuid4()
        expert_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
        
        result = generate_consensus_uuid(
            image_id=image_id,
            annotation_task="grading",
            consensus_method="majority_vote",
            expert_annotation_ids=expert_ids
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_consensus_uuid_is_deterministic(self):
        """Test that generate_consensus_uuid produces same UUID for same inputs."""
        image_id = uuid.uuid4()
        task = "grading"
        method = "majority_vote"
        expert_ids = [uuid.uuid4(), uuid.uuid4(), uuid.uuid4()]
        
        uuid1 = generate_consensus_uuid(image_id, task, method, expert_ids)
        uuid2 = generate_consensus_uuid(image_id, task, method, expert_ids)
        
        assert uuid1 == uuid2

    def test_generate_consensus_uuid_sorts_expert_ids(self):
        """Test that generate_consensus_uuid produces same UUID regardless of expert ID order."""
        image_id = uuid.uuid4()
        task = "grading"
        method = "majority_vote"
        
        expert_id1 = uuid.uuid4()
        expert_id2 = uuid.uuid4()
        expert_id3 = uuid.uuid4()
        
        # Different order, should produce same UUID
        uuid1 = generate_consensus_uuid(
            image_id, task, method, [expert_id1, expert_id2, expert_id3]
        )
        uuid2 = generate_consensus_uuid(
            image_id, task, method, [expert_id3, expert_id1, expert_id2]
        )
        
        assert uuid1 == uuid2

    def test_generate_consensus_uuid_different_methods_produce_different_uuids(self):
        """Test that different consensus methods produce different UUIDs."""
        image_id = uuid.uuid4()
        task = "grading"
        expert_ids = [uuid.uuid4(), uuid.uuid4()]
        
        uuid1 = generate_consensus_uuid(image_id, task, "majority_vote", expert_ids)
        uuid2 = generate_consensus_uuid(image_id, task, "mean", expert_ids)
        
        assert uuid1 != uuid2

    def test_generate_consensus_uuid_with_empty_expert_list(self):
        """Test that generate_consensus_uuid handles empty expert annotation list."""
        image_id = uuid.uuid4()
        
        result = generate_consensus_uuid(
            image_id=image_id,
            annotation_task="grading",
            consensus_method="majority_vote",
            expert_annotation_ids=[]
        )
        assert isinstance(result, uuid.UUID)


class TestProvenanceAndTransformations:
    """Tests for provenance and transformation UUID generation functions."""

    def test_generate_provenance_chain_uuid_returns_uuid(self):
        """Test that generate_provenance_chain_uuid returns a UUID."""
        annotation_ids = [uuid.uuid4(), uuid.uuid4()]
        
        result = generate_provenance_chain_uuid(
            unified_annotation_type="drusen_segmentation",
            source_type="original",
            root_source_raw_data_id=None,
            source_annotation_ids=annotation_ids
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_provenance_chain_uuid_is_deterministic(self):
        """Test that generate_provenance_chain_uuid produces same UUID for same inputs."""
        annotation_type = "drusen_segmentation"
        source_type = "original"
        annotation_ids = [uuid.uuid4(), uuid.uuid4()]
        
        uuid1 = generate_provenance_chain_uuid(
            annotation_type, source_type, None, annotation_ids
        )
        uuid2 = generate_provenance_chain_uuid(
            annotation_type, source_type, None, annotation_ids
        )
        
        assert uuid1 == uuid2

    def test_generate_provenance_chain_uuid_sorts_annotation_ids(self):
        """Test that generate_provenance_chain_uuid produces same UUID regardless of annotation ID order."""
        annotation_type = "drusen_segmentation"
        source_type = "original"
        
        id1 = uuid.uuid4()
        id2 = uuid.uuid4()
        id3 = uuid.uuid4()
        
        uuid1 = generate_provenance_chain_uuid(
            annotation_type, source_type, None, [id1, id2, id3]
        )
        uuid2 = generate_provenance_chain_uuid(
            annotation_type, source_type, None, [id3, id1, id2]
        )
        
        assert uuid1 == uuid2

    def test_generate_provenance_chain_uuid_with_root_source(self):
        """Test that generate_provenance_chain_uuid accepts optional root_source_raw_data_id."""
        annotation_type = "drusen_segmentation"
        source_type = "original"
        root_source_id = uuid.uuid4()
        annotation_ids = [uuid.uuid4()]
        
        result = generate_provenance_chain_uuid(
            annotation_type, source_type, root_source_id, annotation_ids
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_transformation_uuid_returns_uuid(self):
        """Test that generate_transformation_uuid returns a UUID."""
        result = generate_transformation_uuid("resize")
        assert isinstance(result, uuid.UUID)

    def test_generate_transformation_uuid_is_deterministic(self):
        """Test that generate_transformation_uuid produces same UUID for same inputs."""
        operation_type = "resize"
        
        uuid1 = generate_transformation_uuid(operation_type)
        uuid2 = generate_transformation_uuid(operation_type)
        
        assert uuid1 == uuid2

    def test_generate_transformation_uuid_with_hashes(self):
        """Test that generate_transformation_uuid accepts optional hash parameters."""
        operation_type = "resize"
        input_hash = "input_hash_123"
        params_hash = "params_hash_456"
        
        result = generate_transformation_uuid(
            operation_type=operation_type,
            input_data_hash=input_hash,
            operation_parameters_hash=params_hash
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_transformation_uuid_different_operations_produce_different_uuids(self):
        """Test that different operation types produce different UUIDs."""
        uuid1 = generate_transformation_uuid("resize")
        uuid2 = generate_transformation_uuid("crop")
        
        assert uuid1 != uuid2

    def test_generate_provenance_transformation_uuid_returns_uuid(self):
        """Test that generate_provenance_transformation_uuid returns a UUID."""
        chain_id = uuid.uuid4()
        transformation_id = uuid.uuid4()
        
        result = generate_provenance_transformation_uuid(chain_id, transformation_id)
        assert isinstance(result, uuid.UUID)

    def test_generate_provenance_transformation_uuid_is_deterministic(self):
        """Test that generate_provenance_transformation_uuid produces same UUID for same inputs."""
        chain_id = uuid.uuid4()
        transformation_id = uuid.uuid4()
        
        uuid1 = generate_provenance_transformation_uuid(chain_id, transformation_id)
        uuid2 = generate_provenance_transformation_uuid(chain_id, transformation_id)
        
        assert uuid1 == uuid2


class TestAnnotationTables:
    """Tests for annotation table UUID generation functions."""

    def test_generate_segmentation_uuid_returns_uuid(self):
        """Test that generate_segmentation_uuid returns a UUID."""
        image_id = uuid.uuid4()
        annotation_type_id = uuid.uuid4()
        
        result = generate_segmentation_uuid(
            image_id=image_id,
            annotation_type_id=annotation_type_id
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_segmentation_uuid_is_deterministic(self):
        """Test that generate_segmentation_uuid produces same UUID for same inputs."""
        image_id = uuid.uuid4()
        annotation_type_id = uuid.uuid4()
        
        uuid1 = generate_segmentation_uuid(image_id, annotation_type_id)
        uuid2 = generate_segmentation_uuid(image_id, annotation_type_id)
        
        assert uuid1 == uuid2

    def test_generate_segmentation_uuid_with_optional_parameters(self):
        """Test that generate_segmentation_uuid accepts all optional parameters."""
        image_id = uuid.uuid4()
        annotation_type_id = uuid.uuid4()
        expert_annotation_id = uuid.uuid4()
        consensus_id = uuid.uuid4()
        raw_data_id = uuid.uuid4()
        
        result = generate_segmentation_uuid(
            image_id=image_id,
            annotation_type_id=annotation_type_id,
            expert_annotation_id=expert_annotation_id,
            consensus_id=consensus_id,
            raw_data_id=raw_data_id
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_disease_grading_uuid_returns_uuid(self):
        """Test that generate_disease_grading_uuid returns a UUID."""
        image_id = uuid.uuid4()
        scale_id = uuid.uuid4()
        
        result = generate_disease_grading_uuid(
            image_id=image_id,
            disease_type="DR",
            scale_id=scale_id
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_disease_grading_uuid_is_deterministic(self):
        """Test that generate_disease_grading_uuid produces same UUID for same inputs."""
        image_id = uuid.uuid4()
        disease_type = "DR"
        scale_id = uuid.uuid4()
        
        uuid1 = generate_disease_grading_uuid(image_id, disease_type, scale_id)
        uuid2 = generate_disease_grading_uuid(image_id, disease_type, scale_id)
        
        assert uuid1 == uuid2

    def test_generate_disease_grading_uuid_with_optional_parameters(self):
        """Test that generate_disease_grading_uuid accepts all optional parameters."""
        image_id = uuid.uuid4()
        scale_id = uuid.uuid4()
        expert_annotation_id = uuid.uuid4()
        consensus_id = uuid.uuid4()
        raw_data_id = uuid.uuid4()
        
        result = generate_disease_grading_uuid(
            image_id=image_id,
            disease_type="DR",
            scale_id=scale_id,
            expert_annotation_id=expert_annotation_id,
            consensus_id=consensus_id,
            raw_data_id=raw_data_id,
            original_grade="grade_2"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_disease_grading_uuid_different_diseases_produce_different_uuids(self):
        """Test that different disease types produce different UUIDs."""
        image_id = uuid.uuid4()
        scale_id = uuid.uuid4()
        
        uuid1 = generate_disease_grading_uuid(image_id, "DR", scale_id)
        uuid2 = generate_disease_grading_uuid(image_id, "DME", scale_id)
        
        assert uuid1 != uuid2

    def test_generate_localization_uuid_returns_uuid(self):
        """Test that generate_localization_uuid returns a UUID."""
        image_id = uuid.uuid4()
        
        result = generate_localization_uuid(
            image_id=image_id,
            localization_type="bounding_box",
            target_structure="optic_disc"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_localization_uuid_is_deterministic(self):
        """Test that generate_localization_uuid produces same UUID for same inputs."""
        image_id = uuid.uuid4()
        loc_type = "bounding_box"
        target = "optic_disc"
        
        uuid1 = generate_localization_uuid(image_id, loc_type, target)
        uuid2 = generate_localization_uuid(image_id, loc_type, target)
        
        assert uuid1 == uuid2

    def test_generate_localization_uuid_with_optional_parameters(self):
        """Test that generate_localization_uuid accepts all optional parameters."""
        image_id = uuid.uuid4()
        expert_annotation_id = uuid.uuid4()
        consensus_id = uuid.uuid4()
        raw_data_id = uuid.uuid4()
        
        result = generate_localization_uuid(
            image_id=image_id,
            localization_type="keypoint",
            target_structure="fovea",
            expert_annotation_id=expert_annotation_id,
            consensus_id=consensus_id,
            raw_data_id=raw_data_id,
            coordinates_hash="coord_hash_123"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_classification_uuid_returns_uuid(self):
        """Test that generate_classification_uuid returns a UUID."""
        image_id = uuid.uuid4()
        
        result = generate_classification_uuid(
            image_id=image_id,
            task_type="binary",
            class_name="referable"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_classification_uuid_is_deterministic(self):
        """Test that generate_classification_uuid produces same UUID for same inputs."""
        image_id = uuid.uuid4()
        task_type = "binary"
        class_name = "referable"
        
        uuid1 = generate_classification_uuid(image_id, task_type, class_name)
        uuid2 = generate_classification_uuid(image_id, task_type, class_name)
        
        assert uuid1 == uuid2

    def test_generate_classification_uuid_with_optional_parameters(self):
        """Test that generate_classification_uuid accepts all optional parameters."""
        image_id = uuid.uuid4()
        expert_annotation_id = uuid.uuid4()
        consensus_id = uuid.uuid4()
        raw_data_id = uuid.uuid4()
        
        result = generate_classification_uuid(
            image_id=image_id,
            task_type="multi_class",
            class_name="severity",
            expert_annotation_id=expert_annotation_id,
            consensus_id=consensus_id,
            raw_data_id=raw_data_id,
            class_value_hash="value_hash_123"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_quality_uuid_returns_uuid(self):
        """Test that generate_quality_uuid returns a UUID."""
        image_id = uuid.uuid4()
        
        result = generate_quality_uuid(
            image_id=image_id,
            quality_type="overall"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_quality_uuid_is_deterministic(self):
        """Test that generate_quality_uuid produces same UUID for same inputs."""
        image_id = uuid.uuid4()
        quality_type = "gradability"
        
        uuid1 = generate_quality_uuid(image_id, quality_type)
        uuid2 = generate_quality_uuid(image_id, quality_type)
        
        assert uuid1 == uuid2

    def test_generate_quality_uuid_with_optional_parameters(self):
        """Test that generate_quality_uuid accepts all optional parameters."""
        image_id = uuid.uuid4()
        expert_annotation_id = uuid.uuid4()
        raw_data_id = uuid.uuid4()
        
        result = generate_quality_uuid(
            image_id=image_id,
            quality_type="overall",
            expert_annotation_id=expert_annotation_id,
            raw_data_id=raw_data_id,
            quality_score=0.95
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_quality_uuid_with_quality_score_zero(self):
        """Test that generate_quality_uuid handles quality_score of 0.0."""
        image_id = uuid.uuid4()
        
        # Test that 0.0 is treated as a valid score (not None)
        result = generate_quality_uuid(
            image_id=image_id,
            quality_type="overall",
            quality_score=0.0
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_description_uuid_returns_uuid(self):
        """Test that generate_description_uuid returns a UUID."""
        image_id = uuid.uuid4()
        
        result = generate_description_uuid(
            image_id=image_id,
            description_type="clinical_caption"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_description_uuid_is_deterministic(self):
        """Test that generate_description_uuid produces same UUID for same inputs."""
        image_id = uuid.uuid4()
        desc_type = "diagnosis_text"
        
        uuid1 = generate_description_uuid(image_id, desc_type)
        uuid2 = generate_description_uuid(image_id, desc_type)
        
        assert uuid1 == uuid2

    def test_generate_description_uuid_with_optional_parameters(self):
        """Test that generate_description_uuid accepts all optional parameters."""
        image_id = uuid.uuid4()
        expert_id = uuid.uuid4()
        raw_data_id = uuid.uuid4()
        
        result = generate_description_uuid(
            image_id=image_id,
            description_type="notes",
            expert_id=expert_id,
            raw_data_id=raw_data_id,
            description_hash="desc_hash_123"
        )
        assert isinstance(result, uuid.UUID)


class TestKeywords:
    """Tests for keyword UUID generation functions."""

    def test_generate_keyword_uuid_returns_uuid(self):
        """Test that generate_keyword_uuid returns a UUID."""
        dataset_id = uuid.uuid4()
        
        result = generate_keyword_uuid(
            dataset_id=dataset_id,
            keyword_term="hemorrhage",
            keyword_source="diagnostic_keywords"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_keyword_uuid_is_deterministic(self):
        """Test that generate_keyword_uuid produces same UUID for same inputs."""
        dataset_id = uuid.uuid4()
        term = "hemorrhage"
        source = "diagnostic_keywords"
        
        uuid1 = generate_keyword_uuid(dataset_id, term, source)
        uuid2 = generate_keyword_uuid(dataset_id, term, source)
        
        assert uuid1 == uuid2

    def test_generate_keyword_uuid_different_terms_produce_different_uuids(self):
        """Test that different keyword terms produce different UUIDs."""
        dataset_id = uuid.uuid4()
        source = "diagnostic_keywords"
        
        uuid1 = generate_keyword_uuid(dataset_id, "hemorrhage", source)
        uuid2 = generate_keyword_uuid(dataset_id, "exudate", source)
        
        assert uuid1 != uuid2

    def test_generate_keyword_uuid_different_sources_produce_different_uuids(self):
        """Test that different keyword sources produce different UUIDs."""
        dataset_id = uuid.uuid4()
        term = "hemorrhage"
        
        uuid1 = generate_keyword_uuid(dataset_id, term, "diagnostic_keywords")
        uuid2 = generate_keyword_uuid(dataset_id, term, "clinical_description")
        
        assert uuid1 != uuid2

    def test_generate_keyword_annotation_uuid_returns_uuid(self):
        """Test that generate_keyword_annotation_uuid returns a UUID."""
        image_id = uuid.uuid4()
        keyword_id = uuid.uuid4()
        
        result = generate_keyword_annotation_uuid(
            image_id=image_id,
            keyword_id=keyword_id
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_keyword_annotation_uuid_is_deterministic(self):
        """Test that generate_keyword_annotation_uuid produces same UUID for same inputs."""
        image_id = uuid.uuid4()
        keyword_id = uuid.uuid4()
        
        uuid1 = generate_keyword_annotation_uuid(image_id, keyword_id)
        uuid2 = generate_keyword_annotation_uuid(image_id, keyword_id)
        
        assert uuid1 == uuid2

    def test_generate_keyword_annotation_uuid_with_optional_parameters(self):
        """Test that generate_keyword_annotation_uuid accepts optional parameters."""
        image_id = uuid.uuid4()
        keyword_id = uuid.uuid4()
        expert_id = uuid.uuid4()
        raw_data_id = uuid.uuid4()
        
        result = generate_keyword_annotation_uuid(
            image_id=image_id,
            keyword_id=keyword_id,
            expert_id=expert_id,
            raw_data_id=raw_data_id
        )
        assert isinstance(result, uuid.UUID)


class TestDatasetSplits:
    """Tests for dataset split UUID generation functions."""

    def test_generate_dataset_split_uuid_returns_uuid(self):
        """Test that generate_dataset_split_uuid returns a UUID."""
        dataset_id = uuid.uuid4()
        
        result = generate_dataset_split_uuid(
            dataset_id=dataset_id,
            split_name="train",
            split_type="explicit"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_dataset_split_uuid_is_deterministic(self):
        """Test that generate_dataset_split_uuid produces same UUID for same inputs."""
        dataset_id = uuid.uuid4()
        split_name = "train"
        split_type = "explicit"
        
        uuid1 = generate_dataset_split_uuid(dataset_id, split_name, split_type)
        uuid2 = generate_dataset_split_uuid(dataset_id, split_name, split_type)
        
        assert uuid1 == uuid2

    def test_generate_dataset_split_uuid_with_task_type(self):
        """Test that generate_dataset_split_uuid accepts optional task_type."""
        dataset_id = uuid.uuid4()
        
        result = generate_dataset_split_uuid(
            dataset_id=dataset_id,
            split_name="val",
            split_type="metadata_defined",
            task_type="grading"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_dataset_split_uuid_different_splits_produce_different_uuids(self):
        """Test that different split names produce different UUIDs."""
        dataset_id = uuid.uuid4()
        split_type = "explicit"
        
        uuid1 = generate_dataset_split_uuid(dataset_id, "train", split_type)
        uuid2 = generate_dataset_split_uuid(dataset_id, "test", split_type)
        
        assert uuid1 != uuid2

    def test_generate_image_split_uuid_returns_uuid(self):
        """Test that generate_image_split_uuid returns a UUID."""
        image_id = uuid.uuid4()
        split_id = uuid.uuid4()
        
        result = generate_image_split_uuid(
            image_id=image_id,
            split_id=split_id
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_image_split_uuid_is_deterministic(self):
        """Test that generate_image_split_uuid produces same UUID for same inputs."""
        image_id = uuid.uuid4()
        split_id = uuid.uuid4()
        
        uuid1 = generate_image_split_uuid(image_id, split_id)
        uuid2 = generate_image_split_uuid(image_id, split_id)
        
        assert uuid1 == uuid2

    def test_generate_image_split_uuid_with_task_type(self):
        """Test that generate_image_split_uuid accepts optional task_type."""
        image_id = uuid.uuid4()
        split_id = uuid.uuid4()
        
        result = generate_image_split_uuid(
            image_id=image_id,
            split_id=split_id,
            task_type="segmentation"
        )
        assert isinstance(result, uuid.UUID)

    def test_generate_image_split_uuid_different_images_produce_different_uuids(self):
        """Test that different image IDs produce different UUIDs."""
        split_id = uuid.uuid4()
        
        uuid1 = generate_image_split_uuid(uuid.uuid4(), split_id)
        uuid2 = generate_image_split_uuid(uuid.uuid4(), split_id)
        
        assert uuid1 != uuid2
