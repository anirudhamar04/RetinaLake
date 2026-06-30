"""
Tests for annotation type-specific database operations.

Tests based on docstring specifications only.
"""

import pytest
from datetime import datetime
from uuid import UUID

from chaksudb.db.queries.annotation_types import (
    upsert_annotation_type,
    upsert_segmentation_annotation,
    upsert_localization_annotation,
    upsert_classification_annotation,
    upsert_quality_annotation,
    upsert_clinical_description,
    upsert_keyword_vocabulary,
    upsert_keyword_annotation,
)
from chaksudb.db.queries.datasets import upsert_dataset
from chaksudb.db.queries.images import upsert_image
from chaksudb.db.models import (
    Dataset,
    Image,
    AnnotationType,
    SegmentationAnnotation,
    LocalizationAnnotation,
    ClassificationAnnotation,
    QualityAnnotation,
    ClinicalDescription,
    KeywordVocabulary,
    KeywordAnnotation,
)


@pytest.mark.asyncio
async def test_upsert_annotation_type_creates_new_record(db_connection, test_uuids):
    """Test that upsert_annotation_type creates a new annotation type record.
    
    Based on docstring: 'Upsert an annotation type record.'
    """
    annotation_type_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    annotation_type = AnnotationType(
        annotation_type_id=annotation_type_id,
        annotation_type="microaneurysm",
        annotation_description="Small red dots indicating early DR",
    )
    
    await upsert_annotation_type(annotation_type)
    
    # Verify annotation type was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT annotation_type, annotation_description FROM annotation_type WHERE annotation_type_id = %s",
            (annotation_type_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "microaneurysm"
    assert result[1] == "Small red dots indicating early DR"


@pytest.mark.asyncio
async def test_upsert_segmentation_annotation_creates_new_record(db_connection, test_uuids):
    """Test that upsert_segmentation_annotation creates a new segmentation annotation record.
    
    Based on docstring: 'Upsert a segmentation annotation record.'
    """
    # Setup dataset, image, and annotation type
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    image = Image(
        image_id=test_uuids["image_1"],
        dataset_id=test_uuids["dataset_1"],
        original_image_id="IMG_001",
        storage_provider="local",
        file_path="/data/images/img001.jpg",
        file_format="jpg",
        modality="fundus",
        created_at=datetime.now(),
    )
    await upsert_image(image)
    
    annotation_type_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    annotation_type = AnnotationType(
        annotation_type_id=annotation_type_id,
        annotation_type="optic_disc",
        annotation_description="Optic disc segmentation",
    )
    await upsert_annotation_type(annotation_type)
    
    # Create segmentation annotation
    segmentation_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    segmentation = SegmentationAnnotation(
        segmentation_id=segmentation_id,
        image_id=test_uuids["image_1"],
        annotation_type_id=annotation_type_id,
        lesion_subtype="optic_disc",
        mask_file_path="/data/masks/mask001.png",
        group_id=None,
        unified_format="binary_mask",
        original_format="png",
        original_file_path="/data/original/mask001.png",
        raw_data_id=None,
        coordinate_system="pixel",
        expert_annotation_id=None,
        consensus_id=None,
        annotation_method="manual",
        confidence_score=0.95,
        provenance_chain_id=None,
        created_at=datetime.now(),
    )
    
    await upsert_segmentation_annotation(segmentation)
    
    # Verify segmentation annotation was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT image_id, lesion_subtype, mask_file_path FROM segmentation_annotations WHERE segmentation_id = %s",
            (segmentation_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == test_uuids["image_1"]
    assert result[1] == "optic_disc"
    assert result[2] == "/data/masks/mask001.png"


@pytest.mark.asyncio
async def test_upsert_localization_annotation_creates_new_record(db_connection, test_uuids):
    """Test that upsert_localization_annotation creates a new localization annotation record.
    
    Based on docstring: 'Upsert a localization annotation record.'
    """
    # Setup dataset and image
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    image = Image(
        image_id=test_uuids["image_1"],
        dataset_id=test_uuids["dataset_1"],
        original_image_id="IMG_001",
        storage_provider="local",
        file_path="/data/images/img001.jpg",
        file_format="jpg",
        modality="fundus",
        created_at=datetime.now(),
    )
    await upsert_image(image)
    
    # Create localization annotation
    localization_id = UUID("cccccccc-cccc-cccc-cccc-cccccccccccc")
    localization = LocalizationAnnotation(
        localization_id=localization_id,
        image_id=test_uuids["image_1"],
        localization_type="bounding_box",
        target_structure="microaneurysm",
        coordinates={"x1": 100, "y1": 150, "x2": 200, "y2": 250},
        lesion_subtype="small_red_dot",
        raw_data_id=None,
        expert_annotation_id=None,
        consensus_id=None,
        annotation_method="manual",
        provenance_chain_id=None,
        created_at=datetime.now(),
    )
    
    await upsert_localization_annotation(localization)
    
    # Verify localization annotation was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT image_id, localization_type, target_structure FROM localization_annotations WHERE localization_id = %s",
            (localization_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == test_uuids["image_1"]
    assert result[1] == "bounding_box"
    assert result[2] == "microaneurysm"


@pytest.mark.asyncio
async def test_upsert_classification_annotation_creates_new_record(db_connection, test_uuids):
    """Test that upsert_classification_annotation creates a new classification annotation record.
    
    Based on docstring: 'Upsert a classification annotation record.'
    """
    # Setup dataset and image
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    image = Image(
        image_id=test_uuids["image_1"],
        dataset_id=test_uuids["dataset_1"],
        original_image_id="IMG_001",
        storage_provider="local",
        file_path="/data/images/img001.jpg",
        file_format="jpg",
        modality="fundus",
        created_at=datetime.now(),
    )
    await upsert_image(image)
    
    # Create classification annotation
    classification_id = UUID("dddddddd-dddd-dddd-dddd-dddddddddddd")
    classification = ClassificationAnnotation(
        classification_id=classification_id,
        image_id=test_uuids["image_1"],
        task_type="binary",
        task_name="referable_dr",
        class_name="referable_dr",
        concept="DR",
        is_multilabel=False,
        class_index=1,
        class_label="positive",
        sub_key=None,
        class_value={"referable_dr": True},
        raw_data_id=None,
        expert_annotation_id=None,
        consensus_id=None,
        annotation_method="manual",
        confidence_score=0.9,
        provenance_chain_id=None,
        created_at=datetime.now(),
    )
    
    await upsert_classification_annotation(classification)
    
    # Verify classification annotation was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT image_id, task_type, class_name FROM classification_annotations WHERE classification_id = %s",
            (classification_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == test_uuids["image_1"]
    assert result[1] == "binary"
    assert result[2] == "referable_dr"


@pytest.mark.asyncio
async def test_upsert_quality_annotation_creates_new_record(db_connection, test_uuids):
    """Test that upsert_quality_annotation creates a new quality annotation record.
    
    Based on docstring: 'Upsert a quality annotation record.'
    """
    # Setup dataset and image
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    image = Image(
        image_id=test_uuids["image_1"],
        dataset_id=test_uuids["dataset_1"],
        original_image_id="IMG_001",
        storage_provider="local",
        file_path="/data/images/img001.jpg",
        file_format="jpg",
        modality="fundus",
        created_at=datetime.now(),
    )
    await upsert_image(image)
    
    # Create quality annotation
    quality_id = UUID("eeeeeeee-eeee-eeee-eeee-eeeeeeeeeeee")
    quality = QualityAnnotation(
        quality_id=quality_id,
        image_id=test_uuids["image_1"],
        quality_type="gradability",
        quality_score=0.85,
        quality_label="good",
        scale_description="0-1 scale, 1 being best",
        raw_data_id=None,
        expert_annotation_id=None,
        provenance_chain_id=None,
        created_at=datetime.now(),
    )
    
    await upsert_quality_annotation(quality)
    
    # Verify quality annotation was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT image_id, quality_type, quality_score, quality_label FROM quality_annotations WHERE quality_id = %s",
            (quality_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == test_uuids["image_1"]
    assert result[1] == "gradability"
    assert result[2] == 0.85
    assert result[3] == "good"


@pytest.mark.asyncio
async def test_upsert_clinical_description_creates_new_record(db_connection, test_uuids):
    """Test that upsert_clinical_description creates a new clinical description record.
    
    Based on docstring: 'Upsert a clinical description record.'
    """
    # Setup dataset and image
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    image = Image(
        image_id=test_uuids["image_1"],
        dataset_id=test_uuids["dataset_1"],
        original_image_id="IMG_001",
        storage_provider="local",
        file_path="/data/images/img001.jpg",
        file_format="jpg",
        modality="fundus",
        created_at=datetime.now(),
    )
    await upsert_image(image)
    
    # Create clinical description
    description_id = UUID("ffffffff-ffff-ffff-ffff-ffffffffffff")
    description = ClinicalDescription(
        description_id=description_id,
        image_id=test_uuids["image_1"],
        description_text="Fundus photograph shows multiple microaneurysms and dot-blot hemorrhages consistent with moderate NPDR.",
        description_type="clinical_caption",
        raw_data_id=None,
        expert_id=None,
        word_count=15,
        created_at=datetime.now(),
    )
    
    await upsert_clinical_description(description)
    
    # Verify clinical description was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT image_id, description_type, word_count FROM clinical_descriptions WHERE description_id = %s",
            (description_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == test_uuids["image_1"]
    assert result[1] == "clinical_caption"
    assert result[2] == 15


@pytest.mark.asyncio
async def test_upsert_keyword_vocabulary_creates_new_record(db_connection, test_uuids):
    """Test that upsert_keyword_vocabulary creates a new keyword vocabulary record.
    
    Based on docstring: 'Upsert a keyword vocabulary record.'
    """
    # Setup dataset
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    # Create keyword vocabulary
    keyword_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    keyword = KeywordVocabulary(
        keyword_id=keyword_id,
        keyword_term="microaneurysm",
        keyword_source="diagnostic_keywords",
        category="lesion",
        dataset_id=test_uuids["dataset_1"],
        created_at=datetime.now(),
    )
    
    await upsert_keyword_vocabulary(keyword)
    
    # Verify keyword vocabulary was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT keyword_term, keyword_source, category FROM keyword_vocabulary WHERE keyword_id = %s",
            (keyword_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "microaneurysm"
    assert result[1] == "diagnostic_keywords"
    assert result[2] == "lesion"


@pytest.mark.asyncio
async def test_upsert_keyword_annotation_creates_new_record(db_connection, test_uuids):
    """Test that upsert_keyword_annotation creates a new keyword annotation record.
    
    Based on docstring: 'Upsert a keyword annotation record.'
    """
    # Setup dataset, image, and keyword vocabulary
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    image = Image(
        image_id=test_uuids["image_1"],
        dataset_id=test_uuids["dataset_1"],
        original_image_id="IMG_001",
        storage_provider="local",
        file_path="/data/images/img001.jpg",
        file_format="jpg",
        modality="fundus",
        created_at=datetime.now(),
    )
    await upsert_image(image)
    
    keyword_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    keyword = KeywordVocabulary(
        keyword_id=keyword_id,
        keyword_term="microaneurysm",
        keyword_source="diagnostic_keywords",
        category="lesion",
        dataset_id=test_uuids["dataset_1"],
        created_at=datetime.now(),
    )
    await upsert_keyword_vocabulary(keyword)
    
    # Create keyword annotation
    keyword_annotation_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    keyword_annotation = KeywordAnnotation(
        keyword_annotation_id=keyword_annotation_id,
        image_id=test_uuids["image_1"],
        keyword_id=keyword_id,
        keyword_text="microaneurysm",
        raw_data_id=None,
        expert_id=None,
        annotation_method="manual",
        provenance_chain_id=None,
        created_at=datetime.now(),
    )
    
    await upsert_keyword_annotation(keyword_annotation)
    
    # Verify keyword annotation was created
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT image_id, keyword_id, keyword_text FROM keyword_annotations WHERE keyword_annotation_id = %s",
            (keyword_annotation_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == test_uuids["image_1"]
    assert result[1] == keyword_id
    assert result[2] == "microaneurysm"


@pytest.mark.asyncio
async def test_upsert_annotation_type_updates_existing_record(db_connection, test_uuids):
    """Test that upsert_annotation_type updates an existing annotation type record.
    
    Based on docstring: 'Upsert an annotation type record.' (upsert implies insert or update)
    """
    annotation_type_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    
    # Create initial annotation type
    annotation_type_v1 = AnnotationType(
        annotation_type_id=annotation_type_id,
        annotation_type="microaneurysm",
        annotation_description="Initial description",
    )
    await upsert_annotation_type(annotation_type_v1)
    
    # Update the annotation type
    annotation_type_v2 = AnnotationType(
        annotation_type_id=annotation_type_id,
        annotation_type="microaneurysm",
        annotation_description="Updated description with more details",
    )
    await upsert_annotation_type(annotation_type_v2)
    
    # Verify annotation type was updated
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT annotation_description FROM annotation_type WHERE annotation_type_id = %s",
            (annotation_type_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "Updated description with more details"


@pytest.mark.asyncio
async def test_upsert_segmentation_annotation_updates_existing_record(db_connection, test_uuids):
    """Test that upsert_segmentation_annotation updates an existing segmentation annotation record.
    
    Based on docstring: 'Upsert a segmentation annotation record.' (upsert implies insert or update)
    """
    # Setup dataset, image, and annotation type
    dataset = Dataset(
        dataset_id=test_uuids["dataset_1"],
        dataset_name="Test Dataset",
        source_url="https://example.com",
        license="MIT",
        modality_types=["fundus"],
        created_at=datetime.now(),
    )
    await upsert_dataset(dataset)
    
    image = Image(
        image_id=test_uuids["image_1"],
        dataset_id=test_uuids["dataset_1"],
        original_image_id="IMG_001",
        storage_provider="local",
        file_path="/data/images/img001.jpg",
        file_format="jpg",
        modality="fundus",
        created_at=datetime.now(),
    )
    await upsert_image(image)
    
    annotation_type_id = UUID("aaaaaaaa-aaaa-aaaa-aaaa-aaaaaaaaaaaa")
    annotation_type = AnnotationType(
        annotation_type_id=annotation_type_id,
        annotation_type="optic_disc",
        annotation_description="Optic disc segmentation",
    )
    await upsert_annotation_type(annotation_type)
    
    segmentation_id = UUID("bbbbbbbb-bbbb-bbbb-bbbb-bbbbbbbbbbbb")
    
    # Create initial segmentation annotation
    segmentation_v1 = SegmentationAnnotation(
        segmentation_id=segmentation_id,
        image_id=test_uuids["image_1"],
        annotation_type_id=annotation_type_id,
        lesion_subtype="optic_disc",
        mask_file_path="/data/masks/mask001_v1.png",
        unified_format="binary_mask",
        annotation_method="manual",
        confidence_score=0.8,
        created_at=datetime.now(),
    )
    await upsert_segmentation_annotation(segmentation_v1)
    
    # Update the segmentation annotation
    segmentation_v2 = SegmentationAnnotation(
        segmentation_id=segmentation_id,
        image_id=test_uuids["image_1"],
        annotation_type_id=annotation_type_id,
        lesion_subtype="optic_disc",
        mask_file_path="/data/masks/mask001_v2.png",  # Updated
        unified_format="binary_mask",
        annotation_method="manual",
        confidence_score=0.95,  # Updated
        created_at=datetime.now(),
    )
    await upsert_segmentation_annotation(segmentation_v2)
    
    # Verify segmentation annotation was updated
    async with db_connection.cursor() as cur:
        await cur.execute(
            "SELECT mask_file_path, confidence_score FROM segmentation_annotations WHERE segmentation_id = %s",
            (segmentation_id,)
        )
        result = await cur.fetchone()
        
    assert result is not None
    assert result[0] == "/data/masks/mask001_v2.png"
    assert result[1] == 0.95
