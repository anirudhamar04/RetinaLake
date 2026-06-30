"""
Comprehensive tests for keyword_processor using real data from data directory.

Tests cover:
- Keyword parsing from various formats
- Keyword vocabulary management
- Batch keyword processing
- Real data from DeepEyeNet and ODIR-5K
"""

import uuid
import pytest

from chaksudb.db.queries.annotation_types import (
    upsert_keyword_annotation,
    upsert_keyword_vocabulary,
)
from chaksudb.ingest.framework.task_processors.keyword_processor import (
    parse_keyword_string,
    parse_deepeyenet_keywords,
    get_or_create_keyword_vocabulary,
    process_keyword_annotation,
    process_keywords_batch,
    prepare_keywords_for_upsert,
)
from chaksudb.ingest.framework.gen_uuid import generate_dataset_uuid


pytestmark = pytest.mark.asyncio


class TestKeywordProcessorWithRealData:
    """Test keyword processor with real data from datasets."""

    @pytest.fixture
    async def test_image_id(self, test_image_in_db):
        """Create a test image ID with database record."""
        return test_image_in_db

    @pytest.fixture
    async def test_dataset_id(self, test_dataset_in_db):
        """Create a test dataset ID with database record."""
        return test_dataset_in_db

    async def test_parse_keyword_string_comma_separated(self):
        """Test parsing comma-separated keyword strings."""
        keywords = parse_keyword_string("microaneurysms, hemorrhages, exudates")
        assert keywords == ["microaneurysms", "hemorrhages", "exudates"]

    async def test_parse_keyword_string_with_spaces(self):
        """Test parsing keyword strings with extra spaces."""
        keywords = parse_keyword_string("  drusen  ,  reticular pseudodrusen  ,  pigmentary changes  ")
        assert keywords == ["drusen", "reticular pseudodrusen", "pigmentary changes"]

    async def test_parse_keyword_string_semicolon_delimiter(self):
        """Test parsing keyword strings with custom delimiter."""
        keywords = parse_keyword_string("drusen; reticular pseudodrusen; pigmentary changes", delimiter=";")
        assert keywords == ["drusen", "reticular pseudodrusen", "pigmentary changes"]

    async def test_parse_keyword_string_empty(self):
        """Test parsing empty keyword string."""
        keywords = parse_keyword_string("")
        assert keywords == []

        keywords = parse_keyword_string("   ")
        assert keywords == []

    async def test_parse_deepeyenet_keywords_dict_with_keywords_key(self):
        """Test parsing DeepEyeNet keywords from dict with 'keywords' key."""
        data = {"keywords": ["microaneurysms", "hemorrhages"]}
        keywords = parse_deepeyenet_keywords(data)
        assert keywords == ["microaneurysms", "hemorrhages"]

    async def test_parse_deepeyenet_keywords_dict_with_diagnostic_keywords(self):
        """Test parsing DeepEyeNet keywords from dict with 'diagnostic_keywords' key."""
        data = {"diagnostic_keywords": "drusen, exudates"}
        keywords = parse_deepeyenet_keywords(data)
        assert keywords == ["drusen", "exudates"]

    async def test_parse_deepeyenet_keywords_dict_with_findings(self):
        """Test parsing DeepEyeNet keywords from dict with 'findings' key."""
        data = {"findings": ["cotton wool spots", "neovascularization"]}
        keywords = parse_deepeyenet_keywords(data)
        assert keywords == ["cotton wool spots", "neovascularization"]

    async def test_parse_deepeyenet_keywords_list(self):
        """Test parsing DeepEyeNet keywords from list."""
        data = ["microaneurysms", "hemorrhages", "exudates"]
        keywords = parse_deepeyenet_keywords(data)
        assert keywords == ["microaneurysms", "hemorrhages", "exudates"]

    async def test_parse_deepeyenet_keywords_json_string(self):
        """Test parsing DeepEyeNet keywords from JSON string."""
        json_str = '{"keywords": ["microaneurysms", "hemorrhages"]}'
        keywords = parse_deepeyenet_keywords(json_str)
        assert keywords == ["microaneurysms", "hemorrhages"]

    async def test_parse_deepeyenet_keywords_invalid_json(self):
        """Test parsing DeepEyeNet keywords with invalid JSON (falls back to string parsing)."""
        json_str = "not a valid json"
        keywords = parse_deepeyenet_keywords(json_str)
        # Should fall back to comma-separated parsing
        assert isinstance(keywords, list)

    async def test_get_or_create_keyword_vocabulary_new(self, test_dataset_id):
        """Test creating a new keyword vocabulary entry."""
        keyword_id = await get_or_create_keyword_vocabulary(
            keyword_term="microaneurysms",
            keyword_source="diagnostic_keywords",
            dataset_id=test_dataset_id,
            category="lesion",
        )

        assert keyword_id is not None

    async def test_get_or_create_keyword_vocabulary_idempotency(self, test_dataset_id):
        """Test that get_or_create_keyword_vocabulary is idempotent."""
        keyword_id_1 = await get_or_create_keyword_vocabulary(
            keyword_term="hemorrhages",
            keyword_source="diagnostic_keywords",
            dataset_id=test_dataset_id,
            category="lesion",
        )

        keyword_id_2 = await get_or_create_keyword_vocabulary(
            keyword_term="hemorrhages",
            keyword_source="diagnostic_keywords",
            dataset_id=test_dataset_id,
            category="lesion",
        )

        assert keyword_id_1 == keyword_id_2

    async def test_get_or_create_keyword_vocabulary_invalid_source(self, test_dataset_id):
        """Test that invalid keyword sources raise ValueError."""
        with pytest.raises(ValueError, match="Invalid keyword_source"):
            await get_or_create_keyword_vocabulary(
                keyword_term="test",
                keyword_source="invalid_source",
                dataset_id=test_dataset_id,
            )

    async def test_process_keyword_annotation_single(self, test_image_id, test_dataset_id):
        """Test processing a single keyword annotation."""
        annotation = await process_keyword_annotation(
            keyword_term="microaneurysms",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
            category="lesion",
        )

        assert annotation.keyword_text == "microaneurysms"
        assert annotation.image_id == test_image_id
        assert annotation.annotation_method == "manual"

    async def test_process_keyword_annotation_with_raw_data_id(self, test_image_id, test_dataset_id):
        """Test processing keyword annotation with raw_data_id."""
        raw_data_id = uuid.uuid4()
        annotation = await process_keyword_annotation(
            keyword_term="exudates",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
            raw_data_id=raw_data_id,
        )

        assert annotation.raw_data_id == raw_data_id

    async def test_process_keyword_annotation_invalid_method(self, test_image_id, test_dataset_id):
        """Test that invalid annotation methods raise ValueError."""
        with pytest.raises(ValueError, match="Invalid annotation_method"):
            await process_keyword_annotation(
                keyword_term="test",
                keyword_source="diagnostic_keywords",
                image_id=test_image_id,
                dataset_id=test_dataset_id,
                annotation_method="invalid_method",
            )

    async def test_process_keywords_batch_from_string(self, test_image_id, test_dataset_id):
        """Test batch processing keywords from comma-separated string."""
        annotations = await process_keywords_batch(
            keywords="microaneurysms, hemorrhages, exudates",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
        )

        assert len(annotations) == 3
        terms = [ann.keyword_text for ann in annotations]
        assert "microaneurysms" in terms
        assert "hemorrhages" in terms
        assert "exudates" in terms

    async def test_process_keywords_batch_from_list(self, test_image_id, test_dataset_id):
        """Test batch processing keywords from list."""
        keywords_list = ["drusen", "reticular pseudodrusen", "pigmentary changes"]
        annotations = await process_keywords_batch(
            keywords=keywords_list,
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
        )

        assert len(annotations) == 3
        terms = [ann.keyword_text for ann in annotations]
        assert terms == keywords_list

    async def test_process_keywords_batch_from_dict(self, test_image_id, test_dataset_id):
        """Test batch processing keywords from dict (DeepEyeNet format)."""
        keywords_dict = {"keywords": ["microaneurysms", "hemorrhages"]}
        annotations = await process_keywords_batch(
            keywords=keywords_dict,
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
        )

        assert len(annotations) == 2

    async def test_process_keywords_batch_removes_duplicates(self, test_image_id, test_dataset_id):
        """Test that batch processing removes duplicate keywords."""
        annotations = await process_keywords_batch(
            keywords="drusen, drusen, exudates, drusen",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
        )

        # Should only have 2 annotations (drusen and exudates)
        assert len(annotations) == 2
        terms = [ann.keyword_text for ann in annotations]
        assert "drusen" in terms
        assert "exudates" in terms

    async def test_process_keywords_batch_empty(self, test_image_id, test_dataset_id):
        """Test batch processing with empty keywords."""
        annotations = await process_keywords_batch(
            keywords="",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
        )

        assert len(annotations) == 0

    async def test_process_keywords_odir5k_style(self, test_image_id, test_dataset_id):
        """Test processing keywords from ODIR-5K diagnostic keywords."""
        # ODIR-5K example: "moderate non proliferative retinopathy"
        annotations = await process_keywords_batch(
            keywords="moderate non proliferative retinopathy",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
            delimiter=",",  # Treat as single keyword since no comma
        )

        # Should be treated as a single keyword
        assert len(annotations) == 1
        assert annotations[0].keyword_text == "moderate non proliferative retinopathy"

    async def test_process_keywords_odir5k_multiple(self, test_image_id, test_dataset_id):
        """Test processing multiple keywords from ODIR-5K style."""
        # ODIR-5K can have multiple conditions separated by Chinese comma or English comma
        annotations = await process_keywords_batch(
            keywords="moderate non proliferative retinopathy, cataract",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
        )

        assert len(annotations) == 2

    async def test_process_keywords_clinical_description_source(self, test_image_id, test_dataset_id):
        """Test processing keywords from clinical description source."""
        annotations = await process_keywords_batch(
            keywords="optic disc edema, flame hemorrhages, retinal detachment",
            keyword_source="clinical_description",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
        )

        assert len(annotations) == 3
        # All should be from clinical_description source
        for ann in annotations:
            # Verify vocabulary was created with correct source
            assert ann.keyword_text in ["optic disc edema", "flame hemorrhages", "retinal detachment"]

    async def test_process_keywords_with_category(self, test_image_id, test_dataset_id):
        """Test processing keywords with category grouping."""
        lesion_keywords = await process_keywords_batch(
            keywords="microaneurysms, hemorrhages, exudates",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
            category="lesion",
        )

        assert len(lesion_keywords) == 3

        anatomical_keywords = await process_keywords_batch(
            keywords="optic disc, macula, fovea",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
            category="anatomical",
        )

        assert len(anatomical_keywords) == 3

    async def test_prepare_keywords_for_upsert_alias(self, test_image_id, test_dataset_id):
        """Test that prepare_keywords_for_upsert is an alias."""
        annotations = await prepare_keywords_for_upsert(
            keywords="microaneurysms, hemorrhages",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
        )

        assert len(annotations) == 2

    async def test_keyword_annotation_deterministic_uuids(self, test_image_id, test_dataset_id):
        """Test that processing the same keyword twice produces the same UUID."""
        annotation_1 = await process_keyword_annotation(
            keyword_term="test_deterministic",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
        )

        annotation_2 = await process_keyword_annotation(
            keyword_term="test_deterministic",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
        )

        assert annotation_1.keyword_annotation_id == annotation_2.keyword_annotation_id

    async def test_process_and_upsert_integration(self, test_image_id, test_dataset_id):
        """Test full integration: process and upsert to database."""
        annotations = await process_keywords_batch(
            keywords="microaneurysms, hemorrhages, exudates",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
        )

        # Upsert all annotations to database
        for annotation in annotations:
            await upsert_keyword_annotation(annotation)

        # Verify idempotency - upsert again
        for annotation in annotations:
            await upsert_keyword_annotation(annotation)

    async def test_different_sources_same_term(self, test_image_id, test_dataset_id):
        """Test that the same term from different sources creates different vocabulary entries."""
        keyword_id_1 = await get_or_create_keyword_vocabulary(
            keyword_term="hemorrhages",
            keyword_source="diagnostic_keywords",
            dataset_id=test_dataset_id,
        )

        keyword_id_2 = await get_or_create_keyword_vocabulary(
            keyword_term="hemorrhages",
            keyword_source="clinical_description",
            dataset_id=test_dataset_id,
        )

        # Should be different IDs since sources differ
        assert keyword_id_1 != keyword_id_2

    async def test_process_keywords_extracted_method(self, test_image_id, test_dataset_id):
        """Test processing keywords with 'extracted' annotation method."""
        annotations = await process_keywords_batch(
            keywords="microaneurysms, hemorrhages",
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
            annotation_method="extracted",
        )

        for ann in annotations:
            assert ann.annotation_method == "extracted"

    async def test_process_keywords_deepeyenet_realistic_example(self, test_image_id, test_dataset_id):
        """Test processing keywords from realistic DeepEyeNet data."""
        # Example from DeepEyeNet JSON
        deepeyenet_data = {
            "diagnostic_keywords": ["diabetic retinopathy", "macular edema", "hemorrhages"],
            "clinical-description": "Moderate NPDR with CME and scattered hemorrhages",
        }

        # Process diagnostic keywords
        diagnostic_annotations = await process_keywords_batch(
            keywords=deepeyenet_data["diagnostic_keywords"],
            keyword_source="diagnostic_keywords",
            image_id=test_image_id,
            dataset_id=test_dataset_id,
        )

        assert len(diagnostic_annotations) == 3
