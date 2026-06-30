"""
Keyword processor for keywords and clinical descriptions.

This module provides high-level processing functions for keyword annotations
that handle parsing, vocabulary registration, and model preparation.

Key features:
- Parses comma-separated keyword strings
- Handles DeepEyeNet JSON format
- Registers keywords in vocabulary table (idempotent)
- Links keywords to images
- Returns typed KeywordVocabulary and KeywordAnnotation models ready for upsert
- Automatic provenance tracking via context variables
"""

import json
import logging
import uuid
from typing import Any, Optional, Union

from chaksudb.db.models import KeywordAnnotation, KeywordVocabulary
from chaksudb.db.queries.annotation_types import (
    find_keyword_vocabulary_by_id,
    upsert_keyword_vocabulary,
)
from chaksudb.ingest.framework.gen_uuid import (
    generate_keyword_annotation_uuid,
    generate_keyword_uuid,
)
from chaksudb.ingest.framework.provenance_context import get_current_provenance

logger = logging.getLogger(__name__)


# ============================================
# Helper Functions
# ============================================


def parse_keyword_string(
    keyword_string: str,
    delimiter: str = ",",
) -> list[str]:
    """
    Parse a delimited keyword string into a list of keywords.

    Args:
        keyword_string: Delimited string of keywords
        delimiter: Delimiter character (default: ',')

    Returns:
        List of cleaned keyword strings

    Example:
        >>> parse_keyword_string("microaneurysms, hemorrhages, exudates")
        ["microaneurysms", "hemorrhages", "exudates"]
        >>> parse_keyword_string("drusen; reticular pseudodrusen; pigmentary changes", delimiter=";")
        ["drusen", "reticular pseudodrusen", "pigmentary changes"]
    """
    if not keyword_string or not keyword_string.strip():
        return []

    # Split by delimiter and clean
    keywords = []
    for keyword in keyword_string.split(delimiter):
        cleaned = keyword.strip()
        if cleaned:  # Skip empty strings
            keywords.append(cleaned)

    return keywords


def parse_deepeyenet_keywords(
    json_data: Union[str, dict[str, Any]],
) -> list[str]:
    """
    Parse keywords from DeepEyeNet JSON format.

    DeepEyeNet provides keywords in JSON format with various structures.

    Args:
        json_data: JSON string or dict containing keywords

    Returns:
        List of keyword strings

    Example:
        >>> parse_deepeyenet_keywords('{"keywords": ["microaneurysms", "hemorrhages"]}')
        ["microaneurysms", "hemorrhages"]
        >>> parse_deepeyenet_keywords({"diagnostic_keywords": "drusen, exudates"})
        ["drusen", "exudates"]
    """
    # Parse JSON if string
    if isinstance(json_data, str):
        try:
            data = json.loads(json_data)
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse JSON: {e}. Treating as plain string.")
            # Fall back to comma-separated parsing
            return parse_keyword_string(json_data)
    else:
        data = json_data

    keywords = []

    # Handle various JSON structures
    if isinstance(data, dict):
        # Check common key names
        for key in ["keywords", "diagnostic_keywords", "findings", "lesions"]:
            if key in data:
                value = data[key]
                if isinstance(value, list):
                    keywords.extend(str(v) for v in value if v)
                elif isinstance(value, str):
                    keywords.extend(parse_keyword_string(value))

        # If no keywords found, try to extract from all string values
        if not keywords:
            for value in data.values():
                if isinstance(value, str):
                    keywords.extend(parse_keyword_string(value))
                elif isinstance(value, list):
                    keywords.extend(str(v) for v in value if v)

    elif isinstance(data, list):
        # Direct list of keywords
        keywords.extend(str(k) for k in data if k)

    return keywords


# ============================================
# Keyword Vocabulary Management
# ============================================


async def get_or_create_keyword_vocabulary(
    keyword_term: str,
    keyword_source: str,
    dataset_id: uuid.UUID,
    category: Optional[str] = None,
) -> uuid.UUID:
    """
    Get or create a keyword vocabulary entry (idempotent).

    If the keyword exists in the vocabulary, returns its UUID.
    If it doesn't exist, creates it and returns the new UUID.

    Args:
        keyword_term: Keyword term (unique within dataset and source)
        keyword_source: Source ('diagnostic_keywords', 'clinical_description', 'diagnosis_text')
        dataset_id: Dataset UUID
        category: Optional category for grouping keywords

    Returns:
        keyword_id UUID

    Example:
        ```python
        keyword_id = await get_or_create_keyword_vocabulary(
            keyword_term="microaneurysms",
            keyword_source="diagnostic_keywords",
            dataset_id=dataset_id,
            category="lesion",
        )
        ```
    """
    # Validate keyword_source
    valid_sources = {"diagnostic_keywords", "clinical_description", "diagnosis_text"}
    if keyword_source not in valid_sources:
        raise ValueError(
            f"Invalid keyword_source: {keyword_source}. "
            f"Must be one of {valid_sources}"
        )

    # Generate deterministic UUID for the keyword
    keyword_id = generate_keyword_uuid(
        dataset_id=dataset_id,
        keyword_term=keyword_term,
        keyword_source=keyword_source,
    )

    # Check if keyword already exists
    existing_keyword = await find_keyword_vocabulary_by_id(keyword_id)

    if existing_keyword:
        logger.debug(
            f"Keyword '{keyword_term}' already exists in vocabulary (source: {keyword_source})"
        )
        return keyword_id

    # Create new keyword vocabulary entry
    keyword_vocab = KeywordVocabulary(
        keyword_id=keyword_id,
        keyword_term=keyword_term,
        keyword_source=keyword_source,
        category=category,
        dataset_id=dataset_id,
    )

    await upsert_keyword_vocabulary(keyword_vocab)

    logger.debug(
        f"Registered new keyword in vocabulary: '{keyword_term}' "
        f"(source: {keyword_source}, category: {category})"
    )

    return keyword_id


# ============================================
# Main Processing Functions
# ============================================


async def process_keyword_annotation(
    keyword_term: str,
    keyword_source: str,
    image_id: uuid.UUID,
    dataset_id: uuid.UUID,
    category: Optional[str] = None,
    raw_data_id: Optional[uuid.UUID] = None,
    expert_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    provenance_chain_id: Optional[uuid.UUID] = None,
) -> KeywordAnnotation:
    """
    Process a single keyword annotation and prepare it for upsert.

    This function:
    1. Registers the keyword in vocabulary (idempotent)
    2. Generates deterministic UUID for the annotation
    3. Returns a typed KeywordAnnotation model ready for upsert

    Args:
        keyword_term: Keyword term
        keyword_source: Source ('diagnostic_keywords', 'clinical_description', 'diagnosis_text')
        image_id: UUID of the image
        dataset_id: Dataset UUID (for vocabulary registration)
        category: Optional category for grouping keywords
        raw_data_id: Optional raw annotation file UUID
        expert_id: Optional expert UUID
        annotation_method: Method ('manual', 'extracted', 'pseudo')
        provenance_chain_id: Optional provenance chain UUID

    Returns:
        KeywordAnnotation model ready for upsert

    Raises:
        ValueError: If keyword_source or annotation_method is invalid

    Example:
        ```python
        annotation = await process_keyword_annotation(
            keyword_term="microaneurysms",
            keyword_source="diagnostic_keywords",
            image_id=image_id,
            dataset_id=dataset_id,
            category="lesion",
            raw_data_id=raw_file_id,
        )
        await upsert_keyword_annotation(annotation)
        ```
    """
    # Get provenance from context if not explicitly provided
    if raw_data_id is None or provenance_chain_id is None:
        context_raw_id, context_chain_id = get_current_provenance()
        raw_data_id = raw_data_id or context_raw_id
        provenance_chain_id = provenance_chain_id or context_chain_id
    
    # Validate annotation_method
    valid_methods = {"manual", "extracted", "pseudo"}
    if annotation_method not in valid_methods:
        raise ValueError(
            f"Invalid annotation_method: {annotation_method}. "
            f"Must be one of {valid_methods}"
        )

    # Normalize keyword term to lowercase for consistent vocabulary
    keyword_term = keyword_term.strip().lower()

    # Get or create keyword in vocabulary
    keyword_id = await get_or_create_keyword_vocabulary(
        keyword_term=keyword_term,
        keyword_source=keyword_source,
        dataset_id=dataset_id,
        category=category,
    )

    # Generate deterministic UUID for the annotation
    keyword_annotation_id = generate_keyword_annotation_uuid(
        image_id=image_id,
        keyword_id=keyword_id,
        expert_id=expert_id,
        raw_data_id=raw_data_id,
    )

    # Create KeywordAnnotation model
    annotation = KeywordAnnotation(
        keyword_annotation_id=keyword_annotation_id,
        image_id=image_id,
        keyword_id=keyword_id,
        keyword_text=keyword_term,  # Store the actual keyword text for convenience
        raw_data_id=raw_data_id,
        expert_id=expert_id,
        annotation_method=annotation_method,
        provenance_chain_id=provenance_chain_id,
    )

    logger.debug(
        f"Processed keyword annotation: '{keyword_term}' "
        f"(source: {keyword_source}) for image {image_id}"
    )

    return annotation


async def process_keywords_batch(
    keywords: Union[str, list[str], dict[str, Any]],
    keyword_source: str,
    image_id: uuid.UUID,
    dataset_id: uuid.UUID,
    category: Optional[str] = None,
    delimiter: str = ",",
    raw_data_id: Optional[uuid.UUID] = None,
    expert_id: Optional[uuid.UUID] = None,
    annotation_method: str = "manual",
    provenance_chain_id: Optional[uuid.UUID] = None,
) -> list[KeywordAnnotation]:
    """
    Process multiple keywords at once (batch processing).

    This function handles various input formats:
    - String: comma-separated keywords
    - List: list of keyword strings
    - Dict: DeepEyeNet JSON format

    Args:
        keywords: Keywords in various formats
        keyword_source: Source ('diagnostic_keywords', 'clinical_description', 'diagnosis_text')
        image_id: UUID of the image
        dataset_id: Dataset UUID (for vocabulary registration)
        category: Optional category for grouping keywords
        delimiter: Delimiter for string parsing (default: ',')
        raw_data_id: Optional raw annotation file UUID
        expert_id: Optional expert UUID
        annotation_method: Method ('manual', 'extracted', 'pseudo')
        provenance_chain_id: Optional provenance chain UUID

    Returns:
        List of KeywordAnnotation models ready for upsert

    Examples:
        Comma-separated string:
        ```python
        annotations = await process_keywords_batch(
            keywords="microaneurysms, hemorrhages, exudates",
            keyword_source="diagnostic_keywords",
            image_id=image_id,
            dataset_id=dataset_id,
            raw_data_id=raw_file_id,
        )
        ```

        List of keywords:
        ```python
        annotations = await process_keywords_batch(
            keywords=["drusen", "reticular pseudodrusen", "pigmentary changes"],
            keyword_source="diagnostic_keywords",
            image_id=image_id,
            dataset_id=dataset_id,
        )
        ```

        DeepEyeNet JSON:
        ```python
        annotations = await process_keywords_batch(
            keywords={"keywords": ["microaneurysms", "hemorrhages"]},
            keyword_source="diagnostic_keywords",
            image_id=image_id,
            dataset_id=dataset_id,
        )
        ```
    """
    # Get provenance from context if not explicitly provided
    if raw_data_id is None or provenance_chain_id is None:
        context_raw_id, context_chain_id = get_current_provenance()
        raw_data_id = raw_data_id or context_raw_id
        provenance_chain_id = provenance_chain_id or context_chain_id
    
    # Parse keywords based on input format
    if isinstance(keywords, str):
        keyword_list = parse_keyword_string(keywords, delimiter=delimiter)
    elif isinstance(keywords, list):
        keyword_list = [str(k).strip() for k in keywords if k]
    elif isinstance(keywords, dict):
        keyword_list = parse_deepeyenet_keywords(keywords)
    else:
        raise ValueError(f"Invalid keywords type: {type(keywords)}")

    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for keyword in keyword_list:
        if keyword and keyword not in seen:
            seen.add(keyword)
            unique_keywords.append(keyword)

    # Process each keyword
    annotations = []
    for keyword_term in unique_keywords:
        annotation = await process_keyword_annotation(
            keyword_term=keyword_term,
            keyword_source=keyword_source,
            image_id=image_id,
            dataset_id=dataset_id,
            category=category,
            raw_data_id=raw_data_id,
            expert_id=expert_id,
            annotation_method=annotation_method,
            provenance_chain_id=provenance_chain_id,
        )
        annotations.append(annotation)

    logger.debug(
        f"Processed {len(annotations)} keyword annotations for image {image_id}"
    )

    return annotations


# ============================================
# Convenience Function (alias)
# ============================================


async def prepare_keywords_for_upsert(
    keywords: Union[str, list[str], dict[str, Any]],
    keyword_source: str,
    image_id: uuid.UUID,
    dataset_id: uuid.UUID,
    **kwargs,
) -> list[KeywordAnnotation]:
    """
    Alias for process_keywords_batch() for consistency with other processors.

    See process_keywords_batch() for full documentation.
    """
    return await process_keywords_batch(
        keywords=keywords,
        keyword_source=keyword_source,
        image_id=image_id,
        dataset_id=dataset_id,
        **kwargs,
    )
