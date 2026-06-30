"""
Image creation helpers with automatic metadata extraction.

Provides utilities to extract and prepare image metadata for Image objects.
The metadata (resolution, format, storage locator) can be spread into the
Image() constructor.
"""

import logging
import uuid
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, Optional

from chaksudb.config.config import get_data_root
from chaksudb.db.models import Image
from chaksudb.ingest.framework.image_metadata import extract_image_metadata
from chaksudb.storage import create_local_locator
from chaksudb.storage.paths import compute_relative_path, normalize_path

logger = logging.getLogger(__name__)


def _compute_image_hashes(image_path: Path) -> tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (file_hash, content_hash, phash) for an image, degrading gracefully.

    file_hash    = exact bytes (cheap, always attempted).
    content_hash = decoded RGB pixels — encoding-invariant for lossless re-encodes.
    phash        = perceptual dHash — matches across lossy re-encoding / resize.
    Any hash that cannot be computed (e.g. a non-decodable file) is returned as None
    rather than failing image ingestion.
    """
    from chaksudb.ingest.framework.hashing import (
        compute_file_hash,
        compute_pixel_and_perceptual_hashes,
    )

    def _safe(fn):
        try:
            return fn(image_path)
        except Exception as e:  # never let hashing break ingestion
            logger.warning(f"Hash {fn.__name__} failed for {image_path}: {e}")
            return None

    file_hash = _safe(compute_file_hash)
    # content + perceptual hashes share a single image decode (the expensive part)
    pixel_perceptual = _safe(compute_pixel_and_perceptual_hashes)
    content_hash, phash = pixel_perceptual if pixel_perceptual else (None, None)
    return file_hash, content_hash, phash


def get_image_metadata_dict(
    image_path: Path,
    extract_metadata: bool = True,
) -> Dict[str, Any]:
    """
    Extract image metadata and storage locator as a dict for spreading into Image().
    
    This returns a dictionary with storage locator fields and extracted metadata
    (resolution, format) that can be spread directly into the Image constructor.
    
    Args:
        image_path: Path to the image file (must exist)
        extract_metadata: If True, extract resolution and format from file.
            Set to False to skip extraction (faster, for large datasets).
    
    Returns:
        Dictionary with keys: storage_provider, file_path, bucket, object_key,
        version_id, file_format, resolution_width, resolution_height
    
    Raises:
        FileNotFoundError: If image_path does not exist
    
    Example:
        ```python
        from chaksudb.db.models import Image
        from chaksudb.ingest.framework import get_image_metadata_dict
        
        # Image() call stays exactly the same, just spread the metadata dict
        image = Image(
            image_id=image_id,
            dataset_id=dataset_id,
            original_image_id=image_name,
            **get_image_metadata_dict(image_path),  # <-- Add this line
            modality="fundus",
            eye_laterality="left",
        )
        ```
    
    Note:
        - ``file_path`` in the returned dict is relative to the data root when
          the image lies under it; otherwise it is an absolute path (normalized).
        - Resolution extracted automatically via OpenCV/Pillow
        - File format normalized to schema-compliant values
        - Storage locator created for local files
        - If extraction fails, resolution fields are None (graceful degradation)
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Build file_path relative to data root when the image lies under it
    resolved = image_path.resolve()
    data_root = get_data_root().resolve()
    try:
        path_str = compute_relative_path(resolved, data_root)
    except ValueError:
        path_str = normalize_path(str(resolved))
    # Create storage locator (required); file_path will be relative to data root when possible
    locator = create_local_locator(Path(path_str))
    result = locator.to_dict()

    # Duplicate-detection hashes for every image (spread into Image()), so dupes are
    # discoverable without per-script changes:
    #   file_hash    exact bytes,  content_hash  decoded pixels (encoding-invariant),
    #   phash        perceptual (matches across lossy re-encoding / resize).
    result["file_hash"], result["content_hash"], result["phash"] = _compute_image_hashes(image_path)

    # Extract metadata if requested
    if extract_metadata:
        try:
            metadata = extract_image_metadata(image_path)
            result["resolution_width"] = metadata.resolution_width
            result["resolution_height"] = metadata.resolution_height
            result["file_format"] = metadata.file_format
            
            logger.debug(
                f"Extracted metadata for {image_path.name}: "
                f"{metadata.resolution_width}x{metadata.resolution_height}, "
                f"format={metadata.file_format}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to extract metadata from {image_path}: {e}. "
                f"Image will be created without resolution/format."
            )
            # Add None values so keys exist
            result["resolution_width"] = None
            result["resolution_height"] = None
            
            # Try to at least get format from extension
            if image_path.suffix:
                from chaksudb.config.config import constants
                ext = image_path.suffix[1:].lower()
                if ext in constants.FILE_FORMATS:
                    result["file_format"] = ext
    else:
        # No extraction - add None values and try format from extension
        result["resolution_width"] = None
        result["resolution_height"] = None
        
        if image_path.suffix:
            from chaksudb.config.config import constants
            ext = image_path.suffix[1:].lower()
            if ext in constants.FILE_FORMATS:
                result["file_format"] = ext
    
    return result


def create_image_with_metadata(
    image_id: uuid.UUID,
    dataset_id: uuid.UUID,
    image_path: Path,
    original_image_id: Optional[str] = None,
    modality: Optional[str] = None,
    group_id: Optional[uuid.UUID] = None,
    frame_index: Optional[int] = None,
    field_of_view: Optional[int] = None,
    eye_laterality: Optional[str] = None,
    acquisition_date: Optional[date] = None,
    extract_metadata: bool = True,
) -> Image:
    """
    Create an Image object with automatic metadata extraction.
    
    Extracts resolution, file format, and other metadata from the image file
    using OpenCV/Pillow, creates a local storage locator, and returns a fully
    populated Image model ready for upserting to the database.
    
    Args:
        image_id: UUID for the image (use generate_image_uuid)
        dataset_id: UUID of the parent dataset
        image_path: Path to the image file (must exist)
        original_image_id: Original identifier from the dataset (e.g., filename)
        modality: Image modality ('fundus', 'oct', 'fa', 'uwf')
        group_id: Optional group ID for multi-frame images (e.g., OCT volumes)
        frame_index: Optional frame index within a group
        field_of_view: Optional field of view in degrees
        eye_laterality: Optional eye laterality ('left', 'right', 'unknown')
        acquisition_date: Optional date of image acquisition
        extract_metadata: If True, extract resolution and format from file.
            Set to False to skip extraction (e.g., for non-standard formats).
    
    Returns:
        Image model with storage locator and extracted metadata
    
    Raises:
        FileNotFoundError: If image_path does not exist
        ValueError: If image cannot be read (when extract_metadata=True)
    
    Example:
        ```python
        from chaksudb.ingest.framework.gen_uuid import generate_image_uuid
        from chaksudb.ingest.framework.image_helpers import create_image_with_metadata
        
        image_path = data_root / "train" / "image_001.jpeg"
        image_id = generate_image_uuid(dataset_id, "image_001")
        
        # Create image with automatic metadata extraction
        image = create_image_with_metadata(
            image_id=image_id,
            dataset_id=dataset_id,
            image_path=image_path,
            original_image_id="image_001",
            modality="fundus",
            eye_laterality="left"
        )
        
        # Now upsert to database
        await upsert_image(image)
        ```
    
    Note:
        - Resolution (width, height) extracted automatically via OpenCV/Pillow
        - File format normalized to schema-compliant values
        - Storage locator created automatically for local files
        - If metadata extraction fails, Image is still created with None values
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Build file_path relative to data root when the image lies under it
    resolved = image_path.resolve()
    data_root = get_data_root().resolve()
    try:
        path_str = compute_relative_path(resolved, data_root)
    except ValueError:
        path_str = normalize_path(str(resolved))
    locator = create_local_locator(Path(path_str))
    
    # Extract metadata if requested
    resolution_width = None
    resolution_height = None
    file_format = None
    
    if extract_metadata:
        try:
            metadata = extract_image_metadata(image_path)
            resolution_width = metadata.resolution_width
            resolution_height = metadata.resolution_height
            file_format = metadata.file_format
            
            logger.debug(
                f"Extracted metadata for {image_path.name}: "
                f"{resolution_width}x{resolution_height}, format={file_format}"
            )
        except Exception as e:
            logger.warning(
                f"Failed to extract metadata from {image_path}: {e}. "
                f"Creating Image without resolution/format."
            )
            # Continue with None values - Image creation should not fail due to metadata extraction
    
    # If metadata extraction was skipped or failed, try to at least get format from extension
    if file_format is None and image_path.suffix:
        # Simple fallback: use extension without dot, lowercased
        ext = image_path.suffix[1:].lower()
        # Only set if it's a valid format (avoid invalid formats in DB)
        from chaksudb.config.config import constants
        if ext in constants.FILE_FORMATS:
            file_format = ext
    
    # Duplicate-detection hashes (exact bytes / decoded pixels / perceptual)
    file_hash, content_hash, phash = _compute_image_hashes(image_path)

    # Create and return Image model
    return Image(
        image_id=image_id,
        dataset_id=dataset_id,
        original_image_id=original_image_id,
        **locator.to_dict(),
        file_hash=file_hash,
        content_hash=content_hash,
        phash=phash,
        file_format=file_format,
        modality=modality,
        group_id=group_id,
        frame_index=frame_index,
        resolution_width=resolution_width,
        resolution_height=resolution_height,
        field_of_view=field_of_view,
        eye_laterality=eye_laterality,
        acquisition_date=acquisition_date,
        created_at=datetime.now(),
    )


