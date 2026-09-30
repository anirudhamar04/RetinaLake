"""
Storage locator abstraction for unified file storage management.

This module provides a unified way to represent where files are stored, supporting
local filesystem, cloud storage (S3, GCS, Azure), and HTTP URLs. Storage locators
are used throughout the ingestion framework to abstract away storage backend details.

The locator structure matches the database schema constraints:
- For 'local' storage: file_path must be provided
- For cloud storage ('s3', 'gcs', 'azure'): object_key must be provided
- For 'http': object_key (URL) must be provided
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from chaksudb.config.config import constants

logger = logging.getLogger(__name__)


@dataclass
class StorageLocator:
    """
    Storage locator representing a file location across different storage backends.
    
    Attributes:
        storage_provider: Storage provider type ('local', 's3', 'gcs', 'azure', 'http')
        file_path: Local file path (required for 'local' provider)
        bucket: Cloud storage bucket name (optional, for cloud providers)
        object_key: Cloud storage object key or HTTP URL (required for non-local providers)
        version_id: Version identifier for versioned storage (optional)
    
    The locator must satisfy schema constraints:
    - If storage_provider = 'local': file_path must be NOT NULL
    - If storage_provider != 'local': object_key must be NOT NULL
    """

    storage_provider: str
    file_path: Optional[str] = None
    bucket: Optional[str] = None
    object_key: Optional[str] = None
    version_id: Optional[str] = None

    def __post_init__(self):
        """Validate locator after initialization."""
        if not self.validate():
            raise ValueError(f"Invalid storage locator: {self}")

    def validate(self) -> bool:
        """
        Validate that the locator satisfies schema constraints.
        
        Returns:
            True if valid, False otherwise
        """
        # Validate storage provider
        if self.storage_provider not in constants.STORAGE_PROVIDERS:
            logger.error(
                f"Invalid storage_provider: {self.storage_provider}. "
                f"Must be one of {constants.STORAGE_PROVIDERS}"
            )
            return False

        # Validate locator constraints
        if self.storage_provider == "local":
            if not self.file_path:
                logger.error(
                    "Local storage provider requires file_path to be set"
                )
                return False
        else:
            if not self.object_key:
                logger.error(
                    f"Storage provider '{self.storage_provider}' requires "
                    "object_key to be set"
                )
                return False

        return True

    def to_dict(self) -> dict:
        """
        Convert locator to dictionary for database insertion.
        
        Returns:
            Dictionary with locator fields
        """
        return {
            "storage_provider": self.storage_provider,
            "file_path": self.file_path,
            "bucket": self.bucket,
            "object_key": self.object_key,
            "version_id": self.version_id,
        }


def create_local_locator(file_path: Path) -> StorageLocator:
    """
    Create a local storage locator from a file path.
    
    This function creates a storage locator for local filesystem storage.
    The file_path is normalized to use forward slashes for cross-platform
    compatibility.
    
    Args:
        file_path: Path to the local file (absolute or relative)
    
    Returns:
        StorageLocator configured for local storage
    
    Raises:
        ValueError: If file_path is None or empty
    """
    if not file_path:
        raise ValueError("file_path cannot be None or empty")

    # Normalize path to use forward slashes (cross-platform)
    normalized_path = str(file_path).replace("\\", "/")

    return StorageLocator(
        storage_provider="local",
        file_path=normalized_path,
        bucket=None,
        object_key=None,
        version_id=None,
    )


def create_s3_locator(
    bucket: str,
    object_key: str,
    version_id: Optional[str] = None,
) -> StorageLocator:
    """
    Create an S3 storage locator.
    
    Args:
        bucket: S3 bucket name
        object_key: S3 object key (path within bucket)
        version_id: Optional S3 object version ID
    
    Returns:
        StorageLocator configured for S3 storage
    
    Raises:
        ValueError: If bucket or object_key is None or empty
    """
    if not bucket:
        raise ValueError("bucket cannot be None or empty")
    if not object_key:
        raise ValueError("object_key cannot be None or empty")

    return StorageLocator(
        storage_provider="s3",
        file_path=None,
        bucket=bucket,
        object_key=object_key,
        version_id=version_id,
    )


def create_gcs_locator(
    bucket: str,
    object_key: str,
    version_id: Optional[str] = None,
) -> StorageLocator:
    """
    Create a Google Cloud Storage locator.
    
    Args:
        bucket: GCS bucket name
        object_key: GCS object key (path within bucket)
        version_id: Optional GCS object generation number
    
    Returns:
        StorageLocator configured for GCS storage
    
    Raises:
        ValueError: If bucket or object_key is None or empty
    """
    if not bucket:
        raise ValueError("bucket cannot be None or empty")
    if not object_key:
        raise ValueError("object_key cannot be None or empty")

    return StorageLocator(
        storage_provider="gcs",
        file_path=None,
        bucket=bucket,
        object_key=object_key,
        version_id=version_id,
    )


def create_azure_locator(
    bucket: str,
    object_key: str,
    version_id: Optional[str] = None,
) -> StorageLocator:
    """
    Create an Azure Blob Storage locator.
    
    Args:
        bucket: Azure container name
        object_key: Azure blob name (path within container)
        version_id: Optional Azure blob version ID
    
    Returns:
        StorageLocator configured for Azure storage
    
    Raises:
        ValueError: If bucket or object_key is None or empty
    """
    if not bucket:
        raise ValueError("bucket cannot be None or empty")
    if not object_key:
        raise ValueError("object_key cannot be None or empty")

    return StorageLocator(
        storage_provider="azure",
        file_path=None,
        bucket=bucket,
        object_key=object_key,
        version_id=version_id,
    )


def create_http_locator(url: str) -> StorageLocator:
    """
    Create an HTTP/HTTPS URL locator.
    
    Args:
        url: HTTP or HTTPS URL
    
    Returns:
        StorageLocator configured for HTTP storage
    
    Raises:
        ValueError: If url is None or empty
    """
    if not url:
        raise ValueError("url cannot be None or empty")

    return StorageLocator(
        storage_provider="http",
        file_path=None,
        bucket=None,
        object_key=url,
        version_id=None,
    )


def validate_locator(locator: StorageLocator) -> bool:
    """
    Validate a storage locator against schema constraints.
    
    This is a convenience function that calls the locator's validate method.
    Useful for validating locators before database insertion.
    
    Args:
        locator: StorageLocator to validate
    
    Returns:
        True if valid, False otherwise
    """
    return locator.validate()