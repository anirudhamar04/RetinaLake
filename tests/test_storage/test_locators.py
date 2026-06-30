"""
Tests for storage locators (chaksudb/storage/locators.py).

Tests storage locator creation, validation, and conversion based on docstring specifications.
"""

import pytest
from pathlib import Path

from chaksudb.storage.locators import (
    StorageLocator,
    create_local_locator,
    create_s3_locator,
    create_gcs_locator,
    create_azure_locator,
    create_http_locator,
    validate_locator,
)


# ============================================
# StorageLocator.validate() tests
# ============================================


def test_validate_returns_true_for_valid_local_locator():
    """Test that validate returns True for a valid local storage locator."""
    locator = StorageLocator(storage_provider="local", file_path="/path/to/file.jpg")
    assert locator.validate() is True


def test_validate_returns_true_for_valid_s3_locator():
    """Test that validate returns True for a valid S3 storage locator."""
    locator = StorageLocator(
        storage_provider="s3",
        bucket="my-bucket",
        object_key="path/to/file.jpg"
    )
    assert locator.validate() is True


def test_validate_returns_false_for_invalid_storage_provider():
    """Test that validate returns False when storage_provider is not in STORAGE_PROVIDERS."""
    # Bypass __post_init__ validation to test validate() directly
    locator = StorageLocator.__new__(StorageLocator)
    locator.storage_provider = "ftp"  # Invalid provider
    locator.file_path = "/path/to/file.jpg"
    locator.bucket = None
    locator.object_key = None
    locator.version_id = None
    
    assert locator.validate() is False


def test_validate_returns_false_for_local_without_file_path():
    """Test that validate returns False when local storage has no file_path."""
    # Bypass __post_init__ validation to test validate() directly
    locator = StorageLocator.__new__(StorageLocator)
    locator.storage_provider = "local"
    locator.file_path = None
    locator.bucket = None
    locator.object_key = None
    locator.version_id = None
    
    assert locator.validate() is False


def test_validate_returns_false_for_s3_without_object_key():
    """Test that validate returns False when S3 storage has no object_key."""
    # Bypass __post_init__ validation to test validate() directly
    locator = StorageLocator.__new__(StorageLocator)
    locator.storage_provider = "s3"
    locator.file_path = None
    locator.bucket = "my-bucket"
    locator.object_key = None
    locator.version_id = None
    
    assert locator.validate() is False


def test_validate_returns_false_for_http_without_object_key():
    """Test that validate returns False when HTTP storage has no object_key."""
    # Bypass __post_init__ validation to test validate() directly
    locator = StorageLocator.__new__(StorageLocator)
    locator.storage_provider = "http"
    locator.file_path = None
    locator.bucket = None
    locator.object_key = None
    locator.version_id = None
    
    assert locator.validate() is False


# ============================================
# StorageLocator.__post_init__() tests
# ============================================


def test_post_init_raises_error_for_invalid_locator():
    """Test that __post_init__ raises ValueError for invalid locator."""
    with pytest.raises(ValueError, match="Invalid storage locator"):
        StorageLocator(storage_provider="local", file_path=None)


def test_post_init_allows_valid_locator():
    """Test that __post_init__ allows valid locator without raising error."""
    locator = StorageLocator(storage_provider="local", file_path="/path/to/file.jpg")
    assert locator is not None


# ============================================
# StorageLocator.to_dict() tests
# ============================================


def test_to_dict_returns_dictionary_with_all_fields():
    """Test that to_dict returns a dictionary with all locator fields."""
    locator = StorageLocator(
        storage_provider="s3",
        file_path=None,
        bucket="my-bucket",
        object_key="path/to/file.jpg",
        version_id="v123"
    )
    result = locator.to_dict()
    
    assert isinstance(result, dict)
    assert result["storage_provider"] == "s3"
    assert result["file_path"] is None
    assert result["bucket"] == "my-bucket"
    assert result["object_key"] == "path/to/file.jpg"
    assert result["version_id"] == "v123"


def test_to_dict_includes_none_fields():
    """Test that to_dict includes fields even when they are None."""
    locator = StorageLocator(storage_provider="local", file_path="/path/to/file.jpg")
    result = locator.to_dict()
    
    assert "bucket" in result
    assert result["bucket"] is None
    assert "object_key" in result
    assert result["object_key"] is None
    assert "version_id" in result
    assert result["version_id"] is None


# ============================================
# create_local_locator() tests
# ============================================


def test_create_local_locator_returns_valid_locator():
    """Test that create_local_locator returns a valid StorageLocator for local storage."""
    file_path = Path("/path/to/file.jpg")
    locator = create_local_locator(file_path)
    
    assert isinstance(locator, StorageLocator)
    assert locator.storage_provider == "local"
    assert locator.file_path == "/path/to/file.jpg"
    assert locator.bucket is None
    assert locator.object_key is None
    assert locator.version_id is None


def test_create_local_locator_normalizes_path_with_forward_slashes():
    """Test that create_local_locator normalizes path to use forward slashes."""
    file_path = Path(r"C:\Users\test\file.jpg")
    locator = create_local_locator(file_path)
    
    # Should convert backslashes to forward slashes
    assert "\\" not in locator.file_path
    assert "/" in locator.file_path


def test_create_local_locator_accepts_relative_path():
    """Test that create_local_locator accepts relative file paths."""
    file_path = Path("relative/path/to/file.jpg")
    locator = create_local_locator(file_path)
    
    assert locator.file_path == "relative/path/to/file.jpg"


def test_create_local_locator_accepts_absolute_path():
    """Test that create_local_locator accepts absolute file paths."""
    file_path = Path("/absolute/path/to/file.jpg")
    locator = create_local_locator(file_path)
    
    assert locator.file_path == "/absolute/path/to/file.jpg"


def test_create_local_locator_raises_error_for_none_path():
    """Test that create_local_locator raises ValueError if file_path is None."""
    with pytest.raises(ValueError, match="file_path cannot be None or empty"):
        create_local_locator(None)


# ============================================
# create_s3_locator() tests
# ============================================


def test_create_s3_locator_returns_valid_locator():
    """Test that create_s3_locator returns a valid StorageLocator for S3 storage."""
    locator = create_s3_locator(
        bucket="my-bucket",
        object_key="path/to/file.jpg"
    )
    
    assert isinstance(locator, StorageLocator)
    assert locator.storage_provider == "s3"
    assert locator.bucket == "my-bucket"
    assert locator.object_key == "path/to/file.jpg"
    assert locator.file_path is None
    assert locator.version_id is None


def test_create_s3_locator_accepts_version_id():
    """Test that create_s3_locator accepts optional version_id parameter."""
    locator = create_s3_locator(
        bucket="my-bucket",
        object_key="path/to/file.jpg",
        version_id="v12345"
    )
    
    assert locator.version_id == "v12345"


def test_create_s3_locator_raises_error_for_empty_bucket():
    """Test that create_s3_locator raises ValueError if bucket is None or empty."""
    with pytest.raises(ValueError, match="bucket cannot be None or empty"):
        create_s3_locator(bucket="", object_key="path/to/file.jpg")


def test_create_s3_locator_raises_error_for_none_bucket():
    """Test that create_s3_locator raises ValueError if bucket is None."""
    with pytest.raises(ValueError, match="bucket cannot be None or empty"):
        create_s3_locator(bucket=None, object_key="path/to/file.jpg")


def test_create_s3_locator_raises_error_for_empty_object_key():
    """Test that create_s3_locator raises ValueError if object_key is None or empty."""
    with pytest.raises(ValueError, match="object_key cannot be None or empty"):
        create_s3_locator(bucket="my-bucket", object_key="")


def test_create_s3_locator_raises_error_for_none_object_key():
    """Test that create_s3_locator raises ValueError if object_key is None."""
    with pytest.raises(ValueError, match="object_key cannot be None or empty"):
        create_s3_locator(bucket="my-bucket", object_key=None)


# ============================================
# create_gcs_locator() tests
# ============================================


def test_create_gcs_locator_returns_valid_locator():
    """Test that create_gcs_locator returns a valid StorageLocator for GCS storage."""
    locator = create_gcs_locator(
        bucket="my-bucket",
        object_key="path/to/file.jpg"
    )
    
    assert isinstance(locator, StorageLocator)
    assert locator.storage_provider == "gcs"
    assert locator.bucket == "my-bucket"
    assert locator.object_key == "path/to/file.jpg"
    assert locator.file_path is None
    assert locator.version_id is None


def test_create_gcs_locator_accepts_version_id():
    """Test that create_gcs_locator accepts optional version_id (generation number)."""
    locator = create_gcs_locator(
        bucket="my-bucket",
        object_key="path/to/file.jpg",
        version_id="1234567890"
    )
    
    assert locator.version_id == "1234567890"


def test_create_gcs_locator_raises_error_for_empty_bucket():
    """Test that create_gcs_locator raises ValueError if bucket is None or empty."""
    with pytest.raises(ValueError, match="bucket cannot be None or empty"):
        create_gcs_locator(bucket="", object_key="path/to/file.jpg")


def test_create_gcs_locator_raises_error_for_empty_object_key():
    """Test that create_gcs_locator raises ValueError if object_key is None or empty."""
    with pytest.raises(ValueError, match="object_key cannot be None or empty"):
        create_gcs_locator(bucket="my-bucket", object_key="")


# ============================================
# create_azure_locator() tests
# ============================================


def test_create_azure_locator_returns_valid_locator():
    """Test that create_azure_locator returns a valid StorageLocator for Azure storage."""
    locator = create_azure_locator(
        bucket="my-container",  # Azure calls it container
        object_key="path/to/file.jpg"
    )
    
    assert isinstance(locator, StorageLocator)
    assert locator.storage_provider == "azure"
    assert locator.bucket == "my-container"
    assert locator.object_key == "path/to/file.jpg"
    assert locator.file_path is None
    assert locator.version_id is None


def test_create_azure_locator_accepts_version_id():
    """Test that create_azure_locator accepts optional version_id (blob version ID)."""
    locator = create_azure_locator(
        bucket="my-container",
        object_key="path/to/file.jpg",
        version_id="2021-01-01T12:00:00.0000000Z"
    )
    
    assert locator.version_id == "2021-01-01T12:00:00.0000000Z"


def test_create_azure_locator_raises_error_for_empty_bucket():
    """Test that create_azure_locator raises ValueError if bucket is None or empty."""
    with pytest.raises(ValueError, match="bucket cannot be None or empty"):
        create_azure_locator(bucket="", object_key="path/to/file.jpg")


def test_create_azure_locator_raises_error_for_empty_object_key():
    """Test that create_azure_locator raises ValueError if object_key is None or empty."""
    with pytest.raises(ValueError, match="object_key cannot be None or empty"):
        create_azure_locator(bucket="my-container", object_key="")


# ============================================
# create_http_locator() tests
# ============================================


def test_create_http_locator_returns_valid_locator():
    """Test that create_http_locator returns a valid StorageLocator for HTTP storage."""
    locator = create_http_locator(url="https://example.com/path/to/file.jpg")
    
    assert isinstance(locator, StorageLocator)
    assert locator.storage_provider == "http"
    assert locator.object_key == "https://example.com/path/to/file.jpg"
    assert locator.file_path is None
    assert locator.bucket is None
    assert locator.version_id is None


def test_create_http_locator_accepts_http_url():
    """Test that create_http_locator accepts HTTP URLs."""
    locator = create_http_locator(url="http://example.com/file.jpg")
    
    assert locator.object_key == "http://example.com/file.jpg"


def test_create_http_locator_accepts_https_url():
    """Test that create_http_locator accepts HTTPS URLs."""
    locator = create_http_locator(url="https://example.com/file.jpg")
    
    assert locator.object_key == "https://example.com/file.jpg"


def test_create_http_locator_raises_error_for_empty_url():
    """Test that create_http_locator raises ValueError if url is None or empty."""
    with pytest.raises(ValueError, match="url cannot be None or empty"):
        create_http_locator(url="")


def test_create_http_locator_raises_error_for_none_url():
    """Test that create_http_locator raises ValueError if url is None."""
    with pytest.raises(ValueError, match="url cannot be None or empty"):
        create_http_locator(url=None)


# ============================================
# validate_locator() tests
# ============================================


def test_validate_locator_returns_true_for_valid_locator():
    """Test that validate_locator returns True for a valid storage locator."""
    locator = StorageLocator(storage_provider="local", file_path="/path/to/file.jpg")
    assert validate_locator(locator) is True


def test_validate_locator_returns_false_for_invalid_locator():
    """Test that validate_locator returns False for an invalid storage locator."""
    # Create locator that bypasses __post_init__ validation
    locator = StorageLocator.__new__(StorageLocator)
    locator.storage_provider = "invalid_provider"
    locator.file_path = None
    locator.bucket = None
    locator.object_key = None
    locator.version_id = None
    
    assert validate_locator(locator) is False


def test_validate_locator_is_convenience_function():
    """Test that validate_locator is a convenience wrapper for locator.validate()."""
    locator = StorageLocator(storage_provider="local", file_path="/path/to/file.jpg")
    
    # Both should return the same result
    assert validate_locator(locator) == locator.validate()
