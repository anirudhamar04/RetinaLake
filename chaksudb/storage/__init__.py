"""
Storage utilities for locator abstraction and path management.
"""

from chaksudb.storage.locators import (
    StorageLocator,
    create_azure_locator,
    create_gcs_locator,
    create_http_locator,
    create_local_locator,
    create_s3_locator,
    validate_locator,
)
from chaksudb.storage.paths import (
    compute_relative_path,
    ensure_storage_directory,
    generate_storage_path,
    get_file_size,
    get_storage_root,
    normalize_path,
    resolve_storage_path,
)

__all__ = [
    # Locators
    "StorageLocator",
    "create_local_locator",
    "create_s3_locator",
    "create_gcs_locator",
    "create_azure_locator",
    "create_http_locator",
    "validate_locator",
    # Paths
    "normalize_path",
    "compute_relative_path",
    "get_storage_root",
    "ensure_storage_directory",
    "generate_storage_path",
    "resolve_storage_path",
    "get_file_size",
]