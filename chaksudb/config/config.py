"""
Configuration module for the ingestion framework.

Provides database connection settings, storage paths, logging configuration,
and framework constants (UUID namespaces, file extensions, enums).
"""

import os
import uuid
from pathlib import Path
from typing import Set

from dotenv import load_dotenv
from pydantic import Field
from pydantic_settings import SettingsConfigDict, BaseSettings

# Load environment variables from .env file if it exists
load_dotenv()


class DatabaseConfig(BaseSettings):
    """Database connection configuration."""

    host: str = Field(default="localhost", description="PostgreSQL host")
    port: int = Field(default=5342, description="PostgreSQL port")
    database: str = Field(default="chaksu", description="Database name")
    user: str = Field(default="admin", description="Database user")
    password: str = Field(default="admin123", description="Database password")
    min_connections: int = Field(default=2, description="Minimum connection pool size")
    max_connections: int = Field(default=10, description="Maximum connection pool size")
    model_config = SettingsConfigDict(env_prefix="DB_", case_sensitive=False)

    @property
    def connection_string(self) -> str:
        """Generate PostgreSQL connection string."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode=disable"
        )

    @property
    def async_connection_string(self) -> str:
        """Generate PostgreSQL async connection string for psycopg."""
        return (
            f"postgresql://{self.user}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
            f"?sslmode=disable"
            f"&keepalives=1&keepalives_idle=60&keepalives_interval=10&keepalives_count=5"
        )


class StorageConfig(BaseSettings):
    """Storage configuration."""

    local_root: Path = Field(
        default=Path("./storage"),
        description="Root directory for local file storage"
    )
    data_root: Path = Field(
        default=Path("./data"),
        description="Root directory for dataset files"
    )
    image_server_url: str | None = Field(
        default=None,
        description="Base URL of the lab image HTTP server (e.g. http://192.168.1.10:8091). "
                    "When set, missing local files are downloaded from this server and cached."
    )
    image_cache_dir: Path = Field(
        default=Path.home() / ".cache" / "chaksudb",
        description="Local directory where remotely fetched images are cached"
    )
    model_config = SettingsConfigDict(env_prefix="STORAGE_", case_sensitive=False)

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure directories exist
        self.local_root.mkdir(parents=True, exist_ok=True)
        self.data_root.mkdir(parents=True, exist_ok=True)


class LoggingConfig(BaseSettings):
    """Logging configuration."""

    level: str = Field(default="INFO", description="Logging level")
    format: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format string"
    )
    file: Path | None = Field(
        default=None,
        description="Optional log file path"
    )
    model_config = SettingsConfigDict(env_prefix="LOG_", case_sensitive=False)


class FrameworkConstants:
    """Framework constants for UUID namespaces, file extensions, and enums."""

        # ============================================
    # UUID Namespaces (for UUID v5 generation)
    # ============================================
    # Fixed namespace UUIDs for each table type
    # These are deterministic UUIDs used as namespaces for UUID v5 generation
    NAMESPACE_DATASET = uuid.UUID("1c0ed524-dbd7-4a3e-bd01-9c05b2ed0b7b")
    NAMESPACE_IMAGE = uuid.UUID("5cbf22d1-fc0d-4b51-bb67-a28396d8a409")
    NAMESPACE_PATIENT = uuid.UUID("f876947e-6ee2-462e-a49f-abfa4ae203be")
    NAMESPACE_RAW_FILE = uuid.UUID("10927aac-244f-402e-94a8-018c694dbdcb")
    NAMESPACE_IMAGE_GROUP = uuid.UUID("8a0180be-8ef6-420e-8ceb-ca4ba28d1816")
    NAMESPACE_EXPERT = uuid.UUID("4e365b18-2b3f-44c3-9599-d67f85da352c")
    NAMESPACE_MODEL = uuid.UUID("1a213e19-273d-4500-bc0f-aea27faaffe0")
    NAMESPACE_EXPERT_ANNOTATION = uuid.UUID("b018f7ec-8ae9-415d-b0f6-aa668ba934f1")
    NAMESPACE_ANNOTATION_TYPE = uuid.UUID("fcf1146d-810e-4d42-a96f-0d490c06cb22")
    NAMESPACE_GRADING_SCALE = uuid.UUID("44ea7ced-5c03-497e-a9e3-98a7b6693cf1")
    NAMESPACE_GRADING_SCALE_MAPPING = uuid.UUID("0e39d5f7-691c-4fdf-98de-c30bdb7fbeb1")
    NAMESPACE_PROVENANCE_CHAIN = uuid.UUID("851aa4ae-5e63-4905-8f5c-51e9c41267cf")
    NAMESPACE_TRANSFORMATION = uuid.UUID("223a2458-3fe3-4a19-a61a-97e46a7ae2f1")
    NAMESPACE_SEGMENTATION = uuid.UUID("1093ebbd-a51d-46a0-90ac-aed3d915a07f")
    NAMESPACE_CLASSIFICATION = uuid.UUID("7b833ff9-8dcd-4b96-b1c7-43a45d3c3679")
    NAMESPACE_LOCALIZATION = uuid.UUID("d3895239-8c63-4daa-a3c7-f8f6f471f114")
    NAMESPACE_QUALITY = uuid.UUID("83e31150-559d-42c6-ad00-c7493fb02358")
    NAMESPACE_KEYWORD = uuid.UUID("64e683f0-2d6f-472f-9b7d-f94b292bed38")
    NAMESPACE_DESCRIPTION = uuid.UUID("7483cfa6-4937-46b4-bcba-d904a67f592b")
    NAMESPACE_DATASET_SPLIT = uuid.UUID("5b6e05e9-c13e-4a9e-9df5-7e2525df9c46")
    NAMESPACE_PATIENT_IMAGE = uuid.UUID("a1b2c3d4-e5f6-4789-a012-3456789abcde")
    NAMESPACE_CONSENSUS = uuid.UUID("b2c3d4e5-f6a7-4890-b123-456789abcdef")
    NAMESPACE_PROVENANCE_TRANSFORMATION = uuid.UUID("c3d4e5f6-a7b8-4901-c234-56789abcdef0")
    NAMESPACE_DISEASE_GRADING = uuid.UUID("d4e5f6a7-b8c9-4012-d345-6789abcdef01")
    NAMESPACE_KEYWORD_ANNOTATION = uuid.UUID("e5f6a7b8-c9d0-4123-e456-789abcdef012")
    NAMESPACE_IMAGE_SPLIT = uuid.UUID("f6a7b8c9-d0e1-4234-f567-89abcdef0123")

    # ============================================
    # File Extensions
    # ============================================
    IMAGE_EXTENSIONS: Set[str] = {
        ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".ppm", ".bmp", ".dcm", ".dicom",".JPG",".JPEG",".PNG",".TIF",".TIFF",".PPM",".BMP",".DCM",".DICOM"
    }
    ANNOTATION_EXTENSIONS: Set[str] = {
        ".csv", ".json", ".xlsx", ".xls", ".txt", ".xml", ".jsonl", ".mat", ".html", ".wmv"
    }
    MASK_EXTENSIONS: Set[str] = {
        ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp", ".npy", ".npz"
    }

    # ============================================
    # File Formats (matching schema constraints)
    # ============================================
    FILE_FORMATS: Set[str] = {
        "jpg", "jpeg", "png", "tif", "tiff", "ppm", "dicom", "wmv"
    }
    
    # ============================================
    # Modalities (matching schema constraints)
    # ============================================
    MODALITIES: Set[str] = {
        "fundus", "oct", "fa", "uwf"
    }

    # ============================================
    # Storage Providers (matching schema constraints)
    # ============================================
    STORAGE_PROVIDERS: Set[str] = {
        "local", "s3","gcs","azure","http"
    }

    # ============================================
    # Eye Laterality (matching schema constraints)
    # ============================================
    EYE_LATERALITY: Set[str] = {
        "left", "right", "unknown"
    }

    # ============================================
    # Annotation Tasks (matching schema constraints)
    # ============================================
    ANNOTATION_TASKS: Set[str] = {
        "grading", "segmentation", "classification", "localization",
        "quality", "keyword", "description"
    }

    # ============================================
    # Disease Types (matching schema constraints)
    # ============================================
    DISEASE_TYPES: Set[str] = {
        "DR", "DME", "Glaucoma", "AMD", "myopic_maculopathy"
    }

    # ============================================
    # Image Group Types (matching schema constraints)
    # ============================================
    GROUP_TYPES: Set[str] = {
        "oct_volume", "video", "sequence"
    }

    # ============================================
    # Provenance Source Types (matching schema constraints)
    # ============================================
    PROVENANCE_SOURCE_TYPES: Set[str] = {
        "original", "transformed", "pseudo_generated", "consensus"
    }

    # ============================================
    # Confidence Levels (matching schema constraints)
    # ============================================
    CONFIDENCE_LEVELS: Set[str] = {
        "high", "medium", "low"
    }

    # ============================================
    # Sex Values (matching schema constraints)
    # ============================================
    SEX_VALUES: Set[str] = {
        "male", "female", "unknown"
    }

    # ============================================
    # Dataset Split Types (matching schema constraints)
    # ============================================
    SPLIT_TYPES: Set[str] = {
        "explicit", "metadata_defined", "user_defined", "undefined"
    }


# ============================================
# Global Configuration Instances
# ============================================

# Database configuration
db_config = DatabaseConfig()

# Storage configuration
storage_config = StorageConfig()

# Logging configuration
logging_config = LoggingConfig()

# Framework constants
constants = FrameworkConstants()


# ============================================
# Convenience Functions
# ============================================

def get_data_root() -> Path:
    """Get the root directory for dataset files."""
    return storage_config.data_root


def get_storage_root() -> Path:
    """Get the root directory for local file storage."""
    return storage_config.local_root


def get_db_connection_string() -> str:
    """Get the database connection string."""
    return db_config.connection_string


def get_db_async_connection_string() -> str:
    """Get the async database connection string."""
    return db_config.async_connection_string