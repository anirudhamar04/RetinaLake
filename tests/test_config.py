"""
Tests for chaksudb/config/config.py

Tests configuration management, database connection strings, storage paths,
and framework constants based on their docstrings.
"""

import os
import pytest
from pathlib import Path
from uuid import UUID

from chaksudb.config import config
from chaksudb.config.config import (
    DatabaseConfig,
    StorageConfig,
    LoggingConfig,
    FrameworkConstants,
    get_data_root,
    get_storage_root,
    get_db_connection_string,
    get_db_async_connection_string,
)


class TestDatabaseConfig:
    """Tests for DatabaseConfig class."""

    def test_database_config_default_values(self):
        """Test that DatabaseConfig initializes with default values."""
        db = DatabaseConfig()
        assert db.host == "localhost"
        assert db.port == 5342
        assert db.database == "chaksu"
        assert db.user == "admin"
        assert db.password == "admin123"
        assert db.min_connections == 2
        assert db.max_connections == 10

    def test_database_config_custom_values(self):
        """Test that DatabaseConfig accepts custom values."""
        db = DatabaseConfig(
            host="testhost",
            port=5432,
            database="testdb",
            user="testuser",
            password="testpass",
            min_connections=5,
            max_connections=20
        )
        assert db.host == "testhost"
        assert db.port == 5432
        assert db.database == "testdb"
        assert db.user == "testuser"
        assert db.password == "testpass"
        assert db.min_connections == 5
        assert db.max_connections == 20

    def test_connection_string_property(self):
        """Test that connection_string property generates correct PostgreSQL connection string."""
        db = DatabaseConfig(
            host="myhost",
            port=5555,
            database="mydb",
            user="myuser",
            password="mypass"
        )
        expected = "postgresql://myuser:mypass@myhost:5555/mydb"
        assert db.connection_string == expected

    def test_async_connection_string_property(self):
        """Test that async_connection_string property generates correct async connection string."""
        db = DatabaseConfig(
            host="asynchost",
            port=6666,
            database="asyncdb",
            user="asyncuser",
            password="asyncpass"
        )
        expected = "postgresql://asyncuser:asyncpass@asynchost:6666/asyncdb"
        assert db.async_connection_string == expected

    def test_connection_string_with_special_characters(self):
        """Test that connection_string handles special characters in password."""
        db = DatabaseConfig(
            host="localhost",
            port=5432,
            database="testdb",
            user="user",
            password="p@ss:w/rd"
        )
        # Note: This test documents current behavior - may need URL encoding in production
        expected = "postgresql://user:p@ss:w/rd@localhost:5432/testdb"
        assert db.connection_string == expected

    def test_database_config_from_env_prefix(self, monkeypatch):
        """Test that DatabaseConfig reads from environment variables with DB_ prefix."""
        monkeypatch.setenv("DB_HOST", "envhost")
        monkeypatch.setenv("DB_PORT", "7777")
        monkeypatch.setenv("DB_DATABASE", "envdb")
        monkeypatch.setenv("DB_USER", "envuser")
        monkeypatch.setenv("DB_PASSWORD", "envpass")
        
        db = DatabaseConfig()
        assert db.host == "envhost"
        assert db.port == 7777
        assert db.database == "envdb"
        assert db.user == "envuser"
        assert db.password == "envpass"


class TestStorageConfig:
    """Tests for StorageConfig class."""

    def test_storage_config_default_values(self, tmp_path, monkeypatch):
        """Test that StorageConfig initializes with default values."""
        # Use tmp_path to avoid creating real directories
        monkeypatch.chdir(tmp_path)
        storage = StorageConfig()
        assert storage.local_root == Path("./storage")
        assert storage.data_root == Path("./data")

    def test_storage_config_custom_values(self, tmp_path):
        """Test that StorageConfig accepts custom paths."""
        local = tmp_path / "custom_storage"
        data = tmp_path / "custom_data"
        
        storage = StorageConfig(local_root=local, data_root=data)
        assert storage.local_root == local
        assert storage.data_root == data

    def test_storage_config_creates_directories(self, tmp_path):
        """Test that StorageConfig creates directories on initialization."""
        local = tmp_path / "new_storage"
        data = tmp_path / "new_data"
        
        assert not local.exists()
        assert not data.exists()
        
        storage = StorageConfig(local_root=local, data_root=data)
        
        assert local.exists()
        assert data.exists()
        assert local.is_dir()
        assert data.is_dir()

    def test_storage_config_creates_nested_directories(self, tmp_path):
        """Test that StorageConfig creates nested directories with parents=True."""
        local = tmp_path / "level1" / "level2" / "storage"
        data = tmp_path / "a" / "b" / "c" / "data"
        
        storage = StorageConfig(local_root=local, data_root=data)
        
        assert local.exists()
        assert data.exists()

    def test_storage_config_from_env_prefix(self, tmp_path, monkeypatch):
        """Test that StorageConfig reads from environment variables with STORAGE_ prefix."""
        local = tmp_path / "env_storage"
        data = tmp_path / "env_data"
        
        monkeypatch.setenv("STORAGE_LOCAL_ROOT", str(local))
        monkeypatch.setenv("STORAGE_DATA_ROOT", str(data))
        
        storage = StorageConfig()
        assert storage.local_root == local
        assert storage.data_root == data


class TestLoggingConfig:
    """Tests for LoggingConfig class."""

    def test_logging_config_default_values(self):
        """Test that LoggingConfig initializes with default values."""
        log = LoggingConfig()
        assert log.level == "INFO"
        assert log.format == "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        assert log.file is None

    def test_logging_config_custom_values(self, tmp_path):
        """Test that LoggingConfig accepts custom values."""
        log_file = tmp_path / "app.log"
        log = LoggingConfig(
            level="DEBUG",
            format="%(levelname)s: %(message)s",
            file=log_file
        )
        assert log.level == "DEBUG"
        assert log.format == "%(levelname)s: %(message)s"
        assert log.file == log_file

    def test_logging_config_from_env_prefix(self, tmp_path, monkeypatch):
        """Test that LoggingConfig reads from environment variables with LOG_ prefix."""
        log_file = tmp_path / "env.log"
        monkeypatch.setenv("LOG_LEVEL", "WARNING")
        monkeypatch.setenv("LOG_FORMAT", "%(message)s")
        monkeypatch.setenv("LOG_FILE", str(log_file))
        
        log = LoggingConfig()
        assert log.level == "WARNING"
        assert log.format == "%(message)s"
        assert log.file == log_file


class TestFrameworkConstants:
    """Tests for FrameworkConstants class."""

    def test_namespace_uuids_are_valid(self):
        """Test that all namespace UUIDs are valid UUID objects."""
        constants = FrameworkConstants()
        
        # Test a sample of namespaces
        assert isinstance(constants.NAMESPACE_DATASET, UUID)
        assert isinstance(constants.NAMESPACE_IMAGE, UUID)
        assert isinstance(constants.NAMESPACE_PATIENT, UUID)
        assert isinstance(constants.NAMESPACE_EXPERT, UUID)

    def test_namespace_uuids_are_deterministic(self):
        """Test that namespace UUIDs are fixed and deterministic."""
        constants1 = FrameworkConstants()
        constants2 = FrameworkConstants()
        
        assert constants1.NAMESPACE_DATASET == constants2.NAMESPACE_DATASET
        assert constants1.NAMESPACE_IMAGE == constants2.NAMESPACE_IMAGE

    def test_image_extensions_contains_expected_formats(self):
        """Test that IMAGE_EXTENSIONS contains expected file extensions."""
        constants = FrameworkConstants()
        
        assert ".jpg" in constants.IMAGE_EXTENSIONS
        assert ".jpeg" in constants.IMAGE_EXTENSIONS
        assert ".png" in constants.IMAGE_EXTENSIONS
        assert ".tif" in constants.IMAGE_EXTENSIONS
        assert ".dcm" in constants.IMAGE_EXTENSIONS

    def test_annotation_extensions_contains_expected_formats(self):
        """Test that ANNOTATION_EXTENSIONS contains expected file extensions."""
        constants = FrameworkConstants()
        
        assert ".csv" in constants.ANNOTATION_EXTENSIONS
        assert ".json" in constants.ANNOTATION_EXTENSIONS
        assert ".xml" in constants.ANNOTATION_EXTENSIONS

    def test_file_formats_are_lowercase(self):
        """Test that FILE_FORMATS contains lowercase format strings."""
        constants = FrameworkConstants()
        
        assert "jpg" in constants.FILE_FORMATS
        assert "png" in constants.FILE_FORMATS
        assert "dicom" in constants.FILE_FORMATS
        # Should not contain uppercase or with dots
        assert "JPG" not in constants.FILE_FORMATS
        assert ".jpg" not in constants.FILE_FORMATS

    def test_modalities_contains_expected_values(self):
        """Test that MODALITIES contains expected imaging modalities."""
        constants = FrameworkConstants()
        
        assert "fundus" in constants.MODALITIES
        assert "oct" in constants.MODALITIES
        assert "fa" in constants.MODALITIES
        assert "uwf" in constants.MODALITIES

    def test_storage_providers_contains_expected_values(self):
        """Test that STORAGE_PROVIDERS contains expected provider types."""
        constants = FrameworkConstants()
        
        assert "local" in constants.STORAGE_PROVIDERS
        assert "s3" in constants.STORAGE_PROVIDERS
        assert "gcs" in constants.STORAGE_PROVIDERS
        assert "azure" in constants.STORAGE_PROVIDERS

    def test_eye_laterality_contains_expected_values(self):
        """Test that EYE_LATERALITY contains expected values."""
        constants = FrameworkConstants()
        
        assert "left" in constants.EYE_LATERALITY
        assert "right" in constants.EYE_LATERALITY
        assert "unknown" in constants.EYE_LATERALITY


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_get_data_root_returns_path(self):
        """Test that get_data_root returns the data root directory path."""
        result = get_data_root()
        assert isinstance(result, Path)

    def test_get_data_root_returns_storage_config_data_root(self):
        """Test that get_data_root returns storage_config.data_root."""
        expected = config.storage_config.data_root
        result = get_data_root()
        assert result == expected

    def test_get_storage_root_returns_path(self):
        """Test that get_storage_root returns the storage root directory path."""
        result = get_storage_root()
        assert isinstance(result, Path)

    def test_get_storage_root_returns_storage_config_local_root(self):
        """Test that get_storage_root returns storage_config.local_root."""
        expected = config.storage_config.local_root
        result = get_storage_root()
        assert result == expected

    def test_get_db_connection_string_returns_string(self):
        """Test that get_db_connection_string returns a connection string."""
        result = get_db_connection_string()
        assert isinstance(result, str)
        assert result.startswith("postgresql://")

    def test_get_db_connection_string_returns_db_config_connection_string(self):
        """Test that get_db_connection_string returns db_config.connection_string."""
        expected = config.db_config.connection_string
        result = get_db_connection_string()
        assert result == expected

    def test_get_db_async_connection_string_returns_string(self):
        """Test that get_db_async_connection_string returns an async connection string."""
        result = get_db_async_connection_string()
        assert isinstance(result, str)
        assert result.startswith("postgresql://")

    def test_get_db_async_connection_string_returns_db_config_async_connection_string(self):
        """Test that get_db_async_connection_string returns db_config.async_connection_string."""
        expected = config.db_config.async_connection_string
        result = get_db_async_connection_string()
        assert result == expected

    def test_connection_strings_contain_expected_format(self):
        """Test that connection strings contain expected components."""
        result = get_db_connection_string()
        # Should contain user, host, port, database
        assert "@" in result  # user@host separator
        assert ":" in result  # port separator
        assert "/" in result  # database separator


class TestGlobalConfigInstances:
    """Tests for global configuration instances."""

    def test_db_config_is_database_config_instance(self):
        """Test that db_config is an instance of DatabaseConfig."""
        assert isinstance(config.db_config, DatabaseConfig)

    def test_storage_config_is_storage_config_instance(self):
        """Test that storage_config is an instance of StorageConfig."""
        assert isinstance(config.storage_config, StorageConfig)

    def test_logging_config_is_logging_config_instance(self):
        """Test that logging_config is an instance of LoggingConfig."""
        assert isinstance(config.logging_config, LoggingConfig)

    def test_constants_is_framework_constants_instance(self):
        """Test that constants is an instance of FrameworkConstants."""
        assert isinstance(config.constants, FrameworkConstants)