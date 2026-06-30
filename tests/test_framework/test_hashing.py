"""
Tests for chaksudb/ingest/framework/hashing.py

Tests file hashing, content hashing, and JSONB hashing functions based on their docstrings.
"""

import hashlib
import json
import pytest
from pathlib import Path

from chaksudb.ingest.framework.hashing import (
    compute_file_hash,
    compute_content_hash,
    compute_jsonb_hash,
)


class TestComputeFileHash:
    """Tests for compute_file_hash function."""

    def test_compute_file_hash_returns_string(self, tmp_path):
        """Test that compute_file_hash returns a string."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        
        result = compute_file_hash(test_file)
        assert isinstance(result, str)

    def test_compute_file_hash_returns_64_char_hex(self, tmp_path):
        """Test that compute_file_hash returns a 64-character hexadecimal string (SHA256)."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        
        result = compute_file_hash(test_file)
        assert len(result) == 64
        # Verify it's hexadecimal
        int(result, 16)  # Will raise ValueError if not hex

    def test_compute_file_hash_is_deterministic(self, tmp_path):
        """Test that compute_file_hash produces same hash for same file content."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("hello world")
        
        hash1 = compute_file_hash(test_file)
        hash2 = compute_file_hash(test_file)
        
        assert hash1 == hash2

    def test_compute_file_hash_different_content_produces_different_hash(self, tmp_path):
        """Test that different file content produces different hashes."""
        file1 = tmp_path / "file1.txt"
        file2 = tmp_path / "file2.txt"
        
        file1.write_text("content1")
        file2.write_text("content2")
        
        hash1 = compute_file_hash(file1)
        hash2 = compute_file_hash(file2)
        
        assert hash1 != hash2

    def test_compute_file_hash_handles_binary_files(self, tmp_path):
        """Test that compute_file_hash handles binary files correctly."""
        test_file = tmp_path / "binary.bin"
        test_file.write_bytes(b"\x00\x01\x02\xff\xfe\xfd")
        
        result = compute_file_hash(test_file)
        assert len(result) == 64

    def test_compute_file_hash_handles_large_files(self, tmp_path):
        """Test that compute_file_hash handles large files efficiently using chunked reading."""
        test_file = tmp_path / "large.bin"
        # Create a 1MB file
        large_content = b"x" * (1024 * 1024)
        test_file.write_bytes(large_content)
        
        result = compute_file_hash(test_file)
        assert len(result) == 64
        
        # Verify it's the correct hash
        expected = hashlib.sha256(large_content).hexdigest()
        assert result == expected

    def test_compute_file_hash_raises_filenotfound_for_nonexistent_file(self, tmp_path):
        """Test that compute_file_hash raises FileNotFoundError for nonexistent file."""
        nonexistent = tmp_path / "does_not_exist.txt"
        
        with pytest.raises(FileNotFoundError, match="File not found"):
            compute_file_hash(nonexistent)

    def test_compute_file_hash_raises_error_for_directory(self, tmp_path):
        """Test that compute_file_hash raises ValueError when given a directory."""
        directory = tmp_path / "testdir"
        directory.mkdir()
        
        with pytest.raises(ValueError, match="Path is not a file"):
            compute_file_hash(directory)

    def test_compute_file_hash_handles_empty_file(self, tmp_path):
        """Test that compute_file_hash handles empty files."""
        test_file = tmp_path / "empty.txt"
        test_file.write_text("")
        
        result = compute_file_hash(test_file)
        assert len(result) == 64
        
        # Empty file should have known SHA256 hash
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected


class TestComputeContentHash:
    """Tests for compute_content_hash function."""

    def test_compute_content_hash_returns_string(self):
        """Test that compute_content_hash returns a string."""
        result = compute_content_hash(b"hello")
        assert isinstance(result, str)

    def test_compute_content_hash_returns_64_char_hex(self):
        """Test that compute_content_hash returns a 64-character hexadecimal string."""
        result = compute_content_hash(b"hello")
        assert len(result) == 64
        int(result, 16)  # Verify it's hexadecimal

    def test_compute_content_hash_is_deterministic(self):
        """Test that compute_content_hash produces same hash for same content."""
        data = b"test content"
        
        hash1 = compute_content_hash(data)
        hash2 = compute_content_hash(data)
        
        assert hash1 == hash2

    def test_compute_content_hash_different_content_produces_different_hash(self):
        """Test that different content produces different hashes."""
        hash1 = compute_content_hash(b"content1")
        hash2 = compute_content_hash(b"content2")
        
        assert hash1 != hash2

    def test_compute_content_hash_handles_empty_bytes(self):
        """Test that compute_content_hash handles empty bytes."""
        result = compute_content_hash(b"")
        assert len(result) == 64
        
        expected = hashlib.sha256(b"").hexdigest()
        assert result == expected

    def test_compute_content_hash_handles_binary_data(self):
        """Test that compute_content_hash handles binary data."""
        binary_data = b"\x00\x01\x02\xff\xfe\xfd"
        result = compute_content_hash(binary_data)
        assert len(result) == 64

    def test_compute_content_hash_matches_hashlib_sha256(self):
        """Test that compute_content_hash produces same result as hashlib.sha256."""
        data = b"test content for verification"
        
        our_hash = compute_content_hash(data)
        expected_hash = hashlib.sha256(data).hexdigest()
        
        assert our_hash == expected_hash


class TestComputeJsonbHash:
    """Tests for compute_jsonb_hash function."""

    def test_compute_jsonb_hash_returns_string(self):
        """Test that compute_jsonb_hash returns a string."""
        result = compute_jsonb_hash({"key": "value"})
        assert isinstance(result, str)

    def test_compute_jsonb_hash_returns_64_char_hex(self):
        """Test that compute_jsonb_hash returns a 64-character hexadecimal string."""
        result = compute_jsonb_hash({"key": "value"})
        assert len(result) == 64
        int(result, 16)  # Verify it's hexadecimal

    def test_compute_jsonb_hash_is_deterministic(self):
        """Test that compute_jsonb_hash produces same hash for same dictionary."""
        data = {"name": "test", "value": 123, "active": True}
        
        hash1 = compute_jsonb_hash(data)
        hash2 = compute_jsonb_hash(data)
        
        assert hash1 == hash2

    def test_compute_jsonb_hash_is_deterministic_regardless_of_key_order(self):
        """Test that compute_jsonb_hash produces same hash regardless of dictionary key order."""
        data1 = {"a": 1, "b": 2, "c": 3}
        data2 = {"c": 3, "a": 1, "b": 2}
        data3 = {"b": 2, "c": 3, "a": 1}
        
        hash1 = compute_jsonb_hash(data1)
        hash2 = compute_jsonb_hash(data2)
        hash3 = compute_jsonb_hash(data3)
        
        assert hash1 == hash2 == hash3

    def test_compute_jsonb_hash_handles_nested_dictionaries(self):
        """Test that compute_jsonb_hash handles nested dictionaries."""
        data = {
            "outer": {
                "inner": {
                    "value": 123
                }
            }
        }
        
        result = compute_jsonb_hash(data)
        assert len(result) == 64

    def test_compute_jsonb_hash_handles_lists(self):
        """Test that compute_jsonb_hash handles lists in dictionary values."""
        data = {
            "items": [1, 2, 3, 4, 5],
            "tags": ["a", "b", "c"]
        }
        
        result = compute_jsonb_hash(data)
        assert len(result) == 64

    def test_compute_jsonb_hash_handles_mixed_types(self):
        """Test that compute_jsonb_hash handles mixed JSON-serializable types."""
        data = {
            "string": "value",
            "integer": 42,
            "float": 3.14,
            "boolean": True,
            "null": None,
            "list": [1, 2, 3],
            "nested": {"key": "value"}
        }
        
        result = compute_jsonb_hash(data)
        assert len(result) == 64

    def test_compute_jsonb_hash_different_values_produce_different_hash(self):
        """Test that different dictionary values produce different hashes."""
        data1 = {"key": "value1"}
        data2 = {"key": "value2"}
        
        hash1 = compute_jsonb_hash(data1)
        hash2 = compute_jsonb_hash(data2)
        
        assert hash1 != hash2

    def test_compute_jsonb_hash_handles_empty_dict(self):
        """Test that compute_jsonb_hash handles empty dictionary."""
        result = compute_jsonb_hash({})
        assert len(result) == 64

    def test_compute_jsonb_hash_handles_unicode(self):
        """Test that compute_jsonb_hash handles Unicode characters."""
        data = {
            "chinese": "你好",
            "emoji": "🎉",
            "arabic": "مرحبا"
        }
        
        result = compute_jsonb_hash(data)
        assert len(result) == 64

    def test_compute_jsonb_hash_raises_error_for_non_serializable(self):
        """Test that compute_jsonb_hash raises ValueError for non-JSON-serializable types."""
        data = {"function": lambda x: x}  # Functions are not JSON-serializable
        
        with pytest.raises(ValueError, match="Cannot serialize data for hashing"):
            compute_jsonb_hash(data)

    def test_compute_jsonb_hash_rejects_nan_and_inf(self):
        """Test that compute_jsonb_hash rejects NaN and Inf for determinism."""
        data = {"value": float('nan')}
        
        with pytest.raises(ValueError, match="Cannot serialize data for hashing"):
            compute_jsonb_hash(data)

    def test_compute_jsonb_hash_consistent_serialization(self):
        """Test that compute_jsonb_hash uses consistent JSON serialization (no whitespace)."""
        data = {"a": 1, "b": 2}
        
        # Manually compute the expected hash with same serialization settings
        json_bytes = json.dumps(
            data,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        expected = hashlib.sha256(json_bytes).hexdigest()
        
        result = compute_jsonb_hash(data)
        assert result == expected

    def test_compute_jsonb_hash_nested_list_order_matters(self):
        """Test that list order matters in compute_jsonb_hash (lists are ordered)."""
        data1 = {"items": [1, 2, 3]}
        data2 = {"items": [3, 2, 1]}
        
        hash1 = compute_jsonb_hash(data1)
        hash2 = compute_jsonb_hash(data2)
        
        # Different order in list should produce different hash
        assert hash1 != hash2
