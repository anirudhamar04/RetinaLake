"""
Tests for path management utilities (chaksudb/storage/paths.py).

Tests path normalization, relative path computation, directory management,
and storage path generation based on docstring specifications.
"""

import pytest
import tempfile
from pathlib import Path

from chaksudb.storage.paths import (
    normalize_path,
    compute_relative_path,
    get_storage_root,
    ensure_storage_directory,
    generate_storage_path,
    resolve_storage_path,
    get_file_size,
)


# ============================================
# normalize_path() tests
# ============================================


def test_normalize_path_converts_backslashes_to_forward_slashes():
    """Test that normalize_path converts Windows-style backslashes to forward slashes."""
    path = r"C:\Users\test\file.jpg"
    result = normalize_path(path)
    
    assert "\\" not in result
    assert result == "C:/Users/test/file.jpg"


def test_normalize_path_removes_duplicate_slashes():
    """Test that normalize_path removes duplicate slashes from path."""
    path = "path//to//file.jpg"
    result = normalize_path(path)
    
    # Note: function does single pass replacement, so // becomes /
    assert result == "path/to/file.jpg"


def test_normalize_path_preserves_protocol_slashes():
    """Test that normalize_path preserves protocol prefixes like http://."""
    path = "http://example.com//path//to//file.jpg"
    result = normalize_path(path)
    
    # Protocol slashes should be preserved
    assert result.startswith("http://")
    # Other duplicate slashes should be removed
    assert result == "http://example.com/path/to/file.jpg"


def test_normalize_path_handles_https_protocol():
    """Test that normalize_path handles HTTPS protocol correctly."""
    path = "https://example.com//path//file.jpg"
    result = normalize_path(path)
    
    assert result == "https://example.com/path/file.jpg"


def test_normalize_path_returns_empty_string_for_empty_input():
    """Test that normalize_path returns empty string for empty input."""
    result = normalize_path("")
    assert result == ""


def test_normalize_path_returns_none_for_none_input():
    """Test that normalize_path returns None for None input."""
    result = normalize_path(None)
    assert result is None


def test_normalize_path_handles_mixed_slashes():
    """Test that normalize_path handles mixed forward and back slashes."""
    path = r"path/to\file/name.jpg"
    result = normalize_path(path)
    
    assert "\\" not in result
    assert result == "path/to/file/name.jpg"


# ============================================
# compute_relative_path() tests
# ============================================


def test_compute_relative_path_returns_relative_path():
    """Test that compute_relative_path returns relative path from absolute path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir)
        absolute_path = dataset_root / "subdir" / "file.jpg"
        
        result = compute_relative_path(absolute_path, dataset_root)
        
        assert result == "subdir/file.jpg"


def test_compute_relative_path_normalizes_with_forward_slashes():
    """Test that compute_relative_path normalizes result with forward slashes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir)
        absolute_path = dataset_root / "subdir" / "nested" / "file.jpg"
        
        result = compute_relative_path(absolute_path, dataset_root)
        
        # Should use forward slashes
        assert "\\" not in result
        assert "/" in result or result == "file.jpg"  # Single file might not have slash


def test_compute_relative_path_raises_error_when_path_not_within_root():
    """Test that compute_relative_path raises ValueError when path is not within dataset root."""
    with tempfile.TemporaryDirectory() as tmpdir1, tempfile.TemporaryDirectory() as tmpdir2:
        dataset_root = Path(tmpdir1)
        absolute_path = Path(tmpdir2) / "file.jpg"
        
        with pytest.raises(ValueError, match="is not within dataset root"):
            compute_relative_path(absolute_path, dataset_root)


def test_compute_relative_path_resolves_absolute_paths():
    """Test that compute_relative_path resolves paths to absolute before computing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir)
        # Create a subdirectory
        subdir = dataset_root / "subdir"
        subdir.mkdir()
        
        # Use relative path from current directory
        absolute_path = subdir / "file.jpg"
        
        result = compute_relative_path(absolute_path, dataset_root)
        
        assert "subdir" in result


def test_compute_relative_path_handles_same_directory():
    """Test that compute_relative_path handles file in same directory as root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir)
        absolute_path = dataset_root / "file.jpg"
        
        result = compute_relative_path(absolute_path, dataset_root)
        
        assert result == "file.jpg"


# ============================================
# get_storage_root() tests
# ============================================


def test_get_storage_root_returns_path():
    """Test that get_storage_root returns a Path object."""
    result = get_storage_root()
    
    assert isinstance(result, Path)


def test_get_storage_root_returns_storage_config_local_root():
    """Test that get_storage_root is a convenience function that wraps config."""
    from chaksudb.config.config import storage_config
    
    result = get_storage_root()
    
    # Should return the same as storage_config.local_root
    assert result == storage_config.local_root


# ============================================
# ensure_storage_directory() tests
# ============================================


def test_ensure_storage_directory_creates_directory():
    """Test that ensure_storage_directory creates directory if it doesn't exist."""
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = Path(tmpdir) / "new_directory"
        
        assert not new_dir.exists()
        
        ensure_storage_directory(new_dir)
        
        assert new_dir.exists()
        assert new_dir.is_dir()


def test_ensure_storage_directory_creates_parent_directories():
    """Test that ensure_storage_directory creates parent directories."""
    with tempfile.TemporaryDirectory() as tmpdir:
        nested_dir = Path(tmpdir) / "parent" / "child" / "grandchild"
        
        ensure_storage_directory(nested_dir)
        
        assert nested_dir.exists()
        assert nested_dir.is_dir()


def test_ensure_storage_directory_does_nothing_if_directory_exists():
    """Test that ensure_storage_directory takes no action if directory already exists."""
    with tempfile.TemporaryDirectory() as tmpdir:
        existing_dir = Path(tmpdir)
        
        # Should not raise error
        ensure_storage_directory(existing_dir)
        
        assert existing_dir.exists()


def test_ensure_storage_directory_raises_error_if_path_is_file():
    """Test that ensure_storage_directory raises ValueError if path exists but is a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        file_path = Path(tmpdir) / "file.txt"
        file_path.write_text("test content")
        
        with pytest.raises(ValueError, match="Path exists but is not a directory"):
            ensure_storage_directory(file_path)


# ============================================
# generate_storage_path() tests
# ============================================


def test_generate_storage_path_returns_path_with_dataset_name():
    """Test that generate_storage_path returns path organized by dataset name."""
    result = generate_storage_path("EYEPACS")
    
    assert isinstance(result, Path)
    assert "EYEPACS" in str(result)


def test_generate_storage_path_includes_subdirectory():
    """Test that generate_storage_path includes optional subdirectory."""
    result = generate_storage_path("EYEPACS", subdirectory="masks")
    
    assert "EYEPACS" in str(result)
    assert "masks" in str(result)


def test_generate_storage_path_includes_filename():
    """Test that generate_storage_path includes optional filename."""
    result = generate_storage_path("EYEPACS", subdirectory="masks", filename="mask_123.png")
    
    assert "EYEPACS" in str(result)
    assert "masks" in str(result)
    assert "mask_123.png" in str(result)


def test_generate_storage_path_without_subdirectory():
    """Test that generate_storage_path works without subdirectory."""
    result = generate_storage_path("EYEPACS", filename="image.jpg")
    
    assert "EYEPACS" in str(result)
    assert "image.jpg" in str(result)


def test_generate_storage_path_returns_directory_path_without_filename():
    """Test that generate_storage_path returns directory path when filename is None."""
    result = generate_storage_path("EYEPACS", subdirectory="processed")
    
    # Should end with directory name, not a file
    assert str(result).endswith("processed")


def test_generate_storage_path_ensures_parent_directory_exists():
    """Test that generate_storage_path ensures parent directory exists."""
    # This should create the necessary directories
    with tempfile.TemporaryDirectory() as tmpdir:
        # Temporarily override storage root
        from chaksudb.config.config import storage_config
        original_root = storage_config.local_root
        storage_config.local_root = Path(tmpdir)
        
        try:
            result = generate_storage_path("TEST_DATASET", subdirectory="test_subdir", filename="test.txt")
            
            # Parent directory should exist
            assert result.parent.exists()
            assert result.parent.is_dir()
        finally:
            storage_config.local_root = original_root


def test_generate_storage_path_starts_with_storage_root():
    """Test that generate_storage_path returns path starting with storage root."""
    storage_root = get_storage_root()
    result = generate_storage_path("EYEPACS")
    
    # Result should be under storage root
    assert str(result).startswith(str(storage_root))


# ============================================
# resolve_storage_path() tests
# ============================================


def test_resolve_storage_path_returns_absolute_path():
    """Test that resolve_storage_path converts relative path to absolute path."""
    result = resolve_storage_path("relative/path/to/file.jpg")
    
    assert isinstance(result, Path)
    assert result.is_absolute()


def test_resolve_storage_path_with_dataset_root():
    """Test that resolve_storage_path resolves relative to dataset_root when provided."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir)
        relative_path = "subdir/file.jpg"
        
        result = resolve_storage_path(relative_path, dataset_root=dataset_root)
        
        assert result.is_absolute()
        assert str(dataset_root) in str(result)


def test_resolve_storage_path_without_dataset_root():
    """Test that resolve_storage_path resolves relative to current working directory when dataset_root is None."""
    relative_path = "path/to/file.jpg"
    
    result = resolve_storage_path(relative_path, dataset_root=None)
    
    # Should resolve relative to current working directory
    assert result.is_absolute()


def test_resolve_storage_path_resolves_path_object():
    """Test that resolve_storage_path returns resolved Path object."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dataset_root = Path(tmpdir)
        relative_path = "file.jpg"
        
        result = resolve_storage_path(relative_path, dataset_root=dataset_root)
        
        expected = (dataset_root / relative_path).resolve()
        assert result == expected


# ============================================
# get_file_size() tests
# ============================================


def test_get_file_size_returns_file_size_in_bytes():
    """Test that get_file_size returns file size in bytes."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        content = b"Hello, World!"
        tmp_file.write(content)
        tmp_file.flush()
        
        try:
            result = get_file_size(tmp_path)
            
            assert isinstance(result, int)
            assert result == len(content)
        finally:
            tmp_path.unlink()


def test_get_file_size_raises_error_for_nonexistent_file():
    """Test that get_file_size raises FileNotFoundError if file does not exist."""
    non_existent = Path("/this/path/does/not/exist/file.jpg")
    
    with pytest.raises(FileNotFoundError, match="File not found"):
        get_file_size(non_existent)


def test_get_file_size_raises_error_for_directory():
    """Test that get_file_size raises ValueError if path is not a file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        dir_path = Path(tmpdir)
        
        with pytest.raises(ValueError, match="Path is not a file"):
            get_file_size(dir_path)


def test_get_file_size_returns_zero_for_empty_file():
    """Test that get_file_size returns 0 for empty file."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        # Don't write anything, file should be empty
        
        try:
            result = get_file_size(tmp_path)
            
            assert result == 0
        finally:
            tmp_path.unlink()


def test_get_file_size_handles_large_files():
    """Test that get_file_size correctly handles files of various sizes."""
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_path = Path(tmp_file.name)
        # Write 1KB of data
        content = b"x" * 1024
        tmp_file.write(content)
        tmp_file.flush()
        
        try:
            result = get_file_size(tmp_path)
            
            assert result == 1024
        finally:
            tmp_path.unlink()
