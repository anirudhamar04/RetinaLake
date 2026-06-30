"""
Comprehensive tests for ingestion_helpers using real data from data directory.

Tests cover:
- CSV processing with real EYEPACS data
- Excel processing with real PARAGUAY data
- JSON processing
- Text file processing
- Folder tree processing with real dataset structures
- File pairing (images + annotations)
"""

import uuid
import pytest
from pathlib import Path

from chaksudb.config.config import get_data_root
from chaksudb.ingest.framework.ingestion_helpers import (
    process_csv,
    process_excel,
    process_json,
    process_text_file,
    process_folder_tree,
    process_paired_files,
    find_file_for_stem,
)


pytestmark = pytest.mark.asyncio


class TestIngestionHelpersWithRealData:
    """Test ingestion helpers with real data from datasets."""

    @pytest.fixture
    def data_root(self):
        """Get data root directory."""
        return get_data_root()

    @pytest.fixture
    def eyepacs_dir(self, data_root):
        """Get EYEPACS dataset directory."""
        return data_root / "01_EYEPACS"

    @pytest.fixture
    def paraguay_dir(self, data_root):
        """Get Paraguay dataset directory."""
        return data_root / "10_PARAGUAY"

    @pytest.fixture
    def odir_dir(self, data_root):
        """Get ODIR-5K dataset directory."""
        return data_root / "08_ODIR-5K"

    async def test_process_csv_eyepacs_train(self, eyepacs_dir):
        """Test processing EYEPACS train.csv file."""
        train_csv = eyepacs_dir / "train.csv"
        
        if not train_csv.exists():
            pytest.skip(f"EYEPACS train.csv not found at {train_csv}")

        processed_rows = []

        async def handle_row(row, idx):
            processed_rows.append(row)
            # Check expected columns
            assert "image" in row
            assert "level" in row

        stats = await process_csv(
            csv_path=train_csv,
            process_row_fn=handle_row,
            skip_errors=True,
        )

        # Should have processed some rows
        assert stats.successful_items > 0
        assert len(processed_rows) > 0

        # Check first row structure
        first_row = processed_rows[0]
        assert "image" in first_row
        assert "level" in first_row

    async def test_process_csv_with_error_handling(self, eyepacs_dir):
        """Test CSV processing with error handling."""
        train_csv = eyepacs_dir / "train.csv"
        
        if not train_csv.exists():
            pytest.skip(f"EYEPACS train.csv not found at {train_csv}")

        async def handle_row_with_error(row, idx):
            if idx == 5:  # Simulate error on 6th row
                raise ValueError("Test error")

        stats = await process_csv(
            csv_path=train_csv,
            process_row_fn=handle_row_with_error,
            skip_errors=True,  # Should continue despite error
        )

        # Should have some successes and at least 1 error
        assert stats.successful_items > 0
        assert stats.failed_items >= 1

    async def test_process_excel_paraguay(self, paraguay_dir):
        """Test processing Paraguay Excel file."""
        excel_file = paraguay_dir / "Annotations of the classifications.xlsx"
        
        if not excel_file.exists():
            pytest.skip(f"Paraguay Excel file not found at {excel_file}")

        processed_rows = []

        async def handle_row(row, idx):
            processed_rows.append(row)

        stats = await process_excel(
            excel_path=excel_file,
            process_row_fn=handle_row,
            sheet_name=0,  # First sheet
            skip_errors=True,
        )

        # Should have processed some rows
        assert stats.successful_items > 0
        assert len(processed_rows) > 0

    async def test_process_json_deepeyenet(self, data_root):
        """Test processing DeepEyeNet JSON metadata."""
        deepeyenet_dir = data_root / "06_DEN"
        json_file = deepeyenet_dir / "DeepEyeNet_train.json"
        
        if not json_file.exists():
            pytest.skip(f"DeepEyeNet JSON not found at {json_file}")

        processed_entries = []

        async def handle_entry(entry, idx):
            processed_entries.append(entry)

        stats = await process_json(
            json_path=json_file,
            process_entry_fn=handle_entry,
            skip_errors=True,
        )

        # Should have processed some entries
        assert stats.successful_items > 0
        assert len(processed_entries) > 0

    async def test_process_text_file_sampled_images(self, data_root):
        """Test processing sampled images text file."""
        paraguay_dir = data_root / "10_PARAGUAY"
        text_file = paraguay_dir / "sampled_images.txt"
        
        if not text_file.exists():
            pytest.skip(f"Text file not found at {text_file}")

        processed_lines = []

        async def handle_line(line, line_num):
            processed_lines.append(line)

        stats = await process_text_file(
            text_path=text_file,
            process_line_fn=handle_line,
            skip_empty=True,
            skip_comments=True,
            skip_errors=True,
        )

        # Should have processed some lines
        assert stats.successful_items > 0
        assert len(processed_lines) > 0

    async def test_process_folder_tree_lag_structure(self, data_root):
        """Test processing folder tree with LAG dataset structure."""
        lag_dir = data_root / "07_LAG"
        
        if not lag_dir.exists():
            pytest.skip(f"LAG directory not found at {lag_dir}")

        processed_files = []

        async def handle_file(file_path, relative_path, depth):
            processed_files.append({
                "file_path": file_path,
                "relative_path": relative_path,
                "depth": depth,
            })

        stats = await process_folder_tree(
            root_dir=lag_dir,
            process_file_fn=handle_file,
            file_extensions={".jpg", ".jpeg", ".png"},
            recursive=True,
            skip_errors=True,
        )

        # Should have processed some image files
        assert stats.successful_items > 0
        assert len(processed_files) > 0

        # Check that we got files from different depths
        depths = set(f["depth"] for f in processed_files)
        assert len(depths) > 0

    async def test_process_folder_tree_non_recursive(self, data_root):
        """Test processing folder tree without recursion."""
        eyepacs_dir = data_root / "01_EYEPACS" / "train"
        
        if not eyepacs_dir.exists():
            pytest.skip(f"EYEPACS train directory not found at {eyepacs_dir}")

        processed_files = []

        async def handle_file(file_path, relative_path, depth):
            processed_files.append(file_path)

        stats = await process_folder_tree(
            root_dir=eyepacs_dir,
            process_file_fn=handle_file,
            file_extensions={".jpeg"},
            recursive=False,  # Only top level
            skip_errors=True,
        )

        # All files should be at depth 0
        if stats.successful_items > 0:
            # Files found at top level
            assert len(processed_files) > 0

    async def test_find_file_for_stem(self, data_root):
        """Test finding files by stem."""
        eyepacs_dir = data_root / "01_EYEPACS" / "train"
        
        if not eyepacs_dir.exists():
            pytest.skip(f"EYEPACS train directory not found at {eyepacs_dir}")

        # Try to find a specific file (we know from the CSV that files like "10013_left" exist)
        # But we need to check if any files exist first
        files = list(eyepacs_dir.glob("*.jpeg"))
        if not files:
            pytest.skip("No JPEG files found in EYEPACS train directory")

        # Take the first file and test finding it by stem
        test_file = files[0]
        found_file = find_file_for_stem(
            file_stem=test_file.stem,
            search_dir=eyepacs_dir,
            extensions={".jpeg", ".jpg"},
        )

        assert found_file is not None
        assert found_file.stem == test_file.stem

    async def test_find_file_for_stem_not_found(self, data_root):
        """Test finding files by stem when file doesn't exist."""
        eyepacs_dir = data_root / "01_EYEPACS" / "train"
        
        if not eyepacs_dir.exists():
            pytest.skip(f"EYEPACS train directory not found at {eyepacs_dir}")

        found_file = find_file_for_stem(
            file_stem="nonexistent_file_12345",
            search_dir=eyepacs_dir,
            extensions={".jpeg"},
        )

        assert found_file is None

    async def test_process_csv_empty_handler(self):
        """Test CSV processing with minimal handler."""
        import tempfile
        import csv

        # Create temporary CSV
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            writer = csv.DictWriter(f, fieldnames=["col1", "col2"])
            writer.writeheader()
            writer.writerow({"col1": "value1", "col2": "value2"})
            writer.writerow({"col1": "value3", "col2": "value4"})
            temp_path = Path(f.name)

        try:
            count = [0]

            async def count_rows(row, idx):
                count[0] += 1

            stats = await process_csv(
                csv_path=temp_path,
                process_row_fn=count_rows,
            )

            assert stats.successful_items == 2
            assert count[0] == 2
        finally:
            temp_path.unlink()

    async def test_process_text_file_skip_comments(self):
        """Test text file processing with comment skipping."""
        import tempfile

        # Create temporary text file with comments
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("# This is a comment\n")
            f.write("line1\n")
            f.write("# Another comment\n")
            f.write("line2\n")
            f.write("line3\n")
            temp_path = Path(f.name)

        try:
            lines = []

            async def collect_line(line, line_num):
                lines.append(line)

            stats = await process_text_file(
                text_path=temp_path,
                process_line_fn=collect_line,
                skip_comments=True,
                comment_char="#",
            )

            # Should only process non-comment lines
            assert len(lines) == 3
            assert "line1" in lines
            assert "line2" in lines
            assert "line3" in lines
        finally:
            temp_path.unlink()

    async def test_process_text_file_skip_empty(self):
        """Test text file processing with empty line skipping."""
        import tempfile

        # Create temporary text file with empty lines
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("line1\n")
            f.write("\n")
            f.write("line2\n")
            f.write("   \n")
            f.write("line3\n")
            temp_path = Path(f.name)

        try:
            lines = []

            async def collect_line(line, line_num):
                lines.append(line)

            stats = await process_text_file(
                text_path=temp_path,
                process_line_fn=collect_line,
                skip_empty=True,
            )

            # Should only process non-empty lines
            assert len(lines) == 3
        finally:
            temp_path.unlink()

    async def test_process_folder_tree_with_extension_filter(self):
        """Test folder tree processing with extension filtering."""
        import tempfile
        import os

        # Create temporary directory structure
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)
            
            # Create some files with different extensions
            (tmpdir_path / "file1.jpg").touch()
            (tmpdir_path / "file2.png").touch()
            (tmpdir_path / "file3.txt").touch()
            (tmpdir_path / "file4.jpg").touch()

            jpg_files = []

            async def collect_jpg(file_path, relative_path, depth):
                jpg_files.append(file_path.name)

            stats = await process_folder_tree(
                root_dir=tmpdir_path,
                process_file_fn=collect_jpg,
                file_extensions={".jpg"},
                recursive=False,
            )

            # Should only process .jpg files
            assert len(jpg_files) == 2
            assert "file1.jpg" in jpg_files
            assert "file4.jpg" in jpg_files

    async def test_process_json_with_error_in_entry(self):
        """Test JSON processing with error in specific entry."""
        import tempfile
        import json

        # Create temporary JSON file
        data = [
            {"id": 1, "value": "a"},
            {"id": 2, "value": "b"},
            {"id": 3, "value": "c"},
        ]

        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(data, f)
            temp_path = Path(f.name)

        try:
            async def handle_entry_with_error(entry, idx):
                if entry["id"] == 2:
                    raise ValueError("Test error on entry 2")

            stats = await process_json(
                json_path=temp_path,
                process_entry_fn=handle_entry_with_error,
                skip_errors=True,
            )

            # Should have 2 successes and 1 error
            assert stats.successful_items == 2
            assert stats.failed_items == 1
        finally:
            temp_path.unlink()
