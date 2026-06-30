"""
Ingestion framework module.

Provides reusable utilities for dataset ingestion including UUID generation,
file discovery, image processing, mask conversion, and more.
"""

# Annotation file reading
from chaksudb.ingest.framework import annotation_io
from chaksudb.ingest.framework.annotation_io import (
    read_csv_auto,
    read_excel_sheet,
    read_json_file,
)

# File finding and matching
from chaksudb.ingest.framework import file_finder
from chaksudb.ingest.framework.file_finder import (
    find_images,
    find_files_by_extension,
    find_matching_file,
)

# Filename parsing utilities
from chaksudb.ingest.framework import filename_utils
from chaksudb.ingest.framework.filename_utils import (
    extract_laterality,
)

# Folder structure utilities
from chaksudb.ingest.framework import folder_utils
from chaksudb.ingest.framework.folder_utils import (
    get_split_folders,
    get_immediate_subdirs,
)

# Split assignment utilities
from chaksudb.ingest.framework import split_assigner
from chaksudb.ingest.framework.split_assigner import (
    register_dataset_split,
    assign_image_to_split,
    bulk_assign_images_to_split,
    register_standard_splits,
    assign_images_by_split_dict,
)

# Patient registration utilities
from chaksudb.ingest.framework import patient_register
from chaksudb.ingest.framework.patient_register import (
    register_patient,
    link_patient_to_image,
    # Note: The following functions are unused in production scripts:
    # register_patient_with_images, bulk_register_patients,
    # bulk_link_patients_to_images, extract_patient_id_from_filename
    # They are kept for backward compatibility but not exported
)

# Ingestion helpers - generic wrappers
from chaksudb.ingest.framework import ingestion_helpers
from chaksudb.ingest.framework.ingestion_helpers import (
    process_csv,
    process_excel,
    process_json,
    process_text_file,
    process_folder_tree,
    process_paired_files,
    find_file_for_stem,
)

# Image creation helpers with metadata extraction
from chaksudb.ingest.framework import image_helpers
from chaksudb.ingest.framework.image_helpers import (
    get_image_metadata_dict,
    create_image_with_metadata,
)

__all__ = [
    # Submodules
    "annotation_io",
    "file_finder",
    "filename_utils",
    "folder_utils",
    "split_assigner",
    "patient_register",
    "ingestion_helpers",
    "image_helpers",
    # Annotation I/O
    "read_csv_auto",
    "read_excel_sheet",
    "read_json_file",
    # File finder
    "find_images",
    "find_files_by_extension",
    "find_matching_file",
    # Filename utils
    "extract_laterality",
    # Folder utils
    "get_split_folders",
    "get_immediate_subdirs",
    # Split assignment
    "register_dataset_split",
    "assign_image_to_split",
    "bulk_assign_images_to_split",
    "register_standard_splits",
    "assign_images_by_split_dict",
    # Patient registration
    "register_patient",
    "link_patient_to_image",
    # Note: Other patient functions exist but are unused in production
    # Ingestion helpers
    "process_csv",
    "process_excel",
    "process_json",
    "process_text_file",
    "process_folder_tree",
    "process_paired_files",
    "find_file_for_stem",
    # Image helpers
    "get_image_metadata_dict",
    "create_image_with_metadata",
]
