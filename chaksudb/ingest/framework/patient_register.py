"""
Patient registration and patient-image connection utilities.

This module provides utilities for:
- Registering patients with demographic and clinical metadata
- Creating patient-image connections
- Bulk operations for efficient ingestion
"""

import logging
import uuid
from datetime import date, datetime
from typing import Any, Optional, Sequence

from chaksudb.db.models import Patient, PatientImage
from chaksudb.db.queries import bulk_upsert_patient_images, bulk_upsert_patients, upsert_patient, upsert_patient_image
from chaksudb.ingest.framework.gen_uuid import (
    generate_patient_image_uuid,
    generate_patient_uuid,
)

logger = logging.getLogger(__name__)


async def register_patient(
    dataset_id: uuid.UUID,
    original_patient_id: str,
    age: Optional[int] = None,
    sex: Optional[str] = None,
    ethnicity: Optional[str] = None,
    nationality: Optional[str] = None,
    comorbidities: Optional[dict[str, Any]] = None,
) -> uuid.UUID:
    """
    Register a patient record in the database.

    Note: Only register patients if you have actual patient metadata beyond just
    an ID. If you only have a patient ID extracted from a filename, do NOT create
    a patient record - just store the laterality/patient info in the image metadata.

    Args:
        dataset_id: Dataset UUID
        original_patient_id: Original patient identifier from source dataset
        age: Patient age in years
        sex: Patient sex - must be one of: 'male', 'female', 'unknown'
        ethnicity: Patient ethnicity
        nationality: Patient nationality
        comorbidities: Dictionary of comorbidities and other clinical metadata
            e.g., {"diabetes": True, "hypertension": False, "smoking": "former"}

    Returns:
        patient_id UUID of the registered patient

    Example:
        ```python
        # Register a patient with demographic data
        patient_id = await register_patient(
            dataset_id=dataset_id,
            original_patient_id="P12345",
            age=65,
            sex="male",
            ethnicity="Asian",
            comorbidities={
                "diabetes": True,
                "hypertension": True,
                "duration_diabetes_years": 10
            }
        )
        ```
    """
    # Generate deterministic UUID for the patient
    patient_id = generate_patient_uuid(
        dataset_id=dataset_id,
        original_patient_id=original_patient_id,
    )

    # Create patient model
    patient = Patient(
        patient_id=patient_id,
        dataset_id=dataset_id,
        original_patient_id=original_patient_id,
        age=age,
        sex=sex,
        ethnicity=ethnicity,
        nationality=nationality,
        comorbidities=comorbidities,
        created_at=datetime.now(),
    )

    # Store in database (idempotent upsert)
    await upsert_patient(patient)

    logger.debug(
        f"Registered patient '{original_patient_id}' for dataset {dataset_id}"
    )

    return patient_id


async def link_patient_to_image(
    patient_id: uuid.UUID,
    image_id: uuid.UUID,
    exam_date: Optional[date] = None,
) -> uuid.UUID:
    """
    Create a link between a patient and an image.

    Args:
        patient_id: Patient UUID
        image_id: Image UUID
        exam_date: Optional date of the exam/acquisition

    Returns:
        relationship_id UUID of the patient-image relationship

    Example:
        ```python
        # Link a patient to their fundus image
        relationship_id = await link_patient_to_image(
            patient_id=patient_id,
            image_id=image_id,
            exam_date=date(2023, 6, 15)
        )
        ```
    """
    # Generate deterministic UUID for the relationship
    relationship_id = generate_patient_image_uuid(
        patient_id=patient_id,
        image_id=image_id,
    )

    # Create patient-image relationship model
    patient_image = PatientImage(
        relationship_id=relationship_id,
        patient_id=patient_id,
        image_id=image_id,
        exam_date=exam_date,
        created_at=datetime.now(),
    )

    # Store in database (idempotent upsert)
    await upsert_patient_image(patient_image)

    logger.debug(f"Linked patient {patient_id} to image {image_id}")

    return relationship_id


async def register_patient_with_images(
    dataset_id: uuid.UUID,
    original_patient_id: str,
    image_ids: Sequence[uuid.UUID],
    age: Optional[int] = None,
    sex: Optional[str] = None,
    ethnicity: Optional[str] = None,
    nationality: Optional[str] = None,
    comorbidities: Optional[dict[str, Any]] = None,
    exam_date: Optional[date] = None,
) -> tuple[uuid.UUID, list[uuid.UUID]]:
    """
    Register a patient and link them to multiple images in one operation.

    This is a convenience function that combines patient registration with
    image linking.

    Args:
        dataset_id: Dataset UUID
        original_patient_id: Original patient identifier from source dataset
        image_ids: Sequence of image UUIDs to link to this patient
        age: Patient age in years
        sex: Patient sex - must be one of: 'male', 'female', 'unknown'
        ethnicity: Patient ethnicity
        nationality: Patient nationality
        comorbidities: Dictionary of comorbidities and clinical metadata
        exam_date: Optional exam date (applied to all image links)

    Returns:
        Tuple of (patient_id, list of relationship_ids)

    Example:
        ```python
        # Register a patient with both left and right eye images
        patient_id, relationships = await register_patient_with_images(
            dataset_id=dataset_id,
            original_patient_id="P12345",
            image_ids=[left_eye_image_id, right_eye_image_id],
            age=65,
            sex="male",
            exam_date=date(2023, 6, 15)
        )
        ```
    """
    # Register the patient
    patient_id = await register_patient(
        dataset_id=dataset_id,
        original_patient_id=original_patient_id,
        age=age,
        sex=sex,
        ethnicity=ethnicity,
        nationality=nationality,
        comorbidities=comorbidities,
    )

    # Link all images to the patient
    relationship_ids = []
    for image_id in image_ids:
        relationship_id = await link_patient_to_image(
            patient_id=patient_id,
            image_id=image_id,
            exam_date=exam_date,
        )
        relationship_ids.append(relationship_id)

    logger.debug(
        f"Registered patient '{original_patient_id}' and linked to {len(image_ids)} images"
    )

    return patient_id, relationship_ids


async def bulk_register_patients(
    patients_data: Sequence[dict[str, Any]],
    dataset_id: uuid.UUID,
) -> list[uuid.UUID]:
    """
    Bulk register multiple patients efficiently.

    Args:
        patients_data: Sequence of dictionaries containing patient data.
            Each dict should have:
                - original_patient_id: str (required)
                - age: int (optional)
                - sex: str (optional)
                - ethnicity: str (optional)
                - nationality: str (optional)
                - comorbidities: dict (optional)
        dataset_id: Dataset UUID (same for all patients)

    Returns:
        List of patient UUIDs in the same order as input

    Example:
        ```python
        patients = [
            {"original_patient_id": "P001", "age": 65, "sex": "male"},
            {"original_patient_id": "P002", "age": 52, "sex": "female"},
            {"original_patient_id": "P003", "age": 71, "sex": "male"},
        ]
        patient_ids = await bulk_register_patients(patients, dataset_id)
        ```
    """
    patient_models = []
    patient_ids = []

    for patient_data in patients_data:
        patient_id = generate_patient_uuid(
            dataset_id=dataset_id,
            original_patient_id=patient_data["original_patient_id"],
        )
        patient_ids.append(patient_id)

        patient = Patient(
            patient_id=patient_id,
            dataset_id=dataset_id,
            original_patient_id=patient_data["original_patient_id"],
            age=patient_data.get("age"),
            sex=patient_data.get("sex"),
            ethnicity=patient_data.get("ethnicity"),
            nationality=patient_data.get("nationality"),
            comorbidities=patient_data.get("comorbidities"),
            created_at=datetime.now(),
        )
        patient_models.append(patient)

    # Bulk upsert all patients
    await bulk_upsert_patients(patient_models)

    logger.debug(f"Bulk registered {len(patient_models)} patients for dataset {dataset_id}")

    return patient_ids


async def bulk_link_patients_to_images(
    patient_image_pairs: Sequence[tuple[uuid.UUID, uuid.UUID]],
    exam_dates: Optional[Sequence[Optional[date]]] = None,
) -> list[uuid.UUID]:
    """
    Bulk create patient-image links efficiently.

    Args:
        patient_image_pairs: Sequence of (patient_id, image_id) tuples
        exam_dates: Optional sequence of exam dates (same length as pairs)
            If None, all exam_dates will be None

    Returns:
        List of relationship UUIDs in the same order as input

    Example:
        ```python
        # Link multiple patients to their images
        pairs = [
            (patient_id_1, image_id_1),
            (patient_id_2, image_id_2),
            (patient_id_3, image_id_3),
        ]
        dates = [date(2023, 6, 15), date(2023, 6, 16), None]
        relationship_ids = await bulk_link_patients_to_images(pairs, dates)
        ```
    """
    if exam_dates is None:
        exam_dates = [None] * len(patient_image_pairs)

    if len(patient_image_pairs) != len(exam_dates):
        raise ValueError(
            f"Length of patient_image_pairs ({len(patient_image_pairs)}) "
            f"must match length of exam_dates ({len(exam_dates)})"
        )

    patient_image_models = []
    relationship_ids = []

    for (patient_id, image_id), exam_date in zip(patient_image_pairs, exam_dates):
        relationship_id = generate_patient_image_uuid(
            patient_id=patient_id,
            image_id=image_id,
        )
        relationship_ids.append(relationship_id)

        patient_image = PatientImage(
            relationship_id=relationship_id,
            patient_id=patient_id,
            image_id=image_id,
            exam_date=exam_date,
            created_at=datetime.now(),
        )
        patient_image_models.append(patient_image)

    # Bulk upsert all patient-image relationships
    await bulk_upsert_patient_images(patient_image_models)

    logger.debug(f"Bulk linked {len(patient_image_models)} patient-image pairs")

    return relationship_ids


def extract_patient_id_from_filename(
    filename: str,
    pattern: str = "default"
) -> Optional[tuple[str, Optional[str]]]:
    """
    Extract patient ID and laterality from filename using common patterns.

    This is a helper function for parsing patient information from filenames.
    Note: If you only get a patient ID from the filename and have no other
    patient metadata, DO NOT create a patient record. Just store the laterality
    in the image metadata.

    Args:
        filename: The image filename to parse
        pattern: Pattern type to use for extraction:
            - "default": Try common patterns (e.g., "123_left.jpg" -> ("123", "left"))
            - "underscore": Split by underscore (e.g., "P001_right.jpg")
            - "dash": Split by dash (e.g., "P001-left.jpg")

    Returns:
        Tuple of (patient_id, laterality) or None if pattern not matched.
        Laterality will be one of: 'left', 'right', 'unknown', or None

    Example:
        ```python
        # Parse patient ID and laterality from filename
        result = extract_patient_id_from_filename("P12345_left.jpg")
        if result:
            patient_id, laterality = result
            # Only create patient record if you have additional metadata!
        ```
    """
    import re
    from pathlib import Path

    # Remove extension
    name = Path(filename).stem.lower()

    # Common laterality indicators
    laterality_map = {
        "left": "left",
        "l": "left",
        "os": "left",  # Oculus Sinister
        "right": "right",
        "r": "right",
        "od": "right",  # Oculus Dexter
    }

    if pattern in ("default", "underscore"):
        # Try underscore pattern: "PatientID_laterality"
        match = re.match(r"^(.+?)_([a-z]+)$", name)
        if match:
            patient_id = match.group(1)
            lat_str = match.group(2)
            laterality = laterality_map.get(lat_str, "unknown")
            return (patient_id, laterality)

    if pattern in ("default", "dash"):
        # Try dash pattern: "PatientID-laterality"
        match = re.match(r"^(.+?)-([a-z]+)$", name)
        if match:
            patient_id = match.group(1)
            lat_str = match.group(2)
            laterality = laterality_map.get(lat_str, "unknown")
            return (patient_id, laterality)

    # If no pattern matched, return None
    return None
