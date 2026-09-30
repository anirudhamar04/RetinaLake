"""
Foreign key validation and utility query functions.
"""

import uuid

from chaksudb.db.connection import get_connection


async def validate_dataset_exists(dataset_id: uuid.UUID) -> bool:
    """Check if a dataset exists."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1 FROM datasets WHERE dataset_id = %s", (dataset_id,))
            return await cur.fetchone() is not None


async def validate_image_exists(image_id: uuid.UUID) -> bool:
    """Check if an image exists."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1 FROM images WHERE image_id = %s", (image_id,))
            return await cur.fetchone() is not None


async def validate_patient_exists(patient_id: uuid.UUID) -> bool:
    """Check if a patient exists."""
    async with get_connection() as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT 1 FROM patients WHERE patient_id = %s", (patient_id,))
            return await cur.fetchone() is not None
