"""
Patient-related database operations.
"""

from typing import Sequence

import psycopg

from chaksudb.db.connection import get_connection
from chaksudb.db.models import Patient
from chaksudb.db.queries.helpers import _prepare_upsert_query, _serialize_jsonb


async def _execute_upsert(
    conn: psycopg.AsyncConnection,
    table_name: str,
    columns: list[str],
    values: tuple,
    conflict_target: list[str],
    update_columns: list[str] | None = None,
) -> None:
    """Execute a single upsert operation."""
    query = _prepare_upsert_query(table_name, columns, conflict_target, update_columns)
    async with conn.cursor() as cur:
        await cur.execute(query, values)


async def _bulk_upsert(
    conn: psycopg.AsyncConnection,
    table_name: str,
    columns: list[str],
    values_list: Sequence[tuple],
    conflict_target: list[str],
    update_columns: list[str] | None = None,
    batch_size: int = 1000,
) -> int:
    """Execute bulk upsert operations in batches."""
    query = _prepare_upsert_query(table_name, columns, conflict_target, update_columns)
    total_rows = 0

    for i in range(0, len(values_list), batch_size):
        batch = values_list[i : i + batch_size]
        async with conn.cursor() as cur:
            await cur.executemany(query, batch)
            total_rows += len(batch)

    return total_rows


async def upsert_patient(patient: Patient) -> None:
    """Upsert a patient record."""
    async with get_connection() as conn:
        columns = [
            "patient_id",
            "dataset_id",
            "original_patient_id",
            "age",
            "sex",
            "ethnicity",
            "nationality",
            "comorbidities",
            "created_at",
        ]
        values = (
            patient.patient_id,
            patient.dataset_id,
            patient.original_patient_id,
            patient.age,
            patient.sex,
            patient.ethnicity,
            patient.nationality,
            _serialize_jsonb(patient.comorbidities),
            patient.created_at,
        )
        await _execute_upsert(
            conn,
            "patients",
            columns,
            values,
            conflict_target=["dataset_id", "original_patient_id"],
        )


async def bulk_upsert_patients(
    patients: Sequence[Patient], batch_size: int = 1000
) -> int:
    """Bulk upsert patient records."""
    async with get_connection() as conn:
        columns = [
            "patient_id",
            "dataset_id",
            "original_patient_id",
            "age",
            "sex",
            "ethnicity",
            "nationality",
            "comorbidities",
            "created_at",
        ]
        values_list = [
            (
                p.patient_id,
                p.dataset_id,
                p.original_patient_id,
                p.age,
                p.sex,
                p.ethnicity,
                p.nationality,
                _serialize_jsonb(p.comorbidities),
                p.created_at,
            )
            for p in patients
        ]
        return await _bulk_upsert(
            conn,
            "patients",
            columns,
            values_list,
            conflict_target=["dataset_id", "original_patient_id"],
            batch_size=batch_size,
        )
