"""
Model-related database operations.
"""

import psycopg

from chaksudb.db.connection import get_connection
from chaksudb.db.models import Model
from chaksudb.db.queries.helpers import _prepare_upsert_query


async def upsert_model(model: Model) -> None:
    """Upsert a model record."""
    async with get_connection() as conn:
        columns = ["model_id", "model_name", "model_description", "model_url"]
        values = (
            model.model_id,
            model.model_name,
            model.model_description,
            model.model_url,
        )
        query = _prepare_upsert_query("models", columns, conflict_target=["model_id"])
        async with conn.cursor() as cur:
            await cur.execute(query, values)
