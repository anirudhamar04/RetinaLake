"""
Internal helper functions for database operations.

Provides utilities for JSONB serialization and query building.
"""

import json
from typing import Any, Optional

from psycopg import sql


def _serialize_jsonb(value: Optional[dict[str, Any]]) -> Optional[str]:
    """Serialize dict to JSON string for JSONB fields."""
    if value is None:
        return None
    return json.dumps(value, sort_keys=True)


def _prepare_upsert_query(
    table_name: str,
    columns: list[str],
    conflict_target: list[str],
    update_columns: Optional[list[str]] = None,
) -> sql.Composed:
    """
    Build an INSERT ... ON CONFLICT ... DO UPDATE query.

    Args:
        table_name: Table name
        columns: List of column names
        conflict_target: Columns for conflict detection
        update_columns: Columns to update on conflict (None = all except conflict target)

    Returns:
        SQL query object
    """
    if update_columns is None:
        # Update all columns except those in conflict target
        update_columns = [col for col in columns if col not in conflict_target]

    placeholders = sql.SQL(", ").join(sql.Placeholder() * len(columns))
    column_names = sql.SQL(", ").join(map(sql.Identifier, columns))
    conflict_cols = sql.SQL(", ").join(map(sql.Identifier, conflict_target))
    update_set = sql.SQL(", ").join(
        sql.SQL("{} = EXCLUDED.{}").format(
            sql.Identifier(col), sql.Identifier(col)
        )
        for col in update_columns
    )

    query = sql.SQL(
        """
        INSERT INTO {table} ({columns})
        VALUES ({placeholders})
        ON CONFLICT ({conflict_target}) DO UPDATE
        SET {update_set}
        """
    ).format(
        table=sql.Identifier(table_name),
        columns=column_names,
        placeholders=placeholders,
        conflict_target=conflict_cols,
        update_set=update_set,
    )
    return query
