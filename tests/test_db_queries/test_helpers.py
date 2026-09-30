"""
Tests for internal helper functions for database operations.

Tests based on docstring specifications only.
"""

import pytest
import json
from psycopg import sql

from chaksudb.db.queries.helpers import _serialize_jsonb, _prepare_upsert_query


def test_serialize_jsonb_with_dict():
    """Test that _serialize_jsonb serializes dict to JSON string.
    
    Based on docstring: 'Serialize dict to JSON string for JSONB fields.'
    Args: value (Optional[dict[str, Any]])
    Returns: Optional[str]
    """
    test_dict = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
    
    result = _serialize_jsonb(test_dict)
    
    assert result is not None
    assert isinstance(result, str)
    
    # Verify it's valid JSON
    parsed = json.loads(result)
    assert parsed["key1"] == "value1"
    assert parsed["key2"] == 42
    assert parsed["key3"] == [1, 2, 3]


def test_serialize_jsonb_with_none():
    """Test that _serialize_jsonb returns None for None input.
    
    Based on docstring: 'Serialize dict to JSON string for JSONB fields.'
    Args: value (Optional[dict[str, Any]])
    Returns: Optional[str]
    """
    result = _serialize_jsonb(None)
    
    assert result is None


def test_serialize_jsonb_sorts_keys():
    """Test that _serialize_jsonb sorts keys for consistent output.
    
    Based on implementation hint showing sort_keys=True in json.dumps.
    """
    test_dict = {"z": 1, "a": 2, "m": 3}
    
    result = _serialize_jsonb(test_dict)
    
    assert result is not None
    # Keys should be sorted alphabetically in the JSON string
    assert result.index('"a"') < result.index('"m"') < result.index('"z"')


def test_serialize_jsonb_with_nested_dict():
    """Test that _serialize_jsonb handles nested dictionaries.
    
    Based on docstring accepting dict[str, Any] which can include nested dicts.
    """
    test_dict = {
        "level1": {
            "level2": {
                "value": "nested"
            }
        }
    }
    
    result = _serialize_jsonb(test_dict)
    
    assert result is not None
    parsed = json.loads(result)
    assert parsed["level1"]["level2"]["value"] == "nested"


def test_prepare_upsert_query_with_default_update_columns():
    """Test that _prepare_upsert_query builds INSERT...ON CONFLICT...DO UPDATE query.
    
    Based on docstring: 'Build an INSERT ... ON CONFLICT ... DO UPDATE query.'
    Args: table_name, columns, conflict_target, update_columns (None = all except conflict target)
    Returns: SQL query object
    """
    table_name = "test_table"
    columns = ["id", "name", "value"]
    conflict_target = ["id"]
    
    query = _prepare_upsert_query(table_name, columns, conflict_target, update_columns=None)
    
    assert query is not None
    assert isinstance(query, sql.Composed)
    
    # Convert to string to verify structure
    query_str = query.as_string(None)
    assert "INSERT INTO" in query_str
    assert "test_table" in query_str
    assert "ON CONFLICT" in query_str
    assert "DO UPDATE" in query_str
    assert "SET" in query_str


def test_prepare_upsert_query_with_explicit_update_columns():
    """Test that _prepare_upsert_query respects explicit update_columns parameter.
    
    Based on docstring: 'update_columns: Columns to update on conflict (None = all except conflict target)'
    """
    table_name = "test_table"
    columns = ["id", "name", "value", "timestamp"]
    conflict_target = ["id"]
    update_columns = ["name", "value"]  # Explicitly specify which columns to update
    
    query = _prepare_upsert_query(table_name, columns, conflict_target, update_columns=update_columns)
    
    assert query is not None
    assert isinstance(query, sql.Composed)
    
    query_str = query.as_string(None)
    assert "INSERT INTO" in query_str
    assert "ON CONFLICT" in query_str
    assert "DO UPDATE" in query_str


def test_prepare_upsert_query_excludes_conflict_target_from_update():
    """Test that _prepare_upsert_query excludes conflict_target columns from update.
    
    Based on docstring: 'update_columns: Columns to update on conflict (None = all except conflict target)'
    This means conflict target columns should not be in the UPDATE SET clause.
    """
    table_name = "test_table"
    columns = ["id", "name", "value"]
    conflict_target = ["id"]
    
    query = _prepare_upsert_query(table_name, columns, conflict_target, update_columns=None)
    
    assert query is not None
    query_str = query.as_string(None)
    
    # The UPDATE SET clause should include name and value but not id
    assert "name" in query_str
    assert "value" in query_str


def test_prepare_upsert_query_with_composite_conflict_target():
    """Test that _prepare_upsert_query handles composite conflict target.
    
    Based on docstring showing conflict_target as list[str], supporting multiple columns.
    """
    table_name = "test_table"
    columns = ["dataset_id", "patient_id", "name", "value"]
    conflict_target = ["dataset_id", "patient_id"]  # Composite key
    
    query = _prepare_upsert_query(table_name, columns, conflict_target, update_columns=None)
    
    assert query is not None
    assert isinstance(query, sql.Composed)
    
    query_str = query.as_string(None)
    assert "INSERT INTO" in query_str
    assert "ON CONFLICT" in query_str
    assert "DO UPDATE" in query_str


def test_prepare_upsert_query_with_all_columns():
    """Test that _prepare_upsert_query includes all columns in INSERT clause.
    
    Based on docstring showing columns parameter is list[str] for all columns to insert.
    """
    table_name = "test_table"
    columns = ["id", "name", "value", "created_at", "updated_at"]
    conflict_target = ["id"]
    
    query = _prepare_upsert_query(table_name, columns, conflict_target, update_columns=None)
    
    assert query is not None
    query_str = query.as_string(None)
    
    # All columns should be in the INSERT clause
    assert "id" in query_str
    assert "name" in query_str
    assert "value" in query_str
    assert "created_at" in query_str
    assert "updated_at" in query_str


def test_prepare_upsert_query_returns_sql_composed():
    """Test that _prepare_upsert_query returns a SQL Composed object.
    
    Based on docstring: 'Returns: SQL query object'
    The function uses psycopg.sql.Composed for query building.
    """
    table_name = "test_table"
    columns = ["id", "name"]
    conflict_target = ["id"]
    
    query = _prepare_upsert_query(table_name, columns, conflict_target, update_columns=None)
    
    # Should return a SQL Composed object
    assert isinstance(query, sql.Composed)
    
    # Should be able to convert to string
    query_str = query.as_string(None)
    assert isinstance(query_str, str)
    assert len(query_str) > 0
