"""
Context variables for automatic provenance tracking in ingestion callbacks.

This module provides thread-safe context variables that allow framework functions
to automatically pass raw_file_id and chain_id to nested callbacks without
explicit parameter passing. This keeps callback signatures simple and clean.

Usage:
    # In framework function (e.g., process_csv):
    token_raw, token_chain = set_provenance_context(raw_file_id, chain_id)
    try:
        await process_row_fn(row, idx)  # Callback doesn't need provenance params
    finally:
        reset_provenance_context(token_raw, token_chain)
    
    # In task processor (e.g., process_disease_grade):
    raw_data_id, provenance_chain_id = get_current_provenance()
    # Use these IDs when creating annotations
"""

import uuid
from contextvars import ContextVar
from typing import Optional, Tuple

# Context variables for current raw file and provenance chain
_current_raw_file_id: ContextVar[Optional[uuid.UUID]] = ContextVar(
    'raw_file_id', default=None
)
_current_chain_id: ContextVar[Optional[uuid.UUID]] = ContextVar(
    'chain_id', default=None
)


def get_current_provenance() -> Tuple[Optional[uuid.UUID], Optional[uuid.UUID]]:
    """
    Get current raw_file_id and chain_id from context.
    
    This function is called by task processors to automatically get provenance
    information without explicit parameter passing.
    
    Returns:
        Tuple of (raw_file_id, chain_id), both can be None if not set
        
    Example:
        ```python
        # In a task processor
        raw_data_id, provenance_chain_id = get_current_provenance()
        
        grading = DiseaseGrading(
            image_id=image_id,
            raw_data_id=raw_data_id,  # Automatically from context
            provenance_chain_id=provenance_chain_id,  # Automatically from context
            # ... other fields
        )
        ```
    """
    return _current_raw_file_id.get(), _current_chain_id.get()


def set_provenance_context(
    raw_file_id: uuid.UUID,
    chain_id: uuid.UUID
) -> Tuple:
    """
    Set provenance context for current execution scope.
    
    This function is called by framework functions (process_csv, process_folder_tree,
    etc.) to set the current provenance context before processing callbacks.
    
    Args:
        raw_file_id: UUID of the raw annotation file being processed
        chain_id: UUID of the provenance chain
        
    Returns:
        Tuple of (token_raw, token_chain) for cleanup in reset_provenance_context()
        
    Example:
        ```python
        # In framework function
        raw_file_id, chain_id = await register_csv_file(...)
        token_raw, token_chain = set_provenance_context(raw_file_id, chain_id)
        try:
            # Process rows/files - callbacks will have access to provenance
            await process_row_fn(row, idx)
        finally:
            reset_provenance_context(token_raw, token_chain)
        ```
    """
    token_raw = _current_raw_file_id.set(raw_file_id)
    token_chain = _current_chain_id.set(chain_id)
    return token_raw, token_chain


def reset_provenance_context(token_raw, token_chain) -> None:
    """
    Reset provenance context to previous state.
    
    This function must be called in a finally block after set_provenance_context()
    to ensure context is properly cleaned up even if errors occur.
    
    Args:
        token_raw: Token returned from set_provenance_context()
        token_chain: Token returned from set_provenance_context()
        
    Example:
        ```python
        token_raw, token_chain = set_provenance_context(raw_file_id, chain_id)
        try:
            await process_row_fn(row, idx)
        finally:
            reset_provenance_context(token_raw, token_chain)  # Always cleanup
        ```
    """
    _current_raw_file_id.reset(token_raw)
    _current_chain_id.reset(token_chain)
