"""
File hashing utilities for SHA256 file hashing and deterministic JSON hashing.

This module provides utilities for computing SHA256 hashes of files and content,
as well as deterministic hashing of JSONB data structures. These hashes are used
for idempotency checks, file deduplication, and content verification.
"""

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

logger = logging.getLogger(__name__)


def compute_file_hash(file_path: Path) -> str:
    """
    Compute SHA256 hash of a file.

    This function reads the file in chunks to handle large files efficiently
    and computes a SHA256 hash of the file content.

    Args:
        file_path: Path to the file to hash

    Returns:
        Hexadecimal SHA256 hash string (64 characters)

    Raises:
        FileNotFoundError: If the file does not exist
        IOError: If the file cannot be read
    """
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    if not file_path.is_file():
        raise ValueError(f"Path is not a file: {file_path}")

    sha256_hash = hashlib.sha256()
    chunk_size = 8192  # 8KB chunks for efficient reading

    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    except IOError as e:
        logger.error(f"Failed to read file for hashing: {file_path}, error: {e}")
        raise IOError(f"Cannot read file for hashing: {file_path}") from e


def _pixel_hash_from_image(img) -> str:
    """SHA256 over decoded RGB pixels of an already-open PIL image (see compute_pixel_hash)."""
    import numpy as np

    arr = np.asarray(img.convert("RGB"))
    h = hashlib.sha256()
    # include shape so different resolutions never collide
    h.update(str(arr.shape).encode("utf-8"))
    h.update(arr.tobytes())
    return h.hexdigest()


def _perceptual_hash_from_image(img) -> str:
    """64-bit dHash of an already-open PIL image (see compute_perceptual_hash)."""
    import numpy as np
    from PIL import Image as PILImage

    # 9x8 grayscale -> 8x8 horizontal gradient = 64 bits
    small = np.asarray(img.convert("L").resize((9, 8), PILImage.BILINEAR), dtype=np.int16)
    diff = small[:, 1:] > small[:, :-1]          # (8, 8) booleans
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return f"{bits:016x}"


def compute_pixel_hash(file_path: Path) -> str:
    """Compute a SHA256 over the *decoded* pixels, independent of the container format.

    Unlike compute_file_hash (which hashes the raw bytes and therefore changes whenever
    the file is re-saved in a different format), this decodes the image to a canonical
    RGB array and hashes that. The same image stored losslessly under two encodings
    (e.g. PNG and BMP, or a re-save) yields the SAME pixel hash.

    Note: lossy re-encodes (e.g. JPEG) alter the pixels slightly and will NOT match here
    — use compute_perceptual_hash for that case.

    Returns a 64-char hex string, or raises if the image cannot be decoded.
    """
    from PIL import Image as PILImage

    with PILImage.open(file_path) as img:
        return _pixel_hash_from_image(img)


def compute_perceptual_hash(file_path: Path) -> str:
    """Compute a 64-bit perceptual hash (dHash) as a 16-char hex string.

    This is the encoding/resolution-aware fingerprint: it decodes the image, reduces it
    to a tiny grayscale gradient signature, so the same picture under different encodings
    (including lossy JPEG re-compression) or a different resolution produces the same — or
    a very close — hash. Compare two phashes with hamming_distance(); 0 means identical
    signature, small values mean near-duplicates.
    """
    from PIL import Image as PILImage

    with PILImage.open(file_path) as img:
        return _perceptual_hash_from_image(img)


def compute_pixel_and_perceptual_hashes(file_path: Path) -> tuple[str, str]:
    """Return (pixel_hash, perceptual_hash) from a *single* decode of the image.

    Equivalent to calling compute_pixel_hash() and compute_perceptual_hash() but opens
    and decodes the file once instead of twice — the decode dominates the cost on the
    ingestion hot path.
    """
    from PIL import Image as PILImage

    with PILImage.open(file_path) as img:
        return _pixel_hash_from_image(img), _perceptual_hash_from_image(img)


def hamming_distance(hash_a: str, hash_b: str) -> int:
    """Bit-difference between two equal-length hex hash strings (e.g. two perceptual hashes)."""
    return bin(int(hash_a, 16) ^ int(hash_b, 16)).count("1")


def compute_content_hash(data: bytes) -> str:
    """
    Compute SHA256 hash of raw bytes content.

    This function computes a SHA256 hash directly from bytes data without
    reading from a file. Useful for hashing in-memory content or data
    that has already been loaded.

    Args:
        data: Bytes content to hash

    Returns:
        Hexadecimal SHA256 hash string (64 characters)
    """
    sha256_hash = hashlib.sha256()
    sha256_hash.update(data)
    return sha256_hash.hexdigest()


def compute_jsonb_hash(data: Dict[str, Any]) -> str:
    """
    Compute deterministic SHA256 hash of a JSONB-compatible dictionary.

    This function ensures deterministic hashing by:
    - Sorting dictionary keys recursively
    - Using consistent JSON serialization (no whitespace, sorted keys)
    - Handling nested dictionaries and lists consistently
    - Using UTF-8 encoding

    The resulting hash is deterministic: the same dictionary structure
    will always produce the same hash, regardless of key order or formatting.

    Args:
        data: Dictionary to hash (must be JSON-serializable)

    Returns:
        Hexadecimal SHA256 hash string (64 characters)

    Raises:
        TypeError: If data contains non-JSON-serializable types
        ValueError: If JSON serialization fails
    """
    try:
        # Serialize to JSON with sorted keys and no whitespace for determinism
        json_bytes = json.dumps(
            data,
            sort_keys=True,
            ensure_ascii=False,
            separators=(",", ":"),  # No whitespace
            allow_nan=False,  # Reject NaN/Inf for determinism
        ).encode("utf-8")

        return compute_content_hash(json_bytes)
    except (TypeError, ValueError) as e:
        logger.error(f"Failed to serialize data for hashing: {e}")
        raise ValueError(f"Cannot serialize data for hashing: {e}") from e
