"""
Path resolution for export: resolve relative paths to full paths using data root and storage root.

Used so that Parquet files and the torch dataset get consistent full paths for images and
mask files. Mask paths may be relative to data root (original masks) or storage root
(processed masks), so we try both roots when resolving.

When STORAGE_IMAGE_SERVER_URL is set, files not found locally are fetched from the lab
HTTP server and cached under STORAGE_IMAGE_CACHE_DIR (~/.cache/chaksudb by default).
The cache is permanent: a file downloaded once is never re-fetched.
"""

import json
import logging
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from chaksudb.config.config import get_data_root, get_storage_root, storage_config
from chaksudb.storage.paths import resolve_storage_path

logger = logging.getLogger(__name__)


def _relative_key(path_str: str) -> str | None:
    """
    Return the path relative to data_root or storage_root, or None if it can't be computed.

    For absolute server-side paths (e.g. /data/chaksudb/raw/EYEPACS/img.jpg)
    that don't exist on this machine, strip the known root prefix to get the URL key.
    For already-relative paths, return as-is.
    """
    p = Path(path_str)
    if not p.is_absolute():
        return path_str
    for root in (get_data_root(), get_storage_root()):
        try:
            return str(p.relative_to(root))
        except ValueError:
            continue
    # Absolute path outside known roots: use the last two components as a best-effort key
    return "/".join(p.parts[-2:]) if len(p.parts) >= 2 else p.name


def _fetch_from_server(path_str: str) -> Path | None:
    """
    Download path_str from the lab image server into the local cache and return the cached path.

    Returns None if IMAGE_SERVER_URL is not configured or the download fails.
    """
    server_url = storage_config.image_server_url
    if not server_url:
        return None

    key = _relative_key(path_str)
    if not key:
        return None

    cache_path = storage_config.image_cache_dir / key
    if cache_path.exists():
        return cache_path

    encoded_key = "/".join(urllib.parse.quote(part, safe="") for part in key.lstrip("/").split("/"))
    url = server_url.rstrip("/") + "/" + encoded_key
    logger.info("Fetching %s → %s", url, cache_path)
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, cache_path)
        return cache_path
    except Exception as e:
        logger.warning("Failed to fetch %s: %s", url, e)
        if cache_path.exists():
            cache_path.unlink(missing_ok=True)
        return None


def resolve_local_path(relative_path: str) -> Path:
    """
    Resolve a relative path to an absolute path by trying data root first, then storage root.

    Used for both main image file_path and mask_file_path in segmentation_masks.
    Some paths are relative to data root (e.g. original images, original masks);
    others are relative to storage root (e.g. processed masks). Trying both roots
    matches the ingest behaviour (roots_to_try in segmentation_processor).

    When STORAGE_IMAGE_SERVER_URL is set and the file is not found locally, the file
    is downloaded from the lab HTTP server and cached permanently under
    STORAGE_IMAGE_CACHE_DIR. Subsequent calls return the cached path immediately.

    Args:
        relative_path: Path string from the database (usually relative).

    Returns:
        Absolute Path that exists on the filesystem.

    Raises:
        ValueError: If relative_path is empty.
        FileNotFoundError: If the file is not found locally or via the image server.
    """
    if not relative_path or not str(relative_path).strip():
        raise ValueError("resolve_local_path requires a non-empty path")

    path_str = str(relative_path).strip()
    path_obj = Path(path_str)

    # Already absolute and exists: use as-is
    if path_obj.is_absolute():
        resolved = path_obj.resolve()
        if resolved.exists():
            return resolved
        # Try HTTP fallback before raising
        cached = _fetch_from_server(path_str)
        if cached:
            return cached
        raise FileNotFoundError(f"Absolute path does not exist: {resolved}")

    # Try data root first, then storage root
    for root in (get_data_root(), get_storage_root()):
        try:
            resolved = resolve_storage_path(path_str, root)
            if resolved.exists():
                return resolved
        except Exception as e:
            logger.debug("resolve_local_path try root %s: %s", root, e)
            continue

    # Try HTTP fallback before raising
    cached = _fetch_from_server(path_str)
    if cached:
        return cached

    # Not found under any root: raise with the last attempted path for clarity
    resolved_data = resolve_storage_path(path_str, get_data_root())
    resolved_storage = resolve_storage_path(path_str, get_storage_root())
    raise FileNotFoundError(
        f"Path not found under data root or storage root: {path_str!r}. "
        f"Tried: {resolved_data}, {resolved_storage}"
        + (
            f"\nHint: set STORAGE_IMAGE_SERVER_URL=http://<host>:8091 to fetch images "
            f"from the lab server automatically."
            if not storage_config.image_server_url
            else f"\nHTTP fetch from {storage_config.image_server_url} also failed."
        )
    )


def resolve_paths_in_row(row: dict[str, Any]) -> dict[str, Any]:
    """
    Return a new row dict with file_path and segmentation_masks[].mask_file_path resolved to full paths.

    Only resolves when storage_provider is "local". Handles segmentation_masks as a list of dicts
    or a JSON string (e.g. when read from Parquet). Other path-bearing fields can be added here later.

    Args:
        row: One export row (from query or Parquet).

    Returns:
        New dict with resolved path strings; original row is not mutated.
    """
    out = dict(row)

    # Resolve main image file_path for local storage
    if out.get("storage_provider") == "local" and out.get("file_path"):
        try:
            out["file_path"] = str(resolve_local_path(out["file_path"]))
        except FileNotFoundError as e:
            logger.warning("Could not resolve file_path in row: %s", e)
            # Keep original so downstream can decide

    # Resolve mask_file_path in segmentation_masks
    masks = out.get("segmentation_masks")
    if masks is None:
        return out

    # Parquet may store JSONB as string
    if isinstance(masks, str):
        try:
            masks = json.loads(masks)
        except json.JSONDecodeError:
            return out
    if not isinstance(masks, list):
        return out

    resolved_masks = []
    for item in masks:
        if not isinstance(item, dict):
            resolved_masks.append(item)
            continue
        copy = dict(item)
        if copy.get("mask_file_path"):
            try:
                copy["mask_file_path"] = str(resolve_local_path(copy["mask_file_path"]))
            except FileNotFoundError as e:
                logger.warning("Could not resolve mask_file_path: %s", e)
        resolved_masks.append(copy)
    out["segmentation_masks"] = resolved_masks

    return out
