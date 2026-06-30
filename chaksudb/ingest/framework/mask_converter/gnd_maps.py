"""HEI-MED .map.gz exudate map loader.

The HEI-MED dataset encodes pixel-level exudate locations in gzip-compressed
float32 binary files (`*.map.gz`).  Spatial blob labels (Exudate type names)
live in the companion `.GND` text files; however `.GND` contains no pixel
coordinates, so the actual mask must be derived from `.map.gz`.

File format (confirmed by inspection):
    Bytes 0-3   : uint32 or NaN sentinel (ignored)
    Bytes 4-7   : float32 reinterpreted as uint32 → image height  (e.g. 2196)
    Bytes 8-11  : float32 reinterpreted as uint32 → image width   (e.g. 1958)
    Bytes 12+   : height × width float32 values (row-major, C-order)
                  Non-zero values mark exudate pixels; the magnitudes are
                  subnormal floats (1e-45 to 4e-43) and carry no probability
                  meaning — only zero/non-zero matters for segmentation.

Usage::

    from chaksudb.ingest.framework.mask_converter.gnd_maps import (
        load_exudate_map_gz,
        parse_gnd_blob_count,
    )

    # Returns binary uint8 mask (255 = exudate, 0 = background)
    mask = load_exudate_map_gz(Path("image.map.gz"))

    # Optionally count annotated exudate blobs from the .GND text file
    n_blobs = parse_gnd_blob_count(Path("image.GND"))
"""

import gzip
import logging
import struct
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


def _read_map_header(data: bytes) -> Tuple[int, int]:
    """Parse the 12-byte header and return (height, width).

    The second and third float32 words are actually uint32-encoded integers
    representing height and width.  The first word is a NaN sentinel.
    """
    _, h_raw, w_raw = struct.unpack("<fff", data[:12])
    # Reinterpret the bit pattern as uint32 to get the integer dimensions
    h_bytes = struct.pack("<f", h_raw)
    w_bytes = struct.pack("<f", w_raw)
    height = struct.unpack("<I", h_bytes)[0]
    width = struct.unpack("<I", w_bytes)[0]
    return height, width


def load_exudate_map_gz(
    map_gz_path: Path,
    expected_shape: Optional[Tuple[int, int]] = None,
) -> np.ndarray:
    """Load a HEI-MED `.map.gz` file and return a binary exudate mask.

    Args:
        map_gz_path: Path to the `.map.gz` file.
        expected_shape: Optional (height, width) for sanity-check.  If the
            shape encoded in the header does not match, a warning is logged
            but the header shape is used.

    Returns:
        Binary uint8 numpy array of shape (height, width) where 255 marks
        exudate pixels and 0 marks background.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is too short to contain a valid header.
    """
    if not map_gz_path.exists():
        raise FileNotFoundError(f"Map file not found: {map_gz_path}")

    with gzip.open(map_gz_path, "rb") as fh:
        data = fh.read()

    if len(data) < 12:
        raise ValueError(
            f"File too short to contain a valid header: {map_gz_path} "
            f"({len(data)} bytes)"
        )

    height, width = _read_map_header(data)

    expected_pixels = height * width
    available_floats = (len(data) - 12) // 4

    if available_floats < expected_pixels:
        logger.warning(
            "map.gz %s: header claims %dx%d (%d pixels) but only %d float32 "
            "values are available; file may be truncated.",
            map_gz_path.name,
            height,
            width,
            expected_pixels,
            available_floats,
        )
        # Fall back to the data that IS available
        expected_pixels = available_floats
        # Recompute shape as a single row
        height, width = 1, expected_pixels

    if expected_shape is not None and (height, width) != expected_shape:
        logger.warning(
            "map.gz %s: header shape (%d, %d) differs from expected %s.",
            map_gz_path.name,
            height,
            width,
            expected_shape,
        )

    arr = np.frombuffer(data[12 : 12 + expected_pixels * 4], dtype=np.float32).copy()
    arr2d = arr.reshape(height, width)

    # Any finite non-zero value marks an exudate pixel.
    # Explicitly exclude NaN/Inf: numpy evaluates (NaN != 0) as True,
    # which would incorrectly mark NaN pixels as exudate.
    binary_mask = (np.isfinite(arr2d) & (arr2d != 0)).astype(np.uint8) * 255

    logger.debug(
        "Loaded exudate mask from %s: shape=%s, exudate_pixels=%d",
        map_gz_path.name,
        binary_mask.shape,
        np.count_nonzero(binary_mask),
    )

    return binary_mask


def parse_gnd_blob_count(gnd_path: Path) -> int:
    """Return the number of annotated exudate blobs from a `.GND` text file.

    The `.GND` format (HEI-MED):
        Line 1  : integer — number of blobs
        Lines 2…N+1 : blob type label (e.g. "Exudate 1")
        Remaining lines : characteristics/manifestations counts and "NA" entries.

    Args:
        gnd_path: Path to the `.GND` annotation file.

    Returns:
        Number of blobs (0 if file is missing or cannot be parsed).
    """
    if not gnd_path.exists():
        logger.debug("GND file not found: %s", gnd_path)
        return 0

    try:
        with open(gnd_path, "r", encoding="utf-8", errors="replace") as fh:
            first_line = fh.readline().strip()
        return int(first_line)
    except (ValueError, OSError) as exc:
        logger.warning("Could not parse blob count from %s: %s", gnd_path, exc)
        return 0


def parse_meta_file(meta_path: Path) -> dict:
    """Parse a HEI-MED `.meta` file into a key→value dictionary.

    Format: each line is ``~Key~Value`` (tilde-delimited).
    Empty lines and lines not starting with ``~`` are skipped.

    Common keys returned:
        - ``ImageName``
        - ``PatientGender`` (``"M"`` or ``"F"``)
        - ``PatientRace``
        - ``PatientDOB`` (date string)
        - ``QualityValue`` (float string)
        - ``DiabetesType``
        - ``ONrow``, ``ONcol`` (optic nerve head pixel coordinates)

    Args:
        meta_path: Path to the `.meta` file.

    Returns:
        Dictionary of parsed key→value pairs (both as strings).
        Empty dict if file is missing or unreadable.
    """
    result: dict = {}
    if not meta_path.exists():
        logger.debug("Meta file not found: %s", meta_path)
        return result

    try:
        with open(meta_path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line = line.strip()
                if not line.startswith("~"):
                    continue
                parts = line.split("~")
                # Format: ['', 'Key', 'Value'] (leading ~ produces empty first token)
                if len(parts) >= 3:
                    key = parts[1].strip()
                    value = parts[2].strip()
                    if key:
                        result[key] = value
    except OSError as exc:
        logger.warning("Could not read meta file %s: %s", meta_path, exc)

    return result
