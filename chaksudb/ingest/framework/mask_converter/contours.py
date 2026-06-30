"""Contour and polygon conversion utilities."""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Try to import CuPy for GPU acceleration (optional)

def _detect_contour_format(contour_path: Path) -> str:
    """
    Auto-detect contour file format by examining file content.

    Returns: "line_separated", "space_separated", "comma_separated", or "json"
    """
    with open(contour_path, "r") as f:
        content = f.read().strip()

    # Check if JSON
    try:
        data = json.loads(content)
        if isinstance(data, list):
            return "json"
    except (json.JSONDecodeError, ValueError):
        pass

    # Check for newlines (line-separated format)
    if "\n" in content:
        lines = [line.strip() for line in content.split("\n") if line.strip()]
        if lines:
            # Check first line format
            first_line = lines[0]
            if "," in first_line:
                parts = first_line.split(",")
                if len(parts) == 2:
                    return "comma_separated"
            else:
                parts = first_line.split()
                if len(parts) == 2:
                    return "line_separated"
    else:
        # Single line format
        if "," in content:
            return "comma_separated"
        else:
            return "space_separated"

    # Default to line_separated
    return "line_separated"


def _parse_contour_coordinates(contour_path: Path, coordinate_format: Optional[str] = None) -> np.ndarray:
    """
    Parse contour coordinates from file (optimized for large files).
    
    Uses numpy vectorized operations for fast parsing of large files (1M+ lines).
    Falls back to original Python loop if numpy.loadtxt fails for edge cases.

    Args:
        contour_path: Path to contour file
        coordinate_format: Format hint ("line_separated", "space_separated", "comma_separated", "json").
                          If None, auto-detects format.

    Returns:
        Array of coordinates shape (N, 2) with dtype int32
    """
    if coordinate_format is None:
        coordinate_format = _detect_contour_format(contour_path)

    if coordinate_format == "json":
        with open(contour_path, "r") as f:
            coords = json.load(f)
        if not isinstance(coords, list):
            raise ValueError(f"JSON contour must be a list of [x, y] pairs")
        points = np.array(coords, dtype=np.float64)
        if points.shape[1] != 2:
            raise ValueError(f"Each coordinate must have 2 values (x, y), got shape {points.shape}")
    
    elif coordinate_format == "line_separated":
        # One coordinate per line: "x y" - USE NUMPY LOADTXT (FAST!)
        # Try numpy.loadtxt first (fast), fallback to Python loop if it fails
        file_size_mb = contour_path.stat().st_size / (1024 * 1024)
        if file_size_mb > 1.0:
            logger.info(f"Parsing contour file: {contour_path.name} ({file_size_mb:.1f} MB) using numpy.loadtxt...")
        else:
            logger.debug(f"Parsing contour file: {contour_path.name} ({file_size_mb:.1f} MB) using numpy.loadtxt...")
        
        try:
            points = np.loadtxt(
                contour_path,
                dtype=np.float64,
                comments=None,  # Don't treat any character as comment
                delimiter=None,  # Auto-detect whitespace delimiter
                usecols=(0, 1),  # Only read first 2 columns (ignores extra columns)
                unpack=False,  # Return as (N, 2) array
            )
            # Validate shape
            if len(points.shape) != 2 or points.shape[1] != 2:
                raise ValueError(f"Expected 2 columns, got shape {points.shape}")
            
            # Use INFO if >100K points, otherwise DEBUG
            if len(points) > 100000:
                logger.info(f"  ✓ Successfully parsed {len(points):,} points using numpy.loadtxt (fast path)")
            else:
                logger.debug(f"  ✓ Successfully parsed {len(points):,} points using numpy.loadtxt (fast path)")
        except (ValueError, IndexError, IOError) as e:
            # Fallback to original Python loop for edge cases
            logger.warning(f"  ⚠ numpy.loadtxt failed for {contour_path.name}, using fallback parser (slow): {e}")
            logger.warning(f"  Error type: {type(e).__name__}, Error message: {str(e)}")
            import traceback
            logger.debug(f"  Full traceback:\n{traceback.format_exc()}")
            
            with open(contour_path, "r") as f:
                content = f.read().strip()
            lines = content.split("\n")
            points_list = []
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 2:
                    raise ValueError(f"Each line must have exactly 2 values (x, y), got: {line}")
                points_list.append([float(parts[0]), float(parts[1])])
            points = np.array(points_list, dtype=np.float64)
            # Use INFO if >100K points, otherwise DEBUG
            if len(points) > 100000:
                logger.info(f"  ✓ Fallback parser completed: {len(points):,} points parsed")
            else:
                logger.debug(f"  ✓ Fallback parser completed: {len(points):,} points parsed")
    
    elif coordinate_format == "space_separated":
        # Single line: "x1 y1 x2 y2 ..." - USE NUMPY LOADTXT
        file_size_mb = contour_path.stat().st_size / (1024 * 1024)
        logger.info(f"Parsing space-separated contour: {contour_path.name} ({file_size_mb:.1f} MB) using numpy.loadtxt...")
        
        try:
            values = np.loadtxt(
                contour_path,
                dtype=np.float64,
                comments=None,
                delimiter=None,
            )
            if len(values) % 2 != 0:
                raise ValueError(
                    f"Contour file must have even number of values (x, y pairs), got {len(values)}"
                )
            points = values.reshape(-1, 2)
            # Use INFO if >100K points, otherwise DEBUG
            if len(points) > 100000:
                logger.info(f"  ✓ Successfully parsed {len(points):,} points using numpy.loadtxt (fast path)")
            else:
                logger.debug(f"  ✓ Successfully parsed {len(points):,} points using numpy.loadtxt (fast path)")
        except (ValueError, IOError) as e:
            # Fallback to original
            logger.warning(f"  ⚠ numpy.loadtxt failed for {contour_path.name}, using fallback parser (slow): {e}")
            logger.warning(f"  Error type: {type(e).__name__}, Error message: {str(e)}")
            with open(contour_path, "r") as f:
                content = f.read().strip()
            values = content.split()
            if len(values) % 2 != 0:
                raise ValueError(
                    f"Contour file must have even number of values (x, y pairs), got {len(values)}"
                )
            points_list = []
            for i in range(0, len(values), 2):
                points_list.append([float(values[i]), float(values[i + 1])])
            points = np.array(points_list, dtype=np.float64)
            # Use INFO if >100K points, otherwise DEBUG
            if len(points) > 100000:
                logger.info(f"  ✓ Fallback parser completed: {len(points):,} points parsed")
            else:
                logger.debug(f"  ✓ Fallback parser completed: {len(points):,} points parsed")
    
    elif coordinate_format == "comma_separated":
        # "x1,y1 x2,y2 ..." or "x1,y1\nx2,y2\n..."
        with open(contour_path, "r") as f:
            content = f.read().strip()
        
        if "\n" in content:
            # Line-separated: "x1,y1\nx2,y2\n..."
            file_size_mb = contour_path.stat().st_size / (1024 * 1024)
            logger.info(f"Parsing comma-separated contour: {contour_path.name} ({file_size_mb:.1f} MB) using numpy.loadtxt...")
            
            try:
                # Try numpy.loadtxt with comma delimiter
                points = np.loadtxt(
                    contour_path,
                    dtype=np.float64,
                    delimiter=',',
                    comments=None,
                    usecols=(0, 1),
                    unpack=False,
                )
                # Use INFO if >100K points, otherwise DEBUG
                if len(points) > 100000:
                    logger.info(f"  ✓ Successfully parsed {len(points):,} points using numpy.loadtxt (fast path)")
                else:
                    logger.debug(f"  ✓ Successfully parsed {len(points):,} points using numpy.loadtxt (fast path)")
            except (ValueError, IOError) as e:
                # Fallback to original
                logger.warning(f"  ⚠ numpy.loadtxt failed for {contour_path.name}, using fallback parser (slow): {e}")
                logger.warning(f"  Error type: {type(e).__name__}, Error message: {str(e)}")
                lines = content.split("\n")
                points_list = []
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split(",")
                    if len(parts) != 2:
                        raise ValueError(f"Each line must have exactly 2 comma-separated values, got: {line}")
                    points_list.append([float(parts[0]), float(parts[1])])
                points = np.array(points_list, dtype=np.float64)
                # Use INFO if >100K points, otherwise DEBUG
                if len(points) > 100000:
                    logger.info(f"  ✓ Fallback parser completed: {len(points):,} points parsed")
                else:
                    logger.debug(f"  ✓ Fallback parser completed: {len(points):,} points parsed")
        else:
            # Space-separated pairs: "x1,y1 x2,y2 ..."
            # Keep original logic (less common)
            pairs = content.split()
            points_list = []
            for pair in pairs:
                parts = pair.split(",")
                if len(parts) != 2:
                    raise ValueError(f"Each pair must be 'x,y', got: {pair}")
                points_list.append([float(parts[0]), float(parts[1])])
            points = np.array(points_list, dtype=np.float64)
    
    else:
        raise ValueError(f"Unsupported coordinate format: {coordinate_format}")

    if len(points) < 3:
        raise ValueError(f"Contour must have at least 3 points, got {len(points)}")

    # Convert to int32 (clipping happens in convert_contour_to_binary_mask)
    return points.astype(np.int32)


def _simplify_polygon_douglas_peucker(
    points: np.ndarray,
    width: int,
    height: int,
    max_points_target: int = 15000,
) -> Tuple[np.ndarray, int, float]:
    """
    Simplify polygon using Douglas-Peucker algorithm to reduce point count.
    
    CRITICAL: OpenCV fillPoly performance degrades catastrophically with >20k points.
    This function ensures we get down to reasonable point counts (<15k).
    
    Args:
        points: Polygon points as (N, 2) array
        width: Image width
        height: Image height
        max_points_target: Maximum target point count (default 15k)
        
    Returns:
        Tuple of (simplified_points, point_count, epsilon_used)
    """
    # 1) Deduplicate jitter - remove points that are too close together
    diff = np.linalg.norm(np.diff(points, axis=0), axis=1)
    points = points[np.insert(diff > 0.5, 0, True)]

    # 2) Close contour (required by approxPolyDP)
    if not np.array_equal(points[0], points[-1]):
        points = np.vstack([points, points[0:1]])

    # 3) Start aggressive - iterate with escalating epsilon
    epsilon = max(2.5, min(width, height) * 0.0025)

    for iteration in range(3):
        c = points.astype(np.float32).reshape(-1, 1, 2)
        approx = cv2.approxPolyDP(c, epsilon, True)
        pts = approx.reshape(-1, 2).astype(np.int32)

        if len(pts) <= max_points_target:
            return pts, len(pts), epsilon

        epsilon *= 1.6  # escalate

    # Final safety - hard cap if still too many points
    return pts[:max_points_target], max_points_target, epsilon


def _apply_emergency_simplification(
    points: np.ndarray,
    width: int,
    height: int,
) -> Tuple[np.ndarray, int]:
    """
    Apply emergency simplification if polygon still has too many points.
    
    This is a safety net in case normal simplification didn't work.
    
    Args:
        points: Polygon points as (N, 2) array
        width: Image width
        height: Image height
        
    Returns:
        Tuple of (simplified_points, point_count)
    """
    logger.warning(
        f"  ⚠ WARNING: {len(points):,} points still too many for fillPoly. "
        f"Applying emergency simplification to 20k points..."
    )
    
    # Ensure contour is closed
    points_closed = points.copy()
    if not np.array_equal(points_closed[0], points_closed[-1]):
        points_closed = np.vstack([points_closed, points_closed[0:1]])
    
    # Use aggressive epsilon to get down to ~20k
    emergency_epsilon = max(5.0, min(width, height) * 0.005)
    points_float = points_closed.astype(np.float32)
    points_contour = points_float.reshape(-1, 1, 2)
    approximated = cv2.approxPolyDP(points_contour, epsilon=emergency_epsilon, closed=True)
    points_simplified = approximated.reshape(-1, 2).astype(np.int32)
    
    logger.warning(f"  Emergency simplification complete: {len(points_simplified):,} points")
    return points_simplified, len(points_simplified)


def _split_disjoint_contours(points: np.ndarray, jump_threshold: float = 40.0, min_points: int = 30):
    """
    Detect multiple contours concatenated into one sequence by looking for large jumps.
    Returns list of contours in OpenCV shape (-1,1,2).
    """
    if len(points) < 50:
        return [points.reshape(-1, 1, 2)]

    diff = np.linalg.norm(np.diff(points, axis=0), axis=1)
    breaks = np.where(diff > jump_threshold)[0]

    if len(breaks) == 0:
        return [points.reshape(-1, 1, 2)]

    contours = []
    start = 0

    for b in breaks:
        segment = points[start:b+1]
        if len(segment) >= min_points:
            contours.append(segment.reshape(-1, 1, 2))
        start = b + 1

    # last segment
    segment = points[start:]
    if len(segment) >= min_points:
        contours.append(segment.reshape(-1, 1, 2))

    if not contours:
        return [points.reshape(-1, 1, 2)]

    return contours


def convert_contour_to_binary_mask(
    contour_path: Path,
    image_size: Tuple[int, int],
    coordinate_format: Optional[str] = None,
) -> np.ndarray:

    if not contour_path.exists():
        raise FileNotFoundError(f"Contour file not found: {contour_path}")

    width, height = image_size

    # ---- PARSE ----
    points = _parse_contour_coordinates(contour_path, coordinate_format)
    num_points_original = len(points)

    log_level = logger.info if num_points_original > 100000 else logger.debug
    log_level(f"Converting {num_points_original:,} contour points to binary mask ({width}x{height})")

    # ---- CLIP TO BOUNDS ----
    if np.any(points < 0) or np.any(points[:, 0] >= width) or np.any(points[:, 1] >= height):
        logger.warning(f"Clipping coordinates to image bounds ({width}x{height})")
        points[:, 0] = np.clip(points[:, 0], 0, width - 1)
        points[:, 1] = np.clip(points[:, 1], 0, height - 1)

    # ---- SIMPLIFY ----
    if num_points_original > 50000:
        approx_start = time.time()
        points, num_points, epsilon_used = _simplify_polygon_douglas_peucker(points, width, height)
        logger.info(
            f"  Douglas-Peucker approximation: {num_points_original:,} → {num_points:,} "
            f"(epsilon={epsilon_used:.3f}) in {time.time()-approx_start:.2f}s"
        )
    else:
        num_points = num_points_original

    points_int = points.astype(np.int32)

    if num_points > 50000:
        points_int, num_points = _apply_emergency_simplification(points_int, width, height)

    if not points_int.flags['C_CONTIGUOUS']:
        points_int = np.ascontiguousarray(points_int)

    # =========================================================
    # ⭐ NEW PART - MULTI CONTOUR HANDLING ⭐
    # =========================================================
    contours = _split_disjoint_contours(points_int)

    if len(contours) > 1:
        logger.info(f"  Detected {len(contours)} disjoint contours - filling separately")

    # ---- FILL ----
    mask = np.zeros((height, width), dtype=np.uint8)

    fill_start = time.time()
    cv2.fillPoly(mask, contours, 255)
    fill_time = time.time() - fill_start

    logger.info(f"  ✓ Polygon filled in {fill_time:.2f}s ({num_points:,} pts, {len(contours)} contours)")

    # ---- VALIDATION ----
    filled_pixels = int(np.count_nonzero(mask))
    pct = 100.0 * filled_pixels / (width * height)

    # Cup sanity guard
    if filled_pixels < 800 or pct > 25.0:
        logger.error("  ✗ Implausible mask - using convex hull fallback")
        mask[:] = 0
        hull = cv2.convexHull(points_int.reshape(-1, 1, 2))
        cv2.fillPoly(mask, [hull], 255)
        filled_pixels = int(np.count_nonzero(mask))
        pct = 100.0 * filled_pixels / (width * height)

    logger.info(f"  Mask created: {filled_pixels:,} filled pixels ({pct:.2f}% of image)")

    return mask


async def convert_contour_to_binary_mask_async(
    contour_path: Path,
    image_size: Tuple[int, int],
    coordinate_format: Optional[str] = None,
) -> np.ndarray:
    """
    Async version of convert_contour_to_binary_mask that prevents blocking.
    
    Offloads CPU-bound operations to a thread pool. Uses CPU-only processing.
    
    Args:
        contour_path: Path to contour file (text or JSON)
        image_size: (width, height) of target image
        coordinate_format: Optional format hint. If None, auto-detects from file content.
        
    Returns:
        Binary mask as numpy array (uint8, 0 and 255)
    """

    return await asyncio.to_thread(
        convert_contour_to_binary_mask,
        contour_path,
        image_size,
        coordinate_format
    )
