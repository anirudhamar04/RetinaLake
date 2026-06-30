"""Layer boundary annotation loading utilities."""

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def load_layer_boundaries(
    boundaries_path: Path, layer_format: Optional[str] = None
) -> Dict[str, List[Tuple[float, float]]]:
    """
    Load layer boundary annotations as boundary arrays. Returns coordinate data, not binary.

    Layer boundaries should be stored with unified_format="layer_boundaries".
    This function does NOT convert to binary - it returns the boundary coordinates.

    Supports multiple formats:
    - "json": JSON array of layer boundaries, each with coordinates
    - "csv": CSV with columns for layer_id, x, y coordinates
    - "text": Text file with layer boundaries (one per line)

    Args:
        boundaries_path: Path to layer boundaries file
        layer_format: Format of layer boundaries file. If None, auto-detects from extension.

    Returns:
        Dictionary mapping layer_id/name to list of (x, y) coordinate tuples

    Raises:
        FileNotFoundError: If boundaries file does not exist
        ValueError: If format is unsupported or coordinates are invalid
    """
    if not boundaries_path.exists():
        raise FileNotFoundError(f"Layer boundaries file not found: {boundaries_path}")

    # Auto-detect format from extension if not provided
    if layer_format is None:
        ext = boundaries_path.suffix.lower()
        if ext == ".json":
            layer_format = "json"
        elif ext == ".csv":
            layer_format = "csv"
        else:
            layer_format = "text"

    layers = {}

    if layer_format == "json":
        with open(boundaries_path, "r") as f:
            data = json.load(f)

        # Handle different JSON structures
        if isinstance(data, list):
            # List of layers, each with coordinates
            for i, layer in enumerate(data):
                layer_id = layer.get("layer_id") or layer.get("name") or layer.get("id") or f"layer_{i}"
                if "coordinates" in layer:
                    coords = layer["coordinates"]
                elif "points" in layer:
                    coords = layer["points"]
                else:
                    # Assume layer itself is a list of coordinates
                    coords = layer

                if isinstance(coords, list) and len(coords) > 0:
                    # Convert to list of tuples
                    points = []
                    for coord in coords:
                        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                            points.append((float(coord[0]), float(coord[1])))
                    if points:
                        layers[layer_id] = points

        elif isinstance(data, dict):
            # Dictionary with layer names as keys
            for layer_name, coords in data.items():
                if isinstance(coords, list) and len(coords) > 0:
                    points = []
                    for coord in coords:
                        if isinstance(coord, (list, tuple)) and len(coord) >= 2:
                            points.append((float(coord[0]), float(coord[1])))
                    if points:
                        layers[layer_name] = points

    elif layer_format == "csv":
        import csv

        with open(boundaries_path, "r") as f:
            reader = csv.DictReader(f)
            for row in reader:
                layer_id = row.get("layer_id") or row.get("layer") or row.get("name")
                if not layer_id:
                    continue

                x = row.get("x") or row.get("X")
                y = row.get("y") or row.get("Y")
                if x is None or y is None:
                    continue

                if layer_id not in layers:
                    layers[layer_id] = []
                layers[layer_id].append((float(x), float(y)))

    elif layer_format == "text":
        # Text format: "layer_id x1,y1 x2,y2 ..." or "layer_id x1 y1 x2 y2 ..."
        with open(boundaries_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) < 3:
                    continue
                layer_id = parts[0]
                coords_str = " ".join(parts[1:])

                points = []
                # Try comma-separated first
                if "," in coords_str:
                    pairs = coords_str.split()
                    for pair in pairs:
                        try:
                            x, y = pair.split(",")
                            points.append((float(x), float(y)))
                        except ValueError:
                            continue
                else:
                    # Space-separated
                    values = coords_str.split()
                    if len(values) % 2 == 0:
                        for i in range(0, len(values), 2):
                            try:
                                points.append((float(values[i]), float(values[i + 1])))
                            except ValueError:
                                continue

                if points:
                    if layer_id not in layers:
                        layers[layer_id] = []
                    layers[layer_id].extend(points)

    else:
        raise ValueError(f"Unsupported layer format: {layer_format}")

    if not layers:
        raise ValueError(f"No layer boundaries found in {boundaries_path}")

    return layers
