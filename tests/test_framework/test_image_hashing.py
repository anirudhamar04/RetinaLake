"""Encoding-aware image hashing: the same image under different encodings must collide."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from chaksudb.ingest.framework.hashing import (
    compute_file_hash,
    compute_perceptual_hash,
    compute_pixel_hash,
    hamming_distance,
)


@pytest.fixture
def encoded_variants(tmp_path) -> dict[str, Path]:
    rng = np.random.default_rng(42)
    arr = (rng.random((48, 48, 3)) * 255).astype("uint8")
    paths = {}
    for name, kwargs in {"png": {}, "bmp": {}, "jpg": {"quality": 95}}.items():
        p = tmp_path / f"img.{name}"
        Image.fromarray(arr).save(p, **kwargs)
        paths[name] = p
    return paths


def test_file_hash_differs_across_containers(encoded_variants):
    # exact-bytes hash is container-sensitive (by design)
    assert compute_file_hash(encoded_variants["png"]) != compute_file_hash(encoded_variants["bmp"])


def test_content_hash_encoding_invariant_for_lossless(encoded_variants):
    # PNG and BMP decode to identical pixels -> same content hash
    assert compute_pixel_hash(encoded_variants["png"]) == compute_pixel_hash(encoded_variants["bmp"])


def test_perceptual_hash_survives_lossy_reencode(encoded_variants):
    # JPEG alters pixels (content hash differs) but perceptual hash stays within a few bits
    assert compute_pixel_hash(encoded_variants["png"]) != compute_pixel_hash(encoded_variants["jpg"])
    dist = hamming_distance(
        compute_perceptual_hash(encoded_variants["png"]),
        compute_perceptual_hash(encoded_variants["jpg"]),
    )
    assert dist <= 4


def test_hamming_distance_basic():
    assert hamming_distance("00", "00") == 0
    assert hamming_distance("0f", "00") == 4
