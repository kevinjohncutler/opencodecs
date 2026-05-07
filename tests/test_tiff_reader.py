"""Tests for opencodecs.tiff_reader — parallel-pread TIFF tile decoder.

This module had 0% coverage before this file. The module decodes TIFF
tiles using ``os.pread`` from worker threads instead of tifffile's
single-fd serialized reads — meant to win on NAS or other high-latency
storage where the lock around seek+read serializes I/O.

Tests verify correctness against tifffile's own ``imread`` (which is the
ground truth for what the bytes should decode to).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")

from opencodecs.tiff_reader import imread, imread_stack


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_tiled_tiff(path: Path, shape: tuple, dtype=np.uint8,
                     tile=(64, 64), compression: str | None = None) -> np.ndarray:
    """Write a tiled TIFF and return the source array."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, shape, dtype=dtype)
    tifffile.imwrite(
        str(path), arr, tile=tile, compression=compression,
    )
    return arr


def _write_stripped_tiff(path: Path, shape: tuple, dtype=np.uint8,
                         compression: str | None = None) -> np.ndarray:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, shape, dtype=dtype)
    tifffile.imwrite(str(path), arr, compression=compression)
    return arr


# ---------------------------------------------------------------------------
# imread: single-page parallel decode
# ---------------------------------------------------------------------------


def test_imread_tiled_uncompressed_matches_tifffile(tmp_path):
    p = tmp_path / "tiled.tif"
    src = _write_tiled_tiff(p, (256, 256, 3))
    arr = imread(p)
    expected = tifffile.imread(str(p))
    np.testing.assert_array_equal(arr, expected)
    np.testing.assert_array_equal(arr, src)


def test_imread_tiled_compressed_matches_tifffile(tmp_path):
    """Compressed tiles exercise the decode side of the pipeline."""
    p = tmp_path / "tiled_zlib.tif"
    src = _write_tiled_tiff(p, (256, 256, 3), compression="zlib")
    arr = imread(p)
    np.testing.assert_array_equal(arr, src)


def test_imread_n_workers_one(tmp_path):
    """n_workers=1 should still produce correct output (serial through pool)."""
    p = tmp_path / "tiled.tif"
    src = _write_tiled_tiff(p, (128, 128, 3))
    np.testing.assert_array_equal(imread(p, n_workers=1), src)


def test_imread_default_n_workers(tmp_path):
    """n_workers=None uses os.cpu_count() and still decodes correctly."""
    p = tmp_path / "tiled.tif"
    src = _write_tiled_tiff(p, (128, 128, 3))
    np.testing.assert_array_equal(imread(p), src)


def test_imread_grayscale_uint16(tmp_path):
    """Different dtype/sample-format path."""
    p = tmp_path / "gray.tif"
    src = _write_tiled_tiff(p, (256, 256), dtype=np.uint16)
    arr = imread(p)
    np.testing.assert_array_equal(arr, src)


def test_imread_path_object(tmp_path):
    """Path object accepted, not just str."""
    p = tmp_path / "path_obj.tif"
    src = _write_tiled_tiff(p, (128, 128, 3))
    np.testing.assert_array_equal(imread(p), src)


def test_imread_specific_page(tmp_path):
    """Multi-page TIFF: imread(path, page=N) selects that page."""
    p = tmp_path / "multi.tif"
    rng = np.random.default_rng(0)
    pages = [rng.integers(0, 256, (128, 128, 3), dtype=np.uint8) for _ in range(3)]
    with tifffile.TiffWriter(str(p)) as tw:
        for pg in pages:
            tw.write(pg, tile=(64, 64))
    for i, expected in enumerate(pages):
        arr = imread(p, page=i)
        np.testing.assert_array_equal(arr, expected)


# ---------------------------------------------------------------------------
# imread_stack: parallel decode of multiple pages
# ---------------------------------------------------------------------------


def test_imread_stack_all_pages_match_tifffile(tmp_path):
    p = tmp_path / "stack.tif"
    rng = np.random.default_rng(0)
    pages = [rng.integers(0, 256, (128, 128, 3), dtype=np.uint8) for _ in range(4)]
    with tifffile.TiffWriter(str(p)) as tw:
        for pg in pages:
            tw.write(pg, tile=(64, 64))

    stack = imread_stack(p)
    expected = np.stack(pages)
    assert stack.shape == expected.shape
    np.testing.assert_array_equal(stack, expected)


def test_imread_stack_subset_pages(tmp_path):
    """pages=[2, 0, 1] should return those pages in that order."""
    p = tmp_path / "stack.tif"
    rng = np.random.default_rng(0)
    pages = [rng.integers(0, 256, (128, 128, 3), dtype=np.uint8) for _ in range(4)]
    with tifffile.TiffWriter(str(p)) as tw:
        for pg in pages:
            tw.write(pg, tile=(64, 64))

    selected = imread_stack(p, pages=[2, 0, 1])
    assert selected.shape == (3, 128, 128, 3)
    np.testing.assert_array_equal(selected[0], pages[2])
    np.testing.assert_array_equal(selected[1], pages[0])
    np.testing.assert_array_equal(selected[2], pages[1])


def test_imread_stack_default_n_workers(tmp_path):
    p = tmp_path / "stack.tif"
    rng = np.random.default_rng(0)
    pages = [rng.integers(0, 256, (96, 96, 3), dtype=np.uint8) for _ in range(2)]
    with tifffile.TiffWriter(str(p)) as tw:
        for pg in pages:
            tw.write(pg, tile=(48, 48))
    stack = imread_stack(p)
    assert stack.shape == (2, 96, 96, 3)


def test_imread_stack_range_pages(tmp_path):
    """range() argument for pages works."""
    p = tmp_path / "stack.tif"
    rng = np.random.default_rng(0)
    pages = [rng.integers(0, 256, (96, 96, 3), dtype=np.uint8) for _ in range(4)]
    with tifffile.TiffWriter(str(p)) as tw:
        for pg in pages:
            tw.write(pg, tile=(48, 48))
    stack = imread_stack(p, pages=range(1, 3))
    assert stack.shape == (2, 96, 96, 3)
    np.testing.assert_array_equal(stack[0], pages[1])
    np.testing.assert_array_equal(stack[1], pages[2])
