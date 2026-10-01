"""The predictor loops against NumPy, for every layout they branch on.

_undo_rows carries one sample's running sum in a 32-bit register whatever
the sample width, keeps separate fast paths for 1, 2, 3 and 4 samples per
pixel, and walks one chain per sample for more; predictor 3's byte
differencing goes through the same loop. Each branch is checked here
against a reference written in NumPy, with values chosen so the sums
wrap, which is where a wide accumulator could differ from a narrow one.
"""
from __future__ import annotations

import numpy as np
import pytest

from opencodecs.codecs import _tiff

UNDO = {np.uint8: _tiff.undo_horizontal_u8, np.uint16: _tiff.undo_horizontal_u16,
        np.uint32: _tiff.undo_horizontal_u32, np.uint64: _tiff.undo_horizontal_u64}


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.uint32, np.uint64])
@pytest.mark.parametrize("spp", [1, 2, 3, 4, 5])
@pytest.mark.parametrize("cols", [1, 2, 7, 64])
def test_predictor_2_is_a_wrapping_running_sum(dtype, spp, cols):
    rng = np.random.default_rng(cols * 10 + spp)
    info = np.iinfo(dtype)
    # Deltas near the top of the range make every row wrap many times.
    diff = rng.integers(info.max - 50, info.max, (5, cols, spp), dtype=dtype, endpoint=True)
    # A uint64 running sum wraps modulo 2**64; truncating it to the sample
    # width is the same as wrapping modulo the sample's own range.
    expected = np.cumsum(diff.astype(np.uint64), axis=1, dtype=np.uint64)
    got = diff.copy()
    UNDO[dtype](got)
    np.testing.assert_array_equal(got, expected.astype(dtype))


def _reference_undo_float(stored: np.ndarray, bps: int) -> np.ndarray:
    """TIFF Technical Note 3, predictor 3, undone in NumPy for one row set."""
    rows, cols, pixel_bytes = stored.shape
    spp = pixel_bytes // bps
    flat = stored.reshape(rows, cols * pixel_bytes).astype(np.uint64)
    out = np.empty_like(stored)
    for r in range(rows):
        row = flat[r].copy()
        for lane in range(spp):  # byte differencing, one recurrence per sample
            row[lane::spp] = np.cumsum(row[lane::spp]) % 256
        planes = row.astype(np.uint8).reshape(bps, cols * spp)  # most significant first
        samples = planes.T  # (cols * spp, bps), big-endian bytes
        if np.little_endian:
            samples = samples[:, ::-1]
        out[r] = samples.reshape(cols, pixel_bytes)
    return out


@pytest.mark.parametrize("bps", [2, 4, 8])
@pytest.mark.parametrize("spp", [1, 2, 3, 4, 5])
def test_predictor_3_matches_the_technical_note(bps, spp):
    rng = np.random.default_rng(bps * 10 + spp)
    stored = rng.integers(0, 256, (3, 9, spp * bps), dtype=np.uint8)
    got = stored.copy()
    _tiff.undo_floating_point(got, bps)
    np.testing.assert_array_equal(got, _reference_undo_float(stored, bps))
