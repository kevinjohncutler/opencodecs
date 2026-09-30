"""The byte-plane kernels in byteplanes.h, through both of their callers.

Elements of 2, 4 and 8 bytes are moved as words where the destination is
aligned to the element; every other size, and any unaligned destination,
takes the byte loops. Each path is checked against a NumPy transpose, and
the unaligned cases run the byte loops for 2, 4 and 8 on every platform.
"""
from __future__ import annotations

import numpy as np
import pytest

from opencodecs.codecs._bytetools import byteshuffle_decode, byteshuffle_encode

SIZES = [1, 2, 3, 4, 5, 8, 16]
COUNTS = [0, 1, 5, 64, 1001]


def _items(k, n, seed=0):
    return np.random.default_rng(seed + 31 * k + n).integers(0, 256, (n, k), dtype=np.uint8)


def _planes(items):
    return np.ascontiguousarray(items.T).tobytes()


@pytest.mark.parametrize("k", SIZES)
@pytest.mark.parametrize("n", COUNTS)
def test_encode_and_decode_match_a_transpose(k, n):
    items = _items(k, n)
    assert bytes(byteshuffle_encode(items.tobytes(), k, n)) == _planes(items)
    assert bytes(byteshuffle_decode(_planes(items), k, n)) == items.tobytes()


@pytest.mark.parametrize("k", [2, 3, 4, 8])
@pytest.mark.parametrize("shift", [0, 1, 2, 3, 5])
def test_decode_into_a_destination_at_any_offset(k, shift):
    """An offset that breaks the element's alignment takes the byte loop."""
    n = 1001
    items = _items(k, n, seed=shift)
    buf = bytearray(k * n + 16)
    out = memoryview(buf)[shift:shift + k * n]
    byteshuffle_decode(_planes(items), k, n, out=out)
    assert bytes(out) == items.tobytes()
    assert not any(buf[:shift]) and not any(buf[shift + k * n:])


@pytest.mark.parametrize("k", [2, 3, 4, 8])
@pytest.mark.parametrize("shift", [0, 1, 3])
def test_encode_from_a_source_at_any_offset(k, shift):
    n = 777
    items = _items(k, n, seed=shift)
    src = bytearray(shift) + bytearray(items.tobytes())
    assert bytes(byteshuffle_encode(memoryview(src)[shift:], k, n)) == _planes(items)


@pytest.mark.parametrize("k", [2, 3, 4, 8])
@pytest.mark.parametrize("shift", [0, 1])
def test_zstd_fused_unshuffle_matches_at_any_offset(k, shift):
    """The fused zstd decode runs the same kernel into caller storage."""
    zstd = pytest.importorskip("opencodecs.codecs._zstd")
    n = 4099
    items = _items(k, n, seed=7)
    frame = zstd.encode(_planes(items), level=1)
    scratch = bytearray(k * n)
    buf = bytearray(k * n + 8)
    out = memoryview(buf)[shift:shift + k * n]
    zstd.decode_unshuffle_into(frame, scratch, out, k)
    assert bytes(out) == items.tobytes()
