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


@pytest.mark.parametrize("n", [2**62 + 1, -(2**62) + 1, -1])
def test_sizes_whose_product_would_wrap_are_refused(n):
    """4 * (2**62 + 1) wraps to 4, which equals len(b'abcd'); the length
    check alone would pass and the kernels would walk far past the buffers."""
    with pytest.raises(ValueError, match="n_elements"):
        byteshuffle_encode(b"abcd", 4, n)
    with pytest.raises(ValueError, match="n_elements"):
        byteshuffle_decode(b"abcd", 4, n)


@pytest.mark.parametrize("itemsize,samples", [(2, 1), (2, 3), (3, 1), (4, 1), (4, 2), (8, 1)])
def test_windowed_unshuffle_into_rows_at_every_alignment(itemsize, samples):
    """Rows one byte into their buffer, with an odd row width, alternate
    between aligned and unaligned starts, so the windowed decode runs both
    the word kernels and the byte loops with plane != count."""
    z = pytest.importorskip("opencodecs.codecs._zstd")
    rs = np.random.RandomState(5)
    h, w, px = 29, 41, samples * itemsize
    tile = rs.randint(0, 256, (h, w, px)).astype(np.uint8)
    elements = tile.reshape(h * w * samples, itemsize)
    frame = z.encode(np.ascontiguousarray(elements.T).tobytes(), level=1)
    width = 60 * px + 1
    buf = np.full(1 + 40 * width, 7, np.uint8)
    out = buf[1:].reshape(40, width)
    windows = [(3, 20, 5, 33, 2, 9), (0, 29, 0, 41, 10, 0), (20, 29, 30, 41, 1, 45)]
    scratch = bytearray(h * w * px)
    n = z.decode_unshuffle_windows(frame, scratch, out, itemsize, samples, h, w,
                                   windows, z.DecodeContext())
    assert n == len(scratch)
    expected = np.full((40, width), 7, np.uint8)
    for sy0, sy1, sx0, sx1, dy, dx in windows:
        expected[dy:dy + sy1 - sy0, dx * px:(dx + sx1 - sx0) * px] = \
            tile[sy0:sy1, sx0:sx1].reshape(sy1 - sy0, -1)
    np.testing.assert_array_equal(out, expected)
    assert buf[0] == 7
