"""Random access and threading for blosc2 / b2nd.

Both formats store data as independently compressed chunks, which is
what lets a caller read part of a buffer without expanding all of it.
These tests pin the part that is easy to get wrong -- the boundaries.
A partial read that is off by one block, or that ignores typesize, is
still plausible-looking data, so every case here compares against the
exact source bytes rather than against a shape or a checksum.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

_blosc2 = pytest.importorskip("opencodecs.codecs._blosc2")
_b2nd = pytest.importorskip("opencodecs.codecs._b2nd")


@pytest.fixture(scope="module")
def chunk():
    """A chunk big enough to span several blosc2 blocks."""
    n = 1 << 18                      # 2 MB at 8 bytes/item
    a = (np.sin(np.arange(n) / 50.0) * 1000).astype(np.float64)
    raw = a.tobytes()
    return raw, _blosc2.encode(raw, typesize=8, level=5)


@pytest.mark.parametrize("start,count", [
    (0, 1),            # very first item
    (0, 1000),         # from the start, spanning blocks
    (1, 1),            # off the block boundary by one
    (12345, 6789),     # unaligned start and length
    ((1 << 18) - 1, 1),   # very last item
    ((1 << 18) - 100, 100),
])
def test_partial_matches_the_source_bytes(chunk, start, count):
    raw, blob = chunk
    got = _blosc2.decode_partial(blob, start, count)
    assert got == raw[start * 8:(start + count) * 8]


def test_offsets_are_items_of_the_chunks_own_typesize(chunk):
    """Item width comes from the chunk, and cannot be overridden.

    blosc2_getitem addresses items of the width stored in the header,
    so a caller-supplied typesize could only ever mis-size the
    destination buffer -- which is exactly what an earlier version of
    this function did, failing with a bare error code. Asking for a
    different width now says so.
    """
    raw, blob = chunk
    assert _blosc2.chunk_typesize(blob) == 8
    assert _blosc2.decode_partial(blob, 10, 4) == raw[80:112]
    assert _blosc2.decode_partial(blob, 10, 4, typesize=8) == raw[80:112]
    with pytest.raises(ValueError, match="does not match"):
        _blosc2.decode_partial(blob, 10, 4, typesize=1)


def test_typesize_one_chunk_addresses_bytes():
    """With a 1-byte typesize, items and bytes coincide."""
    raw = bytes(range(256)) * 64
    blob = _blosc2.encode(raw, typesize=1, level=5)
    assert _blosc2.chunk_typesize(blob) == 1
    assert _blosc2.decode_partial(blob, 10, 4) == raw[10:14]


def test_partial_zero_items_is_empty(chunk):
    _, blob = chunk
    assert _blosc2.decode_partial(blob, 5, 0) == b""


def test_partial_past_the_end_is_rejected(chunk):
    """A clear error, not blosc2's bare negative return code."""
    _, blob = chunk
    n = 1 << 18
    with pytest.raises(ValueError, match="run past the chunk"):
        _blosc2.decode_partial(blob, n - 5, 10)
    with pytest.raises(ValueError):
        _blosc2.decode_partial(blob, -1, 4)
    with pytest.raises(ValueError):
        _blosc2.decode_partial(blob, 0, -4)


def test_threaded_decode_matches_serial(chunk):
    raw, blob = chunk
    assert _blosc2.decode(blob, numthreads=8) == raw
    assert (_blosc2.decode_partial(blob, 999, 5000, typesize=8, numthreads=8)
            == raw[999 * 8:(999 + 5000) * 8])


def test_blosc2_codec_exposes_partial(chunk):
    raw, blob = chunk
    codec = oc.get_codec("blosc2")
    assert codec.chunked is True
    assert codec.parallel_decode is True
    assert codec.decode_partial(blob, 7, 21, typesize=8) == raw[56:224]


# ---- b2nd: n-dimensional slices -----


@pytest.fixture(scope="module")
def nd():
    shape = (128, 96, 12)
    a = (np.sin(np.arange(int(np.prod(shape))) / 37.0)
         .reshape(shape) * 1000).astype(np.float32)
    return a, _b2nd.encode(a, level=5, compressor="zstd")


@pytest.mark.parametrize("start,stop", [
    ((0, 0, 0), (128, 96, 12)),        # the whole array
    ((0, 0, 0), (1, 1, 1)),            # a single element
    ((1, 1, 1), (9, 9, 9)),            # unaligned interior box
    ((64, 48, 6), (65, 49, 7)),        # a middle corner
    ((127, 95, 11), (128, 96, 12)),    # the last element
    ((0, 0, 0), (0, 0, 0)),            # empty
])
def test_slice_matches_numpy(nd, start, stop):
    a, blob = nd
    got = _b2nd.decode_slice(blob, start, stop)
    want = a[tuple(slice(x, y) for x, y in zip(start, stop))]
    assert got.shape == want.shape
    assert np.array_equal(got, want)


def test_slice_rejects_a_bad_box(nd):
    a, blob = nd
    with pytest.raises(ValueError, match="not within"):
        _b2nd.decode_slice(blob, (0, 0, 0), (129, 96, 12))
    with pytest.raises(ValueError, match="not within"):
        _b2nd.decode_slice(blob, (5, 0, 0), (2, 96, 12))
    with pytest.raises(ValueError, match="must have 3 entries"):
        _b2nd.decode_slice(blob, (0, 0), (5, 5))


def test_slice_threaded_matches_serial(nd):
    a, blob = nd
    got = _b2nd.decode_slice(blob, (3, 4, 5), (60, 70, 11), numthreads=8)
    want = a[3:60, 4:70, 5:11]
    assert np.array_equal(got, want)
    assert np.array_equal(_b2nd.decode(blob, numthreads=8), a)


def test_slice_into_out(nd):
    a, blob = nd
    out = np.empty((10, 10, 3), dtype=a.dtype)
    got = _b2nd.decode_slice(blob, (2, 2, 2), (12, 12, 5), out=out)
    assert got is out
    assert np.array_equal(out, a[2:12, 2:12, 2:5])
    with pytest.raises(ValueError, match="does not match the slice shape"):
        _b2nd.decode_slice(blob, (2, 2, 2), (12, 12, 5),
                           out=np.empty((3, 3, 3), dtype=a.dtype))


def test_b2nd_codec_exposes_slice(nd):
    a, blob = nd
    codec = oc.get_codec("b2nd")
    assert codec.chunked is True
    assert codec.parallel_decode is True
    got = codec.decode_slice(blob, (0, 0, 0), (4, 4, 4))
    assert np.array_equal(got, a[:4, :4, :4])
