"""ISA-L deflate backend tests.

ISA-L is x86_64-only and built only when ``libisal-dev`` is on the
host. The whole test module is skipped on platforms / builds where
the extension didn't compile, so this also serves as the cross-
platform smoke test for the optional-extension dispatch path.
"""

from __future__ import annotations

import numpy as np
import pytest

isal = pytest.importorskip("opencodecs.codecs._isal")

import opencodecs as oc
from opencodecs.codecs import _deflate


def test_isal_version():
    assert isal.version() == "isa-l"


def test_isal_round_trip_bytes():
    data = b"hello opencodecs " * 1000
    blob = isal.encode(data)
    back = isal.decode(blob)
    assert back == data


def test_isal_round_trip_random_bytes():
    rng = np.random.default_rng(0)
    data = rng.bytes(50000)
    blob = isal.encode(data)
    back = isal.decode(blob)
    assert back == data


def test_isal_cross_decodes_with_deflate():
    """ISA-L output is regular zlib — libdeflate / zlib must decode it
    and vice versa. This is the interop contract the codec sells."""
    data = b"ABCDEFGH" * 5000
    isal_blob = isal.encode(data)
    deflate_blob = _deflate.encode(data)
    assert _deflate.decode(isal_blob) == data, \
        "libdeflate failed to decode ISA-L output"
    assert isal.decode(deflate_blob) == data, \
        "ISA-L failed to decode libdeflate output"


def test_isal_levels_change_size_or_speed():
    """All four ISA-L levels (0..3) produce valid zlib that round-trips
    to the same bytes."""
    data = b"some easily compressible text " * 5000
    blobs = [isal.encode(data, level=lvl) for lvl in range(4)]
    for blob in blobs:
        assert isal.decode(blob) == data
    # Higher level should produce smaller-or-equal output.
    assert len(blobs[0]) >= len(blobs[3]), \
        f"level 0 ({len(blobs[0])}) was smaller than level 3 ({len(blobs[3])})"


def test_isal_check_signature():
    data = b"some data to compress"
    blob = isal.encode(data)
    assert isal.check_signature(blob) is True
    assert isal.check_signature(b"not a zlib stream at all") is False


def test_isal_clamps_high_level():
    """Levels above 3 are clamped to 3 (ISA-L's max)."""
    data = b"compressible " * 200
    blob = isal.encode(data, level=9)  # zlib-style level
    assert isal.decode(blob) == data


def test_deflate_codec_backend_isal():
    """The DeflateCodec exposes ISA-L via backend='isal'."""
    data = b"opencodecs deflate via ISA-L backend " * 200
    c = oc.get_codec("deflate")
    blob = c.encode(data, backend="isal")
    back = c.decode(blob, backend="isal")
    assert back == data
    # Cross-decode through libdeflate's default.
    assert c.decode(blob) == data


def test_deflate_codec_default_is_libdeflate():
    """Default backend is libdeflate (smaller output than ISA-L, slightly
    faster decode); pass backend='isal' to opt into the faster encode."""
    data = b"x" * 10000
    c = oc.get_codec("deflate")
    default_blob = c.encode(data)
    isal_blob = c.encode(data, backend="isal")
    # Both must round-trip to the same bytes — they're both zlib.
    assert c.decode(default_blob) == data
    assert c.decode(isal_blob) == data
