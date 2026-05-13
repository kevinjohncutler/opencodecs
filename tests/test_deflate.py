"""opencodecs._deflate (zlib-format) encode/decode round-trip tests.

Cross-validates with stdlib zlib and (when present) imagecodecs.
The backend the test runs against is whatever setup.py picked at
build time — typically libdeflate when ``brew install libdeflate``,
otherwise zlib-ng-compat or stdlib zlib. The encoded bytes are NOT
byte-identical across backends (each picks its own coding choices),
but every encode must decode through every other backend.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

oc = pytest.importorskip("opencodecs.codecs._deflate")


# ---------------------------------------------------------------------------
# Basics + round-trips
# ---------------------------------------------------------------------------


def test_backend_is_one_of_known_strings():
    assert oc.backend() in {"libdeflate", "zlib"}


@pytest.mark.parametrize("size", [0, 1, 1024, 65536, 5 * 1024 * 1024])
def test_round_trip_random(size):
    rng = np.random.default_rng(size or 1)
    data = bytes(rng.integers(0, 256, size=size, dtype=np.uint8))
    enc = oc.encode(data)
    assert oc.decode(enc) == data


@pytest.mark.parametrize("level", [0, 1, 3, 6, 9])
def test_levels_round_trip(level):
    data = (b"opencodecs " * 1024) + bytes(range(256)) * 32
    assert oc.decode(oc.encode(data, level=level)) == data


def test_level_high_clamped_for_libdeflate_or_zlib():
    """libdeflate accepts 0..12; zlib accepts 0..9. Both backends
    should silently clamp out-of-range values, not raise."""
    data = b"A" * 1024
    # 99 → clamp to 12 (libdeflate) or 9 (zlib). Should not raise.
    assert oc.decode(oc.encode(data, level=99)) == data
    # Negative → clamp to 0 (store). Both backends accept that.
    assert oc.decode(oc.encode(data, level=-5)) == data


def test_empty_round_trip():
    enc = oc.encode(b"")
    assert oc.decode(enc) == b""


# ---------------------------------------------------------------------------
# Cross-backend interop
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [256, 4096, 1024 * 1024])
def test_oc_encode_decodes_via_stdlib_zlib(size):
    """Our encode produces a zlib stream stdlib zlib can decode.
    Critical for users mixing opencodecs writers with non-opencodecs
    readers (and vice versa)."""
    rng = np.random.default_rng(size)
    data = bytes(rng.integers(0, 256, size=size, dtype=np.uint8))
    enc = oc.encode(data, level=6)
    assert zlib.decompress(enc) == data


@pytest.mark.parametrize("size", [256, 4096, 1024 * 1024])
def test_oc_decode_handles_stdlib_zlib_output(size):
    rng = np.random.default_rng(size + 1)
    data = bytes(rng.integers(0, 256, size=size, dtype=np.uint8))
    enc = zlib.compress(data, 6)
    assert oc.decode(enc) == data


def test_oc_decode_handles_imagecodecs_output():
    ic = pytest.importorskip("imagecodecs")
    if not hasattr(ic, "zlib_encode"):
        pytest.skip("imagecodecs has no zlib_encode")
    data = bytes(range(256)) * 256
    enc = bytes(ic.zlib_encode(data, level=6))
    assert oc.decode(enc) == data


def test_imagecodecs_decodes_our_output():
    ic = pytest.importorskip("imagecodecs")
    if not hasattr(ic, "zlib_decode"):
        pytest.skip("imagecodecs has no zlib_decode")
    data = bytes(range(256)) * 256
    enc = oc.encode(data, level=6)
    assert bytes(ic.zlib_decode(enc)) == data


# ---------------------------------------------------------------------------
# Signature detection
# ---------------------------------------------------------------------------


def test_check_signature_accepts_real_zlib_streams():
    assert oc.check_signature(oc.encode(b"hello"))
    assert oc.check_signature(zlib.compress(b"hello"))


def test_check_signature_rejects_random_bytes():
    rng = np.random.default_rng(0)
    # Almost all random 2-byte heads fail the CMF*256+FLG % 31 == 0
    # check. Run a small sweep so a single 1-in-31 hit doesn't pass.
    misses = 0
    for _ in range(200):
        head = bytes(rng.integers(0, 256, size=2, dtype=np.uint8))
        if not oc.check_signature(head):
            misses += 1
    assert misses > 100  # well more than half should miss


def test_check_signature_rejects_short_input():
    assert oc.check_signature(b"") is False
    assert oc.check_signature(b"\x78") is False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_decode_rejects_garbage():
    with pytest.raises(Exception):
        oc.decode(b"\xff\xff\xff" * 16)
