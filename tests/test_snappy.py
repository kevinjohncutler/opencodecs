"""opencodecs._snappy raw-block round-trip + cross-validation tests.

Snappy raw block format is identical across implementations (no
framing, no flags), so we cross-decode with imagecodecs and the
``python-snappy`` package when present.
"""

from __future__ import annotations

import numpy as np
import pytest

oc = pytest.importorskip("opencodecs.codecs._snappy")


# ---------------------------------------------------------------------------
# Basics + round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, 1, 1024, 65536, 5 * 1024 * 1024])
def test_round_trip_random(size):
    rng = np.random.default_rng(size or 1)
    data = bytes(rng.integers(0, 256, size=size, dtype=np.uint8))
    enc = oc.encode(data)
    assert oc.decode(enc) == data


@pytest.mark.parametrize("size", [256, 4096, 1024 * 1024])
def test_round_trip_compressible(size):
    data = (b"ABCDEFGHIJKLMNOP" * (size // 16))[:size]
    enc = oc.encode(data)
    # Highly compressible — encoded must be much smaller.
    assert len(enc) < len(data) // 4 if size >= 256 else True
    assert oc.decode(enc) == data


def test_empty_round_trip():
    enc = oc.encode(b"")
    assert oc.decode(enc) == b""


def test_accepts_bytearray_and_memoryview():
    data = b"snappy snappy snappy" * 64
    assert oc.decode(oc.encode(bytearray(data))) == data
    assert oc.decode(oc.encode(memoryview(data))) == data


def test_accepts_numpy_uint8():
    arr = np.arange(1024, dtype=np.uint8)
    enc = oc.encode(arr)
    assert oc.decode(enc) == arr.tobytes()


# ---------------------------------------------------------------------------
# Cross-implementation interop (raw block format is identical)
# ---------------------------------------------------------------------------


def test_imagecodecs_decodes_our_output():
    ic = pytest.importorskip("imagecodecs")
    if not hasattr(ic, "snappy_decode"):
        pytest.skip("imagecodecs has no snappy_decode")
    data = bytes(range(256)) * 256
    enc = oc.encode(data)
    assert bytes(ic.snappy_decode(enc)) == data


def test_oc_decode_handles_imagecodecs_output():
    ic = pytest.importorskip("imagecodecs")
    if not hasattr(ic, "snappy_encode"):
        pytest.skip("imagecodecs has no snappy_encode")
    data = bytes(range(256)) * 256
    enc = bytes(ic.snappy_encode(data))
    assert oc.decode(enc) == data


# ---------------------------------------------------------------------------
# Signature detection
# ---------------------------------------------------------------------------


def test_check_signature_accepts_real_snappy_blocks():
    assert oc.check_signature(oc.encode(b"hello world"))
    assert oc.check_signature(oc.encode(b"A" * 4096))


def test_check_signature_rejects_garbage():
    assert oc.check_signature(b"") is False
    assert oc.check_signature(b"not snappy") is False
    assert oc.check_signature(b"\xff\xfe\xfd\xfc") is False


def test_check_signature_rejects_truncated_valid_block():
    enc = oc.encode(b"some payload" * 64)
    # Lop off the trailing copy/literal bytes — validator should reject.
    assert oc.check_signature(enc[:4]) is False


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_decode_rejects_garbage():
    with pytest.raises(Exception):
        oc.decode(b"\xff\xff\xff" * 16)


def test_decode_rejects_truncated():
    enc = oc.encode(b"hello world" * 1024)
    with pytest.raises(Exception):
        oc.decode(enc[: len(enc) // 2])


# ---------------------------------------------------------------------------
# Codec adapter (high-level API)
# ---------------------------------------------------------------------------


def test_codec_adapter_registered():
    import opencodecs as opc

    codec = opc.get_codec("snappy")
    assert codec is not None
    assert codec.name == "snappy"
    assert ".sz" in codec.file_extensions


def test_codec_adapter_roundtrip():
    import opencodecs as opc

    codec = opc.get_codec("snappy")
    data = b"the quick brown fox" * 1000
    enc = codec.encode(data)
    assert codec.decode(enc) == data
    assert codec.signature(enc) is True
