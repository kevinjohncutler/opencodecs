"""Brunsli tests — lossless JPEG transcoder.

Brunsli repacks an existing JPEG bitstream into a smaller container
that decodes byte-identically back to the source JPEG. The crucial
correctness contract is *byte-identical recovery*, not pixel parity
after re-decoding (which is JPEG's job).
"""

from __future__ import annotations

import numpy as np
import pytest

mod = pytest.importorskip("opencodecs.codecs._brunsli")
import opencodecs as oc


def _jpeg_for(shape, *, level=85, seed=0):
    """Build a real JPEG bitstream for testing."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
    return arr, oc.write(None, arr, format="jpeg", level=level)


def test_brunsli_signature():
    """Brunsli files start with the well-known 4-byte marker 0x0A0442D2."""
    _arr, jpeg = _jpeg_for((96, 96, 3))
    brn = mod.encode_jpeg(jpeg)
    assert mod.check_signature(brn) is True
    assert mod.check_signature(jpeg) is False  # JPEG is not Brunsli


def test_brunsli_jpeg_roundtrip_byte_identical():
    """encode_jpeg + decode_jpeg returns the SAME JPEG bytestream."""
    _arr, jpeg = _jpeg_for((128, 128, 3))
    brn = mod.encode_jpeg(jpeg)
    recovered = mod.decode_jpeg(brn)
    assert bytes(recovered) == bytes(jpeg)


def test_brunsli_smaller_than_jpeg():
    """On a typical photo-quality JPEG, Brunsli is ~15-25% smaller."""
    _arr, jpeg = _jpeg_for((256, 256, 3), level=85)
    brn = mod.encode_jpeg(jpeg)
    assert len(brn) < len(jpeg), \
        f"brunsli ({len(brn)}) should be smaller than jpeg ({len(jpeg)})"


def test_brunsli_rejects_non_jpeg_input():
    with pytest.raises(mod.BrunsliError, match="JPEG"):
        mod.encode_jpeg(b"not a jpeg")


@pytest.mark.parametrize("shape", [(64, 64), (96, 128, 3)])
def test_brunsli_codec_ndarray_path(shape):
    """BrunsliCodec accepts ndarrays — first JPEG-encodes, then transcodes."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
    brn = oc.write(None, arr, format="brunsli", level=85)
    assert mod.check_signature(brn)
    out = oc.read(brn, format="brunsli")
    # JPEG round-trip is lossy, so we only check shape/dtype here.
    assert out.shape == arr.shape
    assert out.dtype == arr.dtype


def test_brunsli_codec_asjpeg_recovers_jpeg_bytes():
    """asjpeg=True returns the raw JPEG bytestream (byte-identical)."""
    _arr, jpeg = _jpeg_for((128, 128, 3))
    brn = oc.write(None, jpeg, format="brunsli")
    out = oc.read(brn, format="brunsli", asjpeg=True)
    assert bytes(out) == bytes(jpeg)


def test_brunsli_codec_registered():
    names = [c["name"] for c in oc.list_codecs()]
    assert "brunsli" in names
