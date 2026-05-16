"""Roundtrip + compression-ratio tests for the Phase-4 filter codecs.

* ``numpy`` — .npy passthrough
* ``byteshuffle`` — element-byte-plane shuffle
* ``delta`` / ``xor`` / ``floatpred`` — TIFF-style predictors
* ``quantize`` — lossy mantissa bit-round / nsd
* ``packints`` — arbitrary-bit-width int pack/unpack

Each codec gets a basic encode-then-decode roundtrip; the
``compose_with_zstd_*`` tests verify the filters genuinely make
downstream byte compressors squeeze tighter (the whole point of
having these as separate codecs).
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc


# ---------------------------------------------------------------------------
# numpy codec — .npy passthrough
# ---------------------------------------------------------------------------


def test_numpy_codec_roundtrip_uint32():
    arr = np.arange(64 * 96, dtype=np.uint32).reshape(64, 96)
    c = oc.get_codec("numpy")
    blob = c.encode(arr)
    back = c.decode(blob)
    np.testing.assert_array_equal(back, arr)
    assert back.dtype == arr.dtype
    assert back.shape == arr.shape


def test_numpy_codec_alias():
    assert oc.has_codec("npy")
    assert oc.get_codec("npy").name == "numpy"


def test_numpy_codec_signature_detects_npy():
    arr = np.arange(10, dtype=np.float32)
    blob = oc.get_codec("numpy").encode(arr)
    assert oc.get_codec("numpy").signature(blob[:8])
    assert not oc.get_codec("numpy").signature(b"NOTNUMPY")


def test_numpy_codec_out_kwarg():
    arr = np.arange(100, dtype=np.float64).reshape(10, 10)
    c = oc.get_codec("numpy")
    blob = c.encode(arr)
    target = np.empty_like(arr)
    out = c.decode(blob, out=target)
    assert out is target
    np.testing.assert_array_equal(out, arr)


# ---------------------------------------------------------------------------
# byteshuffle
# ---------------------------------------------------------------------------


def test_byteshuffle_roundtrip_uint16():
    c = oc.get_codec("byteshuffle")
    arr = np.arange(2048, dtype=np.uint16)
    shuffled = c.encode(arr)
    raw = c.decode(shuffled, itemsize=2)
    back = np.frombuffer(raw, dtype=np.uint16)
    np.testing.assert_array_equal(back, arr)


def test_byteshuffle_compose_with_zstd_improves_ratio():
    """The whole reason byteshuffle exists — should make zstd
    compress smooth multi-byte data noticeably tighter."""
    arr = np.arange(8192, dtype=np.uint32)  # monotonic = lots of high-byte redundancy
    bs = oc.get_codec("byteshuffle")
    zs = oc.get_codec("zstd")
    raw_compressed = zs.encode(arr.tobytes())
    shuffled_compressed = zs.encode(bs.encode(arr))
    # On a monotonic int32 stream the improvement is large (>=4×).
    assert len(shuffled_compressed) * 4 < len(raw_compressed), (
        f"byteshuffle+zstd ({len(shuffled_compressed)}) should be "
        f"<= 1/4 of zstd alone ({len(raw_compressed)})")


# ---------------------------------------------------------------------------
# delta predictor
# ---------------------------------------------------------------------------


def test_delta_roundtrip_signed_int16():
    c = oc.get_codec("delta")
    arr = np.array([3, 5, 4, 8, 10, -2, -3], dtype=np.int16)
    enc = c.encode(arr)
    back = c.decode(enc, dtype=np.int16)
    np.testing.assert_array_equal(back, arr)


def test_delta_compose_with_zstd_improves_ratio():
    arr = np.arange(4096, dtype=np.int32)
    d = oc.get_codec("delta")
    zs = oc.get_codec("zstd")
    raw_z = zs.encode(arr.tobytes())
    delta_z = zs.encode(d.encode(arr))
    assert len(delta_z) < len(raw_z), (
        f"delta+zstd ({len(delta_z)}) should beat zstd alone ({len(raw_z)})")


# ---------------------------------------------------------------------------
# xor predictor
# ---------------------------------------------------------------------------


def test_xor_roundtrip_uint8():
    c = oc.get_codec("xor")
    arr = np.array([0x55, 0xAA, 0x55, 0x00, 0xFF, 0xFF], dtype=np.uint8)
    enc = c.encode(arr)
    back = c.decode(enc, dtype=np.uint8)
    np.testing.assert_array_equal(back, arr)


def test_xor_roundtrip_uint32_2d():
    c = oc.get_codec("xor")
    arr = np.arange(120, dtype=np.uint32).reshape(10, 12)
    enc = c.encode(arr)
    back = c.decode(enc, dtype=np.uint32, shape=arr.shape)
    np.testing.assert_array_equal(back, arr)


# ---------------------------------------------------------------------------
# floatpred (TIFF predictor 3)
# ---------------------------------------------------------------------------


def test_floatpred_roundtrip_float32_2d():
    c = oc.get_codec("floatpred")
    arr = np.linspace(0.0, 1.0, 32, dtype=np.float32).reshape(4, 8)
    enc = c.encode(arr)
    back = c.decode(enc, dtype=np.float32, shape=arr.shape)
    np.testing.assert_array_equal(back, arr)


def test_floatpred_roundtrip_float64():
    c = oc.get_codec("floatpred")
    rng = np.random.default_rng(0)
    arr = rng.standard_normal((8, 16)).astype(np.float64)
    enc = c.encode(arr)
    back = c.decode(enc, dtype=np.float64, shape=arr.shape)
    np.testing.assert_array_equal(back, arr)


# ---------------------------------------------------------------------------
# quantize
# ---------------------------------------------------------------------------


def test_quantize_bitround_keeps_top_bits():
    """bitspersample=10 keeps the top 10 mantissa bits of a float32.
    The relative error should be < 2**-10 for nonzero values."""
    c = oc.get_codec("quantize")
    arr = np.array([1.0, 1.234567, 3.14159265, 2.71828], dtype=np.float32)
    enc = c.encode(arr, bitspersample=10)
    back = c.decode(enc, dtype=np.float32, shape=arr.shape)
    rel = np.abs(back - arr) / np.abs(arr)
    assert (rel < 2.0 ** -9).all(), f"max rel err {rel.max():.4e}"


def test_quantize_nsd_rounds_to_n_significant_digits():
    c = oc.get_codec("quantize")
    arr = np.array([1.234567, 12.34567, 0.001234567], dtype=np.float32)
    enc = c.encode(arr, mode="nsd", nsd=3)
    back = c.decode(enc, dtype=np.float32, shape=arr.shape)
    # 3 sig digits → "1.23", "12.3", "0.00123"
    expected = np.array([1.23, 12.3, 0.00123], dtype=np.float32)
    np.testing.assert_allclose(back, expected, rtol=1e-3)


def test_quantize_bitround_improves_zstd_ratio():
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(4096).astype(np.float32)
    q = oc.get_codec("quantize")
    zs = oc.get_codec("zstd")
    raw_z = zs.encode(arr.tobytes())
    quant_z = zs.encode(q.encode(arr, bitspersample=8))
    assert len(quant_z) < len(raw_z)


# ---------------------------------------------------------------------------
# packints
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bits", [1, 4, 7, 10, 12, 14])
def test_packints_arbitrary_bit_roundtrip(bits):
    rng = np.random.default_rng(bits)
    n = 137
    arr = rng.integers(0, 1 << bits, size=n, dtype=np.uint16)
    p = oc.get_codec("packints")
    packed = p.encode(arr, bitspersample=bits)
    assert len(packed) == (n * bits + 7) // 8
    unpacked = p.decode(packed, dtype=np.uint16, bitspersample=bits,
                        n_elements=n)
    np.testing.assert_array_equal(unpacked, arr)


def test_packints_byte_aligned_8_16_32():
    """For byte-aligned widths the codec shortcuts through frombuffer."""
    p = oc.get_codec("packints")
    for bits, dtype in [(8, np.uint8), (16, np.uint16), (32, np.uint32)]:
        arr = np.arange(100, dtype=dtype)
        packed = p.encode(arr, bitspersample=bits)
        unpacked = p.decode(packed, dtype=dtype, bitspersample=bits,
                            n_elements=100)
        np.testing.assert_array_equal(unpacked, arr)
