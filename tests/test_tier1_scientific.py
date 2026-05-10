"""Tier 1 scientific compressors — round-trip + parameterized edge cases.

Covers the 7 codecs added for v0.2:

    bitshuffle  — bit-transpose filter (lossless, in == out size)
    b2nd        — Blosc2 multidim cframe (lossless, self-describing)
    aec         — CCSDS 121.0-B-2 adaptive entropy (lossless integers)
    lerc        — Esri raster (lossless or near-lossless w/ max_z_error)
    zfp         — fast lossy 1D-4D arrays (rate / precision / accuracy /
                  reversible modes)
    sz3         — error-bounded scientific lossy (float only)
    pcodec      — modern lossless numerical compression
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc


def _need(name: str) -> None:
    if not oc.has_codec(name):
        pytest.skip(f"codec {name!r} not registered on this build")


# ---------------------------------------------------------------------------
# bitshuffle — bytes-in / bytes-out, output size == input size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,itemsize", [
    (np.uint8, 1), (np.uint16, 2), (np.uint32, 4), (np.float32, 4),
    (np.float64, 8),
])
def test_bitshuffle_roundtrip(dtype, itemsize):
    _need("bitshuffle")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 100, size=10000).astype(dtype)
    raw = arr.tobytes()
    shuf = oc.write(None, raw, format="bitshuffle", itemsize=itemsize)
    assert len(shuf) == len(raw), "bitshuffle output size must equal input size"
    back = oc.read(shuf, format="bitshuffle", itemsize=itemsize)
    assert back == raw


def test_bitshuffle_pairs_with_zstd():
    """Bitshuffle's *raison d'être*: pre-filter that radically improves
    LZ77 compression on typed data."""
    _need("bitshuffle")
    _need("zstd")
    arr = np.arange(20000, dtype=np.uint16)
    raw = arr.tobytes()
    raw_zstd = oc.write(None, raw, format="zstd")
    shuf = oc.write(None, raw, format="bitshuffle", itemsize=2)
    shuf_zstd = oc.write(None, shuf, format="zstd")
    # bitshuffle+zstd should be at least 5× smaller than zstd alone on a
    # sequential uint16 array. (Locally observed ratio is ~40×.)
    assert len(shuf_zstd) * 5 < len(raw_zstd), (
        f"bshuf+zstd={len(shuf_zstd)}, raw+zstd={len(raw_zstd)}"
    )


def test_bitshuffle_alias():
    _need("bitshuffle")
    assert oc.has_codec("bshuf")
    raw = b"\x00\x01\x02\x03\x04\x05\x06\x07"
    out = oc.write(None, raw, format="bshuf", itemsize=1)
    assert oc.read(out, format="bshuf", itemsize=1) == raw


# ---------------------------------------------------------------------------
# b2nd — Blosc2 NDim, self-describing array cframe
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,dtype", [
    ((100,),               np.float32),
    ((32, 64),             np.uint16),
    ((8, 16, 32),          np.float64),
    ((4, 8, 16, 32),       np.int32),
])
def test_b2nd_roundtrip(shape, dtype):
    _need("b2nd")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 100, size=shape).astype(dtype)
    blob = oc.write(None, arr, format="b2nd")
    back = oc.read(blob, format="b2nd")
    assert back.shape == arr.shape
    assert back.dtype == arr.dtype
    np.testing.assert_array_equal(back, arr)


def test_b2nd_self_describing_inspect():
    """inspect() returns shape/dtype without decompressing."""
    _need("b2nd")
    arr = np.arange(1000, dtype=np.float32).reshape(10, 100)
    blob = oc.write(None, arr, format="b2nd", shuffle="bit")
    codec = oc.get_codec("b2nd")
    meta = codec.inspect(blob)
    assert meta["ndim"] == 2
    assert meta["shape"] == (10, 100)
    assert meta["itemsize"] == 4


def test_b2nd_compresses_typed_runs():
    """Bitshuffle filter inside b2nd should compress sequential u16
    enormously."""
    _need("b2nd")
    arr = np.arange(512 * 512, dtype=np.uint16).reshape(512, 512)
    blob = oc.write(None, arr, format="b2nd", shuffle="bit", compressor="zstd")
    assert len(blob) * 100 < arr.nbytes, (
        f"sequential u16 should compress >100x, got {arr.nbytes/len(blob):.1f}x"
    )


# ---------------------------------------------------------------------------
# aec — lossless integer entropy coding (CCSDS 121.0-B-2)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.int8, np.uint16, np.int16])
def test_aec_roundtrip(dtype):
    _need("aec")
    rng = np.random.default_rng(0)
    arr = rng.integers(
        np.iinfo(dtype).min // 2, np.iinfo(dtype).max // 2,
        size=5000, dtype=dtype,
    )
    blob = oc.write(None, arr, format="aec")
    back = oc.read(blob, format="aec")
    np.testing.assert_array_equal(np.frombuffer(back, dtype=dtype), arr)


def test_aec_explicit_bits_per_sample():
    """Pass bits_per_sample for sub-word data (12-bit values in u16
    storage — common in satellite imagery)."""
    _need("aec")
    arr = np.random.default_rng(0).integers(0, 4096, size=10000, dtype=np.uint16)
    blob = oc.write(None, arr, format="aec", bits_per_sample=12)
    back = np.frombuffer(oc.read(blob, format="aec"), dtype=np.uint16)
    np.testing.assert_array_equal(back, arr)


def test_aec_compresses_correlated_data():
    """AEC's strength: locally-predictable integer streams."""
    _need("aec")
    arr = np.arange(100000, dtype=np.uint16)  # perfectly predictable
    blob = oc.write(None, arr, format="aec")
    assert len(blob) * 5 < arr.nbytes, (
        f"sequential u16 should compress >=5x, got {arr.nbytes/len(blob):.1f}x"
    )


# ---------------------------------------------------------------------------
# lerc — Esri raster compression, lossless or near-lossless
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32, np.float64])
def test_lerc_lossless_roundtrip(dtype):
    _need("lerc")
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.integer):
        arr = rng.integers(0, np.iinfo(dtype).max // 2, size=(64, 96)).astype(dtype)
    else:
        arr = rng.random(size=(64, 96)).astype(dtype)
    blob = oc.write(None, arr, format="lerc")
    back = oc.read(blob, format="lerc")
    np.testing.assert_array_equal(back, arr)


def test_lerc_near_lossless_respects_error_budget():
    _need("lerc")
    arr = np.random.default_rng(0).random((128, 128)).astype(np.float32)
    blob = oc.write(None, arr, format="lerc", max_z_error=1e-3)
    back = oc.read(blob, format="lerc")
    err = np.abs(arr - back).max()
    assert err <= 1e-3 + 1e-7, f"max_z_error breached: {err}"


def test_lerc_self_describing_info():
    """info() returns shape, dtype, and value range without decoding."""
    _need("lerc")
    codec = oc.get_codec("lerc")
    arr = np.arange(64 * 96, dtype=np.uint16).reshape(64, 96)
    blob = oc.write(None, arr, format="lerc")
    nfo = codec.info(blob)
    assert nfo["n_rows"] == 64
    assert nfo["n_cols"] == 96
    assert nfo["dtype"] == np.dtype(np.uint16)
    assert nfo["z_min"] == float(arr.min())
    assert nfo["z_max"] == float(arr.max())


def test_lerc_signature_check():
    _need("lerc")
    codec = oc.get_codec("lerc")
    arr = np.zeros((8, 8), dtype=np.uint8)
    blob = oc.write(None, arr, format="lerc")
    assert codec.signature(blob) is True
    assert codec.signature(b"not-a-lerc-blob") is False


# ---------------------------------------------------------------------------
# zfp — fast lossy 1D-4D float / int compression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [
    (1024,), (64, 64), (8, 16, 32), (4, 8, 16, 32),
])
def test_zfp_reversible_roundtrip_f32(shape):
    _need("zfp")
    arr = np.random.default_rng(0).random(shape).astype(np.float32)
    blob = oc.write(None, arr, format="zfp", mode="reversible")
    back = oc.read(blob, format="zfp")
    np.testing.assert_array_equal(back, arr)


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.int32, np.int64])
def test_zfp_reversible_roundtrip_dtypes(dtype):
    _need("zfp")
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.integer):
        arr = rng.integers(-100000, 100000, size=(32, 32)).astype(dtype)
    else:
        arr = rng.random((32, 32)).astype(dtype)
    blob = oc.write(None, arr, format="zfp", mode="reversible")
    back = oc.read(blob, format="zfp")
    np.testing.assert_array_equal(back, arr)


def test_zfp_accuracy_mode_respects_budget():
    _need("zfp")
    arr = np.random.default_rng(0).random((64, 64)).astype(np.float32)
    blob = oc.write(None, arr, format="zfp", mode="accuracy", accuracy=1e-3)
    back = oc.read(blob, format="zfp")
    assert np.abs(arr - back).max() <= 1e-3 + 1e-6


def test_zfp_rate_mode_size_predictable():
    """rate=4 means 4 bits per value; size should scale linearly."""
    _need("zfp")
    arr = np.random.default_rng(0).random((128, 128)).astype(np.float32)
    blob = oc.write(None, arr, format="zfp", mode="rate", rate=4)
    # 4 bits/value × 128*128 values = 8192 bytes payload + small header.
    assert len(blob) < arr.nbytes // 2  # at least 2× compression


def test_zfp_signature():
    _need("zfp")
    codec = oc.get_codec("zfp")
    arr = np.zeros((16, 16), dtype=np.float32)
    blob = oc.write(None, arr, format="zfp", mode="reversible")
    assert codec.signature(blob) is True


# ---------------------------------------------------------------------------
# sz3 — error-bounded lossy (float only)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [
    (1024,), (256, 256), (32, 32, 32), (8, 16, 16, 8),
])
def test_sz3_abs_mode_respects_budget(shape):
    _need("sz3")
    arr = np.random.default_rng(0).random(shape).astype(np.float32)
    blob = oc.write(None, arr, format="sz3", mode="abs", abs_err=1e-3)
    back = oc.read(blob, format="sz3")
    assert np.abs(arr - back).max() <= 1e-3 + 1e-6


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_sz3_dtype_roundtrip(dtype):
    _need("sz3")
    arr = np.random.default_rng(0).random((128, 128)).astype(dtype)
    blob = oc.write(None, arr, format="sz3", mode="abs", abs_err=1e-3)
    back = oc.read(blob, format="sz3")
    assert back.dtype == arr.dtype
    np.testing.assert_allclose(back, arr, atol=1e-3)


def test_sz3_rejects_integer_dtype():
    """SZ3 v3 C API only handles floats; reject integers with a clear msg."""
    _need("sz3")
    arr = np.arange(100, dtype=np.int32)
    with pytest.raises(ValueError, match="float32/float64"):
        oc.write(None, arr, format="sz3")


# ---------------------------------------------------------------------------
# pcodec — modern lossless numerical compressor
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [
    np.uint8, np.int8, np.uint16, np.int16, np.float16,
    np.uint32, np.int32, np.float32,
    np.uint64, np.int64, np.float64,
])
def test_pcodec_roundtrip(dtype):
    _need("pcodec")
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        arr = rng.integers(info.min // 2, info.max // 2, size=10000).astype(dtype)
    else:
        # float16 has a smaller range; clamp accordingly.
        arr = rng.random(10000).astype(dtype)
    blob = oc.write(None, arr, format="pcodec")
    back = oc.read(blob, format="pcodec")
    np.testing.assert_array_equal(back, arr)


def test_pcodec_alias():
    _need("pcodec")
    assert oc.has_codec("pco")
    arr = np.arange(100, dtype=np.float32)
    assert np.array_equal(oc.read(oc.write(None, arr, format="pco"), format="pco"), arr)


def test_pcodec_beats_zstd_on_floats():
    """pcodec's pitch — beat zstd on dense numerical float arrays."""
    _need("pcodec")
    _need("zstd")
    rng = np.random.default_rng(0)
    arr = rng.standard_normal(20_000).astype(np.float32)
    pco = oc.write(None, arr, format="pcodec", level=8)
    zst = oc.write(None, arr.tobytes(), format="zstd", level=11)
    assert len(pco) < len(zst), (
        f"pcodec={len(pco)} vs zstd-11={len(zst)} on standard normal floats"
    )


def test_pcodec_multidim_shape_preserved():
    _need("pcodec")
    arr = np.random.default_rng(0).random((8, 16, 32)).astype(np.float32)
    blob = oc.write(None, arr, format="pcodec")
    back = oc.read(blob, format="pcodec")
    assert back.shape == arr.shape
    assert back.dtype == arr.dtype
    np.testing.assert_array_equal(back, arr)
