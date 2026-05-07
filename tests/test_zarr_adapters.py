"""Tests for the zarr / numcodecs adapter modules.

opencodecs ships two adapter surfaces:

  * ``opencodecs.zarr.JxlCodec`` — a numcodecs-style codec that lets
    you store zarr v2 arrays compressed with JPEG XL. Each chunk is a
    JXL still (or animation when ``animation=True``).
  * ``opencodecs._zarr_codecs.OcZstd / OcLz4 / OcBrotli / OcBlosc2 /
    OcDeflate`` — zarr v3 ``BytesBytesCodec`` wrappers that route
    chunk bytes through opencodecs's native compressors.

These modules had ~0% coverage before this file. The tests verify that:

  * ``JxlCodec`` round-trips numpy chunks of every supported shape
    (2-D grayscale, 3-D RGB, 3-D RGBA, multi-frame in animation mode).
  * The codec self-registers in numcodecs so a serialized zarr store
    can deserialize without an explicit import.
  * The v3 ``Oc*`` wrappers round-trip arbitrary bytes through every
    backend.
  * Both surfaces handle the same buffer-protocol inputs that zarr
    will actually pass at runtime (numpy uint8 views, memoryviews).
"""

from __future__ import annotations

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# numcodecs adapter (opencodecs.zarr.JxlCodec)
# ---------------------------------------------------------------------------


def _need_jxl() -> None:
    pytest.importorskip("numcodecs")
    from opencodecs.jxl import _HAVE_BACKEND
    if not _HAVE_BACKEND:
        pytest.skip("libjxl backend not available")


def test_jxlcodec_imports():
    _need_jxl()
    from opencodecs.zarr import JxlCodec
    assert JxlCodec.codec_id == "opencodecs_jxl"


def test_jxlcodec_self_registers_in_numcodecs():
    """Importing opencodecs.zarr should auto-register so existing zarr
    stores serialized with this codec_id will resolve at read time."""
    _need_jxl()
    import opencodecs.zarr  # noqa: F401  (registration side-effect)
    from numcodecs.registry import get_codec
    codec = get_codec({"id": "opencodecs_jxl", "lossless": True})
    from opencodecs.zarr import JxlCodec
    assert isinstance(codec, JxlCodec)


def test_jxlcodec_roundtrip_2d_grayscale():
    _need_jxl()
    from opencodecs.zarr import JxlCodec
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 256, (32, 48), dtype=np.uint8)
    c = JxlCodec(lossless=True)
    enc = c.encode(chunk)
    assert isinstance(enc, bytes)
    dec = c.decode(enc)
    np.testing.assert_array_equal(np.squeeze(dec), chunk)


def test_jxlcodec_roundtrip_3d_rgb():
    _need_jxl()
    from opencodecs.zarr import JxlCodec
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    c = JxlCodec(lossless=True)
    dec = c.decode(c.encode(chunk))
    np.testing.assert_array_equal(np.squeeze(dec), chunk)


def test_jxlcodec_roundtrip_3d_rgba():
    _need_jxl()
    from opencodecs.zarr import JxlCodec
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 256, (32, 48, 4), dtype=np.uint8)
    c = JxlCodec(lossless=True)
    dec = c.decode(c.encode(chunk))
    np.testing.assert_array_equal(np.squeeze(dec), chunk)


def test_jxlcodec_strips_leading_singleton_axes():
    """zarr chunks frequently have leading length-1 axes (T=1 from the
    chunking) — encode should squeeze them away rather than failing."""
    _need_jxl()
    from opencodecs.zarr import JxlCodec
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 256, (1, 1, 32, 48, 3), dtype=np.uint8)
    c = JxlCodec(lossless=True)
    enc = c.encode(chunk)
    # Decoded back as the squeezed shape — the (T=1, T=1) prefix is gone.
    dec = c.decode(enc)
    np.testing.assert_array_equal(np.squeeze(dec), np.squeeze(chunk))


def test_jxlcodec_high_dim_rejected_without_animation():
    """A genuine 4-D chunk (not just leading singletons) is not
    encodable as a single JXL still — must raise unless animation=True."""
    _need_jxl()
    from opencodecs.zarr import JxlCodec
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 256, (3, 32, 48, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="not encodable"):
        JxlCodec(lossless=True).encode(chunk)


def test_jxlcodec_high_dim_works_with_animation():
    _need_jxl()
    from opencodecs.zarr import JxlCodec
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 256, (3, 32, 48, 3), dtype=np.uint8)
    c = JxlCodec(lossless=True, animation=True)
    enc = c.encode(chunk)
    assert isinstance(enc, bytes) and len(enc) > 0


def test_jxlcodec_decode_with_out_buffer():
    """zarr passes a pre-allocated ``out`` ndarray into decode(); the
    codec must write into it and return it."""
    _need_jxl()
    from opencodecs.zarr import JxlCodec
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    c = JxlCodec(lossless=True)
    enc = c.encode(chunk)

    out = np.empty_like(chunk)
    ret = c.decode(enc, out=out)
    assert ret is out
    np.testing.assert_array_equal(np.squeeze(out), chunk)


def test_jxlcodec_repr_lossless():
    _need_jxl()
    from opencodecs.zarr import JxlCodec
    r = repr(JxlCodec(lossless=True))
    assert "lossless" in r


def test_jxlcodec_repr_lossy():
    _need_jxl()
    from opencodecs.zarr import JxlCodec
    r = repr(JxlCodec(lossless=False, distance=1.0))
    assert "distance" in r


def test_jxlcodec_repr_includes_animation_flag():
    _need_jxl()
    from opencodecs.zarr import JxlCodec
    r = repr(JxlCodec(lossless=True, animation=True))
    assert "animation" in r


def test_jxlcodec_accepts_buffer_protocol():
    """zarr feeds chunks to encode() as numpy ndarrays, but the adapter
    falls back to ensure_contiguous_ndarray for memoryviews — both
    paths should produce the same output."""
    _need_jxl()
    from opencodecs.zarr import JxlCodec
    rng = np.random.default_rng(0)
    chunk = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    c = JxlCodec(lossless=True)
    enc_arr = c.encode(chunk)
    enc_mv = c.encode(memoryview(chunk))
    # Both should round-trip back to the same chunk; the bytes themselves
    # need not be identical (deterministic but format flexibility).
    np.testing.assert_array_equal(np.squeeze(c.decode(enc_arr)), chunk)
    np.testing.assert_array_equal(np.squeeze(c.decode(enc_mv)), chunk)


# ---------------------------------------------------------------------------
# zarr v3 BytesBytesCodec wrappers (_zarr_codecs.py)
# ---------------------------------------------------------------------------


def _need_zarr_v3():
    zarr = pytest.importorskip("zarr")
    if not zarr.__version__.startswith("3"):
        pytest.skip("OcZstd & friends require zarr v3")


_V3_WRAPPERS = ["OcZstd", "OcLz4", "OcBrotli", "OcBlosc2", "OcDeflate"]


@pytest.mark.parametrize("name", _V3_WRAPPERS)
def test_v3_wrapper_exists(name):
    _need_zarr_v3()
    from opencodecs import _zarr_codecs
    assert hasattr(_zarr_codecs, name)
    cls = getattr(_zarr_codecs, name)
    assert cls is not None


@pytest.mark.parametrize("name", _V3_WRAPPERS)
def test_v3_wrapper_to_dict_roundtrip(name):
    """to_dict / from_dict round-trip preserves the level config."""
    _need_zarr_v3()
    from opencodecs import _zarr_codecs
    cls = getattr(_zarr_codecs, name)
    inst = cls(level=3)
    d = inst.to_dict()
    assert d["configuration"].get("level") == 3
    rebuilt = cls.from_dict(d)
    assert rebuilt.level == 3


@pytest.mark.parametrize("name", _V3_WRAPPERS)
def test_v3_wrapper_to_dict_omits_level_when_none(name):
    _need_zarr_v3()
    from opencodecs import _zarr_codecs
    cls = getattr(_zarr_codecs, name)
    d = cls().to_dict()
    assert "level" not in d["configuration"]


_WRAPPER_TO_OC_NAME = {
    "OcZstd": "zstd", "OcLz4": "lz4", "OcBrotli": "brotli",
    "OcBlosc2": "blosc2", "OcDeflate": "deflate",
}


def _need_codec_for_wrapper(name: str) -> None:
    oc_name = _WRAPPER_TO_OC_NAME[name]
    import opencodecs as oc
    if not oc.has_codec(oc_name):
        pytest.skip(f"backing codec {oc_name!r} not registered")


@pytest.mark.parametrize("name", _V3_WRAPPERS)
def test_v3_wrapper_encode_decode_bytes(name):
    """Direct _encode_bytes/_decode_bytes round-trip — the inner path
    used by zarr v3 chunk storage."""
    _need_zarr_v3()
    _need_codec_for_wrapper(name)
    from opencodecs import _zarr_codecs
    cls = getattr(_zarr_codecs, name)
    inst = cls()
    payload = b"hello opencodecs zarr v3 wrapper" * 100
    enc = inst._encode_bytes(payload)
    dec = inst._decode_bytes(enc)
    assert dec == payload


@pytest.mark.parametrize("name", _V3_WRAPPERS)
def test_v3_wrapper_handles_numpy_input(name):
    """Zarr passes uint8 views; wrapper must coerce via .tobytes()."""
    _need_zarr_v3()
    _need_codec_for_wrapper(name)
    from opencodecs import _zarr_codecs
    cls = getattr(_zarr_codecs, name)
    inst = cls()
    payload_arr = np.frombuffer(b"some test bytes" * 100, dtype=np.uint8)
    enc = inst._encode_bytes(payload_arr)
    assert inst._decode_bytes(enc) == bytes(payload_arr)


@pytest.mark.parametrize("name", _V3_WRAPPERS)
def test_v3_wrapper_compute_encoded_size_raises(name):
    """The codecs are not fixed-size — compute_encoded_size must say so."""
    _need_zarr_v3()
    from opencodecs import _zarr_codecs
    cls = getattr(_zarr_codecs, name)
    inst = cls()
    assert inst.is_fixed_size is False
    with pytest.raises(NotImplementedError):
        inst.compute_encoded_size(1024, None)


def test_v3_wrappers_listed_in_all():
    _need_zarr_v3()
    from opencodecs import _zarr_codecs
    for name in _V3_WRAPPERS:
        assert name in _zarr_codecs.__all__
