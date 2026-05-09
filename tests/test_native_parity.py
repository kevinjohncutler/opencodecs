"""Parity tests: opencodecs native codecs vs. imagecodecs reference.

Each codec is tested for:
  - Round-trip (encode then decode produces input)
  - imagecodecs decode of opencodecs encode
  - opencodecs decode of imagecodecs encode

Non-trivial fixtures are seeded so failures are reproducible.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import opencodecs as oc

imagecodecs = pytest.importorskip("imagecodecs")


def _need(codec_name: str):
    """Skip the calling test if a codec isn't registered on this host
    (e.g. the system library it wraps isn't installed)."""
    if not oc.has_codec(codec_name):
        pytest.skip(f"codec {codec_name!r} not registered on this host")


# ---------------------------------------------------------------------------
# zstd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, 1, 1024, 64 * 1024, 1024 * 1024])
def test_zstd_random(size: int) -> None:
    _need("zstd")
    data = os.urandom(size)
    oc_enc = oc.write(None, data, format="zstd")
    assert oc.read(oc_enc, format="zstd") == data
    if size:
        assert imagecodecs.zstd_decode(oc_enc) == data
        assert oc.read(imagecodecs.zstd_encode(data), format="zstd") == data


@pytest.mark.parametrize("level", [1, 3, 9, 19])
def test_zstd_levels(level: int) -> None:
    _need("zstd")
    data = os.urandom(64 * 1024)
    oc_enc = oc.write(None, data, format="zstd", level=level)
    assert imagecodecs.zstd_decode(oc_enc) == data


# ---------------------------------------------------------------------------
# LZ4 (frame format)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, 1, 1024, 64 * 1024, 1024 * 1024])
def test_lz4_random(size: int) -> None:
    _need("lz4")
    data = os.urandom(size)
    oc_enc = oc.write(None, data, format="lz4")
    assert oc.read(oc_enc, format="lz4") == data
    if size:
        assert imagecodecs.lz4f_decode(oc_enc) == data
        assert oc.read(imagecodecs.lz4f_encode(data), format="lz4") == data


def test_lz4_compressible() -> None:
    _need("lz4")
    data = b"A" * (256 * 1024)
    oc_enc = oc.write(None, data, format="lz4")
    assert imagecodecs.lz4f_decode(oc_enc) == data
    assert len(oc_enc) < len(data) // 50  # heavily compressible


# ---------------------------------------------------------------------------
# brotli
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, 1, 1024, 64 * 1024, 1024 * 1024])
def test_brotli_random(size: int) -> None:
    _need("brotli")
    data = os.urandom(size)
    oc_enc = oc.write(None, data, format="brotli")
    assert oc.read(oc_enc, format="brotli") == data
    if size:
        assert imagecodecs.brotli_decode(oc_enc) == data
        assert oc.read(imagecodecs.brotli_encode(data), format="brotli") == data


@pytest.mark.parametrize("level", [0, 5, 11])
def test_brotli_levels(level: int) -> None:
    _need("brotli")
    data = os.urandom(64 * 1024)
    oc_enc = oc.write(None, data, format="brotli", level=level)
    assert imagecodecs.brotli_decode(oc_enc) == data


# ---------------------------------------------------------------------------
# QOI
# ---------------------------------------------------------------------------


def test_qoi_rgb() -> None:
    _need("qoi")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    oc_enc = oc.write(None, arr, format="qoi")
    np.testing.assert_array_equal(oc.read(oc_enc, format="qoi"), arr)
    np.testing.assert_array_equal(imagecodecs.qoi_decode(oc_enc), arr)
    np.testing.assert_array_equal(
        oc.read(imagecodecs.qoi_encode(arr), format="qoi"), arr)


def test_qoi_rgba() -> None:
    _need("qoi")
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, (32, 48, 4), dtype=np.uint8)
    oc_enc = oc.write(None, arr, format="qoi")
    np.testing.assert_array_equal(oc.read(oc_enc, format="qoi"), arr)


# ---------------------------------------------------------------------------
# BMP
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape",
    [(16, 16), (32, 40, 3), (32, 40, 4), (5, 7, 3), (200, 256, 3)],
)
def test_bmp_byte_parity(shape: tuple[int, ...]) -> None:
    _need("bmp")
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, shape, dtype=np.uint8)
    oc_enc = oc.write(None, arr, format="bmp")
    try:
        ic_enc = imagecodecs.bmp_encode(arr)
    except ValueError:
        # imagecodecs < 2026.3.x rejects 32-bit RGBA BMP encode. Skip the
        # byte-parity arm of the test on older imagecodecs builds; round-trip
        # still validates our encoder/decoder.
        np.testing.assert_array_equal(oc.read(oc_enc, format="bmp"), arr)
        # imagecodecs<2026.3.x also can't decode BI_BITFIELDS (compression
        # type 3); skip if older.
        try:
            np.testing.assert_array_equal(imagecodecs.bmp_decode(oc_enc), arr)
        except Exception as exc:
            if "compression_type=3" in str(exc):
                pytest.skip("older imagecodecs lacks BI_BITFIELDS support")
            raise
        return
    assert oc_enc == ic_enc, "BMP encode should be byte-identical to imagecodecs"
    np.testing.assert_array_equal(oc.read(oc_enc, format="bmp"), arr)
    np.testing.assert_array_equal(imagecodecs.bmp_decode(oc_enc), arr)
    np.testing.assert_array_equal(
        oc.read(ic_enc, format="bmp"), arr)


# ---------------------------------------------------------------------------
# PNG (via libspng)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((32, 40), np.uint8),
        ((32, 40, 2), np.uint8),
        ((32, 40, 3), np.uint8),
        ((32, 40, 4), np.uint8),
        ((16, 20), np.uint16),
        ((16, 20, 2), np.uint16),
        ((16, 20, 3), np.uint16),
        ((16, 20, 4), np.uint16),
    ],
)
def test_png_roundtrip(shape: tuple[int, ...], dtype) -> None:
    _need("png")
    rng = np.random.default_rng(7)
    high = 65536 if dtype is np.uint16 else 256
    arr = rng.integers(0, high, shape, dtype=dtype)
    oc_enc = oc.write(None, arr, format="png")
    np.testing.assert_array_equal(
        np.squeeze(oc.read(oc_enc, format="png")), np.squeeze(arr))
    np.testing.assert_array_equal(
        np.squeeze(imagecodecs.png_decode(oc_enc)), np.squeeze(arr))
    ic_enc = imagecodecs.png_encode(arr)
    np.testing.assert_array_equal(
        np.squeeze(oc.read(ic_enc, format="png")), np.squeeze(arr))


@pytest.mark.parametrize("level", [0, 3, 6, 9])
def test_png_levels(level: int) -> None:
    _need("png")
    rng = np.random.default_rng(11)
    arr = rng.integers(0, 256, (256, 256, 3), dtype=np.uint8)
    oc_enc = oc.write(None, arr, format="png", level=level)
    np.testing.assert_array_equal(
        imagecodecs.png_decode(oc_enc), arr)


# ---------------------------------------------------------------------------
# blosc2
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, 1, 1024, 64 * 1024, 1024 * 1024])
def test_blosc2_random(size: int) -> None:
    _need("blosc2")
    data = os.urandom(size)
    oc_enc = oc.write(None, data, format="blosc2")
    assert oc.read(oc_enc, format="blosc2") == data
    if size:
        assert imagecodecs.blosc2_decode(oc_enc) == data
        assert oc.read(imagecodecs.blosc2_encode(data), format="blosc2") == data


# ---------------------------------------------------------------------------
# deflate / zlib
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("size", [0, 1, 1024, 64 * 1024, 1024 * 1024])
def test_deflate_random(size: int) -> None:
    _need("deflate")
    data = os.urandom(size)
    oc_enc = oc.write(None, data, format="deflate")
    assert oc.read(oc_enc, format="deflate") == data
    if size:
        assert imagecodecs.zlib_decode(oc_enc) == data
        assert oc.read(imagecodecs.zlib_encode(data), format="deflate") == data


# ---------------------------------------------------------------------------
# JPEG (libjpeg-turbo) — lossy, so test cross-decode equivalence
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(64, 96, 3), (64, 96)])
@pytest.mark.parametrize("level", [50, 75, 95])
def test_jpeg_cross_decode(shape: tuple[int, ...], level: int) -> None:
    _need("jpeg")
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 256, shape, dtype=np.uint8)
    oc_enc = oc.write(None, arr, format="jpeg", level=level)
    ic_enc = imagecodecs.jpeg_encode(arr, level=level)
    # Decoder cross-equality: each encode decoded by both decoders should match.
    oc_dec_oc = oc.read(oc_enc, format="jpeg")
    ic_dec_oc = imagecodecs.jpeg_decode(oc_enc)
    np.testing.assert_array_equal(np.squeeze(oc_dec_oc), np.squeeze(ic_dec_oc))
    oc_dec_ic = oc.read(ic_enc, format="jpeg")
    ic_dec_ic = imagecodecs.jpeg_decode(ic_enc)
    np.testing.assert_array_equal(np.squeeze(oc_dec_ic), np.squeeze(ic_dec_ic))


# ---------------------------------------------------------------------------
# WebP
# ---------------------------------------------------------------------------


def test_webp_lossless_rgb() -> None:
    _need("webp")
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    oc_enc = oc.write(None, arr, format="webp", lossless=True)
    np.testing.assert_array_equal(oc.read(oc_enc, format="webp"), arr)
    np.testing.assert_array_equal(imagecodecs.webp_decode(oc_enc), arr)


def test_webp_lossy_cross() -> None:
    _need("webp")
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    oc_enc = oc.write(None, arr, format="webp", level=80)
    np.testing.assert_array_equal(
        oc.read(oc_enc, format="webp"),
        imagecodecs.webp_decode(oc_enc),
    )
    ic_enc = imagecodecs.webp_encode(arr, level=80)
    np.testing.assert_array_equal(
        oc.read(ic_enc, format="webp"),
        imagecodecs.webp_decode(ic_enc),
    )


# ---------------------------------------------------------------------------
# JPEG-2000
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "shape, dtype",
    [
        ((32, 48), np.uint8),
        ((32, 48, 3), np.uint8),
        ((32, 48, 4), np.uint8),
        ((32, 48), np.uint16),
    ],
)
def test_jpeg2k_lossless(shape, dtype) -> None:
    _need("jpeg2k")
    rng = np.random.default_rng(11)
    high = 65536 if dtype is np.uint16 else 256
    arr = rng.integers(0, high, shape, dtype=dtype)
    oc_enc = oc.write(None, arr, format="jpeg2k", lossless=True)
    np.testing.assert_array_equal(oc.read(oc_enc, format="jpeg2k"), arr)
    np.testing.assert_array_equal(imagecodecs.jpeg2k_decode(oc_enc), arr)
    ic_enc = imagecodecs.jpeg2k_encode(arr)
    np.testing.assert_array_equal(oc.read(ic_enc, format="jpeg2k"), arr)


# ---------------------------------------------------------------------------
# AVIF — lossless RGB / RGBA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [(32, 48, 3), (32, 48, 4)])
def test_avif_lossless(shape: tuple[int, ...]) -> None:
    _need("avif")
    rng = np.random.default_rng(23)
    arr = rng.integers(0, 256, shape, dtype=np.uint8)
    try:
        oc_enc = oc.write(None, arr, format="avif", lossless=True)
    except Exception as exc:
        msg = str(exc).lower()
        if any(s in msg for s in ("no codec available", "unsupported", "encoder")):
            pytest.skip(f"libavif build has no AV1 encoder: {exc}")
        raise
    np.testing.assert_array_equal(oc.read(oc_enc, format="avif"), arr)


# ---------------------------------------------------------------------------
# HEIF — lossy round-trip (imagecodecs wheel may lack heif support)
# ---------------------------------------------------------------------------


def test_heif_roundtrip_smoke() -> None:
    _need("heif")
    rng = np.random.default_rng(33)
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    try:
        oc_enc = oc.write(None, arr, format="heif", level=80)
    except Exception as exc:
        # Linux libheif from APT often ships without an HEVC encoder plugin
        # (x265/aomenc are separate packages). Conda-forge libheif on
        # Windows builds without HEVC for licensing reasons. Decode-only
        # is still useful; mark these as skip rather than fail.
        msg = str(exc).lower()
        if any(s in msg for s in (
            "encoder", "unsupported", "null error text", "heif_writer",
        )):
            pytest.skip(f"no HEVC encoder available: {exc}")
        raise
    out = oc.read(oc_enc, format="heif")
    assert out.shape == arr.shape
    assert out.dtype == arr.dtype


# ---------------------------------------------------------------------------
# zarr v3 codec wrappers (skip when zarr not installed)
# ---------------------------------------------------------------------------


zarr_mod = pytest.importorskip("zarr")


@pytest.mark.parametrize("codec_name", [
    "OcZstd", "OcLz4", "OcBlosc2", "OcBrotli", "OcDeflate",
])
def test_zarr_v3_codec(codec_name: str, tmp_path) -> None:
    # Each Oc* codec wraps an underlying opencodecs codec — skip if absent.
    underlying = {
        "OcZstd": "zstd", "OcLz4": "lz4", "OcBlosc2": "blosc2",
        "OcBrotli": "brotli", "OcDeflate": "deflate",
    }[codec_name]
    _need(underlying)
    import opencodecs._zarr_codecs as ozc
    Codec = getattr(ozc, codec_name, None)
    if Codec is None:
        pytest.skip(f"{codec_name} not available")
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (100, 100), dtype=np.uint8)
    store = zarr_mod.storage.LocalStore(str(tmp_path / "x.zarr"))
    z = zarr_mod.create_array(
        store=store, shape=arr.shape, dtype=arr.dtype, chunks=(50, 50),
        compressors=[Codec(level=5) if codec_name != "OcLz4" else Codec()],
        zarr_format=3,
    )
    z[:] = arr
    np.testing.assert_array_equal(z[:], arr)


# ---------------------------------------------------------------------------
# HDF5 reader (skip when h5py not installed)
# ---------------------------------------------------------------------------


h5py = pytest.importorskip("h5py")


def test_hdf5_reader(tmp_path) -> None:
    arr = np.arange(60, dtype=np.float32).reshape(3, 4, 5)
    p = tmp_path / "test.h5"
    with h5py.File(p, "w") as h5:
        h5.create_dataset("image", data=arr)
    out = oc.read(p)
    np.testing.assert_array_equal(out, arr)
    with oc.get_codec("hdf5").open(p) as r:
        assert r.shape == arr.shape
        assert r.dtype == arr.dtype
        # Random-access frame slicing.
        np.testing.assert_array_equal(r[1], arr[1])


# ---------------------------------------------------------------------------
# CZI native reader (parity with czifile, parallel decode)
# ---------------------------------------------------------------------------


# Uses a real lab CZI; skip when the NAS isn't mounted.
import os.path
_LAB_CZI = (
    "/Volumes/HiprDrive/2024_02_02_GNE_synthetic_community/"
    "2024_02_02_GNEPanelTest_slide1_B1_GNE0001_cellmix01_200nMENC_"
    "20nMCOMP_quarterpower_fov_4_561.czi"
)


@pytest.mark.skipif(
    not os.path.isfile(_LAB_CZI),
    reason="reference lab CZI file not available",
)
def test_czi_codec_unified_api() -> None:
    _need("czi")
    czifile = pytest.importorskip("czifile")

    # End-to-end via opencodecs.read()
    arr_oc = np.squeeze(oc.read(_LAB_CZI))

    # Reference via czifile
    with czifile.CziFile(_LAB_CZI) as cz:
        arr_ref = np.stack(
            [np.squeeze(sb.data()) for sb in cz.subblocks()], axis=0,
        )

    np.testing.assert_array_equal(arr_oc, arr_ref)


@pytest.mark.skipif(
    not os.path.isfile(_LAB_CZI),
    reason="reference lab CZI file not available",
)
def test_czi_codec_signature_and_path() -> None:
    _need("czi")

    # Magic-byte detection
    with open(_LAB_CZI, "rb") as f:
        head = f.read(64)
    codec = oc.codec_for_bytes(head)
    assert codec is not None
    assert codec.name == "czi"

    # Path-based lookup
    assert oc.codec_for_path("anything.czi").name == "czi"

    # Reader interface
    with oc.get_codec("czi").open(_LAB_CZI) as r:
        assert r.is_chunked
        assert r.n_frames > 0
        assert len(r.shape) >= 2
        f0 = r[0]
        assert f0.shape == r.shape[1:]


@pytest.mark.skipif(
    not os.path.isfile(_LAB_CZI),
    reason="reference lab CZI file not available",
)
def test_czi_metadata_access() -> None:
    """metadata_bytes / metadata_xml are bytes-and-str views of the same XML.

    Compares against czifile's reference parse and verifies the cache
    returns the identical bytes object on repeat calls (so downstream
    parsers can use ``is`` comparisons or memoize).
    """
    _need("czi")
    czifile = pytest.importorskip("czifile")

    with oc.get_codec("czi").open(_LAB_CZI) as r:
        mb = r.metadata_bytes
        mx = r.metadata_xml
        assert isinstance(mb, bytes)
        assert isinstance(mx, str)
        assert len(mb) > 0
        assert len(mx) == len(mb)              # ASCII / direct decode
        assert r.metadata_bytes is mb           # cached
        assert r.metadata_xml is mx             # cached
        assert "<ImageDocument" in mx
        assert b"<ImageDocument" in mb

    # Reference: czifile produces equivalent text.
    with czifile.CziFile(_LAB_CZI) as cz:
        ref = cz.metadata()
    assert "<ImageDocument" in ref
    # Sizes match (czifile may apply its own escape fixups but the bulk is the same).
    assert abs(len(ref) - len(mx)) < 64
