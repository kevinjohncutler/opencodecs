"""Tests for the libultrahdr-backed Ultra-HDR / ISO 21496-1 codec.

The binding at ``opencodecs.codecs._uhdr`` is a thin wrapper around
libuhdr's own encode / decode entry points — no Python-side gain-map
reimplementation. These tests exercise the round-trip:

  HDR float ndarray → uhdr.encode → Ultra-HDR JPEG bytes
  → uhdr.decode → fp16 HDR + (optional) gain map + SDR base

plus the trivial probes (version string, signature detection) and the
file-on-disk read / write helpers in ``opencodecs.uhdr``.

Skipped at module level if the Cython extension isn't built (e.g. the
host doesn't have libultrahdr installed, which is the typical CI miss
case).
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

_uhdr_codecs_mod = pytest.importorskip("opencodecs.codecs._uhdr")

from opencodecs import uhdr as oc_uhdr
from opencodecs.codecs._uhdr import (
    compute_gain_map_u8,
    compute_sdr_base_u8,
    decode,
    encode,
    encode_assembled,
    is_uhdr,
    libuhdr_version,
    UhdrError,
)
from opencodecs.uhdr import (
    encode_native,
    decode_native,
    encode_to,
    probe,
    read_thumbnail,
    read_thumbnail_bytes,
    read_thumbnail_hdr,
)


# ---------------------------------------------------------------------------
# Trivial probes
# ---------------------------------------------------------------------------


def test_libuhdr_version_returns_dotted_string():
    v = libuhdr_version()
    assert isinstance(v, str)
    parts = v.split(".")
    assert len(parts) == 3
    # libultrahdr 1.x is the published series; allow forward jumps.
    assert all(p.isdigit() for p in parts)


def test_is_uhdr_false_on_garbage():
    assert is_uhdr(b"") is False
    assert is_uhdr(b"abc") is False
    assert is_uhdr(b"\x00" * 32) is False


def test_is_uhdr_false_on_plain_jpeg():
    """A plain (non-Ultra-HDR) JPEG should not register as Ultra-HDR
    — libuhdr's signature check looks for the gain-map metadata box."""
    plain = pytest.importorskip("imagecodecs")
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    jpg = plain.jpeg_encode(arr, level=95)
    assert is_uhdr(jpg) is False


# ---------------------------------------------------------------------------
# Encode → is_uhdr → decode round-trip
# ---------------------------------------------------------------------------


def _synthetic_hdr_rgb(h=64, w=64) -> np.ndarray:
    """A small linear-light Display-P3 HDR raster with a bright corner
    so libuhdr has actual HDR headroom to encode. 1.0 = SDR white."""
    arr = np.zeros((h, w, 3), dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    arr[..., 0] = (xx / w) * 4.0    # 0 → 4× SDR
    arr[..., 1] = (yy / h) * 4.0
    arr[..., 2] = ((xx + yy) / (w + h)) * 2.0
    return arr


def test_encode_produces_uhdr_jpeg():
    """Round-trip just the encode → signature check path. Confirms
    libuhdr emitted a real ISO 21496-1 container, not a plain JPEG."""
    hdr = _synthetic_hdr_rgb()
    data = encode(hdr, gamut="display-p3", quality=80)
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 0
    assert is_uhdr(data) is True


def test_decode_full_roundtrip():
    """Encode then decode; verify shape + dtype + bounded reconstruction
    error. We can't bit-exact-equal because libuhdr is lossy (JPEG base
    + lossy gain-map encoding), but the decoded HDR should track the
    input rasters' coarse structure."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data = encode(hdr, gamut="display-p3", quality=95)

    info = decode(data, want_hdr=True)
    assert set(info.keys()) >= {"hdr_fp16", "width", "height"}
    rec = info["hdr_fp16"]
    assert rec.dtype == np.float16
    # Decoder returns (H, W, 4) RGBA in canonical linear-light units.
    assert rec.shape[:2] == hdr.shape[:2]
    assert rec.shape[2] in (3, 4)
    # Coarse sanity: per-channel mean tracks the input. Ultra-HDR is
    # double-lossy (JPEG base + quantised gain map), so the budget has
    # to be generous — exact-equal is a much stricter contract than
    # the codec was designed to deliver. We just confirm the output
    # isn't pinned to 0 / constant / wildly off.
    rec_rgb = rec[..., :3].astype(np.float32)
    in_mean = hdr.mean(axis=(0, 1))
    out_mean = rec_rgb.mean(axis=(0, 1))
    # All channels actually have signal (not NaN, not zero).
    assert np.all(np.isfinite(out_mean))
    assert np.all(out_mean > 0.05 * in_mean)
    # And within a factor of 4x of input (HDR signals re-quantised
    # through the gain-map round-trip can lose / gain ~2x depending
    # on the encoder's content-boost choice).
    ratio = out_mean / np.maximum(in_mean, 1e-6)
    assert (ratio > 0.25).all() and (ratio < 4.0).all(), (
        f"per-channel ratio out-of-bounds: in={in_mean}, out={out_mean}"
    )


def test_decode_with_gainmap_and_base():
    """want_gainmap=True returns the raw gain map; want_base=True returns
    the embedded SDR JPEG. Both are standalone artefacts the user can
    introspect or save independently."""
    hdr = _synthetic_hdr_rgb()
    data = encode(hdr, quality=95)
    info = decode(
        data,
        want_hdr=False,
        want_gainmap=True,
        want_base=True,
    )
    assert "gainmap_u8" in info
    assert "gainmap_metadata" in info
    assert "base_compressed" in info
    gm = info["gainmap_u8"]
    assert gm.dtype == np.uint8
    assert gm.ndim == 3 and gm.shape[2] in (3, 4)
    meta = info["gainmap_metadata"]
    assert isinstance(meta, dict)
    assert "max_content_boost" in meta
    # base_compressed is a valid standalone JPEG: starts with SOI
    # marker (0xFFD8). HDR-unaware viewers read just this.
    base = info["base_compressed"]
    assert isinstance(base, (bytes, bytearray))
    assert bytes(base[:2]) == b"\xff\xd8"


# ---------------------------------------------------------------------------
# Public file-on-disk wrappers
# ---------------------------------------------------------------------------


def test_write_then_read_roundtrip(tmp_path: Path):
    hdr = _synthetic_hdr_rgb()
    p = tmp_path / "synth.jpg"
    oc_uhdr.write(str(p), hdr, quality=90)
    assert p.exists()
    info = oc_uhdr.read(str(p), want_hdr=True)
    assert info["hdr_fp16"].shape[:2] == hdr.shape[:2]


# ---------------------------------------------------------------------------
# encode_assembled: pre-encoded layers in → libuhdr container out
# ---------------------------------------------------------------------------


def test_encode_assembled_minimal():
    """encode_assembled takes pre-encoded SDR + gain-map JPEGs plus a
    metadata dict and emits an Ultra-HDR container. Confirms the
    direct-binding entry point works without going through encode()."""
    imagecodecs = pytest.importorskip("imagecodecs")
    # Tiny SDR base + tiny gain map, both as plain JPEGs.
    H, W = 32, 32
    sdr = (np.linspace(0, 255, H * W * 3, dtype=np.uint8)
           .reshape(H, W, 3))
    gain = np.full((H, W, 3), 128, dtype=np.uint8)
    sdr_jpeg = imagecodecs.jpeg_encode(sdr, level=85)
    gain_jpeg = imagecodecs.jpeg_encode(gain, level=85)

    metadata = {
        "max_content_boost": 4.0,
        "min_content_boost": 1.0,
        "gamma": 1.0,
        "offset_sdr": 0.0,
        "offset_hdr": 0.0,
        "hdr_capacity_min": 1.0,
        "hdr_capacity_max": 4.0,
        "use_base_cg": True,
    }
    blob = encode_assembled(
        base_jpeg=sdr_jpeg, gainmap_jpeg=gain_jpeg, metadata=metadata,
    )
    assert is_uhdr(blob) is True


# ---------------------------------------------------------------------------
# encode_native: fused-Cython fast path
# ---------------------------------------------------------------------------


def test_encode_native_produces_uhdr():
    """The native fast path computes SDR + gain map outside libuhdr and
    hands pre-encoded JPEGs to libuhdr's container-assembly api. Output
    must still be a valid Ultra-HDR JPEG."""
    hdr = _synthetic_hdr_rgb()
    data = encode_native(hdr, gamut="display-p3", quality=95)
    assert isinstance(data, (bytes, bytearray))
    assert len(data) > 0
    assert is_uhdr(data) is True


def test_encode_native_roundtrip_matches_libuhdr():
    """encode_native and encode should produce decoded HDR rasters in
    roughly the same neighbourhood — gain-map quantization + JPEG-q95
    noise is the dominant error, not the kernel implementation choice."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data_n = encode_native(hdr, quality=95)
    data_r = encode(hdr, quality=95)
    rec_n = decode(data_n, want_hdr=True)["hdr_fp16"][..., :3].astype(
        np.float32)
    rec_r = decode(data_r, want_hdr=True)["hdr_fp16"][..., :3].astype(
        np.float32)
    # Mean per-channel ratio should be within 2x in either direction
    # (both encoders normalize by ~peak, so they share the same coarse
    # scale; q95 + 8-bit gain-map quantisation contributes the residual).
    mn = rec_n.mean(axis=(0, 1))
    mr = rec_r.mean(axis=(0, 1))
    assert np.all(np.isfinite(mn)) and np.all(np.isfinite(mr))
    ratio = mn / np.maximum(mr, 1e-6)
    assert (ratio > 0.5).all() and (ratio < 2.0).all(), (
        f"native vs ref decoded means diverge: native={mn}, ref={mr}")


def test_encode_native_accepts_user_sdr_base():
    """Caller-supplied SDR base path: encode_native skips its own
    peak-normalize and uses the user's SDR exactly. Confirms the
    'I want my own base supplied' contract — useful for tilescan
    callers that already have a tonemapped SDR."""
    hdr = _synthetic_hdr_rgb(64, 64)
    # Simple peak-normalize SDR base.
    peak = float(hdr.max())
    sdr = compute_sdr_base_u8(hdr, peak=peak)
    data = encode_native(
        hdr, sdr=sdr, quality=95, max_content_boost=peak,
    )
    assert is_uhdr(data) is True
    info = decode(data, want_hdr=True)
    assert info["hdr_fp16"].shape[:2] == hdr.shape[:2]


def test_encode_native_parallel_matches_serial():
    """parallel=True overlaps SDR JPEG encode + gain-map compute +
    gain-map JPEG encode via a ThreadPoolExecutor. Result must be
    byte-identical to the serial path (same kernels, same inputs)."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data_p = encode_native(hdr, parallel=True, quality=90)
    data_s = encode_native(hdr, parallel=False, quality=90)
    assert data_p == data_s


def test_encode_native_rejects_bad_shape():
    with pytest.raises(ValueError):
        encode_native(np.zeros((64, 64), dtype=np.float32))
    with pytest.raises(ValueError):
        encode_native(np.zeros((64, 64, 4), dtype=np.float32))


def test_compute_gain_map_and_sdr_helpers():
    """The standalone helpers (compute_sdr_base_u8 +
    compute_gain_map_u8) should produce shape-correct uint8 outputs
    plus a metadata dict with all libuhdr-required keys."""
    hdr = _synthetic_hdr_rgb(32, 32)
    sdr = compute_sdr_base_u8(hdr)
    assert sdr.dtype == np.uint8
    assert sdr.shape == hdr.shape
    gain, meta = compute_gain_map_u8(hdr, sdr)
    assert gain.dtype == np.uint8
    assert gain.shape == hdr.shape
    for key in ("max_content_boost", "min_content_boost", "gamma",
                "hdr_capacity_min", "hdr_capacity_max"):
        assert key in meta


# ---------------------------------------------------------------------------
# decode_native: fused-Cython fast-path decoder
# ---------------------------------------------------------------------------


def test_decode_native_returns_expected_keys():
    """decode_native returns hdr + sdr_u8 + gainmap_u8 + metadata."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data = encode_native(hdr, quality=95)
    info = decode_native(data)
    assert set(info.keys()) >= {
        "hdr", "sdr_u8", "gainmap_u8", "gainmap_metadata",
        "width", "height",
    }
    assert info["hdr"].dtype == np.float16
    assert info["hdr"].shape == (64, 64, 3)
    assert info["sdr_u8"].dtype == np.uint8
    assert info["gainmap_u8"].dtype == np.uint8
    assert info["width"] == 64 and info["height"] == 64


def test_decode_native_matches_libuhdr_at_sdr_boost():
    """At display_boost=1.0 decode_native should match libuhdr's
    default decode (also display_boost=1.0) within JPEG-q95 + 8-bit
    gain-quantisation noise."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data = encode_native(hdr, quality=95)

    lib = decode(data, want_hdr=True)["hdr_fp16"][..., :3].astype(np.float32)
    nat = decode_native(data, dtype=np.float32, display_boost=1.0)["hdr"]

    # Per-channel mean diff should be small (~JPEG q95 noise).
    diff = np.abs(lib - nat)
    assert diff.mean() < 0.05, (
        f"native vs libuhdr decode mean diff {diff.mean():.4f} > 0.05"
    )
    # Channel-wise means should track each other within ~25%.
    ratio = nat.mean(axis=(0, 1)) / np.maximum(lib.mean(axis=(0, 1)), 1e-6)
    assert (ratio > 0.75).all() and (ratio < 1.25).all(), (
        f"per-channel ratio out of bounds: native={nat.mean(axis=(0,1))}, "
        f"lib={lib.mean(axis=(0,1))}"
    )


def test_decode_native_full_boost_is_brighter_than_sdr():
    """display_boost=hdr_capacity_max should produce a brighter HDR
    raster than display_boost=1.0 (the SDR-equivalent)."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data = encode_native(hdr, quality=95)

    sdr_view = decode_native(data, dtype=np.float32, display_boost=1.0)["hdr"]
    hdr_view = decode_native(data, dtype=np.float32)["hdr"]  # default = full
    # Full-HDR view should have strictly more energy than SDR view.
    assert hdr_view.mean() > sdr_view.mean() * 1.5


def test_decode_native_parallel_matches_serial():
    """parallel=True kicks the two JPEG decodes into threads. Result
    must be bit-identical to the serial path (same kernel inputs)."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data = encode_native(hdr, quality=95)
    a = decode_native(data, parallel=True)["hdr"]
    b = decode_native(data, parallel=False)["hdr"]
    np.testing.assert_array_equal(a, b)


def test_decode_native_dtype_fp32_skips_cast():
    hdr = _synthetic_hdr_rgb(32, 32)
    data = encode_native(hdr, quality=95)
    info = decode_native(data, dtype=np.float32)
    assert info["hdr"].dtype == np.float32


# ---------------------------------------------------------------------------
# encode_to: streaming write to file-like
# ---------------------------------------------------------------------------


def test_encode_to_writes_to_bytesio():
    """encode_to streams Ultra-HDR bytes to a file-like via a zero-copy
    memoryview over libuhdr's internal buffer. Output must match the
    direct-bytes encode_native path byte-for-byte."""
    hdr = _synthetic_hdr_rgb(64, 64)
    direct = encode_native(hdr, quality=95)
    buf = io.BytesIO()
    rv = encode_to(buf, hdr, quality=95)
    assert rv is None  # streaming path returns None, not a byte count
    assert buf.getvalue() == direct
    assert is_uhdr(buf.getvalue()) is True


def test_encode_native_out_streams_no_bytes_intermediate():
    """encode_native(out=fp) is the streaming variant. fp.write()
    sees the same bytes as the default (out=None) path; the function
    returns None when out is given."""
    hdr = _synthetic_hdr_rgb(64, 64)
    direct = encode_native(hdr, quality=95)
    buf = io.BytesIO()
    rv = encode_native(hdr, quality=95, out=buf)
    assert rv is None
    assert buf.getvalue() == direct


def test_encode_native_gain_quality_and_scale_round_trip():
    """gain_quality=70 + gain_scale=2 produces a smaller container
    that's still valid Ultra-HDR + decodes to the same shape."""
    hdr = _synthetic_hdr_rgb(64, 64)
    full = encode_native(hdr, quality=95)
    cut = encode_native(hdr, quality=95, gain_quality=70, gain_scale=2)
    assert is_uhdr(cut) is True
    # Smaller gain map → smaller file in practice; on tiny test
    # rasters the JPEG header dominates so this can flip; just
    # verify the container parses.
    info = probe(cut)
    assert info["gainmap_width"] == 32  # 64 / scale=2
    assert info["gainmap_height"] == 32
    full_info = probe(full)
    assert full_info["gainmap_width"] == 64


def test_decode_native_dct_scale_int():
    """decode_native(scale=N) routes through opencodecs._jpeg with
    libjpeg-turbo's DCT-domain scale knob. Output shape is the source
    dimensions divided by N (with TJSCALED rounding)."""
    hdr = _synthetic_hdr_rgb(64, 96)
    data = encode_native(hdr, quality=95)
    for scale in (2, 4, 8):
        info = decode_native(data, scale=scale)
        H, W = info["hdr"].shape[:2]
        # TJSCALED(d, (1, scale)) = (d + scale - 1) // scale.
        assert H == (64 + scale - 1) // scale, (
            f"scale={scale}: expected H={(64 + scale - 1) // scale}, got {H}")
        assert W == (96 + scale - 1) // scale


def test_decode_native_dct_scale_fraction():
    """Pass an arbitrary supported (num, denom) ratio."""
    hdr = _synthetic_hdr_rgb(80, 80)
    data = encode_native(hdr, quality=95)
    info = decode_native(data, scale=(3, 8))
    H, W = info["hdr"].shape[:2]
    # TJSCALED(80, (3, 8)) = (80*3 + 8 - 1) // 8 = 30
    assert H == 30 and W == 30


def test_decode_native_dct_scale_preserves_hdr_signal():
    """A scaled-down decode should still recover roughly the same
    HDR mean (within the box-average-equivalent tolerance)."""
    hdr = _synthetic_hdr_rgb(96, 96)
    data = encode_native(hdr, quality=95)
    full = decode_native(data, dtype=np.float32, display_boost=1.0)["hdr"]
    half = decode_native(data, dtype=np.float32, display_boost=1.0, scale=2)["hdr"]
    # Half-res mean should track the full-res mean within ~15%.
    fm = full.mean(axis=(0, 1))
    hm = half.mean(axis=(0, 1))
    ratio = hm / np.maximum(fm, 1e-6)
    assert (ratio > 0.85).all() and (ratio < 1.15).all(), (
        f"scaled mean differs too far: full={fm}, half={hm}")



# A lossless SDR base is a SOF3 JPEG, and libultrahdr parses the base
# with whatever libjpeg it was linked against. Homebrew's libjpeg-turbo
# reads SOF3; the conda-forge build our Linux CI links does not, and
# fails with libjpeg's own "Unsupported JPEG process: SOF type 0xc3".
#
# So this is a property of the linked library, not of the platform, and
# it is detected by trying it rather than by checking sys.platform --
# the same Linux box with a different libjpeg gives the other answer.
_LOSSLESS_PROBE: bool | None = None


def _lossless_base_supported() -> bool:
    global _LOSSLESS_PROBE
    if _LOSSLESS_PROBE is None:
        try:
            encode_native(_synthetic_hdr_rgb(16, 16), lossless=True)
            _LOSSLESS_PROBE = True
        except UhdrError:
            _LOSSLESS_PROBE = False
        except Exception:                                # noqa: BLE001
            _LOSSLESS_PROBE = True      # a different failure is a real one
    return _LOSSLESS_PROBE


needs_lossless_jpeg = pytest.mark.skipif(
    not _lossless_base_supported(),
    reason="libultrahdr is linked against a libjpeg that cannot parse a "
           "lossless (SOF3) JPEG, so lossless=True is unavailable here")


@needs_lossless_jpeg
def test_encode_native_lossless_sdr_base():
    """encode_native(lossless=True) produces a valid Ultra-HDR
    container whose SDR base is a libjpeg-turbo lossless JPEG.
    Caveats: libuhdr's wrapped decode() rejects, our decode_native
    works. File is larger than the baseline."""
    hdr = _synthetic_hdr_rgb(64, 64)
    baseline = encode_native(hdr, quality=95)
    lossless = encode_native(hdr, lossless=True)
    assert is_uhdr(lossless) is True
    assert len(lossless) > len(baseline), (
        f"expected lossless > baseline; got {len(lossless)} vs {len(baseline)}")
    # decode_native works on both
    nat = decode_native(lossless, dtype=np.float32, display_boost=1.0)
    assert nat["hdr"].shape[:2] == hdr.shape[:2]
    assert np.isfinite(nat["hdr"]).all()
    # libuhdr's wrapped decode rejects on the colorspace check
    with pytest.raises((UhdrError, RuntimeError)):
        decode(lossless, want_hdr=True)


@needs_lossless_jpeg
def test_encode_native_lossless_is_bit_exact_in_sdr_layer():
    """The defining property of ``lossless=True``: the SDR base JPEG
    inside the assembled Ultra-HDR container decodes byte-for-byte
    back to the caller-supplied SDR raster. With ``lossless=False``
    (baseline DCT q95) the same round-trip has noticeable
    quantization noise."""
    import imagecodecs
    rng = np.random.default_rng(0)
    # Use the user-supplied ``sdr=`` path so encode_native skips the
    # sRGB OETF / peak-normalize step that compute_sdr_base_u8 would
    # apply on bare-float input.
    sdr_in = rng.integers(0, 255, size=(32, 48, 3), dtype=np.uint8)
    hdr_dummy = sdr_in.astype(np.float32)  # shape sentinel; ignored
    data = encode_native(
        hdr_dummy, sdr=sdr_in, quality=95, lossless=True,
        max_content_boost=2.0,
    )
    info = decode_native(data, dtype=np.float32, display_boost=1.0)
    # Pull the round-tripped SDR base raster back out:
    assert info["sdr_u8"].shape[:2] == sdr_in.shape[:2]
    if info["sdr_u8"].shape[-1] == 4:
        rec = info["sdr_u8"][..., :3]
    else:
        rec = info["sdr_u8"]
    # Bit-exact: not a single LSB difference between caller bytes
    # and the bytes that came back through the lossless JPEG layer.
    assert np.array_equal(rec, sdr_in), (
        f"lossless SDR not bit-exact: max diff "
        f"{np.abs(rec.astype(np.int16) - sdr_in.astype(np.int16)).max()}")


def test_encode_native_sdr_subsampling_controls_base_layer_size():
    """``sdr_subsampling='444'`` keeps full chroma in the SDR base
    layer, producing a larger file than the default ``'420'``."""
    hdr = _synthetic_hdr_rgb(96, 96)
    blob_420 = encode_native(hdr, quality=95, sdr_subsampling="420")
    blob_444 = encode_native(hdr, quality=95, sdr_subsampling="444")
    assert is_uhdr(blob_420) is True
    assert is_uhdr(blob_444) is True
    assert len(blob_444) > len(blob_420), (
        f"expected 4:4:4 to produce a larger SDR base than 4:2:0; "
        f"got {len(blob_444)} vs {len(blob_420)}")
    # Both still HDR-decode through decode_native.
    for blob in (blob_420, blob_444):
        info = decode_native(blob, dtype=np.float32)
        assert info["hdr"].shape[:2] == hdr.shape[:2]


def test_encode_native_no_thumbnail_by_default():
    """encode_native with no thumbnail_size produces a file with no
    embedded thumbnail; read_thumbnail returns None."""
    hdr = _synthetic_hdr_rgb(64, 64)
    data = encode_native(hdr, quality=95)
    assert read_thumbnail_bytes(data) is None
    assert read_thumbnail(data) is None


def test_encode_native_embeds_exif_thumbnail():
    """thumbnail_size=N embeds an EXIF APP1 thumbnail readable by
    read_thumbnail; the thumbnail JPEG decodes to ≤ N px per side."""
    hdr = _synthetic_hdr_rgb(96, 96)
    data = encode_native(hdr, quality=95, thumbnail_size=32)
    raw = read_thumbnail_bytes(data)
    assert raw is not None
    assert raw.startswith(b"\xff\xd8")  # SOI marker — it's a JPEG
    decoded = read_thumbnail(data)
    assert decoded.dtype == np.uint8
    assert decoded.shape[2] == 3
    assert max(decoded.shape[:2]) <= 32, (
        f"thumbnail larger than requested: {decoded.shape}")


def test_thumbnail_doesnt_break_uhdr_or_decode():
    """Embedding a thumbnail must leave the file a conforming Ultra-
    HDR JPEG: is_uhdr, probe, libuhdr's decode, and our decode_native
    must all succeed unchanged."""
    hdr = _synthetic_hdr_rgb(96, 96)
    plain = encode_native(hdr, quality=95)
    with_thumb = encode_native(hdr, quality=95, thumbnail_size=32)

    assert is_uhdr(with_thumb) is True
    p = probe(with_thumb)
    assert p["width"] == probe(plain)["width"]
    assert p["height"] == probe(plain)["height"]
    # libuhdr's wrapped decode shouldn't get confused by the extra
    # APP1 segment in front of the main image
    info_lib = decode(with_thumb, want_hdr=True)
    assert info_lib["hdr_fp16"].shape[:2] == (96, 96)
    # Our decode_native too
    info_native = decode_native(with_thumb, dtype=np.float32, display_boost=1.0)
    assert info_native["hdr"].shape[:2] == (96, 96)


def test_thumbnail_size_caps_largest_axis():
    """For a non-square image, thumbnail_size bounds the *larger* axis;
    the other shrinks by the same integer stride."""
    rng = np.random.default_rng(0)
    hdr = rng.random((48, 96, 3), dtype=np.float32) * 4.0
    data = encode_native(hdr, quality=95, thumbnail_size=24)
    t = read_thumbnail(data)
    # stride = ceil(96/24) = 4; so dims become 96//4=24, 48//4=12
    assert t.shape == (12, 24, 3), f"got {t.shape}"


def test_thumbnail_smaller_than_source_no_upscale():
    """If the source SDR is already smaller than thumbnail_size, the
    thumbnail is the source's own size — never upscaled."""
    hdr = _synthetic_hdr_rgb(32, 32)
    data = encode_native(hdr, quality=95, thumbnail_size=256)
    t = read_thumbnail(data)
    assert t.shape == (32, 32, 3)


def test_thumbnail_is_uhdr_by_default():
    """thumbnail_size=N defaults to an UHDR-formatted thumbnail (SDR
    base + gain map + MPF) so HDR-aware viewers preserve peak brightness
    when previewing. SDR-only readers still see a plain JPEG."""
    hdr = _synthetic_hdr_rgb(96, 96)
    data = encode_native(hdr, quality=95, thumbnail_size=32)
    raw = read_thumbnail_bytes(data)
    assert raw is not None
    assert is_uhdr(raw) is True, (
        "default thumbnail should be Ultra-HDR; got plain JPEG")


def test_thumbnail_hdr_false_emits_plain_sdr_jpeg():
    """thumbnail_hdr=False opts out of UHDR thumbnails and embeds a
    plain SDR JPEG (smaller; legacy-EXIF-reader-safe)."""
    hdr = _synthetic_hdr_rgb(96, 96)
    data_hdr = encode_native(hdr, quality=95, thumbnail_size=32,
                             thumbnail_hdr=True)
    data_sdr = encode_native(hdr, quality=95, thumbnail_size=32,
                             thumbnail_hdr=False)
    raw_hdr = read_thumbnail_bytes(data_hdr)
    raw_sdr = read_thumbnail_bytes(data_sdr)
    assert is_uhdr(raw_hdr) is True
    assert is_uhdr(raw_sdr) is False
    # Plain-SDR thumbnail should be strictly smaller than the UHDR one
    # (saves the gain map + MPF overhead). Hold this loose — at 32 px
    # the difference can be small.
    assert len(raw_sdr) < len(raw_hdr)


def test_read_thumbnail_hdr_preserves_main_peak():
    """For the default (UHDR) thumbnail, read_thumbnail_hdr decodes the
    thumbnail through the gain-map pipeline → peak HDR brightness
    matches the main image's, not the SDR-clipped 1.0 ceiling."""
    # Use a synthetic HDR with a clear bright dye-spot-like peak so
    # the stride-decimation has something to preserve.
    rng = np.random.default_rng(0)
    hdr = (rng.random((128, 128, 3), dtype=np.float32) * 0.5
           + 0.1).astype(np.float32)
    # Plant a bright peak at a centered-stride grid point so stride
    # decimation lands on it (stride = 128//32 = 4; off=2; (2, 2) is on
    # the grid → thumb pixel (0, 0) samples there).
    hdr[2, 2, :] = 6.0
    data = encode_native(hdr, quality=95,
                         thumbnail_size=32, thumbnail_quality=95)

    th_hdr = read_thumbnail_hdr(data)
    assert th_hdr.dtype == np.float32
    assert th_hdr.shape == (32, 32, 3)
    # The thumbnail's peak should be well above SDR-white (1.0), close
    # to the planted 6.0 (some loss from JPEG quantization + gain-map
    # round-trip is expected; require ≥ 3× SDR-white).
    assert th_hdr.max() > 3.0, (
        f"thumbnail HDR peak {th_hdr.max():.3f} suggests SDR-clipped path")


def test_read_thumbnail_hdr_fallback_on_sdr_thumb():
    """When thumbnail_hdr=False, read_thumbnail_hdr returns fp32 [0, 1]
    (no HDR boost available, scaled from the SDR uint8). Caller can
    detect via peak ≤ 1.0."""
    hdr = _synthetic_hdr_rgb(96, 96)
    data = encode_native(hdr, quality=95, thumbnail_size=32,
                         thumbnail_hdr=False)
    th = read_thumbnail_hdr(data)
    assert th.dtype == np.float32
    assert th.max() <= 1.0


def test_read_thumbnail_hdr_none_when_no_thumbnail():
    hdr = _synthetic_hdr_rgb(32, 32)
    data = encode_native(hdr, quality=95)
    assert read_thumbnail_hdr(data) is None


def test_probe_returns_dimensions_and_metadata():
    """probe(data) parses just the MPF metadata without any pixel
    decode, ~100× faster than decode() for indexing workloads."""
    hdr = _synthetic_hdr_rgb(64, 96)
    data = encode_native(hdr, quality=95)
    info = probe(data)
    assert info["width"] == 96  # X / W axis
    assert info["height"] == 64
    assert info["gainmap_width"] == 96
    assert info["gainmap_height"] == 64
    meta = info["gainmap_metadata"]
    assert "max_content_boost" in meta
    assert "hdr_capacity_max" in meta


# ---------------------------------------------------------------------------
# UhdrError surfaces on malformed input
# ---------------------------------------------------------------------------


def test_decode_garbage_raises():
    with pytest.raises((UhdrError, ValueError, RuntimeError)):
        decode(b"not an ultra-hdr blob, not even close")
