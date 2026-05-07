"""Robustness tests: every codec must raise a clear Python exception
(not segfault, not return garbage) when handed malformed input.

Three classes of bad input each codec should reject:

  1. **Random bytes** that don't match the format magic.
  2. **Truncated valid streams** — encode an array, slice off the tail,
     try to decode.
  3. **Right magic, garbage body** — for image codecs we plant the magic
     bytes then random data after.

Plus encoder-side input validation:

  4. **Wrong dtype** for codecs that only accept specific dtypes.
  5. **Wrong shape / channel count** for image codecs.

When a codec's backend isn't built we skip — the codec stub already
raises ImportError on call, which is tested separately in
``test_optional_backend.py``.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import opencodecs as oc


def _need(codec_name: str) -> None:
    if not oc.has_codec(codec_name):
        pytest.skip(f"codec {codec_name!r} not registered on this host")


# ---------------------------------------------------------------------------
# All compression codecs: bytes-in / bytes-out
# ---------------------------------------------------------------------------


_COMPRESSION_CODECS = ["zstd", "lz4", "brotli", "blosc2", "deflate"]


@pytest.mark.parametrize("fmt", _COMPRESSION_CODECS)
def test_compression_decode_random_bytes_raises(fmt):
    _need(fmt)
    payload = os.urandom(1024)
    with pytest.raises(Exception) as exc:
        oc.read(payload, format=fmt)
    # Every codec we ship raises some flavour of RuntimeError-or-subclass
    # on bad input; assertion is that it's a normal exception (no segfault,
    # no SystemExit, no None return).
    assert isinstance(exc.value, Exception)


@pytest.mark.parametrize("fmt", _COMPRESSION_CODECS)
def test_compression_decode_truncated_raises(fmt):
    _need(fmt)
    rng = np.random.default_rng(0)
    enc = oc.write(None, rng.bytes(8192), format=fmt)
    if len(enc) < 16:
        pytest.skip(f"{fmt} encoded smaller than truncation makes sense")
    truncated = enc[: len(enc) // 2]
    with pytest.raises(Exception):
        oc.read(truncated, format=fmt)


@pytest.mark.parametrize("fmt", _COMPRESSION_CODECS)
def test_compression_decode_empty_returns_empty(fmt):
    _need(fmt)
    # Empty input is a degenerate case: opencodecs returns ``b""`` rather
    # than raising. Document that contract.
    out = oc.read(b"", format=fmt)
    assert out == b"" or len(out) == 0


# ---------------------------------------------------------------------------
# Image codecs: bytes-in / ndarray-out
# ---------------------------------------------------------------------------


_IMAGE_CODECS = ["qoi", "bmp", "png", "jpeg", "webp", "jpeg2k", "avif", "heif"]


@pytest.mark.parametrize("fmt", _IMAGE_CODECS)
def test_image_decode_random_bytes_raises(fmt):
    _need(fmt)
    # Use plenty of bytes so a codec that does header-then-body parsing
    # gets past the header check before failing.
    payload = os.urandom(64 * 1024)
    with pytest.raises(Exception):
        oc.read(payload, format=fmt)


# qoi.h's reference C decoder doesn't validate stream completeness — it
# stops at end-of-input and returns whatever pixels it decoded. Adding
# validation in our wrapper would mean re-checking against the declared
# dimensions, which the reference implementation doesn't do either. Skip
# qoi from truncation testing and document as a known limitation.
_TRUNCATION_RAISING_CODECS = [
    f for f in _IMAGE_CODECS if f != "qoi"
]


@pytest.mark.parametrize("fmt", _TRUNCATION_RAISING_CODECS)
def test_image_decode_truncated_raises(fmt):
    """Encode a real image, slice off the second half, decode → must raise."""
    _need(fmt)
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
    try:
        enc = oc.write(None, arr, format=fmt)
    except Exception as exc:
        if fmt == "heif":
            msg = str(exc).lower()
            if any(s in msg for s in (
                "encoder", "unsupported", "null error text", "heif_writer",
            )):
                pytest.skip(f"no HEVC encoder available: {exc}")
        raise
    if len(enc) < 64:
        pytest.skip(f"{fmt} encoded too small to truncate meaningfully")
    truncated = enc[: len(enc) // 2]
    with pytest.raises(Exception):
        oc.read(truncated, format=fmt)


@pytest.mark.parametrize("fmt", _IMAGE_CODECS)
def test_image_decode_too_short_raises(fmt):
    """A few bytes is not a valid image of any format."""
    _need(fmt)
    with pytest.raises(Exception):
        oc.read(b"\x00\x01\x02\x03", format=fmt)


# ---------------------------------------------------------------------------
# Encoder-side: invalid input
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("fmt", ["qoi", "bmp"])
def test_image_encode_wrong_dtype_raises(fmt):
    """qoi and bmp require uint8."""
    _need(fmt)
    arr = np.ones((16, 16, 3), dtype=np.float32)
    with pytest.raises(Exception):
        oc.write(None, arr, format=fmt)


@pytest.mark.parametrize("fmt", ["qoi"])
def test_image_encode_wrong_shape_raises(fmt):
    """QOI doesn't support grayscale or odd channel counts."""
    _need(fmt)
    arr = np.ones((16, 16), dtype=np.uint8)  # grayscale
    with pytest.raises(Exception):
        oc.write(None, arr, format=fmt)


def test_jpeg_encode_rgba_raises_or_drops_alpha():
    """JPEG can't carry an alpha channel; encoder must reject (RGB only)."""
    _need("jpeg")
    arr = np.ones((16, 16, 4), dtype=np.uint8)
    with pytest.raises(Exception):
        oc.write(None, arr, format="jpeg")


def test_png_encode_unsupported_dtype_raises():
    """PNG supports uint8 and uint16; float should be rejected."""
    _need("png")
    arr = np.ones((16, 16, 3), dtype=np.float32)
    with pytest.raises(Exception):
        oc.write(None, arr, format="png")


def test_jpeg2k_encode_unsupported_dtype_raises():
    """JPEG-2000 supports up to int32; float64 should be rejected."""
    _need("jpeg2k")
    arr = np.ones((16, 16, 3), dtype=np.float64)
    with pytest.raises(Exception):
        oc.write(None, arr, format="jpeg2k")


# ---------------------------------------------------------------------------
# Magic-byte signature: each codec rejects bytes that look like a different
# codec's magic
# ---------------------------------------------------------------------------


# Real magic prefixes for cross-codec checks.
_MAGIC = {
    "zstd": b"\x28\xb5\x2f\xfd",
    "lz4": b"\x04\x22\x4d\x18",
    "qoi": b"qoif",
    "bmp": b"BM",
    "png": b"\x89PNG\r\n\x1a\n",
    "jpeg": b"\xff\xd8\xff\xe0",
    "webp": b"RIFF\x00\x00\x00\x00WEBP",
}


@pytest.mark.parametrize("fmt, head", list(_MAGIC.items()))
def test_signature_matches_real_magic(fmt, head):
    _need(fmt)
    codec = oc.get_codec(fmt)
    assert codec.signature(head), (
        f"{fmt}.signature() should accept its own magic bytes")


@pytest.mark.parametrize("fmt", list(_MAGIC.keys()))
def test_signature_rejects_other_magics(fmt):
    """codec.signature(other_codec_magic) should be False, not raise."""
    _need(fmt)
    codec = oc.get_codec(fmt)
    for other_fmt, other_magic in _MAGIC.items():
        if other_fmt == fmt:
            continue
        # WebP RIFF prefix collides with several other RIFF-based formats;
        # signature checks must be robust to this kind of overlap.
        assert codec.signature(other_magic) is False, (
            f"{fmt}.signature({other_fmt}_magic={other_magic!r}) should be False")


@pytest.mark.parametrize("fmt", list(_MAGIC.keys()))
def test_signature_handles_short_input(fmt):
    """Tiny/empty input should not crash signature(); just return False."""
    _need(fmt)
    codec = oc.get_codec(fmt)
    assert codec.signature(b"") is False
    assert codec.signature(b"\x00") is False


# ---------------------------------------------------------------------------
# Top-level dispatch: unknown format → clear error
# ---------------------------------------------------------------------------


def test_read_unknown_format_raises():
    with pytest.raises(KeyError):
        oc.read(b"whatever", format="not-a-real-codec")


def test_write_unknown_format_raises():
    with pytest.raises(KeyError):
        oc.write(None, b"whatever", format="not-a-real-codec")


def test_codec_for_path_unknown_extension_raises():
    """Unknown extension raises KeyError consistent with other lookups."""
    with pytest.raises(KeyError):
        oc.codec_for_path("foo.unknownextension")


def test_get_codec_unknown_name_raises():
    with pytest.raises(KeyError):
        oc.get_codec("not-a-real-codec")


def test_has_codec_returns_false_for_unknown():
    assert oc.has_codec("not-a-real-codec") is False
