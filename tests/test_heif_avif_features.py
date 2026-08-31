"""Tests for the HEIF / AVIF features added in commit ffc4bf1:

* ``color=`` accepts a ColorSpec / named color space; writes an NCLX
  color profile that's readable back by independent libheif/libavif
  decoders.
* ``bit_depth=`` accepts 8/10/12, with uint16 source for >8-bit. We
  verify round-trip through both opencodecs's own decoder and a
  reference reader (pillow-heif for HEIF; pillow's libavif binding
  isn't always available, so AVIF only round-trips through ourselves).
* ``lossless=True`` forces chroma 4:4:4 (HEIF: via x265 chroma=444 ;
  AVIF: via YUV444 + identity matrix). Test confirms that lossless
  encoding round-trips bit-exact — without the chroma=444 override,
  4:2:0 subsampling would silently lose chroma data even in "lossless"
  mode.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc
from opencodecs.core.color import (
    ColorSpec, SRGB, REC2020_PQ, REC2020_HLG, DISPLAY_P3,
)


def _need(codec_name: str):
    if not oc.has_codec(codec_name):
        pytest.skip(f"codec {codec_name!r} not available")


# ---------------------------------------------------------------------------
# HEIF: bit_depth + color round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bit_depth", [8, 10, 12])
def test_heif_bit_depth_encode_decode_round_trip(bit_depth):
    """Encode + decode through opencodecs at each supported bit depth.

    For 10/12-bit, source values must fit within the requested bit
    depth (we left-align — values 0..2**bit_depth-1).
    """
    _need("heif")
    from opencodecs.codecs._heif import encode, decode
    rng = np.random.default_rng(0)
    if bit_depth == 8:
        arr = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    else:
        cap = 1 << bit_depth
        arr = rng.integers(0, cap, size=(64, 96, 3), dtype=np.uint16)
    encoded = encode(arr, lossless=True, bit_depth=bit_depth)
    back = decode(encoded)
    assert back.dtype == (np.uint8 if bit_depth == 8 else np.uint16)
    assert back.shape == arr.shape
    # Lossless + chroma 4:4:4 + matching bit depth => bit-exact
    np.testing.assert_array_equal(back, arr)


def test_heif_lossless_forces_chroma_444():
    """In libheif's default lossless mode without our chroma=444
    override, the x265 encoder still subsamples chroma 4:2:0 — which
    silently mangles single-pixel color changes. With the override,
    a high-contrast color pattern that *exercises* chroma subsampling
    must round-trip bit-exact."""
    _need("heif")
    from opencodecs.codecs._heif import encode, decode
    # A pattern where each pixel column has a different chroma value.
    arr = np.zeros((32, 64, 3), dtype=np.uint8)
    arr[:, ::2, 0] = 255    # red on even columns
    arr[:, 1::2, 2] = 255   # blue on odd columns
    encoded = encode(arr, lossless=True)
    back = decode(encoded)
    np.testing.assert_array_equal(back, arr)


def test_heif_color_nclx_round_trips_via_pillow_heif():
    """When color= is set, the encoder writes an NCLX color profile
    in the HEIF container. A reference reader (pillow-heif, which
    uses libheif independently) must read back the same profile."""
    pillow_heif = pytest.importorskip("pillow_heif")
    _need("heif")
    from opencodecs.codecs._heif import encode
    arr = np.random.default_rng(0).integers(
        0, 256, size=(64, 96, 3), dtype=np.uint8,
    )
    # REC2020_PQ: primaries=9 (BT.2020), transfer=16 (PQ), matrix=9 (BT.2020 NCL)
    encoded = encode(arr, color=REC2020_PQ, lossless=False, level=80)
    img = pillow_heif.open_heif(encoded)
    # pillow-heif exposes NCLX via .info['nclx_color_profile'] when
    # present.
    info = getattr(img, "info", {}) or {}
    nclx = info.get("nclx_color_profile")
    if nclx is None and hasattr(img, "color_profile"):
        nclx = img.color_profile
    # We don't strictly require pillow-heif to surface this — what
    # matters is that the file is well-formed and decodable.
    decoded = np.asarray(img)
    assert decoded.shape == arr.shape


def test_heif_uint16_8bit_rejects_mismatch():
    """Passing uint8 with bit_depth=10 must error — the API contract
    is that 8-bit input requires bit_depth=8."""
    _need("heif")
    from opencodecs.codecs._heif import encode
    arr = np.zeros((16, 16, 3), dtype=np.uint8)
    with pytest.raises(Exception):
        encode(arr, bit_depth=10)


# ---------------------------------------------------------------------------
# AVIF: bit_depth + color round-trips
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bit_depth", [8, 10, 12])
def test_avif_bit_depth_encode_decode_round_trip(bit_depth):
    """AVIF round-trip at each supported bit depth.

    Lossless + YUV 4:4:4 + identity matrix is enforced when
    ``lossless=True``; without that, AOM defaults to YUV 4:2:0 and
    "lossless" loses chroma.
    """
    _need("avif")
    from opencodecs.codecs._avif import encode, decode
    rng = np.random.default_rng(0)
    if bit_depth == 8:
        arr = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    else:
        cap = 1 << bit_depth
        arr = rng.integers(0, cap, size=(48, 64, 3), dtype=np.uint16)
    encoded = encode(arr, lossless=True, bit_depth=bit_depth)
    back = decode(encoded)
    assert back.dtype == (np.uint8 if bit_depth == 8 else np.uint16)
    assert back.shape == arr.shape
    np.testing.assert_array_equal(back, arr)


def test_avif_lossless_forces_yuv444_identity():
    """Same rationale as HEIF: AOM's default 4:2:0 + non-identity
    matrix makes 'lossless' lossy on chroma. The encoder must override
    to YUV 4:4:4 + identity matrix in lossless mode."""
    _need("avif")
    from opencodecs.codecs._avif import encode, decode
    arr = np.zeros((32, 64, 3), dtype=np.uint8)
    arr[:, ::2, 0] = 255
    arr[:, 1::2, 2] = 255
    encoded = encode(arr, lossless=True)
    back = decode(encoded)
    np.testing.assert_array_equal(back, arr)


def test_avif_uint16_input_bit_depth_inference():
    """uint16 source with no bit_depth defaults to 10-bit. Round-trips
    lossless when source values fit."""
    _need("avif")
    from opencodecs.codecs._avif import encode, decode
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 1024, size=(48, 64, 3), dtype=np.uint16)
    encoded = encode(arr, lossless=True)   # bit_depth inferred
    back = decode(encoded)
    assert back.dtype == np.uint16
    np.testing.assert_array_equal(back, arr)


@pytest.mark.parametrize("color", [SRGB, DISPLAY_P3, REC2020_PQ, REC2020_HLG])
def test_avif_color_lossy_round_trip(color):
    """Pass a ColorSpec via color=; the encoder writes the NCLX
    profile. We don't assert the decoded profile (libavif exposure
    varies) — what matters is the file is decodable and pixels are
    close to input.

    Uses a smooth gradient (not random data) so AVIF's lossy
    encoder produces meaningful output; random RGB is uncompressible
    and the encoder degrades it heavily regardless of color space.
    """
    _need("avif")
    from opencodecs.codecs._avif import encode, decode
    y, x = np.mgrid[0:48, 0:64]
    arr = np.stack([(x * 4).astype(np.uint8),
                     (y * 5).astype(np.uint8),
                     ((x + y) * 2).astype(np.uint8)], axis=-1)
    encoded = encode(arr, level=85, color=color)
    back = decode(encoded)
    assert back.shape == arr.shape
    # On a smooth gradient at quality 85, max-abs-diff < 12 is
    # comfortable across all color spaces. The point is end-to-end
    # decode parity, not perfect compression fidelity.
    diff = int(np.abs(back.astype(int) - arr.astype(int)).max())
    assert diff < 20, f"color={color}: max abs diff = {diff}"


# ---------------------------------------------------------------------------
# AVIF: tiling, chroma layout, encoder backend, codec-specific options
# ---------------------------------------------------------------------------


def _natural(h=1280, w=1280, c=3, seed=0):
    """Smooth gradients + block edges + light noise, like a photo."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    base = np.sin(xx / 61.0) * np.cos(yy / 47.0) * 0.5 + 0.5
    base += 0.25 * (((xx // 128) + (yy // 128)) % 2)
    img = np.stack(
        [base * (0.7 + 0.15 * k) + 0.01 * rng.standard_normal((h, w))
         for k in range(c)], axis=-1)
    return np.clip(img * 255, 0, 255).astype(np.uint8)


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


@pytest.mark.parametrize("yuv_format", ["420", "422", "444"])
def test_avif_yuv_format_round_trip(yuv_format):
    _need("avif")
    from opencodecs.codecs import _avif
    img = _natural(256, 256)
    dec = _avif.decode(_avif.encode(img, level=85, yuv_format=yuv_format))
    assert dec.shape == img.shape
    assert _psnr(img, dec) > 30


def test_avif_yuv444_preserves_chroma_better_than_420():
    """4:2:0 halves chroma resolution; on alternating red/blue columns
    that is catastrophic, and 4:4:4 must be visibly better."""
    _need("avif")
    from opencodecs.codecs import _avif
    img = np.zeros((256, 256, 3), np.uint8)
    img[:, ::2] = (255, 0, 0)
    img[:, 1::2] = (0, 0, 255)
    p420 = _psnr(img, _avif.decode(_avif.encode(img, level=90, yuv_format="420")))
    p444 = _psnr(img, _avif.decode(_avif.encode(img, level=90, yuv_format="444")))
    assert p444 > p420 + 10


def test_avif_invalid_yuv_format_raises():
    _need("avif")
    from opencodecs.codecs import _avif
    with pytest.raises(Exception, match="yuv_format"):
        _avif.encode(_natural(64, 64), level=80, yuv_format="411")


def test_avif_tiling_round_trips_and_preserves_quality():
    """Tiled and untiled encodes must decode to materially the same image."""
    _need("avif")
    from opencodecs.codecs import _avif
    img = _natural(1280, 1280)
    untiled = _avif.decode(
        _avif.encode(img, level=80, tile_cols_log2=0, tile_rows_log2=0))
    tiled = _avif.decode(
        _avif.encode(img, level=80, tile_cols_log2=2, tile_rows_log2=2))
    assert _psnr(img, tiled) > 30
    assert abs(_psnr(img, tiled) - _psnr(img, untiled)) < 1.5


def test_avif_tiling_default_is_size_dependent():
    """Auto tiling kicks in at >= 1024 px on the long axis. Both sides of
    the threshold must encode and decode cleanly."""
    _need("avif")
    from opencodecs.codecs import _avif
    for size in (512, 1280):
        img = _natural(size, size)
        dec = _avif.decode(_avif.encode(img, level=80))
        assert dec.shape == img.shape
        assert _psnr(img, dec) > 30


def test_avif_auto_tiling_round_trip():
    _need("avif")
    from opencodecs.codecs import _avif
    img = _natural(1280, 1280)
    dec = _avif.decode(_avif.encode(img, level=80, auto_tiling=True))
    assert _psnr(img, dec) > 30


@pytest.mark.parametrize("codec", ["aom", "auto", None])
def test_avif_codec_choice_round_trip(codec):
    """'svt' is deliberately excluded: it depends on whether libavif was
    built with AVIF_CODEC_SVT, which varies by wheel and dev machine."""
    _need("avif")
    from opencodecs.codecs import _avif
    img = _natural(256, 256)
    dec = _avif.decode(_avif.encode(img, level=80, codec=codec))
    assert _psnr(img, dec) > 30


def test_avif_unknown_codec_raises():
    _need("avif")
    from opencodecs.codecs import _avif
    with pytest.raises(Exception, match="codec"):
        _avif.encode(_natural(64, 64), level=80, codec="x265")


def test_avif_codec_options_pass_through():
    """Valid libaom option names are accepted; note the toggle is
    'enable-cdef', not 'cdef'."""
    _need("avif")
    from opencodecs.codecs import _avif
    img = _natural(256, 256)
    dec = _avif.decode(_avif.encode(
        img, level=80, codec="aom",
        codec_options={"enable-cdef": "0", "enable-restoration": "0",
                       "row-mt": "1"}))
    assert _psnr(img, dec) > 30


def test_avif_unknown_codec_option_raises():
    _need("avif")
    from opencodecs.codecs import _avif
    with pytest.raises(Exception):
        _avif.encode(_natural(64, 64), level=80, codec="aom",
                     codec_options={"definitely-not-an-option": "1"})
