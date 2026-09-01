"""RGBE / Radiance HDR codec tests.

Validates against:

* Round-trip: encode + decode our own output, confirm pixel-equal
  within RGBE's quantization (~0.4% of channel magnitude).
* imagecodecs reference reader where available: imagecodecs.rgbe_decode
  reads our encoded bytes and produces matching pixels.
"""

from __future__ import annotations

import numpy as np
import pytest

from opencodecs.codecs._rgbe import encode, decode, RgbeError


def _make_hdr(shape=(48, 96, 3), seed=0):
    """Synthesize an HDR-ish RGB float32 array."""
    rng = np.random.default_rng(seed)
    base = rng.uniform(0.05, 8.0, size=shape).astype(np.float32)
    # Hotspots up to ~100 in the upper-left quadrant — exercises the
    # high-exponent path.
    base[: shape[0] // 4, : shape[1] // 4, :] *= 12
    # Some pixels exactly zero — exercises the all-zero exponent path.
    base[-1, -1] = 0.0
    return base


def test_rgbe_basic_round_trip():
    arr = _make_hdr()
    raw = encode(arr)
    back = decode(raw)
    # RGBE quantization error: 8-bit mantissa shared across channels,
    # so per-pixel relative error can be up to ~1% on channels far
    # smaller than the per-pixel max (the shared exponent quantizes
    # to the brightest channel). Compare relative max-diff per channel.
    pixel_max = np.maximum(np.abs(arr), np.abs(back)).max(axis=-1, keepdims=True)
    rel = np.abs(arr - back) / np.maximum(pixel_max, 1e-6)
    assert rel.max() < 0.01, f"max rel diff = {rel.max()}"


def test_rgbe_zero_pixel_preserved():
    arr = _make_hdr()
    arr[5, 7] = 0.0
    back = decode(encode(arr))
    assert np.all(back[5, 7] == 0.0)


def test_rgbe_high_dynamic_range():
    """Values across many stops of magnitude all encode correctly."""
    arr = np.zeros((4, 4, 3), dtype=np.float32)
    arr[0, 0] = 1e-3
    arr[0, 1] = 1.0
    arr[0, 2] = 1e3
    arr[0, 3] = 1e5
    back = decode(encode(arr))
    for col in range(4):
        rel = np.abs(arr[0, col] - back[0, col]).max() / max(arr[0, col].max(), 1e-6)
        assert rel < 0.01, f"col {col}: arr={arr[0, col]} back={back[0, col]}"


def test_rgbe_imagecodecs_can_read_our_output():
    """imagecodecs.rgbe_decode reads what we wrote (it's the
    well-tested reference reader)."""
    imagecodecs = pytest.importorskip("imagecodecs")
    if not hasattr(imagecodecs, "rgbe_decode"):
        pytest.skip("imagecodecs has no rgbe_decode in this build")
    arr = _make_hdr(shape=(32, 64, 3), seed=1)
    raw = encode(arr)
    via_ic = imagecodecs.rgbe_decode(raw)
    # imagecodecs's rgbe_decode also returns float32 (H, W, 3).
    assert via_ic.shape == arr.shape
    # Compare against our own decoder (both should give the same
    # RGBE-quantized pixels).
    via_us = decode(raw)
    np.testing.assert_array_equal(via_us, via_ic)


def test_rgbe_we_can_read_imagecodecs_output():
    """And the reverse: we decode what imagecodecs encoded."""
    imagecodecs = pytest.importorskip("imagecodecs")
    if not hasattr(imagecodecs, "rgbe_encode"):
        pytest.skip("imagecodecs has no rgbe_encode in this build")
    arr = _make_hdr(shape=(40, 80, 3), seed=2)
    raw = imagecodecs.rgbe_encode(arr)
    back = decode(raw)
    via_ic = imagecodecs.rgbe_decode(raw)
    np.testing.assert_array_equal(back, via_ic)


def test_rgbe_short_width_uncompressed_path():
    """Widths < 8 can't use new-style RLE (the RLE header would clash
    with a literal byte). We fall back to uncompressed; both encode
    and decode must handle that."""
    rng = np.random.default_rng(3)
    arr = rng.uniform(0.0, 5.0, size=(4, 4, 3)).astype(np.float32)
    back = decode(encode(arr))
    rel = np.abs(arr - back).max() / arr.max()
    assert rel < 0.01


def test_rgbe_invalid_header_raises():
    with pytest.raises(RgbeError):
        decode(b"not a hdr file")


def test_rgbe_exponent_table_matches_ldexp_bit_exactly():
    """The decoder scales each pixel by 2**(e-136) from a 256-entry table
    instead of calling ldexp() per pixel, which was a libm call in the
    hot loop and cost about 40% of decode time on a 4 MP image.

    The table has to be exactly equivalent, not just close: entries below
    2**-126 are denormal and a sloppy build could flush them to zero.
    Cross-checked against imagecodecs, which still uses the per-pixel
    ldexp path, over inputs spanning 40 orders of magnitude plus zeros
    (which exercise the e == 0 entry).
    """
    imagecodecs = pytest.importorskip("imagecodecs")
    rng = np.random.default_rng(0)
    for _ in range(8):
        img = (rng.standard_normal((64, 64, 3)).astype(np.float32)
               * float(10.0 ** int(rng.integers(-20, 20)))).astype(np.float32)
        img[rng.random((64, 64)) < 0.1] = 0.0
        encoded = encode(img)
        np.testing.assert_array_equal(
            decode(encoded),
            np.asarray(imagecodecs.rgbe_decode(encoded), dtype=np.float32))


def test_rgbe_encoder_matches_the_frexp_formula_exactly():
    """The encoder derives its scale from the float's exponent bits and a
    table instead of calling frexp() per pixel.

    frexp(v,&e)*256/v is 2**(8-e), and for a positive normal float e is
    the biased exponent minus 126, so the scale is a lookup on the
    exponent byte. This must reproduce the original formula bit for bit,
    including the e+128 exponent byte. Width 4 keeps the payload flat so
    the quadruples can be read directly rather than through the RLE.

    OpenImageIO still calls frexpf here, so this one is not shared with
    the reference implementation and gets its own check.
    """
    import math

    def reference(r, g, b):
        v = max(r, g, b)
        if v < 1e-32:
            return (0, 0, 0, 0)
        m, e = math.frexp(v)
        s = np.float32(np.float32(m) * np.float32(256.0) / np.float32(v))
        return (int(np.float32(r) * s) & 0xFF, int(np.float32(g) * s) & 0xFF,
                int(np.float32(b) * s) & 0xFF, (e + 128) & 0xFF)

    rng = np.random.default_rng(0)
    h, w = 200, 4
    img = np.abs(rng.standard_normal((h, w, 3)).astype(np.float32)) * np.float32(1e-3)
    for i in range(0, h, 7):                      # span the exponent range
        img[i] *= np.float32(10.0) ** int(rng.integers(-15, 15))
    img[rng.random((h, w)) < 0.15] = 0.0          # and the v < 1e-32 path

    encoded = bytes(encode(img))
    body = encoded[encoded.index(b"-Y"):]
    body = body[body.index(b"\n") + 1:]
    assert len(body) == h * w * 4
    got = np.frombuffer(body, np.uint8).reshape(h, w, 4)
    for y in range(h):
        for x in range(w):
            assert tuple(int(v) for v in got[y, x]) == reference(*img[y, x])
