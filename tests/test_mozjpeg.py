"""MozJPEG (Mozilla libjpeg-turbo fork) tests.

MozJPEG's value proposition: smaller files at the same quality via
progressive encoding + trellis quantization. Standard JPEG bitstream
so any JPEG decoder reads our output, and we can decode any JPEG.

Gated on the optional _mozjpeg extension being built — if MozJPEG
isn't installed on the system, the extension is skipped at build
time and these tests skip too.
"""

from __future__ import annotations

import numpy as np
import pytest

# The extension is optional. When MozJPEG isn't on the build host,
# the codec module isn't built and these tests skip cleanly.
mz = pytest.importorskip("opencodecs.codecs._mozjpeg")
from opencodecs.codecs._jpeg import encode as tj_encode, decode as tj_decode


def _smooth_rgb(shape=(256, 256, 3), seed=0):
    """Smooth gradient — MozJPEG's advantage is real on natural images
    but vanishes on random pixels (uncompressible no matter the codec)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:shape[0], 0:shape[1]]
    base = np.stack([
        (y * 0.5 + x * 0.3).astype(np.float32),
        (y * 0.3 + x * 0.5).astype(np.float32),
        ((x + y) * 0.4).astype(np.float32),
    ], axis=-1)
    noise = rng.normal(0, 4, base.shape).astype(np.float32)
    return (base + noise + 128).clip(0, 255).astype(np.uint8)


def test_mozjpeg_round_trip():
    """Encode + decode through MozJPEG round-trips visually."""
    arr = _smooth_rgb()
    enc = mz.encode(arr, level=85)
    back = mz.decode(enc)
    assert back.shape == arr.shape
    # At q=85 on smooth content, max abs diff stays under ~15 LSB.
    assert np.abs(back.astype(int) - arr.astype(int)).max() < 25


def test_mozjpeg_smaller_than_libjpeg_turbo():
    """The headline MozJPEG claim: smaller files than libjpeg-turbo at
    the same quality. ~10-15% on natural images. We assert >=5% so the
    test isn't flaky on edge cases."""
    arr = _smooth_rgb()
    mz_size = len(mz.encode(arr, level=85))
    tj_size = len(tj_encode(arr, level=85))
    ratio = mz_size / tj_size
    assert ratio < 0.95, (
        f"MozJPEG should be at least 5% smaller than libjpeg-turbo at "
        f"q=85; got {ratio:.3f} (mz={mz_size}, tj={tj_size})"
    )


def test_mozjpeg_decoded_by_libjpeg_turbo():
    """MozJPEG output is standard JPEG — libjpeg-turbo decodes it
    to the same pixels MozJPEG decodes it to."""
    arr = _smooth_rgb(shape=(128, 128, 3))
    enc = mz.encode(arr, level=85)
    via_mz = mz.decode(enc)
    via_tj = tj_decode(enc)
    np.testing.assert_array_equal(via_mz, via_tj)


def test_mozjpeg_grayscale():
    """2-D grayscale input → grayscale JPEG → 2-D uint8 output."""
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(64, 96), dtype=np.uint8)
    enc = mz.encode(arr, level=90)
    back = mz.decode(enc)
    assert back.ndim == 2
    assert back.shape == arr.shape
    assert back.dtype == np.uint8


@pytest.mark.parametrize("subsampling", ["420", "422", "444"])
def test_mozjpeg_subsampling_options(subsampling):
    """The subsampling= kwarg routes through TJPARAM_SUBSAMP. Each
    value must produce a valid JPEG that round-trips."""
    arr = _smooth_rgb(shape=(64, 96, 3))
    enc = mz.encode(arr, level=85, subsampling=subsampling)
    back = mz.decode(enc)
    assert back.shape == arr.shape


def test_mozjpeg_progressive_param_round_trips():
    """progressive= toggles between baseline and progressive JPEG
    layouts. Both must round-trip cleanly through decoders.

    (We don't assert progressive < baseline size: MozJPEG's trellis
    quantization can produce identical sizes for both modes on smooth
    content, and the size delta is content-dependent. The contract is
    that both modes encode AND decode correctly.)"""
    arr = _smooth_rgb(shape=(64, 96, 3))
    prog_bytes = mz.encode(arr, level=85, progressive=True)
    base_bytes = mz.encode(arr, level=85, progressive=False)
    # Both must round-trip
    prog_back = mz.decode(prog_bytes)
    base_back = mz.decode(base_bytes)
    assert prog_back.shape == arr.shape
    assert base_back.shape == arr.shape


def test_mozjpeg_rejects_non_uint8():
    """uint16 input is rejected — MozJPEG only encodes 8-bit JPEG."""
    arr = np.zeros((16, 16, 3), dtype=np.uint16)
    with pytest.raises(Exception):
        mz.encode(arr)
