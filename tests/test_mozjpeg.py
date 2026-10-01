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


def _sof_marker(stream):
    i = 2
    while True:
        m = stream[i + 1]
        if 0xC0 <= m <= 0xCF and m not in (0xC4, 0xC8, 0xCC):
            return m
        i += 2 + (stream[i + 2] << 8 | stream[i + 3])


@pytest.mark.parametrize("shape", [(64, 96, 3), (64, 96)])
def test_mozjpeg_progressive_false_raises(shape):
    """MozJPEG's TurboJPEG API writes progressive scans (SOF2) whatever
    the flags say, so progressive=False cannot be honored and raises,
    rather than writing the progressive bytes it used to. imagecodecs'
    mozjpeg_encode, which drives libjpeg directly, writes a sequential
    frame (SOF0) for it; that is the meaning being refused."""
    import opencodecs as oc
    arr = _smooth_rgb(shape=(64, 96, 3))
    if len(shape) == 2:
        arr = np.ascontiguousarray(arr[..., 0])
    prog = mz.encode(arr, level=85)
    assert _sof_marker(prog) == 0xC2
    assert mz.encode(arr, level=85, progressive=True) == prog
    assert mz.decode(prog).shape == arr.shape
    with pytest.raises(NotImplementedError, match="progressive"):
        mz.encode(arr, level=85, progressive=False)
    with pytest.raises(NotImplementedError, match="progressive"):
        oc.get_codec("mozjpeg").encode(arr, progressive=False)
    with pytest.raises(NotImplementedError, match="progressive"):
        oc.write(None, arr, format="mozjpeg", progressive=False)
    ic = _ic()
    if hasattr(ic, "mozjpeg_encode") and getattr(
            getattr(ic, "MOZJPEG", None), "available", False):
        assert _sof_marker(ic.mozjpeg_encode(arr, progressive=False)) == 0xC0


def test_mozjpeg_rejects_non_uint8():
    """uint16 input is rejected — MozJPEG only encodes 8-bit JPEG."""
    arr = np.zeros((16, 16, 3), dtype=np.uint16)
    with pytest.raises(Exception):
        mz.encode(arr)


def test_mozjpeg_list_input_is_not_cast():
    # Like imagecodecs, encode takes the dtype NumPy gives the input; a
    # list of Python ints is a signed integer array and raises instead
    # of being cast to uint8, which 0.4.0 did.
    ic = pytest.importorskip("imagecodecs")
    with pytest.raises(Exception):
        ic.jpeg8_encode([[1, 2], [3, 4]])
    with pytest.raises(mz.MozJpegError, match="dtype"):
        mz.encode([[1, 2], [3, 4]])
    with pytest.raises(mz.MozJpegError, match="dtype"):
        mz.encode([[1.0, 2.0]])


# ---------------------------------------------------------------------------
# 0.5.0: decode coverage, (H, W, 1), and parameters that are honored or
# raise. imagecodecs writes the reference streams: its mozjpeg is often a
# stub, but jpeg8 links libjpeg-turbo, and decode is standard JPEG.
# ---------------------------------------------------------------------------


def _ic():
    ic = pytest.importorskip("imagecodecs")
    if not getattr(getattr(ic, "JPEG8", None), "available", False):
        pytest.skip("imagecodecs has no jpeg8")
    return ic


def _streams(ic):
    rng = np.random.default_rng(11)
    u12 = rng.integers(0, 4096, (9, 11), dtype=np.uint16)
    u16 = rng.integers(0, 65536, (9, 11), dtype=np.uint16)
    c4 = rng.integers(0, 256, (16, 24, 4), dtype=np.uint8)
    return {
        "12-bit DCT": ic.jpeg8_encode(u12),
        "12-bit lossless": ic.jpeg8_encode(u12, lossless=True),
        "16-bit lossless": ic.jpeg8_encode(u16, lossless=True,
                                           bitspersample=16),
        "8-bit lossless": ic.jpeg8_encode(_smooth_rgb((16, 24, 3)),
                                          lossless=True),
        "cmyk": ic.jpeg8_encode(c4, colorspace="cmyk", outcolorspace="cmyk"),
        "ycck": ic.jpeg8_encode(c4, colorspace="cmyk", outcolorspace="ycck"),
        "four components, no Adobe marker": ic.jpeg8_encode(c4),
    }


@pytest.mark.parametrize("kind", [
    "12-bit DCT", "12-bit lossless", "16-bit lossless", "8-bit lossless",
    "cmyk", "ycck", "four components, no Adobe marker"])
def test_mozjpeg_decodes_what_imagecodecs_writes(kind):
    import opencodecs as oc
    ic = _ic()
    stream = _streams(ic)[kind]
    expect = ic.jpeg8_decode(stream)
    np.testing.assert_array_equal(mz.decode(stream), expect)
    np.testing.assert_array_equal(oc.get_codec("mozjpeg").decode(stream),
                                  expect)
    with oc.get_codec("mozjpeg").decoder() as dec:
        np.testing.assert_array_equal(dec.decode(stream), expect)


@pytest.mark.parametrize("kind", ["12-bit DCT", "16-bit lossless"])
def test_mozjpeg_finds_the_frame_header_past_other_markers(kind):
    # The decoder reads the frame header itself to route these streams
    # to the jpeg codec. T.81 B.1.1.2 lets fill bytes (0xFF) precede any
    # marker, and application segments may come before the frame; the
    # walk has to step over both, and take any buffer type.
    ic = _ic()
    stream = _streams(ic)[kind]
    app15 = b"\xff\xef" + (2 + 300).to_bytes(2, "big") + bytes(range(256)) \
        + bytes(44)
    padded = stream[:2] + b"\xff\xff" + app15 + stream[2:]
    expect = ic.jpeg8_decode(padded)
    np.testing.assert_array_equal(expect, ic.jpeg8_decode(stream))
    for buf in (padded, bytearray(padded), memoryview(padded)):
        np.testing.assert_array_equal(mz.decode(buf), expect)


def test_mozjpeg_trailing_singleton_channel_is_grayscale():
    import opencodecs as oc
    rng = np.random.default_rng(12)
    g = rng.integers(0, 256, (32, 48, 1), dtype=np.uint8)
    enc = mz.encode(g)
    assert enc == mz.encode(g[:, :, 0])
    assert oc.write(None, g, format="mozjpeg") == enc
    assert tj_decode(enc).shape == (32, 48)


@pytest.mark.parametrize("cs", ["ycbcr", "cmyk", "ycck"])
def test_mozjpeg_unsupported_input_names_the_codec_that_writes_it(cs):
    # The jpeg codec writes YCbCr and CMYK input and refuses YCCK input,
    # so the error points there for the first two only.
    arr = _smooth_rgb((16, 24, 3))
    if cs != "ycbcr":
        arr = np.dstack([arr, arr[..., :1]])
    with pytest.raises(NotImplementedError) as info:
        mz.encode(arr, colorspace=cs)
    message = str(info.value)
    assert message.startswith(f"MozJPEG encode: {cs} input")
    if cs == "ycck":
        assert "'jpeg'" not in message
        with pytest.raises(NotImplementedError):
            tj_encode(arr, colorspace=cs)
    else:
        assert message.endswith("the 'jpeg' codec writes it")
        assert tj_encode(arr, colorspace=cs)[:2] == b"\xff\xd8"


@pytest.mark.parametrize("kwargs", [
    dict(optimize=False), dict(notrellis=True), dict(quanttable=1),
    dict(smoothing=5), dict(outcolorspace="rgb"),
    dict(colorspace="cmyk"),
])
def test_mozjpeg_unavailable_options_raise(kwargs):
    import opencodecs as oc
    arr = _smooth_rgb((16, 24, 3))
    if kwargs.get("colorspace") == "cmyk":
        arr = np.dstack([arr, arr[..., :1]])
    with pytest.raises(NotImplementedError):
        oc.get_codec("mozjpeg").encode(arr, **kwargs)


def test_mozjpeg_honors_subsampling_spellings_and_defaults():
    import opencodecs as oc
    arr = _smooth_rgb((32, 48, 3))
    codec = oc.get_codec("mozjpeg")
    assert codec.encode(arr, subsampling=(1, 1)) == \
        codec.encode(arr, subsampling="444")
    # Values equal to the fixed behavior are accepted.
    assert codec.encode(arr, optimize=True, notrellis=False,
                        smoothing=0) == codec.encode(arr)
    gray = codec.encode(arr, outcolorspace="gray")
    assert tj_decode(gray).ndim == 2


def test_mozjpeg_decode_options_match_imagecodecs():
    ic = _ic()
    stream = ic.jpeg8_encode(_smooth_rgb((37, 53, 3)))
    for kwargs in (dict(outcolorspace="gray"), dict(outcolorspace="rgba"),
                   dict(colorspace="rgb"), dict(outcolorspace="ycbcr")):
        np.testing.assert_array_equal(mz.decode(stream, **kwargs),
                                      ic.jpeg8_decode(stream, **kwargs))
    np.testing.assert_array_equal(mz.decode(stream, fancyupsampling=False),
                                  tj_decode(stream, fancyupsampling=False))
