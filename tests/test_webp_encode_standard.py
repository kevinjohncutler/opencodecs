"""WebP encode: exact lossless, level as libwebp effort, one code path.

References: the WebP lossless bitstream stores every pixel exactly,
including RGB under alpha 0 (RFC 9649, section 3); libwebp's
WebPConfig.quality is the compression effort for lossless encoding
(encode.h); imagecodecs.webp_encode is the API these parameters mirror.
Decoding goes through imagecodecs (its own libwebp) where possible, so
the round trip is not judged by the code under test alone.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("opencodecs.codecs._webp")

from opencodecs._webp_codec import WebpCodec  # noqa: E402
from opencodecs.codecs import _webp  # noqa: E402


def _rgba_with_hidden_rgb(seed=1, shape=(29, 41)):
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 256, shape + (4,), dtype=np.uint8)
    image[..., 3] = rng.integers(1, 256, shape)
    image[rng.random(shape) < 0.1, 3] = 0
    assert (image[..., 3] == 0).sum() > 50
    return image


def _textured_rgb(seed=2, shape=(128, 160)):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:shape[0], :shape[1]]
    image = np.stack([x, y, (x + y) // 2], -1) + rng.integers(0, 30, shape + (3,))
    return np.clip(image, 0, 255).astype(np.uint8)


def _decoders():
    yield "opencodecs", _webp.decode
    try:
        import imagecodecs
    except ImportError:
        return
    yield "imagecodecs", imagecodecs.webp_decode


@pytest.mark.parametrize("encode", [
    lambda a: WebpCodec().encode(a),
    lambda a: WebpCodec().encode(a, lossless=True),
    lambda a: WebpCodec().encode(a, numthreads=1),
    lambda a: WebpCodec().encode(a, numthreads=4, method=6),
    lambda a: WebpCodec().encode(a, method=0, level=0),
    lambda a: _webp.encode(a),
    lambda a: _webp.encode(a, lossless=True, numthreads=None),
    lambda a: _webp.encode(a, lossless=False, level=-1),
], ids=["codec", "codec-lossless", "codec-1thread", "codec-threads-m6",
        "codec-m0-l0", "native-default", "native-lossless", "native-negative-level"])
def test_lossless_rgba_keeps_rgb_under_transparent_pixels(encode):
    image = _rgba_with_hidden_rgb()
    encoded = encode(image)
    for name, decode in _decoders():
        np.testing.assert_array_equal(decode(encoded), image, err_msg=name)


def test_lossless_level_is_effort():
    image = _textured_rgb()
    outputs = {}
    for level in (0, 75, 100):
        encoded = _webp.encode(image, level=level, lossless=True)
        np.testing.assert_array_equal(_webp.decode(encoded), image)
        outputs[level] = encoded
    # libwebp uses quality as the lossless effort, so each level selects
    # a different encoder configuration. level used to be ignored in
    # lossless mode and all three were the same bytes. (Effort is not
    # strictly monotonic in size on every image, so only distinctness
    # is asserted here; exact bytes are pinned against imagecodecs below.)
    assert len(set(outputs.values())) == 3
    assert _webp.encode(image) == outputs[75]


@pytest.mark.parametrize("lossless", [True, False])
def test_bytes_do_not_depend_on_threads_or_entry_point(lossless):
    image = _textured_rgb()
    for level in (None, 0, 90):
        outputs = {
            _webp.encode(image, level=level, lossless=lossless,
                         numthreads=threads, method=method)
            for threads in (None, 0, 1, 4)
            for method in (None, 4)
        }
        assert len(outputs) == 1, level


def test_method_is_clamped_like_imagecodecs():
    # imagecodecs clamps method to [0, 6] with None meaning 4, so -1 is
    # method 0. It used to mean libwebp's default, 4, here.
    image = _textured_rgb()
    assert _webp.encode(image, method=9) == _webp.encode(image, method=6)
    assert _webp.encode(image, method=-1) == _webp.encode(image, method=0)
    assert _webp.encode(image, method=-5) == _webp.encode(image, method=0)
    assert _webp.encode(image, method=None) == _webp.encode(image, method=4)
    assert _webp.encode(image, method=-1) != _webp.encode(image, method=4)


@pytest.mark.parametrize("encode", [
    lambda a, **kw: _webp.encode(a, **kw),
    lambda a, **kw: WebpCodec().encode(a, **kw),
], ids=["native", "codec"])
def test_lossless_takes_any_truth_value(encode):
    # A Cython ``bool`` annotation refused lossless=1, lossless=0 and
    # NumPy bools with TypeError; imagecodecs takes any value int() reads.
    image = _textured_rgb()
    lossless = encode(image, lossless=True)
    lossy = encode(image, lossless=False)
    assert lossless != lossy
    for value in (1, np.True_, np.int64(7)):
        assert encode(image, lossless=value) == lossless
    for value in (0, np.False_, np.int64(0)):
        assert encode(image, lossless=value) == lossy
    np.testing.assert_array_equal(_webp.decode(lossless), image)


def test_fractional_level_is_kept():
    # An ``int`` annotation truncated level=33.5 to 33, and level=-0.5 to
    # 0, which encoded lossy instead of lossless. imagecodecs passes the
    # float to libwebp.
    image = _textured_rgb()
    assert any(_webp.encode(image, level=level + 0.5, lossless=False)
               != _webp.encode(image, level=level, lossless=False)
               for level in range(20, 60))
    negative = _webp.encode(image, level=-0.5, lossless=False)
    np.testing.assert_array_equal(_webp.decode(negative), image)
    assert negative == _webp.encode(image, level=75, lossless=True)
    assert _webp.encode(image, level=150, lossless=False) == \
        _webp.encode(image, level=100, lossless=False)


def test_native_and_segment_defaults_are_lossless():
    # _webp.encode defaulted to lossy while WebpCodec, imagecodecs and
    # tifffile default to lossless, so TIFF compression="webp" silently
    # wrote lossy tiles.
    from opencodecs.core.segment_compression import encode_segment
    image = _textured_rgb()
    for encoded in (_webp.encode(image), encode_segment(image, "webp")):
        for name, decode in _decoders():
            np.testing.assert_array_equal(decode(encoded), image, err_msg=name)


def test_tiff_writer_webp_default_is_exact(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    pytest.importorskip("imagecodecs")
    from opencodecs._tiff_writer import imwrite
    image = _textured_rgb(shape=(64, 96))
    path = tmp_path / "webp.tif"
    imwrite(path, image, compression="webp")
    np.testing.assert_array_equal(tifffile.imread(path), image)


def test_lossy_is_still_lossy():
    image = _textured_rgb()
    encoded = _webp.encode(image, lossless=False, level=50)
    decoded = _webp.decode(encoded)
    assert not np.array_equal(decoded, image)
    assert np.abs(decoded.astype(int) - image).mean() < 8


def test_unknown_option_raises():
    with pytest.raises(TypeError, match="unexpected option"):
        WebpCodec().encode(_textured_rgb(), quality=50)


def test_bytes_match_imagecodecs():
    """Same arguments, same libwebp: the same bytes as imagecodecs.

    Byte identity only holds when both link the same libwebp release,
    so the comparison is gated on the reported versions.
    """
    imagecodecs = pytest.importorskip("imagecodecs")
    if _webp.version() != imagecodecs.webp_version():
        pytest.skip(f"{_webp.version()} vs imagecodecs {imagecodecs.webp_version()}")
    rgb = _textured_rgb(shape=(48, 64))
    rgba = _rgba_with_hidden_rgb()
    for image in (rgb, rgba):
        for lossless in (None, True, False, 0, 1, np.True_):
            for level in (None, -5, -0.5, 0, 33.5, 50, 100, 150):
                for method in (None, -1, 0, 6, 9):
                    theirs = imagecodecs.webp_encode(image, lossless=lossless,
                                                     level=level, method=method)
                    for ours in (
                        WebpCodec().encode(image, lossless=lossless, level=level,
                                           method=method),
                        _webp.encode(image, lossless=lossless, level=level,
                                     method=method),
                    ):
                        assert ours == theirs, (image.shape, lossless, level, method)


@pytest.mark.parametrize("encode", [
    lambda a, **kw: _webp.encode(a, **kw),
    lambda a, **kw: WebpCodec().encode(a, **kw),
], ids=["native", "codec"])
def test_lossless_is_read_with_int_like_imagecodecs(encode):
    """imagecodecs computes int(lossless is None or lossless or level < 0).

    A string that int() cannot read raised nothing here and meant
    lossless ('no' is a true value); imagecodecs raises ValueError.
    Values int() reads keep imagecodecs' meaning, '0' and 0.5 lossy.
    """
    imagecodecs = pytest.importorskip("imagecodecs")
    image = _textured_rgb()
    for value in ("no", "yes"):
        with pytest.raises(ValueError):
            imagecodecs.webp_encode(image, lossless=value)
        with pytest.raises(ValueError, match="lossless"):
            encode(image, lossless=value)
    for value in ("0", "1", 0.5, 2.0):
        assert encode(image, lossless=value) == \
            imagecodecs.webp_encode(image, lossless=value)
    # A negative level still means lossless whatever lossless says.
    assert encode(image, lossless="", level=-1) == \
        imagecodecs.webp_encode(image, lossless="", level=-1)


def test_encode_out_none_is_accepted_and_a_buffer_is_refused():
    """0.4.0 accepted out= (imagecodecs defines it) and dropped it.

    out=None must still work and give the same bytes; a real buffer is
    refused loudly rather than silently left unwritten.
    """
    imagecodecs = pytest.importorskip("imagecodecs")
    image = _textured_rgb()
    expected = imagecodecs.webp_encode(image)
    assert WebpCodec().encode(image, out=None) == expected
    for out in (bytearray(1 << 20), 1 << 20):
        with pytest.raises(TypeError, match="out="):
            WebpCodec().encode(image, out=out)
