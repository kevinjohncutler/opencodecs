"""Pyramids synthesized from one scalable codestream.

JPEG 2000, HTJ2K and JPEG all decode a smaller image directly out of a
single stored image. These tests pin the two things that matter and
that are easy to get wrong:

  * the reduction is real -- the output is the reduced geometry the
    headers promised, and it is the *right pixels*, not garbage or a
    silently-full-resolution decode;
  * the pyramid surface agrees with itself -- a region read equals the
    same slice of the whole level, and level shapes halve.

The "is it actually cheaper" claim is deliberately not asserted by
timing here; wall-clock in a test suite is how flaky tests are born.
It is measured in the module docstrings, where the numbers can be
re-derived on demand instead of failing a build on a busy machine.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc


def _smooth(h=512, w=512):
    """Content whose wavelet lowpass and a plain decimation agree.

    Reductions are lowpass filters, so a stripe pattern legitimately
    disagrees with `img[::2, ::2]` and would make a correctness check
    look like a failure. Smooth content removes that confound.
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    s = 128 + 90 * np.sin(yy / 40) * np.cos(xx / 33)
    return np.clip(np.dstack([s, s * 0.9 + 12, s * 1.05 - 8]),
                   0, 255).astype(np.uint8)


def _codec_or_skip(name):
    if not oc.has_codec(name):
        pytest.skip(f"{name} codec not built")
    return oc.get_codec(name)


CASES = [
    ("jpeg2k", "Jpeg2kPyramidReader", {}),
    ("htj2k", "Htj2kPyramidReader", {"num_decomp": 5}),
    ("jpeg", "JpegPyramidReader", {"level": 90}),
    ("mozjpeg", "MozjpegPyramidReader", {"level": 90}),
]


@pytest.fixture(scope="module")
def img():
    return _smooth()


@pytest.mark.parametrize("name,cls_name,enc_kw", CASES)
def test_levels_halve_and_start_at_full_resolution(name, cls_name, enc_kw, img):
    codec = _codec_or_skip(name)
    reader = getattr(oc, cls_name)(codec.encode(img, **enc_kw))
    levels = reader.levels
    assert len(levels) >= 2, "a pyramid with one level is not a pyramid"
    assert levels[0].shape[:2] == img.shape[:2]
    for i, L in enumerate(levels):
        assert L.downscale == (2 ** i, 2 ** i), (i, L.downscale)
        assert L.shape[0] == levels[0].shape[0] // (2 ** i)
    reader.close()


@pytest.mark.parametrize("name,cls_name,enc_kw", CASES)
def test_reduced_level_is_the_right_pixels(name, cls_name, enc_kw, img):
    """The half-size level must look like the image, not like noise.

    This is the test that would catch a reduction wired up to the wrong
    subbands, or one that quietly returned an uninitialized buffer: on
    smooth content the reduced level and a decimation of the full
    decode agree closely, and nothing else does.
    """
    codec = _codec_or_skip(name)
    reader = getattr(oc, cls_name)(codec.encode(img, **enc_kw))
    full = reader.read_level(0).astype(np.float64)
    half = reader.read_level(1).astype(np.float64)
    assert half.shape[:2] == (full.shape[0] // 2, full.shape[1] // 2)
    err = np.abs(half - full[::2, ::2]).mean()
    assert err < 6.0, f"{name}: reduced level differs by {err:.2f} counts"
    reader.close()


@pytest.mark.parametrize("name,cls_name,enc_kw", CASES)
def test_region_equals_slice_of_level(name, cls_name, enc_kw, img):
    codec = _codec_or_skip(name)
    reader = getattr(oc, cls_name)(codec.encode(img, **enc_kw))
    lvl = 1
    whole = reader.read_level(lvl)
    region = reader.read_region(lvl, y=(7, 61), x=(13, 100))
    assert np.array_equal(region, whole[7:61, 13:100])
    reader.close()


@pytest.mark.parametrize("name,cls_name,enc_kw", CASES)
def test_best_level_for_picks_a_fitting_level(name, cls_name, enc_kw, img):
    codec = _codec_or_skip(name)
    reader = getattr(oc, cls_name)(codec.encode(img, **enc_kw))
    lvl = reader.best_level_for(max_pixels_y=100)
    assert reader.levels[lvl].shape[0] <= 100
    assert reader.read_region(lvl).shape[:2] == reader.levels[lvl].shape[:2]
    reader.close()


# ---- the codec-level reduction arguments -----


def test_jpeg2k_decode_info_matches_decode():
    """decode_info must not be allowed to drift from decode.

    It exists so enumerating levels costs only a header parse; if it
    ever disagreed with what decode produces, every pyramid built on it
    would allocate the wrong shape.
    """
    _codec_or_skip("jpeg2k")
    from opencodecs.codecs import _jpeg2k
    blob = _jpeg2k.encode(_smooth(256, 256))
    for r in range(0, 4):
        info = _jpeg2k.decode_info(blob, reduce=r)
        assert _jpeg2k.decode(blob, reduce=r).shape == info["shape"]


def test_htj2k_decode_info_matches_decode():
    _codec_or_skip("htj2k")
    from opencodecs.codecs import _openjph
    blob = _openjph.encode(_smooth(256, 256), num_decomp=4)
    for r in range(0, 4):
        d = _openjph.decode_info(blob, reduce=r)
        want = (d["height"], d["width"], d["components"])
        assert _openjph.decode(blob, reduce=r).shape == want


def test_reduce_beyond_the_codestream_raises():
    """Too large a reduction must fail loudly.

    Silently clamping to full resolution would be worse than an error:
    a caller asking for a thumbnail would get the whole image and only
    notice from the memory.
    """
    _codec_or_skip("jpeg2k")
    from opencodecs.codecs import _jpeg2k
    blob = _jpeg2k.encode(_smooth(128, 128))
    with pytest.raises(Exception):
        _jpeg2k.decode(blob, reduce=30)
    with pytest.raises(ValueError):
        _jpeg2k.decode(blob, reduce=-1)


def test_mozjpeg_scaled_decode_matches_the_jpeg_codec():
    """Two JPEG decoders taking the same argument must agree.

    mozjpeg reaches DCT scaling through TurboJPEG v2 (destination size)
    and jpeg through v3 (tj3SetScalingFactor). Different mechanisms,
    and the output has to be identical or `scale=` means two things.
    """
    _codec_or_skip("mozjpeg")
    _codec_or_skip("jpeg")
    from opencodecs.codecs import _jpeg, _mozjpeg
    blob = _jpeg.encode(_smooth(512, 512), level=85)
    for denom in (1, 2, 4, 8):
        a = _mozjpeg.decode(blob, scale=(1, denom))
        b = _jpeg.decode(blob, scale=(1, denom))
        assert a.shape == b.shape
        assert np.array_equal(a, b), f"scale 1/{denom} disagrees"


def test_mozjpeg_rejects_an_unsupported_ratio():
    _codec_or_skip("mozjpeg")
    from opencodecs.codecs import _mozjpeg
    blob = _mozjpeg.encode(_smooth(128, 128), level=85)
    with pytest.raises(ValueError, match="not supported"):
        _mozjpeg.decode(blob, scale=(5, 7))


@pytest.mark.parametrize("ext,cls_name,name,enc_kw", [
    (".jp2", "Jpeg2kPyramidReader", "jpeg2k", {}),
    (".jph", "Htj2kPyramidReader", "htj2k", {"num_decomp": 4}),
    (".jpg", "JpegPyramidReader", "jpeg", {"level": 85}),
])
def test_open_pyramid_dispatches_by_extension(ext, cls_name, name, enc_kw,
                                              tmp_path, img):
    codec = _codec_or_skip(name)
    p = tmp_path / f"sample{ext}"
    p.write_bytes(codec.encode(img, **enc_kw))
    with oc.open_pyramid(str(p)) as pyr:
        assert type(pyr) is getattr(oc, cls_name)
        assert pyr.levels[0].shape[:2] == img.shape[:2]
