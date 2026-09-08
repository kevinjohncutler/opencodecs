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
from opencodecs._scaled_pyramid import ScaledCodestreamPyramid


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


# ---- the shared base's own surface -----


class _NoLevels(ScaledCodestreamPyramid):
    codec_name = "empty"

    def _probe_levels(self):
        return []


class _Unimplemented(ScaledCodestreamPyramid):
    codec_name = "stub"


def test_a_codestream_with_no_levels_is_an_error_not_an_empty_pyramid():
    """Returning zero levels would make best_level_for index [0].

    An IndexError deep in a viewer is a worse failure than a message
    naming the codec, so the base refuses up front.
    """
    with pytest.raises(ValueError, match="exposes no levels"):
        _NoLevels(b"whatever").levels


def test_the_subclass_hooks_are_required():
    stub = _Unimplemented(b"whatever")
    with pytest.raises(NotImplementedError):
        stub._probe_levels()
    with pytest.raises(NotImplementedError):
        stub._decode_level(0)


def test_max_levels_truncates(img):
    codec = _codec_or_skip("jpeg2k")
    blob = codec.encode(img)
    full = oc.Jpeg2kPyramidReader(blob)
    assert len(full.levels) > 2
    capped = oc.Jpeg2kPyramidReader(blob, max_levels=2)
    assert len(capped.levels) == 2
    assert capped.levels[0].shape == full.levels[0].shape


def test_repr_survives_an_unprobed_codestream():
    """__repr__ runs in debuggers and tracebacks, where the object is
    often broken. It must not raise and hide the real error."""
    r = repr(_NoLevels(b"whatever"))
    assert "unprobed" in r
    assert "_NoLevels" in r


def test_repr_lists_the_levels(img):
    _codec_or_skip("jpeg2k")
    r = repr(oc.Jpeg2kPyramidReader(oc.get_codec("jpeg2k").encode(img)))
    assert "jpeg2k" in r and "512" in r


def test_close_drops_the_cached_level(img):
    """The cache holds one whole decoded level, including level 0.

    close() has to release it or a viewer that opens many images keeps
    a full-resolution array alive per reader.
    """
    _codec_or_skip("jpeg2k")
    p = oc.Jpeg2kPyramidReader(oc.get_codec("jpeg2k").encode(img))
    p.read_region(1)
    assert p._cache is not None
    p.close()
    assert p._cache is None and p._cache_key is None


def test_region_reads_reuse_one_decoded_level(img):
    """Repeated regions at one zoom must not re-decode each time.

    Counted rather than timed: the reader is asked for several regions
    of the same level and the underlying decode must happen once.
    """
    _codec_or_skip("jpeg2k")
    p = oc.Jpeg2kPyramidReader(oc.get_codec("jpeg2k").encode(img))
    calls = []
    inner = p._decode_level

    def counting(reduction):
        calls.append(reduction)
        return inner(reduction)

    p._decode_level = counting
    for x in range(0, 60, 20):
        p.read_region(1, y=(0, 20), x=(x, x + 20))
    assert calls == [p.levels[1].reader], calls
    # A different level is a different decode.
    p.read_region(2)
    assert len(calls) == 2


def test_probe_propagates_a_failure_at_full_resolution():
    """A codestream that cannot decode at all is a real error.

    Reductions past the end stop the walk quietly, which is how the
    level count is discovered -- but that must not swallow a failure
    at reduction 0, or corrupt input would look like an empty pyramid.
    """
    from opencodecs._scaled_pyramid import probe_by_reduction

    def always_fails(r):
        raise RuntimeError("bad codestream")

    with pytest.raises(RuntimeError, match="bad codestream"):
        probe_by_reduction(always_fails, max_reduction=4)


def test_probe_stops_when_a_level_gets_too_small():
    from opencodecs._scaled_pyramid import probe_by_reduction
    import numpy as np

    def shrinking(r):
        return (max(0, 64 >> r), max(0, 64 >> r)), np.dtype("u1")

    levels = probe_by_reduction(shrinking, max_reduction=20, min_size=8)
    assert [s[1][0] for s in levels] == [64, 32, 16, 8]


def test_open_pyramid_accepts_bytes_and_a_file_object(tmp_path, img):
    """_pyramid_bytes has to handle every source open() does."""
    codec = _codec_or_skip("jpeg2k")
    blob = codec.encode(img)
    with oc.open_pyramid(blob, format="jpeg2k") as p:
        assert p.levels[0].shape[:2] == img.shape[:2]
    f = tmp_path / "x.jp2"
    f.write_bytes(blob)
    with f.open("rb") as fh, oc.open_pyramid(fh, format="jpeg2k") as p:
        assert p.levels[0].shape[:2] == img.shape[:2]


def test_open_pyramid_rejects_a_source_it_cannot_read():
    with pytest.raises(TypeError, match="cannot read a codestream"):
        oc.open_pyramid(object(), format="jpeg2k")


def test_open_pyramid_names_the_formats_it_knows():
    with pytest.raises(ValueError, match="jpeg2k"):
        oc.open_pyramid("mystery.unknown")


def test_open_pyramid_dispatches_mozjpeg_by_name(img):
    codec = _codec_or_skip("mozjpeg")
    blob = codec.encode(img, level=85)
    with oc.open_pyramid(blob, format="mozjpeg") as p:
        assert type(p) is oc.MozjpegPyramidReader
        assert p.levels[0].shape[:2] == img.shape[:2]


# ---- mozjpeg's scale argument, in every form it accepts -----


def test_mozjpeg_scale_accepts_int_float_tuple_and_pair():
    """All four spellings must land on the same factor.

    _resolve_scale is duplicated between the jpeg and mozjpeg modules
    (different C APIs underneath), so the two can drift. Each form is
    checked against the shape it should produce.
    """
    _codec_or_skip("mozjpeg")
    from opencodecs.codecs import _mozjpeg
    blob = _mozjpeg.encode(_smooth(512, 512), level=85)
    quarter = (128, 128, 3)
    assert _mozjpeg.decode(blob, scale=4).shape == quarter
    assert _mozjpeg.decode(blob, scale=(1, 4)).shape == quarter
    assert _mozjpeg.decode(blob, scale_num=1, scale_denom=4).shape == quarter
    assert _mozjpeg.decode(blob, scale=0.25).shape == quarter
    assert _mozjpeg.decode(blob).shape == (512, 512, 3)
    assert _mozjpeg.decode(blob, scale=1).shape == (512, 512, 3)


def test_mozjpeg_scale_rejects_nonsense():
    _codec_or_skip("mozjpeg")
    from opencodecs.codecs import _mozjpeg
    blob = _mozjpeg.encode(_smooth(128, 128), level=85)
    with pytest.raises(ValueError, match="must be >= 1"):
        _mozjpeg.decode(blob, scale=0)
    with pytest.raises(ValueError, match="must be > 0"):
        _mozjpeg.decode(blob, scale=-0.5)
    with pytest.raises(ValueError, match="must be \\(num, denom\\)"):
        _mozjpeg.decode(blob, scale=(1, 2, 3))


def test_mozjpeg_lists_its_scaling_factors():
    _codec_or_skip("mozjpeg")
    from opencodecs.codecs import _mozjpeg
    factors = _mozjpeg.supported_scaling_factors()
    assert (1, 1) in factors and (1, 8) in factors
    assert all(isinstance(n, int) and isinstance(d, int) for n, d in factors)


def test_jpeg_denominators_come_from_the_library_and_are_cached():
    """Probing and decoding must index the SAME list.

    They each computed it separately once, which is a standing
    invitation for the two to disagree about what level 2 means.
    """
    _codec_or_skip("jpeg")
    p = oc.JpegPyramidReader(oc.get_codec("jpeg").encode(_smooth(256, 256),
                                                         level=85))
    d = p.denominators()
    assert d[0] == 1, "level 0 must be full resolution"
    assert list(d) == sorted(d), "levels must go coarsest-last"
    assert p.denominators() is d, "recomputed instead of cached"
    for i, denom in enumerate(d):
        assert p.read_level(i).shape[0] == 256 // denom


def test_a_backend_with_no_scaling_factors_has_no_levels(monkeypatch):
    """If a build reports no supported ratios, say so rather than
    indexing an empty list."""
    _codec_or_skip("jpeg")
    p = oc.JpegPyramidReader(oc.get_codec("jpeg").encode(_smooth(128, 128),
                                                         level=85))
    monkeypatch.setattr(p, "denominators", lambda: ())
    with pytest.raises(ValueError, match="exposes no levels"):
        p.levels


def test_pyramid_accepts_a_memoryview_source():
    """open() hands these readers whatever the caller had."""
    _codec_or_skip("jpeg2k")
    blob = oc.get_codec("jpeg2k").encode(_smooth(128, 128))
    p = oc.Jpeg2kPyramidReader(memoryview(blob))
    assert p.levels[0].shape[:2] == (128, 128)


def test_htj2k_max_levels_truncates():
    codec = _codec_or_skip("htj2k")
    blob = codec.encode(_smooth(256, 256), num_decomp=4)
    assert len(oc.Htj2kPyramidReader(blob, max_levels=2).levels) == 2


def test_htj2k_reader_forwards_ignore_unsupported():
    """The flag has to reach the decoder, not sit unused on the reader."""
    codec = _codec_or_skip("htj2k")
    blob = codec.encode(_smooth(128, 128), num_decomp=3)
    p = oc.Htj2kPyramidReader(blob, ignore_unsupported=True)
    assert p._ignore_unsupported is True
    assert p.read_level(0).shape[:2] == (128, 128)


def test_htj2k_pyramid_over_16_bit_data():
    """Scientific imagery is 16-bit, so the levels must be too.

    The dtype is derived from the codestream's bit depth, and a
    pyramid that quietly handed back uint8 would lose the top 8 bits
    of every pixel while still looking like a working reader.
    """
    codec = _codec_or_skip("htj2k")
    yy, xx = np.mgrid[0:256, 0:256].astype(np.float64)
    img16 = (2000 + 20000 * (0.5 + 0.5 * np.sin(yy / 30) * np.cos(xx / 24))
             ).astype(np.uint16)
    p = oc.Htj2kPyramidReader(codec.encode(img16, num_decomp=4))
    assert all(L.dtype == np.uint16 for L in p.levels), \
        [L.dtype for L in p.levels]
    full = p.read_level(0)
    assert full.dtype == np.uint16
    assert full.max() > 255, "16-bit range was truncated to 8 bits"
    half = p.read_level(1)
    assert half.dtype == np.uint16
    assert half.shape[:2] == (128, 128)
    assert abs(int(half.mean()) - int(full.mean())) < 200


def test_jpeg_max_levels_truncates(img):
    codec = _codec_or_skip("jpeg")
    blob = codec.encode(img, level=85)
    assert len(oc.JpegPyramidReader(blob).levels) > 2
    assert len(oc.JpegPyramidReader(blob, max_levels=2).levels) == 2


def test_pyramid_accepts_a_buffer_that_is_not_bytes():
    """np.frombuffer / mmap results reach these readers routinely."""
    _codec_or_skip("jpeg2k")
    blob = oc.get_codec("jpeg2k").encode(_smooth(128, 128))
    arr = np.frombuffer(blob, dtype=np.uint8)
    p = oc.Jpeg2kPyramidReader(arr)
    assert p.levels[0].shape[:2] == (128, 128)
    assert np.array_equal(p.read_level(0), oc.Jpeg2kPyramidReader(blob).read_level(0))
