"""JPEG 2000 against independent references: imagecodecs, tifffile, the spec.

Each test pins a disagreement that used to exist:

* signed components (ISO/IEC 15444-1 Annex A.5.1 Ssiz bit 7, no DC
  level shift per Annex G.1) decode to signed values, not to unsigned
  values offset by 2**(prec - 1);
* images under 32 pixels on a side encode, with imagecodecs' resolution
  count, and the codestream is byte-identical to imagecodecs';
* ``level`` is imagecodecs' PSNR target in dB, and is never silently
  ignored;
* the imagecodecs keywords are implemented, judged by byte identity
  with ``imagecodecs.jpeg2k_encode``.
"""

from __future__ import annotations

import io
import math

import numpy as np
import pytest
from _tifffile_guard import requires_patchable_tifffile

oj = pytest.importorskip("opencodecs.codecs._jpeg2k")
imagecodecs = pytest.importorskip("imagecodecs")

import opencodecs as oc  # noqa: E402
from opencodecs import get_codec  # noqa: E402
from _ic_reference import skip_if_old_imagecodecs  # noqa: E402

pytestmark = skip_if_old_imagecodecs

try:
    imagecodecs.jpeg2k_encode(np.zeros((8, 8), np.uint8))
except Exception as exc:  # noqa: BLE001
    pytest.skip(f"imagecodecs has no JPEG 2000 backend: {exc}",
                allow_module_level=True)


def _image(shape, dtype, seed=0):
    rng = np.random.default_rng(seed)
    info = np.iinfo(dtype)
    lo, hi = float(info.min), float(info.max)
    y, x = np.mgrid[: shape[0], : shape[1]]
    base = (3.0 * x + 5.0 * y)
    if len(shape) == 3:
        base = base[..., None] + 40.0 * np.arange(shape[2])
    a = lo + base / max(base.max(), 1.0) * (hi - lo) * 0.9
    a = a + rng.normal(0, (hi - lo) / 60, shape)
    return np.clip(a, lo, hi).astype(dtype)


def _marker(codestream: bytes, code: bytes) -> int:
    return codestream.index(code)


def _ssiz(codestream: bytes) -> int:
    """Ssiz of component 0 (ISO/IEC 15444-1 Table A.9): SIZ marker,
    Lsiz, Rsiz, eight 32-bit extents and offsets, Csiz, then Ssiz."""
    return codestream[_marker(codestream, b"\xff\x51") + 40]


def _num_decompositions(codestream: bytes) -> int:
    """SPcod's decomposition level count (Table A.15): COD marker, Lcod,
    Scod, the four SGcod bytes, then NL."""
    return codestream[_marker(codestream, b"\xff\x52") + 9]


# ---------------------------------------------------------------------------
# Signed components
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("codecformat", ["j2k", "jp2"])
@pytest.mark.parametrize("shape", [(23, 37), (31, 29, 3)])
@pytest.mark.parametrize("dtype", [np.int8, np.int16])
def test_signed_codestream_decodes_to_signed(dtype, shape, codecformat):
    """Used to come back as uint, every sample + 2**(prec - 1)."""
    a = _image(shape, dtype, seed=1)
    info = np.iinfo(dtype)
    a.flat[:4] = [info.min, info.max, -1, 0]
    enc = imagecodecs.jpeg2k_encode(a, level=0, codecformat=codecformat)
    assert _ssiz(enc) & 0x80, "imagecodecs wrote a signed component"
    out = oj.decode(enc)
    assert out.dtype == dtype
    np.testing.assert_array_equal(out, a)
    assert oj.decode_info(enc)["dtype"] == dtype
    assert oj.decode_info(enc)["signed"] is True
    np.testing.assert_array_equal(oc.read(enc, format="jpeg2k"), a)


@pytest.mark.parametrize("dtype", [np.int8, np.int16])
def test_signed_encode_matches_imagecodecs(dtype):
    """Signed input used to raise; it is now a signed codestream,
    byte-identical to imagecodecs' and decoded by it exactly."""
    a = _image((31, 29, 3), dtype, seed=2)
    ours = oj.encode(a, codec="j2k")
    assert _ssiz(ours) == 0x80 | (8 * a.itemsize - 1)
    assert ours == imagecodecs.jpeg2k_encode(a, codecformat="j2k")
    np.testing.assert_array_equal(imagecodecs.jpeg2k_decode(ours), a)
    np.testing.assert_array_equal(oj.decode(ours), a)


def test_tiff_signed_jpeg2000_tiles():
    """tifffile writes int16 with JPEG 2000 compression; opencodecs'
    TIFF reader used to return it with the sign bit flipped."""
    tifffile = pytest.importorskip("tifffile")
    a = _image((64, 64), np.int16, seed=3)
    buf = io.BytesIO()
    tifffile.imwrite(buf, a, compression="jpeg2000",
                     compressionargs={"level": 0})
    out = oc.read(buf.getvalue(), format="tiff")
    assert out.dtype == np.int16
    np.testing.assert_array_equal(out, a)


@pytest.mark.parametrize("dtype", [np.int32, np.uint32])
def test_more_than_16_bits_decodes(dtype):
    """Above 16 bits used to raise; imagecodecs returns (u)int32.

    imagecodecs writes 32-bit input as 26-bit components and its
    unsigned path does not round-trip every sample, so the reference
    for uint32 is imagecodecs' decode of the same codestream; int32
    round-trips and is checked against the source as well.
    """
    rng = np.random.default_rng(4)
    lo = -(1 << 20) if dtype == np.int32 else 0
    a = rng.integers(lo, 1 << 20, (21, 33)).astype(dtype)
    enc = imagecodecs.jpeg2k_encode(a, level=0)
    out = oj.decode(enc)
    assert out.dtype == dtype
    assert oj.decode_info(enc)["precision"] > 16
    np.testing.assert_array_equal(out, imagecodecs.jpeg2k_decode(enc))
    if dtype == np.int32:
        np.testing.assert_array_equal(out, a)


def test_planar_decode():
    a = _image((24, 30, 3), np.uint8, seed=5)
    enc = imagecodecs.jpeg2k_encode(a)
    np.testing.assert_array_equal(oj.decode(enc, planar=True),
                                  np.moveaxis(a, -1, 0))
    np.testing.assert_array_equal(
        get_codec("jpeg2k").decode(enc, planar=True),
        imagecodecs.jpeg2k_decode(enc, planar=True))


# ---------------------------------------------------------------------------
# Small images and the resolution count
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,dtype", [
    ((1, 1), np.uint8), ((7, 5), np.uint8), ((8, 8), np.uint8),
    ((16, 16), np.uint16), ((31, 29), np.uint8), ((64, 31), np.uint8),
    ((31, 29, 3), np.uint16), ((32, 32), np.uint8), ((37, 53), np.uint8),
    ((33, 200), np.uint16), ((256, 256), np.uint8),
])
@pytest.mark.parametrize("codec", ["jp2", "j2k"])
def test_small_images_encode_like_imagecodecs(shape, dtype, codec):
    """A fixed 6 resolutions made OpenJPEG refuse anything under 32 px.
    imagecodecs' count is min(6, max(1, int(log2(min(h, w)) - 2)))."""
    a = _image(shape, dtype, seed=6)
    ours = oj.encode(a, codec=codec)
    expected = min(6, max(1, int(math.log(min(shape[:2])) / math.log(2) - 2)))
    assert _num_decompositions(ours) + 1 == expected
    assert ours == imagecodecs.jpeg2k_encode(a, codecformat=codec)
    np.testing.assert_array_equal(imagecodecs.jpeg2k_decode(ours), a)
    np.testing.assert_array_equal(oc.read(oc.write(None, a, format="jpeg2k"),
                                          format="jpeg2k"), a)


def test_resolutions_caps_the_count():
    a = _image((64, 80, 3), np.uint8, seed=7)
    for r in (1, 2, 3, 9):
        ours = oj.encode(a, resolutions=r)
        assert ours == imagecodecs.jpeg2k_encode(a, resolutions=r)
        assert _num_decompositions(ours) + 1 == min(r, 4)


# ---------------------------------------------------------------------------
# level: imagecodecs' PSNR target
# ---------------------------------------------------------------------------


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return math.inf if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


@pytest.mark.parametrize("level", [10, 25, 40, 60, 100, 1000])
def test_level_is_a_psnr_target_like_imagecodecs(level):
    """Was a compression ratio of 100/level, and ignored entirely
    unless lossless=False was also passed."""
    a = _image((200, 300), np.uint8, seed=8)
    # imagecodecs encodes on one thread by default; OpenJPEG's
    # multithreaded rate allocation is not deterministic at an
    # unreachable target such as 1000 dB.
    ref = imagecodecs.jpeg2k_encode(a, level=level)
    assert oj.encode(a, level=level, numthreads=1) == ref
    assert oc.write(None, a, format="jpeg2k", level=level,
                    numthreads=1) == ref
    if level <= 40:
        # The independent check: OpenJPEG stops adding coding passes
        # once the target is met, so the PSNR lands at or a little
        # above it. A ratio of 100/level would give 30 to 45 dB here.
        got = _psnr(oj.decode(ref), a)
        assert level - 1.0 <= got <= level + 5.0, got


@pytest.mark.parametrize("level", [None, 0, 0.5, 1001])
def test_level_outside_1_to_1000_is_lossless(level):
    a = _image((64, 80), np.uint8, seed=9)
    ours = oj.encode(a, level=level)
    assert ours == imagecodecs.jpeg2k_encode(a, level=level)
    np.testing.assert_array_equal(oj.decode(ours), a)


def test_lossless_true_with_a_lossy_level_raises():
    a = _image((64, 80), np.uint8, seed=10)
    with pytest.raises(ValueError, match="lossless=True"):
        oj.encode(a, level=40, lossless=True)
    with pytest.raises(ValueError, match="lossless=True"):
        oc.write(None, a, format="jpeg2k", level=40, lossless=True)
    with pytest.raises(ValueError, match="lossless=True"):
        oj.encode(a, lossless=True, reversible=False)


def test_lossless_false_with_level_is_the_same_psnr_encode():
    a = _image((64, 80), np.uint8, seed=11)
    assert (oj.encode(a, level=40, lossless=False, numthreads=1)
            == imagecodecs.jpeg2k_encode(a, level=40))


def test_ratio_is_a_compression_ratio():
    a = _image((128, 160), np.uint8, seed=12)
    enc = oj.encode(a, ratio=20, codec="j2k")
    assert len(enc) < a.nbytes / 15
    assert oj.decode(enc).shape == a.shape
    with pytest.raises(ValueError):
        oj.encode(a, ratio=20, level=40)


# ---------------------------------------------------------------------------
# The other imagecodecs keywords
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("kw", [
    {"level": 30, "reversible": True}, {"reversible": False},
    {"mct": False}, {"colorspace": "GRAY"}, {"codecformat": "j2k"},
    {"level": 45, "codecformat": "J2K"},
])
def test_encode_options_match_imagecodecs(kw):
    a = _image((64, 80, 3), np.uint8, seed=13)
    ref = imagecodecs.jpeg2k_encode(a, **kw)
    assert oj.encode(a, numthreads=1, **kw) == ref
    assert get_codec("jpeg2k").encode(a, numthreads=1, **kw) == ref


def test_planar_input_and_heuristic():
    a = np.ascontiguousarray(np.moveaxis(_image((20, 30, 3), np.uint8), -1, 0))
    assert oj.encode(a, planar=True) == imagecodecs.jpeg2k_encode(a, planar=True)
    # (3, 20, 30): last axis > 4 and first <= 4 reads as planar, as in
    # imagecodecs.
    assert oj.encode(a) == imagecodecs.jpeg2k_encode(a)


def test_bitspersample():
    a = _image((40, 50), np.uint16, seed=14) >> 4  # 12-bit data
    ours = oj.encode(a, bitspersample=12)
    assert ours == imagecodecs.jpeg2k_encode(a, bitspersample=12)
    assert oj.decode_info(ours)["precision"] == 12
    np.testing.assert_array_equal(oj.decode(ours), a)
    with pytest.raises(ValueError, match="would be lost"):
        oj.encode(a, bitspersample=10)


def test_tile_raises_like_imagecodecs():
    with pytest.raises(NotImplementedError):
        oj.encode(np.zeros((64, 64), np.uint8), tile=(32, 32))


def test_codec_rejects_unknown_options():
    codec = get_codec("jpeg2k")
    a = np.zeros((16, 16), np.uint8)
    with pytest.raises(TypeError, match="unsupported"):
        codec.encode(a, quality=50)
    with pytest.raises(TypeError, match="unsupported"):
        codec.decode(codec.encode(a), skipres=1)


# ---------------------------------------------------------------------------
# Callers that used to get a lossy file without asking
# ---------------------------------------------------------------------------


@requires_patchable_tifffile
def test_tifffile_patch_defaults_to_lossless():
    """The imagecodecs-compatible shim passed lossless=bool(None), which
    made every tifffile JPEG 2000 write lossy."""
    from opencodecs import tifffile_patch
    a = _image((64, 80), np.uint16, seed=15)
    enc = tifffile_patch.jpeg2k_encode(a)
    np.testing.assert_array_equal(imagecodecs.jpeg2k_decode(enc), a)
    enc = tifffile_patch.jpeg2k_encode(a, level=0, codecformat=0)
    np.testing.assert_array_equal(imagecodecs.jpeg2k_decode(enc), a)


def _tiff_jpeg2000(tifffile, a, patch, use_patch, **kw):
    """A tiled JPEG 2000 TIFF written by tifffile, through the patch or
    through imagecodecs, leaving the patch as it was found."""
    was = patch._installed
    (patch.install if use_patch else patch.uninstall)()
    try:
        buf = io.BytesIO()
        tifffile.imwrite(buf, a, compression="jpeg2000", tile=(32, 32), **kw)
        return buf.getvalue()
    finally:
        (patch.install if was else patch.uninstall)()


def _first_tile(tifffile, data):
    with tifffile.TiffFile(io.BytesIO(data)) as tif:
        page = tif.pages[0]
        off, n = page.dataoffsets[0], page.databytecounts[0]
    return data[off:off + n]


@pytest.mark.parametrize("dtype,shape,kw", [
    (np.uint8, (64, 80), {}),
    (np.uint16, (64, 80, 3), {"photometric": "rgb"}),
    (np.int16, (64, 80), {}),
    (np.uint16, (3, 64, 80), {"photometric": "rgb",
                              "planarconfig": "separate"}),
    (np.int16, (64, 80), {"compressionargs": {"level": 40}}),
    (np.uint8, (64, 80, 3), {"photometric": "rgb",
                             "compressionargs": {"level": 40}}),
])
@requires_patchable_tifffile
def test_tifffile_patch_writes_raw_codestreams_like_imagecodecs(dtype, shape,
                                                                kw):
    """tifffile asks for codecformat=0 (a raw J2K codestream). 0.4.0's
    patch dropped it and wrapped every tile in JP2 boxes; now the file is
    the one tifffile writes with imagecodecs, byte for byte."""
    tifffile = pytest.importorskip("tifffile")
    from opencodecs import tifffile_patch
    a = _image(shape, dtype, seed=31)
    ours = _tiff_jpeg2000(tifffile, a, tifffile_patch, True, **kw)
    ref = _tiff_jpeg2000(tifffile, a, tifffile_patch, False, **kw)
    assert ours == ref
    assert _first_tile(tifffile, ours)[:4] == b"\xff\x4f\xff\x51"


@requires_patchable_tifffile
def test_tifffile_patch_reads_jp2_tiles_written_by_040(monkeypatch):
    """Files 0.4.0 wrote through the patch hold JP2-boxed tiles. That
    container still reads, through the patch and through imagecodecs.
    (0.4.0 also made them lossy by default; these tiles are lossless so
    the samples can be compared exactly.)"""
    tifffile = pytest.importorskip("tifffile")
    from opencodecs import tifffile_patch
    from opencodecs.codecs import _jpeg2k

    def old_encode(data, level=None, lossless=None, out=None, **kw):
        return _jpeg2k.encode(data, lossless=True, codec="jp2")

    a = _image((64, 80, 3), np.uint16, seed=32)
    was = tifffile_patch._installed
    tifffile_patch.uninstall()
    monkeypatch.setitem(tifffile_patch._OVERRIDES, "jpeg2k_encode",
                        old_encode)
    data = _tiff_jpeg2000(tifffile, a, tifffile_patch, True,
                          photometric="rgb")
    monkeypatch.undo()
    tifffile_patch.uninstall()
    try:
        assert _first_tile(tifffile, data)[4:8] == b"jP  "
        np.testing.assert_array_equal(tifffile.imread(io.BytesIO(data)), a)
        tifffile_patch.install()
        np.testing.assert_array_equal(tifffile.imread(io.BytesIO(data)), a)
    finally:
        (tifffile_patch.install if was else tifffile_patch.uninstall)()


@requires_patchable_tifffile
def test_tifffile_patch_out():
    """decode(out=) fills the caller's array, as imagecodecs does;
    encode(out=) raises instead of being dropped."""
    from opencodecs import tifffile_patch
    a = _image((40, 48, 3), np.uint8, seed=33)
    enc = imagecodecs.jpeg2k_encode(a)
    out = np.empty_like(a)
    got = tifffile_patch.jpeg2k_decode(enc, out=out)
    assert np.shares_memory(got, out)
    np.testing.assert_array_equal(out, a)
    with pytest.raises(TypeError, match="out"):
        tifffile_patch.jpeg2k_encode(a, out=bytearray(1 << 16))


def test_tiff_writer_jpeg2000_is_lossless():
    tifffile = pytest.importorskip("tifffile")
    a = _image((64, 80), np.uint16, seed=16)
    buf = io.BytesIO()
    oc.tiff_imwrite(buf, a, compression="jpeg2000")
    np.testing.assert_array_equal(tifffile.imread(io.BytesIO(buf.getvalue())), a)


# ---------------------------------------------------------------------------
# bitspersample: imagecodecs' bands, and a clear error outside them
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype,bands", [
    (np.uint8, range(1, 9)), (np.int8, range(1, 9)),
    (np.uint16, range(9, 17)), (np.int16, range(9, 17)),
])
def test_bitspersample_in_band_matches_imagecodecs(dtype, bands):
    for bps in bands:
        info = np.iinfo(dtype)
        hi = (1 << (bps - 1)) - 1 if info.min < 0 else (1 << bps) - 1
        lo = -(1 << (bps - 1)) if info.min < 0 else 0
        a = np.random.default_rng(bps).integers(lo, hi, (21, 26),
                                                endpoint=True).astype(dtype)
        ours = oj.encode(a, bitspersample=bps, codecformat="j2k",
                         numthreads=1)
        assert ours == imagecodecs.jpeg2k_encode(
            a, bitspersample=bps, codecformat="j2k", numthreads=1)
        assert oj.decode_info(ours)["precision"] == bps
        got = oj.decode(ours)
        assert got.dtype == a.dtype
        np.testing.assert_array_equal(got, a)


@pytest.mark.parametrize("dtype,bps", [
    (np.uint16, 8), (np.uint16, 1), (np.int16, 8), (np.uint16, 17),
    (np.uint8, 9), (np.uint8, 16), (np.uint8, 0), (np.uint16, 0),
])
def test_bitspersample_outside_band_raises(dtype, bps):
    """imagecodecs writes the dtype's full width here, ignoring the
    value; for uint16 with 8 bits that is the only way to keep the
    decoded dtype, so the request is refused rather than dropped."""
    a = np.ones((16, 16), dtype)
    with pytest.raises(ValueError, match="bitspersample"):
        oj.encode(a, bitspersample=bps)


# ---------------------------------------------------------------------------
# verbose: OpenJPEG's messages, at imagecodecs' thresholds
# ---------------------------------------------------------------------------


def _messages(caplog, name):
    return [r.getMessage() for r in caplog.records if r.name == name]


def test_verbose_logs_openjpeg_errors_like_imagecodecs(caplog):
    import logging
    enc = imagecodecs.jpeg2k_encode(_image((64, 64), np.uint8, seed=30),
                                    codecformat="j2k", numthreads=1)
    bad = enc[:120]
    for verbose in (None, 0, False):
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(Exception):
                oj.decode(bad, verbose=verbose)
        assert _messages(caplog, "opencodecs") == []
    for verbose in (1, 2, 3, True):
        caplog.clear()
        with caplog.at_level(logging.DEBUG):
            with pytest.raises(Exception):
                imagecodecs.jpeg2k_decode(bad, verbose=verbose)
            with pytest.raises(Exception):
                oj.decode(bad, verbose=verbose)
        ref = _messages(caplog, "imagecodecs")
        ours = _messages(caplog, "opencodecs")
        assert ours, verbose
        assert [m for m in ours if m.startswith("JPEG2K error:")] == \
            [m for m in ref if m.startswith("JPEG2K error:")]
        has_info = any(m.startswith("JPEG2K info:") for m in ours)
        assert has_info == (verbose is not True and verbose > 2)


def test_verbose_on_encode_and_codec(caplog):
    import logging
    a = _image((40, 40), np.uint8, seed=31)
    with caplog.at_level(logging.DEBUG):
        enc = oj.encode(a, verbose=3, numthreads=1)
    assert enc == oj.encode(a, numthreads=1)
    assert any(m.startswith("JPEG2K info:")
               for m in _messages(caplog, "opencodecs"))
    caplog.clear()
    with caplog.at_level(logging.DEBUG):
        got = get_codec("jpeg2k").decode(enc, verbose=3)
    np.testing.assert_array_equal(got, a)
    assert any(m.startswith("JPEG2K info:")
               for m in _messages(caplog, "opencodecs"))


def test_encode_out_and_component_limit_raise():
    """imagecodecs.jpeg2k_encode defines out=; opencodecs refuses it on
    encode. Like imagecodecs, encode takes at most 4095 components."""
    from opencodecs.codecs import _jpeg2k
    a = _image((16, 20), np.uint8)
    with pytest.raises(TypeError, match="out"):
        _jpeg2k.encode(a, out=bytearray(1 << 16))
    with pytest.raises(TypeError, match="out"):
        get_codec("jpeg2k").encode(a, out=bytearray(1 << 16))
    ok = np.zeros((4095, 8, 8), np.uint8)
    np.testing.assert_array_equal(
        imagecodecs.jpeg2k_decode(_jpeg2k.encode(ok, planar=True),
                                  planar=True), ok)
    with pytest.raises(Exception, match="4096"):
        imagecodecs.jpeg2k_encode(np.zeros((4096, 8, 8), np.uint8),
                                  planar=True)
    with pytest.raises(_jpeg2k.Jpeg2kError, match="4096"):
        _jpeg2k.encode(np.zeros((4096, 8, 8), np.uint8), planar=True)


@pytest.mark.parametrize("shape,kw", [
    ((42, 40, 5), {}),
    ((18, 40, 5), {"rows_per_strip": 16}),
    ((8, 40, 6), {"rows_per_strip": 2}),
    ((9, 33, 7), {"rows_per_strip": 4}),
    ((40, 48, 5), {"tile": (16, 16)}),
])
def test_tiff_writer_short_strips_with_many_samples(tmp_path, shape, kw):
    """A TIFF strip is (rows, width, samples). With 5 or more samples
    and 4 or fewer rows, the codec's planar=None rule would read it as
    (C, H, W) and write a wrong codestream, so the TIFF writer encodes
    with planar=False. tifffile decoding through imagecodecs is the
    reference."""
    tifffile = pytest.importorskip("tifffile")
    from opencodecs import tifffile_patch
    a = _image(shape, np.uint8)
    p = tmp_path / "many_samples.tif"
    with oc.TiffWriter(p) as w:
        w.write_page(a, compression="jpeg2000", photometric="minisblack",
                     **kw)
    was = tifffile_patch._installed
    tifffile_patch.uninstall()
    try:
        with tifffile.TiffFile(p) as tif:
            page = tif.pages[0]
            fh = tif.filehandle
            fh.seek(page.dataoffsets[-1])
            last = fh.read(page.databytecounts[-1])
            ref = page.asarray()
    finally:
        if was:
            tifffile_patch.install()
    np.testing.assert_array_equal(ref, a)
    np.testing.assert_array_equal(oc.read(p), a)
    # The last segment's SIZ: Xsiz is the image (or tile) width, and
    # Csiz the samples per pixel.
    siz = _marker(last, b"\xff\x51")
    xsiz = int.from_bytes(last[siz + 6:siz + 10], "big")
    csiz = int.from_bytes(last[siz + 38:siz + 40], "big")
    width = kw["tile"][1] if "tile" in kw else shape[1]
    assert (xsiz, csiz) == (width, shape[2])


def test_segment_encoders_read_strips_as_rows_width_samples():
    """encode_segment and bind_segment_encoder take a (rows, width,
    samples) strip; imagecodecs decodes the result to the same array."""
    from opencodecs.core.segment_compression import (
        JPEG2000, bind_segment_encoder, encode_segment)
    a = _image((2, 40, 5), np.uint16)
    for enc in (encode_segment(a, JPEG2000, lossless=True),
                bind_segment_encoder(JPEG2000, lossless=True)(a)):
        np.testing.assert_array_equal(
            imagecodecs.jpeg2k_decode(bytes(enc)), a)
    # An explicit planar=True is still honored.
    planar = np.ascontiguousarray(np.moveaxis(a, -1, 0))
    enc = encode_segment(planar, JPEG2000, lossless=True, planar=True)
    np.testing.assert_array_equal(imagecodecs.jpeg2k_decode(bytes(enc)), a)


# ---------------------------------------------------------------------------
# NDTiffWriter: compression_level is the codec's level
# ---------------------------------------------------------------------------


def _ndtiff_jpeg2000_strip(tmp_path, a, **kw):
    tifffile = pytest.importorskip("tifffile")
    from opencodecs._ndtiff_writer import NDTiffWriter

    with NDTiffWriter(tmp_path, compression="jpeg2000", **kw) as w:
        w.write_frame(a, {"z": 0})
    path = tmp_path / "NDTiffStack.tif"
    with tifffile.TiffFile(path) as tf:
        page = tf.pages[0]
        assert page.compression == 34712
        assert len(page.dataoffsets) == 1
        with open(path, "rb") as fh:
            fh.seek(page.dataoffsets[0])
            strip = fh.read(page.databytecounts[0])
        # tifffile decodes the strip with imagecodecs.
        return strip, page.asarray()


@pytest.mark.parametrize("level", [25, 40])
def test_ndtiff_writer_jpeg2000_level_is_a_psnr_target(tmp_path, level):
    """0.4.0 forced lossless=True and dropped the level; with the level
    now a PSNR target that combination raised on the first frame."""
    a = _image((48, 64), np.uint8, seed=21)
    strip, got = _ndtiff_jpeg2000_strip(
        tmp_path, a, compression_level=level,
        compression_options={"numthreads": 1})
    assert strip == imagecodecs.jpeg2k_encode(a, level=level)
    psnr = _psnr(got, a)
    assert level - 1.0 <= psnr <= level + 5.0, psnr


def test_ndtiff_writer_jpeg2000_without_level_is_lossless(tmp_path):
    a = _image((48, 64), np.uint16, seed=22)
    strip, got = _ndtiff_jpeg2000_strip(tmp_path, a)
    assert strip == imagecodecs.jpeg2k_encode(a)
    np.testing.assert_array_equal(got, a)
    with pytest.raises(ValueError, match="lossless=True"):
        _ndtiff_jpeg2000_strip(tmp_path / "x", a, compression_level=40,
                               compression_options={"lossless": True})
