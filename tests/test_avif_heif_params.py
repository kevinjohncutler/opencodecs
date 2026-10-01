"""AVIF and HEIF: bit depth, gray, and the meaning of ``level``.

Three defects these pin, each against something outside our encoder:

* uint16 data defaulted to 10 bits with no range check, so values above
  1023 came back clamped, and an explicit ``bit_depth`` clamped too.
  AV1 (AOMedia AV1 specification 5.5.2, color_config) stores 8, 10 or 12
  bits; data that does not fit must raise, not change.
* Gray input was stored as three equal color planes and every file,
  including real monochrome ones, decoded as RGB. AV1 has mono_chrome
  (4:0:0) and HEVC has chroma_format_idc 0; gray is coded that way now.
  AVIF decodes it to (H, W) or (H, W, 2) as imagecodecs.avif_decode
  does; HEIF decodes it to RGB(A) as imagecodecs.heif_decode does, and
  to gray with imagecodecs' ``photometric='monochrome'``.
* ``level=`` was ignored unless ``lossless=False`` was also passed. In
  imagecodecs a level alone selects lossy output (AVIF lossless at 100,
  HEIF above 100), and that is the meaning here now.

References: imagecodecs' AVIF codec (run in this process; AVIF has no
LERC-style clash), the av1C and colr boxes of the files themselves, and
libheif driven directly through ctypes for HEIF (imagecodecs ships no
HEIF encoder).
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc
from _ic_reference import skip_if_old_imagecodecs  # noqa: E402

pytestmark = skip_if_old_imagecodecs

_avif = pytest.importorskip("opencodecs.codecs._avif")
avif = oc.get_codec("avif")


def _box(blob: bytes, fourcc: bytes) -> bytes:
    i = blob.find(fourcc)
    assert i > 0, fourcc
    return blob[i + 4:]


def _av1c(blob: bytes) -> dict:
    """Fields of the av1C box (AV1 Codec ISO Media File Format 2.3)."""
    b = _box(blob, b"av1C")
    return {"high_bitdepth": (b[2] >> 6) & 1, "twelve_bit": (b[2] >> 5) & 1,
            "monochrome": (b[2] >> 4) & 1, "subsampling": (b[2] >> 2) & 3}


def _nclx(blob: bytes) -> tuple:
    b = _box(blob, b"colr")
    assert b[:4] == b"nclx"
    return tuple(int.from_bytes(b[4 + 2 * i:6 + 2 * i], "big") for i in range(3))


def _imagecodecs_avif():
    ic = pytest.importorskip("imagecodecs")
    if not getattr(ic, "AVIF", None) or not ic.AVIF.available:
        pytest.skip("imagecodecs AVIF backend unavailable")
    return ic


# ---------------------------------------------------------------------------
# AVIF bit depth
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("top,depth", [(1023, 10), (4095, 12)])
def test_avif_uint16_picks_a_depth_that_holds_the_data(top, depth):
    rng = np.random.default_rng(top)
    arr = rng.integers(0, top + 1, (32, 48, 3)).astype(np.uint16)
    arr[0, 0, 0] = top
    blob = avif.encode(arr)
    box = _av1c(blob)
    assert box["high_bitdepth"] == 1 and box["twelve_bit"] == (depth == 12)
    np.testing.assert_array_equal(avif.decode(blob), arr)


def test_avif_uint16_matches_imagecodecs_at_the_same_depth():
    """With the depth we pick, the lossless bytes are imagecodecs' bytes."""
    ic = _imagecodecs_avif()
    rng = np.random.default_rng(5)
    arr = rng.integers(0, 4096, (32, 48, 3)).astype(np.uint16)
    ours = avif.encode(arr)
    assert ours == bytes(ic.avif_encode(arr, bitspersample=12))
    np.testing.assert_array_equal(ic.avif_decode(ours), arr)


def test_avif_refuses_data_it_cannot_store():
    rng = np.random.default_rng(6)
    full = rng.integers(0, 65536, (16, 16, 3)).astype(np.uint16)
    full[0, 0, 0] = 65535
    with pytest.raises(_avif.AvifError, match="12 bits"):
        avif.encode(full)
    twelve = np.full((16, 16, 3), 1024, np.uint16)
    with pytest.raises(_avif.AvifError, match="bit_depth=10"):
        avif.encode(twelve, bit_depth=10)
    # Lossy too: clamping is range loss, not quantization.
    with pytest.raises(_avif.AvifError):
        avif.encode(twelve, bit_depth=10, level=50)
    with pytest.raises(_avif.AvifError):
        avif.encode(full, bit_depth=12)
    with pytest.raises(_avif.AvifError):
        avif.encode(np.zeros((8, 8, 3), np.uint16), bit_depth=8)


# ---------------------------------------------------------------------------
# AVIF gray
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape,dtype,top", [
    ((37, 53), np.uint8, 255), ((37, 53, 2), np.uint8, 255),
    ((32, 48), np.uint16, 1023), ((32, 48, 1), np.uint8, 255)])
def test_avif_gray_is_coded_monochrome_and_keeps_its_shape(shape, dtype, top):
    rng = np.random.default_rng(7)
    arr = rng.integers(0, top + 1, shape).astype(dtype)
    blob = avif.encode(arr)
    assert _av1c(blob)["monochrome"] == 1
    back = avif.decode(blob)
    np.testing.assert_array_equal(back, arr.reshape(back.shape))
    assert back.shape == (shape if shape[-1] != 1 else shape[:2])
    with avif.open(blob) as reader:
        assert reader.shape == back.shape


def test_avif_color_is_not_coded_monochrome():
    arr = np.random.default_rng(8).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert _av1c(avif.encode(arr))["monochrome"] == 0


def test_avif_gray_matches_imagecodecs_both_ways():
    ic = _imagecodecs_avif()
    rng = np.random.default_rng(9)
    gray = rng.integers(0, 256, (37, 53), dtype=np.uint8)
    gray_alpha = rng.integers(0, 256, (33, 47, 2), dtype=np.uint8)
    gray10 = rng.integers(0, 1024, (32, 48)).astype(np.uint16)
    for arr, kwargs in ((gray, {}), (gray_alpha, {}),
                        (gray10, {"bitspersample": 10})):
        theirs = bytes(ic.avif_encode(arr, **kwargs))
        assert _av1c(theirs)["monochrome"] == 1
        np.testing.assert_array_equal(avif.decode(theirs), arr)
        np.testing.assert_array_equal(ic.avif_decode(avif.encode(arr)), arr)


def test_avif_gray_rejects_a_chroma_layout():
    with pytest.raises(_avif.AvifError):
        avif.encode(np.zeros((8, 8), np.uint8), level=50, yuv_format="420")
    with pytest.raises(_avif.AvifError):
        avif.encode(np.zeros((8, 8, 3), np.uint8), yuv_format="420")


# ---------------------------------------------------------------------------
# AVIF level / lossless
# ---------------------------------------------------------------------------


def test_avif_level_alone_is_lossy_like_imagecodecs():
    ic = _imagecodecs_avif()
    rng = np.random.default_rng(10)
    arr = rng.integers(0, 256, (64, 48, 3), dtype=np.uint8)
    lossless = avif.encode(arr)
    # No level and level=100 are lossless, and identical to imagecodecs.
    assert lossless == avif.encode(arr, level=100) == bytes(ic.avif_encode(arr))
    assert lossless == bytes(ic.avif_encode(arr, level=100))
    for level in (30, 90):
        ours = avif.encode(arr, level=level)
        theirs = bytes(ic.avif_encode(arr, level=level))
        assert len(ours) < len(lossless) and len(theirs) < len(lossless)
        assert not np.array_equal(avif.decode(ours), arr)
    assert avif.encode(arr, level=30) == avif.encode(arr, level=30,
                                                     lossless=False)


def test_avif_negative_level_is_libavifs_default_quality():
    """imagecodecs maps level -1 and lower to AVIF_QUALITY_DEFAULT.

    That is libavif's own default quality, not quality 0, which is what
    clamping gave: on a gradient imagecodecs' level=-1 file had maxerr
    12 while ours had 21, the same as level=0.
    """
    ic = _imagecodecs_avif()
    y, x = np.mgrid[:37, :53]
    arr = np.dstack([x * 4, y * 6, (x + y) * 2]).astype(np.uint8)
    default = avif.encode(arr, level=-1)
    assert avif.encode(arr, level=-7) == default
    assert _avif.encode(arr, level=-1) == default
    assert default != avif.encode(arr, level=0)
    theirs = bytes(ic.avif_encode(arr, level=-1))
    ours_err = np.abs(avif.decode(default).astype(int) - arr).max()
    their_err = np.abs(ic.avif_decode(theirs).astype(int) - arr).max()
    zero_err = np.abs(avif.decode(avif.encode(arr, level=0)).astype(int)
                      - arr).max()
    assert ours_err <= their_err + 2, (ours_err, their_err)
    assert ours_err < zero_err
    assert _av1c(default)["subsampling"] == _av1c(theirs)["subsampling"] == 0
    with pytest.raises(ValueError):
        avif.encode(arr, level=-1, lossless=True)


def test_avif_lossless_and_lossy_level_contradict():
    arr = np.zeros((16, 16, 3), np.uint8)
    with pytest.raises(ValueError):
        avif.encode(arr, level=30, lossless=True)
    assert avif.encode(arr, level=100, lossless=True) == avif.encode(arr)


def test_avif_lossy_color_is_tagged_bt601():
    """Lossy output names the matrix libavif converted with (H.273 MC 6).

    Gray has no chroma for a matrix to act on and keeps MC 2
    (unspecified), the tag imagecodecs writes for it.
    """
    arr = np.random.default_rng(11).integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert _nclx(avif.encode(arr, level=50))[2] == 6
    assert _nclx(avif.encode(arr))[2] == 0          # lossless: identity
    gray = arr[..., 0]
    assert _nclx(avif.encode(gray))[2] == 2
    assert _nclx(avif.encode(gray, level=50))[2] == 2
    assert _nclx(avif.encode(gray, matrix=1))[2] == 1


def test_avif_gray_lossless_bytes_match_imagecodecs():
    """Gray used to be tagged MC 6 and came out 1-3 bytes off imagecodecs."""
    ic = _imagecodecs_avif()
    rng = np.random.default_rng(13)
    for arr, kwargs in (
            (rng.integers(0, 256, (37, 53), dtype=np.uint8), {}),
            (rng.integers(0, 256, (33, 47, 2), dtype=np.uint8), {}),
            (rng.integers(0, 4096, (32, 48)).astype(np.uint16),
             {"bitspersample": 12})):
        theirs = bytes(ic.avif_encode(arr, **kwargs))
        assert avif.encode(arr, **kwargs) == theirs
        assert _nclx(theirs)[2] == 2



def test_avif_documented_differences_from_imagecodecs():
    """The two differences the docs state, checked against imagecodecs.

    imagecodecs codes gray AVIF lossless whatever ``level`` says, and
    here the level is honored. imagecodecs does not tile by default and
    this codec tiles 4x4 from 1024 px on the long axis, so only
    ``tilelog2=(0, 0)`` gives the untiled stream there.
    """
    ic = _imagecodecs_avif()
    rng = np.random.default_rng(15)
    gray = rng.integers(0, 256, (40, 56), dtype=np.uint8)
    np.testing.assert_array_equal(
        ic.avif_decode(ic.avif_encode(gray, level=30)), gray)
    ours = avif.decode(avif.encode(gray, level=30))
    assert not np.array_equal(ours, gray)
    np.testing.assert_array_equal(avif.decode(avif.encode(gray)), gray)

    wide = rng.integers(0, 256, (40, 1030), dtype=np.uint8)
    untiled = avif.encode(wide, tilelog2=(0, 0))
    assert avif.encode(wide) != untiled
    assert bytes(ic.avif_encode(wide)) == bytes(
        ic.avif_encode(wide, tilelog2=(0, 0)))
    for blob in (avif.encode(wide), untiled):
        np.testing.assert_array_equal(avif.decode(blob), wide)
    np.testing.assert_array_equal(ic.avif_decode(untiled), wide)


def _sharp_rgba():
    y, x = np.mgrid[:64, :96]
    rgb = np.zeros((64, 96, 3), np.uint8)
    rgb[..., 0] = np.where((x // 8 + y // 8) % 2, 255, 0)
    rgb[..., 1] = x * 2
    rgb[..., 2] = 255 - y * 3
    alpha = np.random.default_rng(14).integers(0, 256, (64, 96), dtype=np.uint8)
    return np.dstack([rgb, alpha])


def test_avif_lossy_defaults_follow_imagecodecs():
    """Lossy color is 4:4:4 and alpha lossless, as imagecodecs codes them.

    The defaults were 4:2:0 color and alpha at the color quality, and the
    codec wrapper's speed was 0 where the Cython codec's was 6. A red
    checkerboard then lost up to 219 levels at level=90 (imagecodecs: 16).
    The two packages link different libavif builds, so lossy bytes are
    compared through the av1C box and the decoded error, not byte for
    byte.
    """
    ic = _imagecodecs_avif()
    rgba = _sharp_rgba()
    rgb = np.ascontiguousarray(rgba[..., :3])
    for level in (30, 60, 90):
        ours = avif.encode(rgba, level=level)
        theirs = bytes(ic.avif_encode(rgba, level=level))
        assert _av1c(ours)["subsampling"] == _av1c(theirs)["subsampling"] == 0
        back = avif.decode(ours)
        np.testing.assert_array_equal(back[..., 3], rgba[..., 3])
        np.testing.assert_array_equal(ic.avif_decode(theirs)[..., 3],
                                      rgba[..., 3])
        err = np.abs(back[..., :3].astype(int) - rgb).max()
        ref = np.abs(ic.avif_decode(theirs)[..., :3].astype(int) - rgb).max()
        assert err <= ref + 4, (level, err, ref)
    # The codec wrapper and the Cython codec now share one default.
    assert avif.encode(rgb, level=60) == _avif.encode(rgb, level=60)
    # Subsampling is still there when asked for.
    assert _av1c(avif.encode(rgb, level=60, yuv_format="420"))["subsampling"] == 3


def test_avif_speed_is_clamped_like_imagecodecs():
    """imagecodecs clamps speed to 0..10; we used to drop it silently."""
    rgb = np.ascontiguousarray(_sharp_rgba()[:32, :32, :3])
    assert avif.encode(rgb, level=60, speed=20) == avif.encode(
        rgb, level=60, speed=10)
    assert avif.encode(rgb, level=60, speed=-5) == avif.encode(
        rgb, level=60, speed=0)


def test_avif_imagecodecs_keywords_and_unknown_options():
    rng = np.random.default_rng(12)
    arr = rng.integers(0, 1024, (16, 16, 3)).astype(np.uint16)
    assert avif.encode(arr, bitspersample=12) == avif.encode(arr, bit_depth=12)
    rgb = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    assert (avif.encode(rgb, level=50, pixelformat="yuv444")
            == avif.encode(rgb, level=50, yuv_format="444"))
    assert _av1c(avif.encode(rgb, level=50, pixelformat="444"))["subsampling"] == 0
    with pytest.raises(ValueError):
        avif.encode(arr, bit_depth=10, bitspersample=12)
    with pytest.raises(TypeError):
        avif.encode(rgb, bogus=1)


# ---------------------------------------------------------------------------
# HEIF
# ---------------------------------------------------------------------------


_heif = pytest.importorskip("opencodecs.codecs._heif")
heif = oc.get_codec("heif")


def _heif_can_encode():
    try:
        heif.encode(np.zeros((16, 16, 3), np.uint8))
    except Exception:
        return False
    return True


needs_heif_encoder = pytest.mark.skipif(
    not _heif_can_encode(), reason="libheif has no HEVC encoder here")


@needs_heif_encoder
@pytest.mark.parametrize("shape,dtype,top", [
    ((37, 53), np.uint8, 255), ((37, 53, 2), np.uint8, 255),
    ((32, 48), np.uint16, 1023), ((32, 48, 2), np.uint16, 4095),
    ((33, 47, 3), np.uint8, 255), ((33, 47, 4), np.uint16, 4095)])
def test_heif_lossless_round_trip_keeps_values_and_shape(tmp_path, shape,
                                                         dtype, top):
    from _libheif_writer import primary_image_layout
    rng = np.random.default_rng(20)
    arr = rng.integers(0, top + 1, shape).astype(dtype)
    arr.flat[0] = top
    blob = heif.encode(arr)
    gray = arr.ndim == 2 or arr.shape[2] == 2
    back = heif.decode(blob, photometric="monochrome" if gray else None)
    assert back.shape == arr.shape and back.dtype == arr.dtype
    np.testing.assert_array_equal(back, arr)
    if gray:
        # imagecodecs.heif_decode's default for a monochrome image: RGB(A)
        # with the gray plane in every color channel.
        planes = [arr] if arr.ndim == 2 else [arr[..., 0]]
        expected = np.dstack(planes * 3 + ([arr[..., 1]] if arr.ndim == 3
                                           else []))
        np.testing.assert_array_equal(heif.decode(blob), expected)
    path = tmp_path / "x.heic"
    path.write_bytes(blob)
    layout = primary_image_layout(path)
    if layout is not None:
        mono, alpha, bits = layout
        assert mono == (arr.ndim == 2 or arr.shape[2] == 2)
        assert alpha == (arr.ndim == 3 and arr.shape[2] in (2, 4))
        assert bits == (8 if dtype == np.uint8 else (10 if top < 1024 else 12))


@needs_heif_encoder
def test_heif_reads_libheif_monochrome(tmp_path):
    """A gray HEIF written by libheif itself, decoded both ways.

    imagecodecs.heif_decode returns a monochrome image as RGB unless
    ``photometric`` asks for monochrome; that is the default here too.
    """
    from _libheif_writer import write_monochrome_heif, primary_image_layout
    gray = np.random.default_rng(21).integers(0, 256, (37, 53), dtype=np.uint8)
    path = tmp_path / "gray.heic"
    if not write_monochrome_heif(path, gray):
        pytest.skip("libheif could not write a monochrome image here")
    assert primary_image_layout(path)[0] is True
    blob = path.read_bytes()
    rgb = np.dstack([gray] * 3)
    np.testing.assert_array_equal(heif.decode(blob), rgb)
    np.testing.assert_array_equal(
        heif.decode(blob, photometric="monochrome"), gray)
    with heif.open(blob) as reader:
        assert reader.shape == (37, 53, 3)
        np.testing.assert_array_equal(reader[0], rgb)
    with heif.open(blob, photometric="monochrome") as reader:
        assert reader.shape == (37, 53)
        np.testing.assert_array_equal(reader[0], gray)


@needs_heif_encoder
def test_heif_decode_photometric_follows_imagecodecs(tmp_path):
    """Every value imagecodecs.heif_decode accepts, and nothing else.

    imagecodecs takes a libheif colorspace number (0 YCbCr, 1 RGB,
    2 monochrome, 99 undefined) or a name; only monochrome changes the
    output. The keyword used to be swallowed by ``**opts``.
    """
    from _libheif_writer import write_monochrome_heif
    gray = np.random.default_rng(23).integers(0, 256, (16, 24), dtype=np.uint8)
    path = tmp_path / "gray.heic"
    if not write_monochrome_heif(path, gray):
        pytest.skip("libheif could not write a monochrome image here")
    blob = path.read_bytes()
    rgb = np.dstack([gray] * 3)
    for value in (None, "rgb", "RGB", "ycbcr", 0, 1, 99):
        np.testing.assert_array_equal(
            heif.decode(blob, photometric=value), rgb, err_msg=repr(value))
    for value in ("monochrome", "gray", "minisblack", "MINISWHITE", 2):
        np.testing.assert_array_equal(
            heif.decode(blob, photometric=value), gray, err_msg=repr(value))
    for value in ("cmyk", "palette", 3, 5, 1.5):
        with pytest.raises(ValueError):
            heif.decode(blob, photometric=value)
    with pytest.raises(TypeError):
        heif.decode(blob, bogus=1)
    # A color image has no gray plane; imagecodecs returns its red
    # channel, which is not a gray image, so this raises instead.
    color = heif.encode(np.random.default_rng(24).integers(
        0, 256, (16, 24, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="color"):
        heif.decode(color, photometric="monochrome")


@needs_heif_encoder
@pytest.mark.parametrize("shape", [(9, 20), (33, 47)])
@pytest.mark.parametrize("ybits,abits", [(10, 8), (8, 10), (8, 8),
                                         (10, 10), (12, 12)])
def test_heif_alpha_at_its_own_depth(tmp_path, shape, ybits, abits):
    """HEIF codes alpha as its own image, so its depth can differ.

    libheif writes the fixture plane by plane. A mismatched alpha used to
    come back as the wrong values (an 8-bit alpha read as 16-bit words
    reached 65517) or as a numpy broadcast error, depending on the
    width; it now raises HeifError on every path. Matching depths decode
    exactly both ways.
    """
    from _libheif_writer import (write_planes_heif, CHANNEL_Y,
                                 CHANNEL_ALPHA)
    rng = np.random.default_rng(25)
    y = rng.integers(0, 1 << ybits, shape)
    a = rng.integers(0, 1 << abits, shape)
    path = tmp_path / "ya.heic"
    if not write_planes_heif(path, [(CHANNEL_Y, y, ybits),
                                    (CHANNEL_ALPHA, a, abits)]):
        pytest.skip("libheif could not write this image here")
    blob = path.read_bytes()
    if ybits != abits:
        for photometric in (None, "monochrome"):
            with pytest.raises(_heif.HeifError, match="alpha plane"):
                heif.decode(blob, photometric=photometric)
        return
    dtype = np.uint8 if ybits <= 8 else np.uint16
    np.testing.assert_array_equal(
        heif.decode(blob, photometric="monochrome"),
        np.dstack([y, a]).astype(dtype))
    np.testing.assert_array_equal(heif.decode(blob),
                                  np.dstack([y, y, y, a]).astype(dtype))


@needs_heif_encoder
@pytest.mark.parametrize("ybits,abits", [(10, 8), (8, 10), (8, 8)])
def test_heif_color_alpha_at_its_own_depth(tmp_path, ybits, abits):
    """The same check on a color image, where libheif's RGBA conversion
    kept the wrong bits of a 10-bit alpha under 8-bit color."""
    from _libheif_writer import (write_planes_heif, CHANNEL_Y, CHANNEL_CB,
                                 CHANNEL_CR, CHANNEL_ALPHA)
    rng = np.random.default_rng(26)
    shape = (12, 20)
    planes = [(ch, rng.integers(0, 1 << ybits, shape), ybits)
              for ch in (CHANNEL_Y, CHANNEL_CB, CHANNEL_CR)]
    a = rng.integers(0, 1 << abits, shape)
    path = tmp_path / "rgba.heic"
    if not write_planes_heif(path, planes + [(CHANNEL_ALPHA, a, abits)],
                             color=True):
        pytest.skip("libheif could not write this image here")
    blob = path.read_bytes()
    if ybits != abits:
        # Where libheif reports the alpha plane's depth (1.21, which the
        # wheels bundle), decode refuses the mismatch. libheif 1.23 does
        # not report it and rescales the alpha to the image's depth itself:
        # up by bit replication, down by dropping low bits.
        try:
            got = heif.decode(blob)
        except _heif.HeifError as exc:
            assert "alpha plane" in str(exc)
            return
        if abits < ybits:
            d = ybits - abits
            rescaled = (a << d) | (a >> (abits - d))
        else:
            rescaled = a >> (abits - ybits)
        np.testing.assert_array_equal(got[..., 3], rescaled)
        return
    np.testing.assert_array_equal(heif.decode(blob)[..., 3], a)


@needs_heif_encoder
def test_heif_refuses_data_it_cannot_store():
    full = np.full((16, 16, 3), 65535, np.uint16)
    with pytest.raises(_heif.HeifError, match="12 bits"):
        heif.encode(full)
    with pytest.raises(_heif.HeifError, match="bit_depth=10"):
        heif.encode(np.full((16, 16, 3), 2000, np.uint16), bit_depth=10)


def _hvcc_chroma_format(blob: bytes) -> int:
    """chroma_format_idc of the hvcC box (ISO/IEC 14496-15 8.3.3.1.2)."""
    return _box(blob, b"hvcC")[16] & 3


@needs_heif_encoder
def test_heif_color_is_coded_444_lossy_too():
    """imagecodecs sets x265's chroma to 4:4:4 for every color encode.

    Lossy color here used to be x265's 4:2:0 default, which smears sharp
    color edges; gray stays monochrome (chroma_format_idc 0).
    """
    rgba = _sharp_rgba()
    rgb = np.ascontiguousarray(rgba[..., :3])
    for arr in (rgb, rgba):
        assert _hvcc_chroma_format(heif.encode(arr)) == 3
        for level in (30, 90):
            lossy = heif.encode(arr, level=level)
            assert _hvcc_chroma_format(lossy) == 3
    assert _hvcc_chroma_format(heif.encode(rgb[..., 0], level=50)) == 0
    err = np.abs(heif.decode(heif.encode(rgb, level=90)).astype(int) - rgb)
    assert err.max() < 16


@needs_heif_encoder
def test_heif_level_follows_imagecodecs():
    """imagecodecs.heif_encode: lossless when level is None or > 100."""
    rng = np.random.default_rng(22)
    arr = rng.integers(0, 256, (64, 48, 3), dtype=np.uint8)
    lossless = heif.encode(arr)
    np.testing.assert_array_equal(heif.decode(lossless), arr)
    assert heif.encode(arr, level=101) == lossless
    for level in (30, 100):
        lossy = heif.encode(arr, level=level)
        assert len(lossy) < len(lossless)
        assert not np.array_equal(heif.decode(lossy), arr)
    with pytest.raises(ValueError):
        heif.encode(arr, level=100, lossless=True)
    assert heif.encode(arr, bitspersample=8) == lossless
    with pytest.raises(TypeError):
        heif.encode(arr, bogus=1)
    with pytest.raises(ValueError):
        heif.encode(arr, compression="av1")
    with pytest.raises(ValueError):
        heif.encode(arr, photometric="monochrome")
    assert heif.encode(arr, photometric="rgb", compression="hevc") == lossless


@needs_heif_encoder
def test_heif_encode_takes_imagecodecs_enum_values():
    """compression and photometric as libheif integers, as imagecodecs.

    imagecodecs' heif_encode accepts heif_compression_format and
    heif_colorspace values; 1 is HEVC and 0, 1, 2, 99 are YCbCr, RGB,
    monochrome and undefined. The linked libheif is asked what format 1
    is, rather than trusting our own constant.
    """
    from _libheif_writer import encoder_name_for_format
    name = encoder_name_for_format(1)
    if name is None:
        pytest.skip("libheif not loadable through ctypes")
    assert "hevc" in name.lower() or "x265" in name.lower(), name

    rng = np.random.default_rng(23)
    rgb = rng.integers(0, 256, (32, 40, 3), dtype=np.uint8)
    gray = np.ascontiguousarray(rgb[..., 0])
    lossless = heif.encode(rgb)
    gray_lossless = heif.encode(gray)
    assert b"hvcC" in lossless
    assert heif.encode(rgb, compression=1) == lossless
    assert heif.encode(rgb, compression=np.int32(1)) == lossless
    for value in (0, 1, 99, "RGB", "rgba", "YCbCr"):
        assert heif.encode(rgb, photometric=value) == lossless, value
    for value in (2, 99, "minisblack", "MINISWHITE", "gray"):
        assert heif.encode(gray, photometric=value) == gray_lossless, value
    np.testing.assert_array_equal(
        heif.decode(heif.encode(gray, photometric=2), photometric=2), gray)
    # A value that disagrees with the array, an unknown one, a bool and
    # any format other than HEVC raise rather than being ignored.
    for value in (2, "monochrome"):
        with pytest.raises(ValueError):
            heif.encode(rgb, photometric=value)
    for value in (0, 1, "rgb"):
        with pytest.raises(ValueError):
            heif.encode(gray, photometric=value)
    for value in (3, True, "separated"):
        with pytest.raises(ValueError):
            heif.encode(rgb, photometric=value)
    for value in (0, 2, 4, True, "av1"):
        with pytest.raises(ValueError):
            heif.encode(rgb, compression=value)
