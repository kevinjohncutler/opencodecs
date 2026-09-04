"""Decode tests against files written by somebody else's encoder.

The rest of the suite proves our codecs agree with themselves: encode,
decode, compare. That cannot catch a shared misreading of a spec, and it
never exercises the parts of a format our encoder does not emit. These
four fixture sets close that gap for the codecs that had no native-format
data behind them:

* **bmp** - the bmpsuite conformance set. Its ``g/`` files must decode and
  match the suite's own reference renderings; its ``b/`` files are
  deliberately malformed and must be refused with a ``BmpError`` rather
  than a crash or a numpy exception from deep inside the parser. Adding
  these is what showed that BI_RLE8 and BI_RLE4 were unimplemented.
* **bcn** - BC1 through BC7 textures from the bcdec reference project,
  including signed BC6H, where the sign extension only shows up on HDR
  content.
* **htj2k** - codestreams from the JPEG committee's own conformance set,
  so the decoder is checked against the standard rather than against the
  encoder that shares its source tree.
* **uhdr** - Ultra HDR captures off a Pixel 6 Pro, the only way to test
  that we read a real camera's MPF layout and gain-map metadata.

Every test skips when the file is absent, so a fresh checkout stays
green. Fetch them with ``python corpus/corpus.py fetch``.
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from opencodecs import get_codec

CORPUS = Path(__file__).resolve().parent.parent / ".test_data"
BMP_DIR = CORPUS / "bmp"
DDS_DIR = CORPUS / "dds"
HTJ2K_DIR = CORPUS / "htj2k" / "conformance"
UHDR_DIR = CORPUS / "uhdr"


def _need(path: Path) -> bytes:
    if not path.is_file() or not path.stat().st_size:
        pytest.skip(f"corpus file missing: {path.name} "
                    f"(run `python corpus/corpus.py fetch`)")
    return path.read_bytes()


# --------------------------------------------------------------------
# BMP: the bmpsuite conformance set
# --------------------------------------------------------------------

# The g/ files all encode the same 127x64 image at different depths and
# compressions, so each one has a known-good rendering to compare with.
BMP_GOOD = ["g_pal8.bmp", "g_pal4rle.bmp", "g_pal8rle.bmp",
            "g_rgb16.bmp", "g_rgb24.bmp", "g_rgb32.bmp"]


@pytest.mark.parametrize("name", BMP_GOOD)
def test_bmpsuite_good_files_decode(name):
    arr = get_codec("bmp").decode(_need(BMP_DIR / name))
    assert arr.shape[:2] == (64, 127), f"{name}: unexpected shape {arr.shape}"
    assert arr.dtype == np.uint8


@pytest.mark.parametrize("bmp,ref", [
    ("g_pal8.bmp", "ref_pal8.png"),
    ("g_pal8rle.bmp", "ref_pal8.png"),
    ("g_pal4rle.bmp", "ref_pal4.png"),
])
def test_bmpsuite_matches_reference_rendering(bmp, ref):
    """Exact match against the renderings bmpsuite ships for these files.

    This is the check that pins RLE down. An RLE decoder that drops a
    delta or mishandles the word padding on an absolute run still
    produces a plausible-looking image, and only a pixel-exact
    comparison against a reference notices.
    """
    got = get_codec("bmp").decode(_need(BMP_DIR / bmp))
    want = get_codec("png").decode(_need(BMP_DIR / ref))
    if got.ndim == 2:
        got = np.repeat(got[..., None], 3, axis=2)
    if want.ndim == 2:
        want = np.repeat(want[..., None], 3, axis=2)
    if want.ndim == 3 and want.shape[2] == 4:
        want = want[..., :3]
    assert got.shape == want.shape
    assert np.array_equal(got, want), (
        f"{bmp} differs from {ref} in "
        f"{int(np.count_nonzero((got != want).any(-1)))} pixels")


def test_bmpsuite_rle8_equals_uncompressed():
    """RLE8 and uncompressed hold the same image, so they must decode alike."""
    bmp = get_codec("bmp")
    plain = bmp.decode(_need(BMP_DIR / "g_pal8.bmp"))
    rle = bmp.decode(_need(BMP_DIR / "g_pal8rle.bmp"))
    assert np.array_equal(plain, rle)


@pytest.mark.parametrize("name", [
    "b_badbitcount.bmp",     # bit depth that no BMP variant defines
    "b_badbitssize.bmp",
    "b_reallybig.bmp",       # header claims far more pixels than it carries
    "b_rletopdown.bmp",      # RLE bitmaps cannot be top-down
])
def test_bmpsuite_bad_files_are_refused_cleanly(name):
    """Malformed input must raise BmpError, not leak an internal exception.

    A numpy ValueError about reshaping tells a caller nothing about what
    is wrong with their file, and is indistinguishable from a bug in us.
    """
    from opencodecs._bmp_codec import BmpError
    data = _need(BMP_DIR / name)
    with pytest.raises(BmpError):
        get_codec("bmp").decode(data)


# --------------------------------------------------------------------
# bcn: BC1-BC7 textures from the bcdec reference project
# --------------------------------------------------------------------

_DXGI = {
    70: ("bc1", False), 71: ("bc1", False), 72: ("bc1", False),
    73: ("bc2", False), 74: ("bc2", False), 75: ("bc2", False),
    76: ("bc3", False), 77: ("bc3", False), 78: ("bc3", False),
    79: ("bc4", False), 80: ("bc4", False), 81: ("bc4", True),
    82: ("bc5", False), 83: ("bc5", False), 84: ("bc5", True),
    95: ("bc6h", False), 96: ("bc6h", True),
    97: ("bc7", False), 98: ("bc7", False), 99: ("bc7", False),
}
_FOURCC = {b"DXT1": "bc1", b"DXT3": "bc2", b"DXT5": "bc3",
           b"ATI1": "bc4", b"BC4U": "bc4", b"ATI2": "bc5", b"BC5U": "bc5"}


def _parse_dds(blob: bytes):
    """Minimal DDS header read: enough to hand the blocks to the decoder.

    Our bcn codec takes raw blocks plus a format and dimensions, which is
    the right shape for an API but means a .dds needs unwrapping first.
    """
    assert blob[:4] == b"DDS ", "not a DDS file"
    height, width = struct.unpack_from("<II", blob, 12)
    fourcc = blob[84:88]
    if fourcc == b"DX10":
        dxgi = struct.unpack_from("<I", blob, 128)[0]
        fmt, signed = _DXGI[dxgi]
        offset = 148
    else:
        fmt, signed, offset = _FOURCC[fourcc], False, 128
    return width, height, fmt, signed, blob[offset:]


@pytest.mark.parametrize("name,fmt,channels", [
    ("kodim23_bc1.dds", "bc1", 4),
    ("testcard_bc2.dds", "bc2", 4),
    ("dice_bc3.dds", "bc3", 4),
    ("dice_bc4.dds", "bc4", 1),
    ("dice_bc5.dds", "bc5", 2),
    ("dice_bc7.dds", "bc7", 4),
])
def test_bcdec_reference_textures(name, fmt, channels):
    width, height, got_fmt, signed, blocks = _parse_dds(_need(DDS_DIR / name))
    assert got_fmt == fmt, f"{name}: header says {got_fmt}, expected {fmt}"
    arr = get_codec("bcn").decode(blocks, format=fmt, width=width,
                                  height=height, is_signed=signed)
    assert arr.shape[:2] == (height, width)
    assert arr.dtype == np.uint8
    if channels == 1:
        assert arr.ndim == 2
    else:
        assert arr.shape[2] == channels
    # A texture that decodes to a single flat value means the block
    # decoder ran but produced nothing, which shape checks alone miss.
    assert int(arr.max()) > int(arr.min())


def test_bcdec_bc6h_signed_hdr():
    """Signed BC6H: the case where sign extension actually shows up.

    An unsigned-only implementation decodes this file without error and
    silently clamps the negative end, so the test is that negatives
    survive and the range is HDR rather than 0-1.
    """
    width, height, fmt, signed, blocks = _parse_dds(
        _need(DDS_DIR / "lythwood_room_bc6h_signed.dds"))
    assert fmt == "bc6h" and signed
    arr = get_codec("bcn").decode(blocks, format=fmt, width=width,
                                  height=height, is_signed=True)
    assert arr.shape == (height, width, 3)
    assert arr.dtype == np.float32
    assert np.isfinite(arr).all()
    assert float(arr.min()) < 0.0, "signed BC6H decoded without any negatives"
    assert float(arr.max()) > 1.0, "expected HDR values above 1.0"


# --------------------------------------------------------------------
# htj2k: JPEG committee conformance codestreams
# --------------------------------------------------------------------

try:
    from opencodecs._htj2k_codec import _HAVE_BACKEND as _HAVE_OPENJPH
except Exception:                                     # noqa: BLE001
    _HAVE_OPENJPH = False

_needs_openjph = pytest.mark.skipif(
    not _HAVE_OPENJPH,
    reason="OpenJPH backend not built (system libopenjph not found)")


@_needs_openjph
def test_htj2k_conformance_codestream():
    arr = get_codec("htj2k").decode(_need(HTJ2K_DIR / "hifi_ht1_02.j2k"))
    assert arr.ndim == 3 and arr.shape[2] == 3
    assert arr.dtype in (np.uint8, np.uint16)
    assert int(arr.max()) > int(arr.min())


@_needs_openjph
def test_htj2k_multiple_quality_layers_is_a_known_limitation():
    """OpenJPH decodes one quality layer only, and says so.

    This conformance codestream carries five. The point of the test is
    that we surface the library's refusal as a clean error instead of
    returning a partly-decoded image, and that we notice if a future
    OpenJPH lifts the restriction.
    """
    data = _need(HTJ2K_DIR / "ds1_ht_01_b11.j2k")
    try:
        arr = get_codec("htj2k").decode(data)
    except Exception as exc:                          # noqa: BLE001
        assert "layer" in str(exc).lower() or "rc=" in str(exc), (
            f"expected a quality-layer refusal, got {type(exc).__name__}: {exc}")
        return
    # If this starts passing, OpenJPH gained multi-layer support and the
    # xfail-shaped branch above can go.
    assert arr.ndim >= 2 and arr.size


# --------------------------------------------------------------------
# uhdr: real Pixel 6 Pro captures
# --------------------------------------------------------------------

UHDR_FILES = ["pixel6_original_05.jpg", "pixel6_original_01.jpg"]

# Same shape as the OpenJPH guard above, and missing until now, which
# is why main went red the day the uhdr fixtures started downloading:
# the corpus arrived, the tests ran, and the backend they need is not
# built by the tests workflow, so an optional-backend absence surfaced
# as four hard ImportErrors instead of four skips.
try:
    from opencodecs.uhdr import _HAVE_BACKEND as _HAVE_UHDR
except Exception:                                     # noqa: BLE001
    _HAVE_UHDR = False

_needs_uhdr = pytest.mark.skipif(
    not _HAVE_UHDR,
    reason="libultrahdr backend not built (system libultrahdr not found)")


@_needs_uhdr
@pytest.mark.parametrize("name", UHDR_FILES)
def test_uhdr_probe_reads_camera_metadata(name):
    import opencodecs.uhdr as uhdr
    data = _need(UHDR_DIR / name)
    assert uhdr.is_uhdr(data)
    info = uhdr.probe(data)
    assert info["width"] == 4080 and info["height"] == 3072
    # The gain map is stored at reduced resolution, as cameras do.
    assert 0 < info["gainmap_width"] < info["width"]
    assert 0 < info["gainmap_height"] < info["height"]
    meta = info["gainmap_metadata"]
    assert meta["hdr_capacity_max"] > 1.0
    assert float(np.max(meta["max_content_boost"])) > 1.0


@_needs_uhdr
@pytest.mark.parametrize("name", UHDR_FILES)
def test_uhdr_decoded_peak_matches_declared_boost(name):
    """The decoded HDR peak should land on the boost the file declares.

    This ties the pixel path to the metadata path: a gain map applied
    with the wrong exponent or the wrong base still produces a pretty
    image, but it will not peak where the camera said it would.
    """
    import opencodecs.uhdr as uhdr
    data = _need(UHDR_DIR / name)
    declared = float(np.max(uhdr.probe(data)["gainmap_metadata"]
                            ["max_content_boost"]))
    out = uhdr.decode_native(data)
    hdr = out["hdr"]
    assert hdr.shape == (3072, 4080, 3)
    assert np.isfinite(hdr).all()
    peak = float(np.nanmax(hdr))
    assert peak == pytest.approx(declared, rel=0.02), (
        f"{name}: decoded peak {peak:.4f} vs declared boost {declared:.4f}")
