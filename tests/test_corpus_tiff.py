"""TIFF decoder validation against real-world TIFF corpora.

Four corpora:
  * **libtiff_pics**: canonical TIFF compatibility set (25 files).
    Covers compression variants, bit depths, color types, strip vs
    tile, big-endian vs little-endian.
  * **rio-cogeo fixtures**: real Cloud-Optimized GeoTIFFs.
  * **GDAL GeoTIFF samples**: small/medium GeoTIFFs (tile + strip).
  * **OpenSlide Aperio SVS**: pathology whole-slide image (TIFF +
    Aperio extensions).

Tests are skipif-gated on file presence — run
``bash tests/download_test_corpus.sh --light`` from the repo root.

A subset of the corpus exercises code paths we don't yet support
(1-bit images, LogLuv, Old-JPEG=6, YCbCr subsampling, PlanarConfig=
Separate). Those cases are tested for **graceful failure** (raise
NotImplementedError or ValueError, NOT segfault).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc

CORPUS = Path(__file__).resolve().parent.parent / ".test_data" / "tiff"
PICS = CORPUS / "libtiff_pics"
COG = CORPUS / "cog"
GEOTIFF = CORPUS / "geotiff"
WSI = CORPUS / "wsi"

_HINT = (
    "Run `bash tests/download_test_corpus.sh --light` from the repo "
    "root to populate the TIFF corpus."
)


def _filter_macos_resource_forks(paths):
    """Strip macOS SMB resource-fork files (``._foo.tif``)."""
    return [p for p in paths if not p.name.startswith("._")]


# ---------------------------------------------------------------------------
# libtiff_pics — must-decode list
# ---------------------------------------------------------------------------

# These files use compression variants and color types we DO support.
# Cross-validated against tifffile. The rest of the corpus is tested
# separately for graceful failure.
PICS_MUST_DECODE = [
    "cramps.tif",       # 8-bit grayscale, PACKBITS
    "jello.tif",        # palette
    "jim___cg.tif",
    "jim___dg.tif",
    "jim___gg.tif",
    "jim___ah.tif",     # bilevel (1-bit), NONE
    "ladoga.tif",
    "pc260001.tif",
    "quad-jpeg.tif",    # JPEG-in-TIFF
    "quad-lzw.tif",     # LSB-first late-change LZW — was a SEGV before
                        # the late-change/LSB-first fix
    "strike.tif",
    # Tier 5 session 2 additions (parser/decoder gaps closed):
    "caspian.tif",      # PlanarConfig=Separate, ADOBE_DEFLATE, float64
    "oxford.tif",       # PlanarConfig=Separate, LZW, uint8
    "cramps-tile.tif",  # TileWidth+TileLength but strip-tagged (legacy)
    "quad-tile.tif",    # ditto, LZW
]


@pytest.mark.skipif(not PICS.exists(), reason=_HINT)
@pytest.mark.parametrize("name", PICS_MUST_DECODE)
def test_libtiff_pics_must_decode(name):
    """Each listed file MUST decode without error and the result MUST
    byte-match tifffile."""
    tifffile = pytest.importorskip("tifffile")
    path = PICS / name
    if not path.exists():
        pytest.skip(f"{name} missing from corpus")
    arr = np.squeeze(oc.get_codec("tiff").decode(str(path)))
    ref = np.squeeze(tifffile.imread(str(path)))
    # tifffile exposes PlanarConfig=Separate files as (C, H, W); our
    # API normalizes everything to standard (H, W, C). Transpose ref
    # to match when shapes disagree by exactly that permutation.
    if arr.ndim == 3 and ref.ndim == 3 and arr.shape != ref.shape:
        if arr.shape == ref.shape[1:] + (ref.shape[0],):
            ref = np.transpose(ref, (1, 2, 0))
    assert np.array_equal(arr, ref), f"{name}: decoded data mismatch"


# ---------------------------------------------------------------------------
# libtiff_pics — known-unsupported (must raise, not segfault)
# ---------------------------------------------------------------------------

# These exercise features explicitly deferred or out of scope. They
# must raise an exception cleanly, never segfault.
PICS_KNOWN_UNSUPPORTED = {
    "fax2d.tif":    "CCITT Fax 3 (compression=3) — legacy, deferred",
    "g3test.tif":   "CCITT Fax 3 (compression=3) — legacy, deferred",
    "text.tif":     "Thunderscan + 4-bit images — legacy, out of scope",
    "off_l16.tif":  "LogLuv compression (34676) — legacy, out of scope",
    "off_luv24.tif": "LogLuv compression (34677) — legacy, out of scope",
    "off_luv32.tif": "LogLuv compression (34676) — legacy, out of scope",
    "smallliz.tif": "Old-JPEG (compression=6) — even tifffile fails",
    "zackthecat.tif": "Old-JPEG (compression=6) — even tifffile fails",
}


@pytest.mark.skipif(not PICS.exists(), reason=_HINT)
@pytest.mark.parametrize(
    "name", sorted(PICS_KNOWN_UNSUPPORTED.keys()),
    ids=lambda n: n,
)
def test_libtiff_pics_known_unsupported_raises(name):
    """Known-unsupported features must raise — graceful failure, NOT
    silent corruption or segfault. The SEGV in quad-lzw.tif before
    the late-change LZW fix was the kind of failure we want to
    prevent."""
    path = PICS / name
    if not path.exists():
        pytest.skip(f"{name} missing from corpus")
    with pytest.raises((NotImplementedError, ValueError)):
        oc.get_codec("tiff").decode(str(path))


# ---------------------------------------------------------------------------
# libtiff_pics — known limitations with real-data parser bugs
# ---------------------------------------------------------------------------

# These reveal real parser gaps (PlanarConfig=Separate, YCbCr
# subsampling, tile layout edge cases). They xfail until the
# underlying parser is extended.
PICS_KNOWN_BUGS = [
    pytest.param(
        "dscf0013.tif", marks=pytest.mark.xfail(
            reason="real-camera TIFF with odd row-padding — even tifffile "
                   "raises NotImplementedError on this one",
            strict=False,
        ),
    ),
    pytest.param(
        "ycbcr-cat.tif", marks=pytest.mark.xfail(
            reason="LZW + YCbCr chroma subsampling — even tifffile "
                   "raises NotImplementedError on this combination",
            strict=False,
        ),
    ),
]


@pytest.mark.skipif(not PICS.exists(), reason=_HINT)
@pytest.mark.parametrize("name", PICS_KNOWN_BUGS)
def test_libtiff_pics_known_bugs(name):
    """Files whose decode is BROKEN (not just unsupported). xfail
    markers serve as a regression detector — if these start passing,
    the corresponding parser limitation got fixed and we should
    remove the marker."""
    tifffile = pytest.importorskip("tifffile")
    path = PICS / name
    arr = oc.get_codec("tiff").decode(str(path))
    ref = tifffile.imread(str(path))
    assert np.array_equal(np.squeeze(arr), np.squeeze(ref))


# ---------------------------------------------------------------------------
# Cloud-Optimized GeoTIFF (rio-cogeo fixtures)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not COG.exists(), reason=_HINT)
def test_cog_image_2000px_decodes():
    """Real COG with full overview pyramid (3-level). Validates that
    our TIFF reader handles COG layout (overviews first, then header)
    and decodes the full-res IFD."""
    tifffile = pytest.importorskip("tifffile")
    path = COG / "image_2000px.tif"
    arr = oc.get_codec("tiff").decode(str(path))
    ref = tifffile.imread(str(path))
    assert np.array_equal(np.squeeze(arr), np.squeeze(ref))


@pytest.mark.skipif(not COG.exists(), reason=_HINT)
def test_cog_image_colormap_decodes():
    """COG with a palette/colormap. Decodes the palette indices."""
    tifffile = pytest.importorskip("tifffile")
    path = COG / "image_colormap.tif"
    arr = oc.get_codec("tiff").decode(str(path))
    ref = tifffile.imread(str(path))
    assert np.array_equal(np.squeeze(arr), np.squeeze(ref))


@pytest.mark.skipif(not COG.exists(), reason=_HINT)
def test_cog_image_2000px_single_level_reader():
    """rio-cogeo's image_2000px.tif is a tiled COG layout but does NOT
    carry overviews — verified by inspecting it with tifffile (1 IFD,
    no SubIFDs). TiffPyramidReader should report exactly one level for
    it. (Real multi-level pyramids are tested via the retina_pyramid
    OME-TIFF fixture and the Aperio SVS fixture below.)"""
    path = COG / "image_2000px.tif"
    with oc.TiffPyramidReader(str(path)) as p:
        assert p.n_levels == 1, (
            f"image_2000px.tif has no overviews — expected 1 level, "
            f"got {p.n_levels}"
        )
        assert p.shapes[0] == (1500, 1500, 3)


# ---------------------------------------------------------------------------
# Real pyramid TIFFs in the corpus
# ---------------------------------------------------------------------------


OMETIFF_PYRAMID = (CORPUS / ".." / "ome_tiff" / "retina_pyramid.ome.tiff").resolve()


@pytest.mark.skipif(not OMETIFF_PYRAMID.exists(), reason=_HINT)
def test_real_ometiff_pyramid_levels_via_subifds():
    """retina_pyramid.ome.tiff is bioformats' SubIFD-layout pyramid:
    each top-level IFD has 2 sub-IFDs at /2, /4 resolution. The
    TiffPyramidReader should expose all 3 levels of the first scene."""
    with oc.TiffPyramidReader(str(OMETIFF_PYRAMID)) as p:
        assert p.n_levels == 3, (
            f"retina_pyramid: expected 3 levels (full + /2 + /4), got {p.n_levels}"
        )
        # Largest first.
        assert p.shapes[0][0] > p.shapes[1][0] > p.shapes[2][0]


@pytest.mark.skipif(not WSI.exists(), reason=_HINT)
def test_aperio_svs_pyramid_excludes_label_and_thumbnail():
    """Aperio SVS files store the slide label, macro, and thumbnail
    alongside the pyramid levels. Our pyramid reader must include
    page 0 (full image) and any reduced-resolution overviews but
    exclude the thumbnail (NSFT=0 but smaller than page 0) and the
    label (NSFT bit 3 set)."""
    path = WSI / "CMU-1-Small-Region.svs"
    with oc.TiffPyramidReader(str(path)) as p:
        # CMU-1-Small-Region has page 0 (full) + page 2 (macro reduced
        # overview); page 1 is a separate "main" thumbnail and page 3
        # is the Aperio label (NSFT=9). Pyramid levels = [page 0, page 2].
        assert p.n_levels == 2, (
            f"expected 2 pyramid levels (full + macro), got {p.n_levels}: "
            f"shapes={p.shapes}"
        )
        # First level is the full slide; second is the macro overview.
        assert p.shapes[0][0] > p.shapes[1][0]
        # Neither should be the 431x1280 label or 768x574 thumbnail.
        for s in p.shapes:
            assert s != (431, 1280, 3), "label page should be filtered"


# ---------------------------------------------------------------------------
# GeoTIFF / DEM
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not GEOTIFF.exists(), reason=_HINT)
def test_geotiff_i30dem_decodes():
    """USGS strip-based DEM (elevation float32). Strip-layout, no
    compression — exercises the GeoTIFF strip path."""
    tifffile = pytest.importorskip("tifffile")
    path = GEOTIFF / "i30dem.tif"
    arr = oc.get_codec("tiff").decode(str(path))
    ref = tifffile.imread(str(path))
    assert np.array_equal(np.squeeze(arr), np.squeeze(ref))


# ---------------------------------------------------------------------------
# Aperio SVS pathology WSI
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    reason="Aperio SVS tiles are JPEG-compressed with YCbCr photometric. "
           "We decode the JPEG bytes but don't apply the YCbCr->RGB "
           "color conversion (TIFF tag 262 photometric=6).",
    strict=False,
)
@pytest.mark.skipif(not WSI.exists(), reason=_HINT)
def test_aperio_svs_first_page_decodes():
    """SVS is a multi-page TIFF with Aperio-specific extensions.
    The full multi-page decode trips on per-IFD shape variation
    (each pyramid level is a different size), but the first IFD
    (level 0) should decode cleanly via TiffStream."""
    tifffile = pytest.importorskip("tifffile")
    path = WSI / "CMU-1-Small-Region.svs"
    with oc.get_codec("tiff").open(str(path)) as stream:
        page0 = stream.page(0).asarray()
        with tifffile.TiffFile(str(path)) as tf:
            ref = tf.pages[0].asarray()
    assert np.array_equal(np.squeeze(page0), np.squeeze(ref))


@pytest.mark.skipif(not WSI.exists(), reason=_HINT)
def test_aperio_svs_is_multi_page():
    """Aperio SVS files have a pyramid stored as top-level IFDs (the
    "old style" before SubIFD was standard). The TiffStream reader
    should expose them."""
    path = WSI / "CMU-1-Small-Region.svs"
    with oc.get_codec("tiff").open(str(path)) as stream:
        # Aperio CMU-1-Small-Region typically has 2 IFDs:
        # a full-res slide image + a thumbnail
        assert stream.n_frames >= 2
