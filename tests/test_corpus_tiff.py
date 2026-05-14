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
    "cramps.tif",     # 1-bit B&W, NONE (no, wait — it's 8-bit grayscale)
    "jello.tif",      # palette
    "jim___cg.tif",
    "jim___dg.tif",
    "jim___gg.tif",
    "ladoga.tif",
    "pc260001.tif",
    "quad-jpeg.tif",  # JPEG-in-TIFF
    "quad-lzw.tif",   # LSB-first late-change LZW — was a SEGV before
                      # the late-change/LSB-first fix
    "strike.tif",
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
    arr = oc.get_codec("tiff").decode(str(path))
    ref = tifffile.imread(str(path))
    # tifffile may squeeze singleton dims; our decode preserves the
    # (pages, H, W, C) shape. Squeeze both to compare content.
    assert np.array_equal(np.squeeze(arr), np.squeeze(ref)), (
        f"{name}: decoded data mismatch"
    )


# ---------------------------------------------------------------------------
# libtiff_pics — known-unsupported (must raise, not segfault)
# ---------------------------------------------------------------------------

# These exercise features explicitly deferred or out of scope. They
# must raise an exception cleanly, never segfault.
PICS_KNOWN_UNSUPPORTED = {
    "fax2d.tif":    "1-bit packed images (deferred)",
    "g3test.tif":   "1-bit packed images (deferred)",
    "jim___ah.tif": "1-bit packed images (deferred)",
    "text.tif":     "4-bit images (deferred)",
    "off_l16.tif":  "LogLuv compression (out of scope)",
    "off_luv24.tif": "LogLuv compression (out of scope)",
    "off_luv32.tif": "LogLuv compression (out of scope)",
    "smallliz.tif": "Old-JPEG (compression=6) (out of scope)",
    "zackthecat.tif": "Old-JPEG (compression=6) (out of scope)",
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
        "caspian.tif", marks=pytest.mark.xfail(
            reason="PlanarConfig=Separate not yet supported",
            strict=False,
        ),
    ),
    pytest.param(
        "oxford.tif", marks=pytest.mark.xfail(
            reason="PlanarConfig=Separate not yet supported",
            strict=False,
        ),
    ),
    pytest.param(
        "dscf0013.tif", marks=pytest.mark.xfail(
            reason="real-camera TIFF with odd row-padding",
            strict=False,
        ),
    ),
    pytest.param(
        "ycbcr-cat.tif", marks=pytest.mark.xfail(
            reason="YCbCr subsampling not yet supported",
            strict=False,
        ),
    ),
    pytest.param(
        "cramps-tile.tif", marks=pytest.mark.xfail(
            reason="tile-decode dispatch bug — TileOffsets tag handling",
            strict=False,
        ),
    ),
    pytest.param(
        "quad-tile.tif", marks=pytest.mark.xfail(
            reason="tile-decode dispatch bug — TileOffsets tag handling",
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


@pytest.mark.xfail(
    reason="rio-cogeo emits overviews as SubIFDs (tag 330) — our "
           "TiffPyramidReader currently walks only top-level IFDs",
    strict=False,
)
@pytest.mark.skipif(not COG.exists(), reason=_HINT)
def test_cog_pyramid_reader_levels():
    """A real COG should expose a 3-level pyramid via
    TiffPyramidReader (full / /2 / /4 overviews)."""
    path = COG / "image_2000px.tif"
    with oc.TiffPyramidReader(str(path)) as p:
        # rio-cogeo's image_2000px has overviews at 1x, 2x, 4x.
        assert p.n_levels >= 2, (
            f"expected pyramid but got {p.n_levels} level(s)"
        )
        # Level 0 should be the largest
        assert p.shapes[0][0] >= p.shapes[1][0]


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
