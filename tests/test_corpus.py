"""Tests against the public reference corpus in ``.test_data/``.

These tests use real-world files downloaded from public repositories
(see ``.test_data/README.md`` for sources). They are gated on file
presence — when the corpus isn't downloaded, the tests cleanly skip
with a hint about the download script.

To enable them locally::

    bash tests/download_test_corpus.sh
    pytest tests/test_corpus.py

Each format has its own coverage strategy:

* CZI pyramid — full metadata cross-validation against czifile. The
  same logic is also covered in ``test_czi_pyramid.py`` but is
  re-tested here as a corpus sanity check.
* OME-TIFF pyramid (bioformats output) — exercises decode on a real
  bioformats-emitted file, including the SubIFD-based pyramid layout
  that we don't yet fully expose through ``TiffPyramidReader``.
* OME-Zarr (IDR partial sample) — exercises ``OmeZarrArray`` on a
  real NGFF v0.4 dataset with blosc(lz4) compression.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest


CORPUS_ROOT = Path(".test_data")
CZI_OME_PYRAMID = CORPUS_ROOT / "czi" / "ome_axioscan_pyramid.czi"
OMETIFF_PYRAMID = CORPUS_ROOT / "ome_tiff" / "retina_pyramid.ome.tiff"
OMEZARR_SAMPLE = CORPUS_ROOT / "ome_zarr" / "idr_sample.zarr"

_CORPUS_HINT = (
    "Corpus file missing. Run `bash tests/download_test_corpus.sh` "
    "to download the public reference samples."
)


# ---------------------------------------------------------------------------
# OME-TIFF pyramid (bioformats output)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not OMETIFF_PYRAMID.exists(), reason=_CORPUS_HINT)
def test_real_pyramid_ometiff_basic_read():
    """The retina pyramid OME-TIFF is a 192-IFD file emitted by
    bioformats with SubIFD-based pyramid storage. Our TiffStream reader
    walks the top-level IFD chain; the SubIFD pyramid levels aren't
    exposed yet, but the top-level decode must work.
    """
    import opencodecs as oc
    if not oc.has_codec("tiff"):
        pytest.skip("tiff codec not built")

    with oc.get_codec("tiff").open(str(OMETIFF_PYRAMID)) as r:
        assert r.n_frames > 0
        page = r.page(0)
        assert page.shape == (1567, 2048)
        assert page.dtype == np.dtype("uint8")
        # bioformats writes Adobe Deflate (32946) or plain Deflate (8)
        # depending on the version; both decode through the same code path.
        assert page.compression in (8, 32946)
        # Smoke test: decode the first page in full
        out = page.asarray()
        assert out.shape == (1567, 2048)
        assert out.dtype == np.dtype("uint8")


@pytest.mark.skipif(not OMETIFF_PYRAMID.exists(), reason=_CORPUS_HINT)
def test_real_pyramid_ometiff_matches_tifffile():
    """Per-page decode of the real OME-TIFF must agree with tifffile,
    pixel-for-pixel, on the first IFD."""
    tifffile = pytest.importorskip("tifffile")
    import opencodecs as oc
    if not oc.has_codec("tiff"):
        pytest.skip("tiff codec not built")

    with oc.get_codec("tiff").open(str(OMETIFF_PYRAMID)) as r:
        oc_page0 = r.page(0).asarray()
    with tifffile.TiffFile(str(OMETIFF_PYRAMID)) as tf:
        tf_page0 = tf.pages[0].asarray()
    np.testing.assert_array_equal(oc_page0, tf_page0)


# ---------------------------------------------------------------------------
# OME-Zarr (partial IDR sample)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not OMEZARR_SAMPLE.exists(), reason=_CORPUS_HINT)
def test_real_idr_omezarr_partial_read():
    """The IDR-cached subset has chunks for (c=0,1, z={0,100,200,235}).
    Read each one and confirm it decodes to non-zero data through our
    OmeZarrArray's blosc-lz4 codepath."""
    from opencodecs._omezarr import OmeZarrArray

    arr = OmeZarrArray(OMEZARR_SAMPLE / "2")
    assert arr.shape == (2, 236, 68, 67)
    assert arr.dtype == np.dtype("uint16")
    assert arr.chunks == (1, 1, 68, 67)

    for c in (0, 1):
        for z in (0, 100, 200, 235):
            out = arr.read_region((slice(c, c + 1), slice(z, z + 1),
                                    slice(0, 68), slice(0, 67)))
            assert out.shape == (1, 1, 68, 67)
            assert out.dtype == np.dtype("uint16")
            assert int(out.sum()) > 0, (
                f"chunk c={c} z={z} decoded to all-zero — likely a "
                f"decompression bug, since IDR samples are non-empty"
            )


@pytest.mark.skipif(not OMEZARR_SAMPLE.exists(), reason=_CORPUS_HINT)
def test_real_idr_omezarr_matches_zarr_python():
    """Same data via zarr-python (the reference reader) — pixels equal."""
    zarr = pytest.importorskip("zarr")
    from opencodecs._omezarr import OmeZarrArray

    oc_arr = OmeZarrArray(OMEZARR_SAMPLE / "2")
    # Use zarr-python to read the same array; it will use whatever
    # store backend matches the on-disk layout.
    z = zarr.open(str(OMEZARR_SAMPLE / "2"), mode="r", zarr_format=2)
    for c in (0, 1):
        for z_idx in (0, 100, 200, 235):
            oc_chunk = np.squeeze(oc_arr.read_region(
                (slice(c, c + 1), slice(z_idx, z_idx + 1),
                 slice(0, 68), slice(0, 67))
            ))
            zp_chunk = np.squeeze(np.asarray(z[c, z_idx, :, :]))
            np.testing.assert_array_equal(oc_chunk, zp_chunk)


# ---------------------------------------------------------------------------
# CZI pyramid (covered in test_czi_pyramid.py too; replicated here as
# a corpus sanity check)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not CZI_OME_PYRAMID.exists(), reason=_CORPUS_HINT)
def test_real_pyramid_czi_corpus_metadata_quick_check():
    """Quick smoke test on the public pyramid CZI — full validation is
    in test_czi_pyramid.py. This one just confirms the corpus file is
    intact and our reader sees the expected pyramid structure."""
    import opencodecs as oc
    if not oc.has_codec("czi"):
        pytest.skip("czi codec not built")
    r = oc.get_codec("czi").open(str(CZI_OME_PYRAMID))
    try:
        assert r.is_pyramidal
        assert r.scale_factors_per_level() == [
            (1.0, 1.0), (2.0, 2.0), (4.0, 4.0),
            (8.0, 8.0), (16.0, 16.0), (32.0, 32.0),
        ]
        assert len(r.entries) == 481
    finally:
        r.close()
