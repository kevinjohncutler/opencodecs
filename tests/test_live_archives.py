"""Live smoke tests against public scientific-imaging archives.

These tests hit real endpoints (NASA GSFC FITS samples, EMBL-EBI IDR
OME-Zarr) to verify the Tier 3 streaming readers work end-to-end on
production data — not just our synthetic fixtures. Each test:

* Probes its endpoint with a 5-second HEAD before doing real work and
  skips cleanly when the network is unavailable (offline dev,
  firewalled CI, archive down for maintenance).
* Uses small, stable resources (< 1 MB) to keep wall-clock cost low.
* Doesn't assert on exact byte counts — archives recompress or
  reorganize over time, which would break tight assertions without
  reflecting a real regression on our side.
* Marked ``slow``: ``pytest -m "not slow"`` skips them, ``-m slow``
  runs them alone. Default ``pytest`` runs include them so a passing
  ``pytest`` means the cloud paths still work.

The point is to catch the class of bugs synthetic tests can't:
HTTPS retries on intermittent failures, ``Content-Type`` quirks,
``Content-Length`` discrepancies, server-side recompression, etc.

DICOMweb live testing is deferred to a follow-up: IDC's WADO-RS
endpoint URL pattern is in flux as the v3 API rolls out, and a
stale test there would be more noise than signal.
"""

from __future__ import annotations

import urllib.error
import urllib.request

import numpy as np
import pytest


def _network_reachable(url: str, timeout: float = 5.0) -> bool:
    """HEAD-probe ``url`` and return whether it's reachable. Used as
    the skip predicate so the tests don't hang when the archive is
    down or we're offline. Any 2xx response counts; redirects (3xx)
    do NOT — we want the exact resource available so the test that
    follows actually has bytes to fetch."""
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return 200 <= r.status < 300
    except (urllib.error.URLError, urllib.error.HTTPError, OSError):
        return False


# ---------------------------------------------------------------------------
# FITS — NASA GSFC small sample (700 KB HST WFPC2 image)
# ---------------------------------------------------------------------------


_FITS_URL = "https://fits.gsfc.nasa.gov/samples/WFPC2u5780205r_c0fx.fits"


@pytest.mark.slow
@pytest.mark.skipif(
    not _network_reachable(_FITS_URL),
    reason="NASA GSFC FITS samples archive unreachable",
)
def test_live_fits_open_and_read_image_via_http():
    """Open a real NASA-hosted FITS over HTTP, walk the HDU chain,
    decode the primary (or first data-bearing) image."""
    from opencodecs._fits import FitsStream
    from opencodecs._tiff_http import HTTPDataSource

    # Open with adaptive prefetch ON — this is exactly the workload
    # the adaptive trigger was designed for (sequential small reads
    # walking the HDU header chain).
    ds = HTTPDataSource(
        _FITS_URL,
        prefetch_bytes=64 * 1024,
        adaptive_window=64 * 1024,
    )
    try:
        with FitsStream(ds) as fs:
            assert fs.n_hdus >= 1, "real FITS file should have >= 1 HDU"
            # Headers must parse with standard BITPIX values.
            for hdu in fs._hdus:
                bitpix = int(hdu.header.get("BITPIX", 0))
                assert bitpix in (8, 16, 32, 64, -32, -64), (
                    f"unexpected BITPIX={bitpix} in real FITS"
                )
            # Decode the first data-bearing HDU.
            data_hdus = [h for h in fs._hdus if h.header.get("NAXIS", 0)]
            assert data_hdus, "real FITS file should have at least one image HDU"
            arr = data_hdus[0].asarray()
            assert isinstance(arr, np.ndarray)
            assert arr.size > 0
    finally:
        ds.close()


def test_live_fits_partial_read_pulls_less_than_whole_file():
    """The HTTP-range path should NOT download the whole file just to
    open it. Verifies the streaming-thesis claim on real data."""
    if not _network_reachable(_FITS_URL):
        pytest.skip("NASA GSFC FITS samples archive unreachable")
    from opencodecs._fits import FitsStream
    from opencodecs._tiff_http import HTTPDataSource

    ds = HTTPDataSource(_FITS_URL, prefetch_bytes=64 * 1024)
    try:
        with FitsStream(ds) as fs:
            # Just open + iterate headers (no asarray on big HDUs).
            for h in fs._hdus:
                _ = h.header
        # Headers should fit in well under the whole file (~700 KB).
        # HEAD + initial 64KB prefetch + a few more small range reads
        # at worst.
        full_size = ds._total_size or 700_000
        assert ds._total_bytes_fetched < full_size * 0.5, (
            f"opened FITS pulled {ds._total_bytes_fetched} bytes "
            f"of {full_size} ({100 * ds._total_bytes_fetched / full_size:.1f}%)"
        )
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# OME-Zarr — EMBL-EBI IDR (Image Data Resource)
# ---------------------------------------------------------------------------


# IDR hosts thousands of OME-Zarr datasets. idr0062A's image 6001240 is
# one of the smaller stable ones. Shape (2, 236, 275, 271) uint16 with
# Blosc/lz4 compression — the full multi-resolution pyramid is ~30 MB.
_OMEZARR_BASE = (
    "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"
)


@pytest.mark.slow
@pytest.mark.skipif(
    not _network_reachable(_OMEZARR_BASE + "/0/0/0/0/0"),
    reason="IDR OME-Zarr archive unreachable",
)
def test_live_omezarr_open_and_decode_chunk():
    """Open an OME-Zarr v0.4 from IDR's public S3 mirror, decode one
    chunk via the Blosc/lz4 codec path."""
    from opencodecs._omezarr import OmeZarrArray, _HttpStore

    # Resolution level 0 is the full-resolution array. ``_HttpStore``
    # handles GET-per-key + LRU caching internally.
    arr = OmeZarrArray(store=_HttpStore(_OMEZARR_BASE + "/0"))
    # idr0062A is a 4-D dataset.
    assert len(arr.shape) == 4
    assert arr.dtype.kind in ("u", "i", "f"), f"odd dtype {arr.dtype}"
    # Read a small region from the first chunk via OmeZarrArray's
    # slice path (this exercises codec dispatch + chunk decode end-
    # to-end).
    region = arr.read_region((slice(0, 1), slice(0, 1),
                              slice(0, 32), slice(0, 32)))
    assert region.dtype == arr.dtype
    assert region.size > 0
    # IDR images contain real cell data, not noise — the region
    # should have *some* variance, not be all zeros.
    assert region.min() != region.max(), (
        "IDR chunk decoded to a flat array — codec dispatch likely wrong"
    )
