"""TIFF pyramid reader tests + bench-style HTTPS-range demonstration.

Builds COG-style pyramidal TIFFs via tifffile (full-res page + N
reduced-resolution pages marked with SubfileType=1), then exercises
:class:`TiffPyramidReader` for:

  * Level enumeration and dimensions
  * ``best_level_for`` selection
  * Full-level reads
  * Tile-aware region reads (the COG selling point)
  * Region reads over a local HTTP server with HTTPDataSource —
    proves that we fetch only the tiles inside the bbox
"""

from __future__ import annotations

import http.server
import io
import socketserver
import threading
from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc

tifffile = pytest.importorskip("tifffile")

from opencodecs._tiff_pyramid import TiffPyramidReader
from opencodecs._tiff_http import HTTPDataSource


def _need_tiff():
    if not oc.has_codec("tiff"):
        pytest.skip("native TIFF reader not built")


# ---------------------------------------------------------------------------
# Pyramid fixture builder
# ---------------------------------------------------------------------------


def _build_pyramidal_tiff(arr: np.ndarray, levels: int = 3, tile: int = 64) -> bytes:
    """Build a multi-IFD pyramidal TIFF: full-res + N-1 overviews.

    Uses tifffile's TiffWriter with subfiletype=1 on the overviews —
    the COG layout. Levels are 2× downscaled each.
    """
    buf = io.BytesIO()
    with tifffile.TiffWriter(buf) as tw:
        # Level 0: full-resolution main page.
        tw.write(arr, tile=(tile, tile), compression=None,
                 subfiletype=0)
        cur = arr
        for _ in range(levels - 1):
            # Naive 2× downscale (block-average via reshape stride).
            h, w = cur.shape[:2]
            h2 = (h // 2) * 2
            w2 = (w // 2) * 2
            cropped = cur[:h2, :w2]
            if cur.ndim == 2:
                ds = cropped.reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))
            else:
                ds = cropped.reshape(h2 // 2, 2, w2 // 2, 2, -1).mean(axis=(1, 3))
            ds = ds.astype(arr.dtype)
            tw.write(ds, tile=(tile, tile), compression=None,
                     subfiletype=1)   # NewSubfileType bit 0 = overview
            cur = ds
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Basic level enumeration / metadata
# ---------------------------------------------------------------------------


def test_pyramid_enumerates_levels(tmp_path):
    _need_tiff()
    arr = np.arange(512 * 384, dtype=np.uint16).reshape(384, 512)
    data = _build_pyramidal_tiff(arr, levels=3, tile=64)
    with TiffPyramidReader(data) as p:
        assert p.n_levels == 3
        # Level 0 is the full image.
        assert p.level(0).shape == arr.shape
        # Downscale should be 2× (within rounding).
        assert p.level(0).downscale == (1, 1)
        assert p.level(1).downscale == (2, 2)
        assert p.level(2).downscale == (4, 4)
        # Each level's reader has the right shape.
        assert p.level(1).shape == (192, 256)
        assert p.level(2).shape == (96, 128)


def test_pyramid_best_level_for(tmp_path):
    _need_tiff()
    arr = np.arange(1024 * 768, dtype=np.uint16).reshape(768, 1024)
    data = _build_pyramidal_tiff(arr, levels=4, tile=128)
    # Levels: 0=1024×768, 1=512×384, 2=256×192, 3=128×96
    with TiffPyramidReader(data) as p:
        # Full requested → full res.
        assert p.best_level_for(max_pixels_y=10_000, max_pixels_x=10_000) == 0
        # Want max 400 rows → level 1 (height 384) is the highest-
        # resolution level that fits within the envelope.
        assert p.best_level_for(max_pixels_y=400) == 1
        # Want max 100 rows → level 3 (96) is the only one that fits.
        assert p.best_level_for(max_pixels_y=100) == 3


def test_pyramid_dtype_and_shape_props(tmp_path):
    _need_tiff()
    arr = np.zeros((200, 300), dtype=np.uint8)
    data = _build_pyramidal_tiff(arr, levels=2, tile=64)
    with TiffPyramidReader(data) as p:
        assert p.dtype == np.dtype(np.uint8)
        assert len(p.shapes) == 2
        assert p.shapes[0] == (200, 300)


# ---------------------------------------------------------------------------
# Region reads — the COG selling point
# ---------------------------------------------------------------------------


def test_pyramid_read_full_level(tmp_path):
    _need_tiff()
    arr = np.arange(384 * 512, dtype=np.uint16).reshape(384, 512)
    data = _build_pyramidal_tiff(arr, levels=3, tile=64)
    with TiffPyramidReader(data) as p:
        full = p.read_region(level=0)
        assert full.shape == arr.shape
        np.testing.assert_array_equal(full, arr)


def test_pyramid_read_subregion(tmp_path):
    _need_tiff()
    arr = np.arange(512 * 512, dtype=np.uint16).reshape(512, 512)
    data = _build_pyramidal_tiff(arr, levels=3, tile=128)
    with TiffPyramidReader(data) as p:
        # Read a 200×200 patch that crosses tile boundaries.
        # Tiles are 128×128, so y=50..250 spans tile rows 0+1, x same.
        patch = p.read_region(level=0, y=(50, 250), x=(60, 260))
        assert patch.shape == (200, 200)
        np.testing.assert_array_equal(patch, arr[50:250, 60:260])


def test_pyramid_read_region_at_overview_level(tmp_path):
    _need_tiff()
    arr = np.arange(512 * 512, dtype=np.uint16).reshape(512, 512)
    data = _build_pyramidal_tiff(arr, levels=3, tile=128)
    with TiffPyramidReader(data) as p:
        # The level-2 page is 128×128 (the same as the tile).
        full_2 = p.read_region(level=2)
        assert full_2.shape == (128, 128)
        # Subregion of an overview.
        patch = p.read_region(level=2, y=(0, 64), x=(0, 64))
        assert patch.shape == (64, 64)


def test_pyramid_read_region_striped(tmp_path):
    """Striped pyramid pages — non-tiled COG variant."""
    _need_tiff()
    arr = np.arange(300 * 256, dtype=np.uint16).reshape(300, 256)
    buf = io.BytesIO()
    with tifffile.TiffWriter(buf) as tw:
        tw.write(arr, compression=None, rowsperstrip=64, subfiletype=0)
        # 2× downscaled.
        ds = arr[:296, :].reshape(148, 2, 256).mean(axis=1)[:, ::2].astype(arr.dtype)
        tw.write(ds, compression=None, rowsperstrip=64, subfiletype=1)
    data = buf.getvalue()
    with TiffPyramidReader(data) as p:
        assert p.n_levels == 2
        # Sub-region read on the striped full-res page.
        out = p.read_region(level=0, y=(50, 200), x=(20, 200))
        assert out.shape == (150, 180)
        np.testing.assert_array_equal(out, arr[50:200, 20:200])


# ---------------------------------------------------------------------------
# HTTPS pyramid — partial-fetch demonstration
# ---------------------------------------------------------------------------


@pytest.fixture
def http_pyramid(tmp_path):
    """Serve a pyramidal TIFF over a local HTTP server with Range
    support. Yields (HTTPDataSource, expected_array)."""
    arr = np.arange(1024 * 1024, dtype=np.uint16).reshape(1024, 1024)
    path = tmp_path / "pyramid.tif"
    path.write_bytes(_build_pyramidal_tiff(arr, levels=4, tile=128))

    class H(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            rng = self.headers.get("Range")
            data = (Path(self.directory) / Path(self.path).name).read_bytes()
            if rng:
                s, e = rng.split("=", 1)[1].split("-")
                s = int(s); e = int(e) if e else len(data) - 1
                chunk = data[s:e + 1]
                self.send_response(206)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Range", f"bytes {s}-{e}/{len(data)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def log_message(self, *a, **k):
            pass

    server = socketserver.ThreadingTCPServer(
        ("127.0.0.1", 0),
        lambda *a, **kw: H(*a, directory=str(tmp_path), **kw),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        url = f"http://127.0.0.1:{server.server_address[1]}/pyramid.tif"
        yield url, path, arr
    finally:
        server.shutdown()
        server.server_close()


def test_pyramid_overview_via_http_partial_fetch(http_pyramid):
    """Reading the lowest-resolution overview must fetch far less data
    than the full file — proving the pyramid pays off on HTTPS."""
    _need_tiff()
    url, path, arr = http_pyramid
    full_size = path.stat().st_size

    src = HTTPDataSource(url, prefetch_bytes=64 * 1024)
    try:
        with TiffPyramidReader(src) as p:
            overview = p.read_region(level=p.n_levels - 1)
            assert overview.shape == p.level(-1).shape
    finally:
        stats = src.stats
        src.close()

    # Overview level is 128×128 u16 ≈ 32 KB raw + IFD overhead. Even
    # generous fetch (prefetch + IFD scan + the level-3 tiles) should
    # be a tiny fraction of the full file.
    assert stats["bytes_fetched"] < full_size // 4, (
        f"fetched {stats['bytes_fetched']}/{full_size}; overview read "
        f"should be a tiny fraction"
    )


def test_pyramid_region_via_http_minimal_tiles(http_pyramid):
    """Reading a 256×256 region from level 0 fetches only the 4 tiles
    overlapping it — not the whole 1024×1024 image."""
    _need_tiff()
    url, path, arr = http_pyramid
    full_size = path.stat().st_size

    src = HTTPDataSource(url, prefetch_bytes=64 * 1024)
    try:
        with TiffPyramidReader(src) as p:
            patch = p.read_region(level=0, y=(0, 256), x=(0, 256))
            assert patch.shape == (256, 256)
            np.testing.assert_array_equal(patch, arr[:256, :256])
    finally:
        stats = src.stats
        src.close()

    # 256×256 region with 128×128 tiles = 4 tiles ≈ 128 KB raw + IFD
    # walk overhead. Should be way under the full file size.
    assert stats["bytes_fetched"] < full_size // 2, (
        f"fetched {stats['bytes_fetched']}/{full_size}; tile-aware read "
        f"should fetch ~4 tiles + headers, not the whole file"
    )


# ---------------------------------------------------------------------------
# SubIFD-based pyramid (bioformats / OME-TIFF convention)
# ---------------------------------------------------------------------------


def _build_subifd_pyramid_tiff(tmp_path, base_size=512):
    """Build a TIFF whose pyramid is encoded via SubIFDs (tag 330)
    rather than via separate top-level IFDs. This is the layout
    bioformats emits.

    tifffile's ``subifds=N`` argument writes N reduced-resolution
    IFDs attached to the just-written page as sub-IFDs.
    """
    p = tmp_path / "subifd_pyr.tif"
    full = np.arange(base_size * base_size, dtype=np.uint16).reshape(
        base_size, base_size,
    )
    half = full[::2, ::2].copy()
    quarter = full[::4, ::4].copy()
    with tifffile.TiffWriter(str(p)) as tw:
        # Main page + 2 sub-resolutions reachable via SubIFDs.
        tw.write(full, subifds=2, tile=(128, 128), compression=None)
        tw.write(half, subfiletype=1, tile=(128, 128), compression=None)
        tw.write(quarter, subfiletype=1, tile=(128, 128), compression=None)
    return p, [full, half, quarter]


def test_subifd_pyramid_detected_automatically(tmp_path):
    """When IFD 0 has SubIFDs, the pyramid reader uses them as levels
    automatically (no ``ifd_index`` kwarg required)."""
    _need_tiff()
    p, levels = _build_subifd_pyramid_tiff(tmp_path)
    with TiffPyramidReader(p) as r:
        assert r.n_levels == 3
        assert r.level(0).shape == levels[0].shape
        assert r.level(1).shape == levels[1].shape
        assert r.level(2).shape == levels[2].shape
        assert r.level(1).downscale == (2, 2)
        assert r.level(2).downscale == (4, 4)


def test_subifd_pyramid_round_trip_via_region(tmp_path):
    """Each level reads back pixel-equal to source."""
    _need_tiff()
    p, levels = _build_subifd_pyramid_tiff(tmp_path)
    with TiffPyramidReader(p) as r:
        for i, src in enumerate(levels):
            out = r.read_region(i, y=(0, src.shape[0]), x=(0, src.shape[1]))
            np.testing.assert_array_equal(out, src)


def test_subifd_pyramid_explicit_ifd_index(tmp_path):
    """``ifd_index=N`` anchors the pyramid on a specific top-level IFD —
    useful for multi-series files where each IFD has its own SubIFD
    pyramid (one acquisition plane per IFD, levels in SubIFDs)."""
    _need_tiff()
    # Two independent pyramid-IFDs written back-to-back.
    p = tmp_path / "two_series_subifd.tif"
    rng = np.random.default_rng(0)
    p0_full = rng.integers(0, 256, size=(256, 256), dtype=np.uint8)
    p0_half = p0_full[::2, ::2].copy()
    p1_full = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    p1_half = p1_full[::2, ::2].copy()
    with tifffile.TiffWriter(str(p)) as tw:
        tw.write(p0_full, subifds=1, tile=(64, 64), compression=None)
        tw.write(p0_half, subfiletype=1, tile=(64, 64), compression=None)
        tw.write(p1_full, subifds=1, tile=(64, 64), compression=None)
        tw.write(p1_half, subfiletype=1, tile=(64, 64), compression=None)

    # Default auto-detect → anchors on IFD 0 → returns the p0 pyramid
    with TiffPyramidReader(p) as r:
        assert r.n_levels == 2
        assert r.level(0).shape == (256, 256)
        np.testing.assert_array_equal(
            r.read_region(0, y=(0, 256), x=(0, 256)), p0_full,
        )

    # When written with subifds=N, tifffile stores the sub-IFDs OUT
    # of the top-level chain — so n_frames counts only main writes.
    # ifd_index=1 anchors on the second pyramid.
    with TiffPyramidReader(p, ifd_index=1) as r:
        assert r.n_levels == 2
        assert r.level(0).shape == (128, 128)
        np.testing.assert_array_equal(
            r.read_region(0, y=(0, 128), x=(0, 128)), p1_full,
        )


# ---------------------------------------------------------------------------
# Real bioformats OME-TIFF — gated on the test corpus being present
# ---------------------------------------------------------------------------


_REAL_OMETIFF = Path(".test_data/ome_tiff/retina_pyramid.ome.tiff")


@pytest.mark.skipif(
    not _REAL_OMETIFF.exists(),
    reason="run tests/download_test_corpus.sh to enable",
)
def test_real_ometiff_subifd_pyramid_3_levels():
    """The OME-TIFF in our corpus has bioformats's SubIFD pyramid
    layout. With SubIFD support, our reader now exposes 3 levels
    (1567x2048, 783x1024, 391x512) anchored on IFD 0 — series 0 of
    bioformats's terminology."""
    _need_tiff()
    with TiffPyramidReader(str(_REAL_OMETIFF)) as r:
        assert r.n_levels == 3
        assert r.level(0).shape == (1567, 2048)
        assert r.level(1).shape == (783, 1024)
        assert r.level(2).shape == (391, 512)
        assert r.level(1).downscale == (2, 2)
        assert r.level(2).downscale == (4, 4)


@pytest.mark.skipif(
    not _REAL_OMETIFF.exists(),
    reason="run tests/download_test_corpus.sh to enable",
)
def test_real_ometiff_subifd_level_matches_tifffile():
    """Pixels at every level on the real bioformats OME-TIFF must
    match tifffile's per-level decode."""
    _need_tiff()
    import tifffile as tff
    with tff.TiffFile(str(_REAL_OMETIFF)) as tf:
        levels = tf.series[0].levels
        ref = [np.asarray(lvl.asarray()[0, 0]) for lvl in levels]
    with TiffPyramidReader(str(_REAL_OMETIFF)) as r:
        for i, expected in enumerate(ref):
            out = r.read_region(
                i, y=(0, expected.shape[0]), x=(0, expected.shape[1]),
            )
            np.testing.assert_array_equal(
                out, expected,
                err_msg=f"level {i}: pixels disagree with tifffile",
            )
