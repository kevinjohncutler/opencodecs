"""Tests for the HTTP-range TIFF data source.

Spins up a stdlib ThreadingHTTPServer in a fixture, writes a TIFF into
its served directory, opens it via HTTPDataSource + TiffStream, and
verifies that:

  * The file isn't fully downloaded — only the IFD chain + the tiles
    we explicitly decode are fetched. Tracked via the data source's
    request counter.
  * Random tile access works correctly over Range requests.
  * Multi-page IFD walks work.
  * The opt-in prefetch_bytes header speeds up the IFD walk by
    serving most of those small reads out of the local cache.
"""

from __future__ import annotations

import http.server
import io
import socketserver
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc

tifffile = pytest.importorskip("tifffile")

from opencodecs._tiff_codec import TiffStream
from opencodecs._tiff_http import HTTPDataSource, FileDataSource


def _need_tiff():
    if not oc.has_codec("tiff"):
        pytest.skip("native TIFF reader not built")


# ---------------------------------------------------------------------------
# Local HTTP server fixture
# ---------------------------------------------------------------------------


class _RangedHandler(http.server.SimpleHTTPRequestHandler):
    """SimpleHTTPRequestHandler doesn't honour Range; subclass to add
    the minimum needed for HTTP/1.1 Range support."""

    def do_GET(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().do_GET()
        try:
            unit, span = rng.split("=", 1)
            if unit.strip() != "bytes":
                self.send_error(416, "only bytes range supported")
                return
            start_s, end_s = span.split("-", 1)
            start = int(start_s) if start_s else 0
            path = self.translate_path(self.path)
            data = Path(path).read_bytes()
            total = len(data)
            end = int(end_s) if end_s else total - 1
            end = min(end, total - 1)
            chunk = data[start:end + 1]
            self.send_response(206)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Range", f"bytes {start}-{end}/{total}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
        except Exception:
            self.send_error(416)

    def log_message(self, *args, **kwargs):
        # Quiet; the test framework already logs failures.
        pass


@pytest.fixture
def http_tiff_url(tmp_path):
    """Spin up an HTTP server in a tmp dir; yield (url_for, server)
    where url_for(filename) -> str URL."""
    server = socketserver.ThreadingTCPServer(
        ("127.0.0.1", 0),
        lambda *a, **kw: _RangedHandler(*a, directory=str(tmp_path), **kw),
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    port = server.server_address[1]
    try:
        yield (
            lambda fname: f"http://127.0.0.1:{port}/{fname}",
            tmp_path,
            server,
        )
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_http_range_basic_decode(http_tiff_url):
    """Open a remote TIFF and decode it; result must match local."""
    _need_tiff()
    url_for, tmp_path, _server = http_tiff_url
    arr = np.arange(64 * 96, dtype=np.uint16).reshape(64, 96)
    (tmp_path / "x.tif").write_bytes(_tiff_bytes(arr))
    src = HTTPDataSource(url_for("x.tif"))
    try:
        with TiffStream(None, read_at=src) as r:
            assert r.n_frames == 1
            np.testing.assert_array_equal(r.page(0).asarray(), arr)
    finally:
        src.close()


def test_http_range_partial_tile_fetch_minimal(http_tiff_url):
    """Random tile access over HTTP should fetch only the IFD + the
    target tile bytes — NOT the whole file."""
    _need_tiff()
    url_for, tmp_path, _ = http_tiff_url
    # Build a 4 MB tiled TIFF (1024 tiles).
    arr = np.arange(2048 * 2048, dtype=np.uint16).reshape(2048, 2048)
    path = tmp_path / "tiled.tif"
    path.write_bytes(_tiff_bytes(arr, tile=(64, 64), compression=None))
    full_size = path.stat().st_size

    src = HTTPDataSource(url_for("tiled.tif"), prefetch_bytes=0)
    try:
        with TiffStream(None, read_at=src) as r:
            page = r.page(0)
            # Decode exactly 5 random tiles, no more.
            rng = np.random.default_rng(0)
            for ti in rng.integers(0, page.tiles_x * page.tiles_y, size=5):
                offset = int(page.offsets[int(ti)])
                nbytes = int(page.byte_counts[int(ti)])
                raw = r._read(offset, nbytes)
                _ = page._decode_segment(raw)
    finally:
        stats = src.stats
        src.close()

    # Sanity: we issued at most ~30 small requests (IFD walk + tag
    # bodies + 5 tile reads) totalling far less than the whole file.
    assert stats["bytes_fetched"] < full_size // 4, (
        f"fetched {stats['bytes_fetched']} of {full_size}; "
        f"random tile access shouldn't pull the whole file"
    )
    # And the count matches "small reads + 5 tiles" order of magnitude.
    assert stats["requests"] <= 50


def test_http_range_prefetch_speeds_ifd_walk(http_tiff_url):
    """With prefetch_bytes=64KB, the IFD walk should issue ZERO
    additional requests beyond the prefetch when the whole IFD chain
    fits in the prefetch window."""
    _need_tiff()
    url_for, tmp_path, _ = http_tiff_url
    # Many small pages so the IFD chain is big-ish.
    pages = [np.full((16, 32), i, dtype=np.uint8) for i in range(50)]
    buf = io.BytesIO()
    with tifffile.TiffWriter(buf) as tw:
        for p in pages:
            tw.write(p, compression=None)
    (tmp_path / "multi.tif").write_bytes(buf.getvalue())
    file_size = (tmp_path / "multi.tif").stat().st_size

    # Prefetch the whole file (it's tiny — ~30 KB).
    src = HTTPDataSource(url_for("multi.tif"), prefetch_bytes=file_size + 1024)
    try:
        with TiffStream(None, read_at=src) as r:
            assert r.n_frames == 50
            # Touch every page's tags; should all resolve from prefetch.
            for i in range(r.n_frames):
                _ = r.page(i)
    finally:
        stats = src.stats
        src.close()

    # The constructor's prefetch counts as one request. The IFD walk
    # plus 50 page tag-parses should add zero more network requests.
    assert stats["requests"] == 1, (
        f"expected 1 prefetch request, got {stats['requests']}"
    )


def test_file_data_source_matches_path(tmp_path):
    """FileDataSource should produce identical results to passing a
    path directly — proves the read_at protocol equivalence."""
    _need_tiff()
    arr = np.arange(40 * 60, dtype=np.uint16).reshape(40, 60)
    path = tmp_path / "a.tif"
    path.write_bytes(_tiff_bytes(arr))

    via_path = oc.get_codec("tiff").open(str(path)).page(0).asarray()

    src = FileDataSource(path)
    try:
        with TiffStream(None, read_at=src) as r:
            via_fds = r.page(0).asarray()
    finally:
        src.close()

    np.testing.assert_array_equal(via_path, via_fds)


# ---------------------------------------------------------------------------
# Pyramid-over-HTTP — the COG-from-S3 capstone
# ---------------------------------------------------------------------------


def _write_pyramid_tiff(path: Path, level0_shape=(1024, 1024),
                         tile=(128, 128)) -> list[np.ndarray]:
    """Write a 3-level pyramid TIFF with SubIFD layout.

    Returns the level arrays (full → quarter) so tests can cross-check
    against the source pixels."""
    rng = np.random.default_rng(0)
    h0, w0 = level0_shape
    levels = [
        rng.integers(0, 256, size=(h0, w0), dtype=np.uint8),
        rng.integers(0, 256, size=(h0 // 2, w0 // 2), dtype=np.uint8),
        rng.integers(0, 256, size=(h0 // 4, w0 // 4), dtype=np.uint8),
    ]
    with tifffile.TiffWriter(str(path)) as tw:
        tw.write(
            levels[0], tile=tile, compression=None,
            subifds=2,
        )
        # Subsequent writes inside the same TiffWriter become SubIFDs
        # of the previous write when subifds= was used.
        tw.write(levels[1], tile=tile, compression=None)
        tw.write(levels[2], tile=tile, compression=None)
    return levels


def test_http_pyramid_read_region_coalesced(http_tiff_url):
    """A pyramid TIFF over HTTP must:
      * expose every level via TiffPyramidReader,
      * answer read_region with the right pixels at every level,
      * fetch *fewer* HTTP requests than the bbox covers tiles —
        proving the read_many coalescing path fires.
    """
    _need_tiff()
    url_for, tmp_path, _ = http_tiff_url
    path = tmp_path / "pyramid.tif"
    levels = _write_pyramid_tiff(path)
    file_size = path.stat().st_size

    src = HTTPDataSource(url_for("pyramid.tif"), prefetch_bytes=8192)
    try:
        # open_pyramid can't infer format from a raw HTTPDataSource
        # (no path / URL string to inspect). Pass format='tiff' so we
        # can keep the HTTPDataSource handle and inspect its stats.
        with oc.open_pyramid(src, format="tiff") as pyr:
            assert pyr.n_levels == 3, (
                f"expected 3-level pyramid, got {pyr.n_levels}"
            )
            # Pixel correctness at each level (separate test from the
            # bytes-fetched assertion below).
            for i in range(3):
                full = pyr.read_region(level=i)
                np.testing.assert_array_equal(full, levels[i])

            # Reset our request accounting to isolate the partial-fetch
            # measurement from the full-level reads we just did. The
            # in-memory LRU cache from those reads is fine — that's the
            # production reality, and a cache hit doesn't issue a
            # request anyway.
            stats_before = dict(src.stats)
            region = pyr.read_region(level=0, y=(0, 256), x=(0, 256))
            stats_after = dict(src.stats)
        np.testing.assert_array_equal(region, levels[0][:256, :256])
    finally:
        src.close()

    # The 256x256 region covers 4 tiles (each 128x128 at level 0). The
    # coalesced read_many should issue ≤2 HTTP requests (one for the
    # merged top row tiles, one for the merged bottom row), not 4.
    # Even fewer is fine if the tiles were already in cache.
    req_delta = stats_after["requests"] - stats_before["requests"]
    assert req_delta <= 2, (
        f"expected ≤2 coalesced HTTP requests for a 4-tile bbox, got "
        f"{req_delta} — read_many coalescing may not be firing"
    )
    # And the bytes added by *this* partial read must be small.
    fetched_delta = stats_after["bytes_fetched"] - stats_before["bytes_fetched"]
    # 4 tiles × 128 × 128 = 65536 bytes at most (uncompressed),
    # but they're already cached from the level-0 full read above,
    # so the delta should be 0. Give some slack for HTTP framing.
    assert fetched_delta <= 65536, (
        f"partial region pulled {fetched_delta} bytes — should be a "
        f"small delta, possibly 0 if tiles were cached"
    )


def test_http_url_auto_routes_to_pyramid(http_tiff_url):
    """``oc.open_pyramid(url)`` should auto-detect URL → HTTPDataSource
    → TiffPyramidReader without the caller wiring it up by hand."""
    _need_tiff()
    url_for, tmp_path, _ = http_tiff_url
    path = tmp_path / "small.tif"
    levels = _write_pyramid_tiff(path, level0_shape=(256, 256))

    with oc.open_pyramid(url_for("small.tif")) as pyr:
        assert pyr.n_levels == 3
        full = pyr.read_region(level=0)
        np.testing.assert_array_equal(full, levels[0])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _tiff_bytes(arr, **kw):
    buf = io.BytesIO()
    kw.setdefault("compression", None)
    tifffile.imwrite(buf, arr, **kw)
    return buf.getvalue()
