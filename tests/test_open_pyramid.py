"""Tests for the top-level ``opencodecs.open_pyramid`` dispatch helper.

Coverage:
* TIFF / COG local path → TiffPyramidReader
* TIFF / COG HTTP URL → TiffPyramidReader + HTTPDataSource (range
  requests for tile-level random access)
* Format hints (``format="tiff"``) override extension sniffing
* Read region of a sub-bbox returns only those tiles' contents
"""

from __future__ import annotations

import http.server
import os
import socketserver
import tempfile
import threading

import numpy as np
import pytest

import opencodecs as oc


def _make_pyramid_tiff(path: str, full: np.ndarray) -> None:
    """Write a 3-level pyramidal tiled TIFF (full / /2 / /4)."""
    half = full[::2, ::2]
    qtr = full[::4, ::4]
    with oc.TiffWriter(path) as w:
        w.write_page(
            full, tile=(256, 256), compression="zstd",
            photometric="minisblack",
        )
        w.write_page(
            half, tile=(256, 256), compression="zstd", subfiletype=1,
        )
        w.write_page(
            qtr,  tile=(256, 256), compression="zstd", subfiletype=1,
        )


@pytest.fixture
def pyramid_path(tmp_path):
    p = str(tmp_path / "cog.tif")
    full = np.arange(1024 * 1024, dtype=np.uint16).reshape(1024, 1024)
    _make_pyramid_tiff(p, full)
    return p, full


def test_open_pyramid_local_path(pyramid_path):
    path, full = pyramid_path
    with oc.open_pyramid(path) as p:
        assert isinstance(p, oc.TiffPyramidReader)
        assert p.n_levels == 3
        assert p.shapes == ((1024, 1024), (512, 512), (256, 256))


def test_open_pyramid_format_override(pyramid_path, tmp_path):
    # Rename to a non-tiff extension and pass format= explicitly
    path, full = pyramid_path
    weird = str(tmp_path / "data.bin")
    os.rename(path, weird)
    with oc.open_pyramid(weird, format="tiff") as p:
        assert p.n_levels == 3
        crop = p.read_region(0, y=(0, 100), x=(0, 100))
        assert np.array_equal(crop.squeeze(), full[:100, :100])


def test_open_pyramid_random_access_crop(pyramid_path):
    path, full = pyramid_path
    with oc.open_pyramid(path) as p:
        crop = p.read_region(0, y=(500, 700), x=(300, 600))
        assert np.array_equal(crop.squeeze(), full[500:700, 300:600])


def test_open_pyramid_best_level(pyramid_path):
    path, _ = pyramid_path
    with oc.open_pyramid(path) as p:
        # Want max 600 px on Y axis — L1 (512) fits, L0 (1024) doesn't
        L = p.best_level_for(max_pixels_y=600)
        assert L == 1


def test_open_pyramid_rejects_unknown_format(tmp_path):
    # No extension hint, no format= override
    weird = str(tmp_path / "data.bin")
    with open(weird, "wb") as f:
        f.write(b"\x00" * 16)
    with pytest.raises(ValueError):
        oc.open_pyramid(weird)


# ---------------------------------------------------------------------------
# HTTP path — serve a real COG over a local HTTP server and verify
# range-request tile fetching works end-to-end.
# ---------------------------------------------------------------------------


@pytest.fixture
def http_pyramid_server(tmp_path):
    """Spin up a tiny HTTPServer rooted at tmp_path serving cog.tif."""
    full = np.arange(2048 * 2048, dtype=np.uint16).reshape(2048, 2048)
    path = str(tmp_path / "cog.tif")
    _make_pyramid_tiff(path, full)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(tmp_path), **kw)

    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{port}/cog.tif", full
    finally:
        httpd.shutdown()


def test_open_pyramid_http_url(http_pyramid_server):
    url, full = http_pyramid_server
    with oc.open_pyramid(url) as p:
        assert isinstance(p, oc.TiffPyramidReader)
        assert p.n_levels == 3
        # Tile-aware bbox crop via HTTP Range requests
        crop = p.read_region(0, y=(500, 550), x=(500, 550))
        assert np.array_equal(crop.squeeze(), full[500:550, 500:550])


def test_open_pyramid_http_passes_http_opts(http_pyramid_server):
    url, _ = http_pyramid_server
    # http_opts kwarg should reach HTTPDataSource without raising
    with oc.open_pyramid(url, http_opts={"prefetch_bytes": 4096}) as p:
        assert p.n_levels == 3
