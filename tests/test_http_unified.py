"""Unified HTTPS readers — CZI + JXL via the same patterns as TIFF / NDTiff.

CziReader: accepts a ``buffer=`` parameter for non-local sources, plus
a ``from_http(url)`` convenience that fetches the whole file.

JxlReader: ``opencodecs.jxl.open_http(url)`` fetches and decodes.
"""

from __future__ import annotations

import http.server
import io
import socketserver
import threading
from pathlib import Path

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Tiny HTTP server (same shape as test_tiff_http.py)
# ---------------------------------------------------------------------------


@pytest.fixture
def http_server(tmp_path):
    """Spin up a stdlib ThreadingHTTPServer that serves tmp_path.

    Yields (url_for, tmp_path, server). Range requests are honored
    for parity with HTTPDataSource's tile-fetch path.
    """

    class H(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            rng = self.headers.get("Range")
            data = (Path(self.directory) / Path(self.path).name).read_bytes()
            if rng:
                s, e = rng.split("=", 1)[1].split("-")
                s = int(s)
                e = int(e) if e else len(data) - 1
                chunk = data[s:e + 1]
                self.send_response(206)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Range",
                                 f"bytes {s}-{e}/{len(data)}")
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
        yield (
            lambda fname: f"http://127.0.0.1:{server.server_address[1]}/{fname}",
            tmp_path,
            server,
        )
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# CZI over HTTPS
# ---------------------------------------------------------------------------


def test_czi_from_http_round_trip(http_server):
    """Write a tiny synthetic CZI, serve it over HTTP, open via from_http."""
    import opencodecs as oc
    if not oc.has_codec("czi"):
        pytest.skip("CZI codec not registered")

    from _czi_fixture import czi_bytes
    from opencodecs._czi_reader import CziReader

    url_for, tmp_path, _ = http_server
    arr = np.arange(64 * 96, dtype=np.uint16).reshape(64, 96)
    czi = czi_bytes(arr)
    (tmp_path / "x.czi").write_bytes(czi)

    with CziReader.from_http(url_for("x.czi")) as r:
        assert r.n_frames >= 1
        decoded = r[0]
        np.testing.assert_array_equal(decoded, arr)


def test_czi_buffer_constructor(http_server, tmp_path):
    """CziReader accepts a buffer= directly; should match path-based read."""
    import opencodecs as oc
    if not oc.has_codec("czi"):
        pytest.skip("CZI codec not registered")

    from _czi_fixture import czi_bytes
    from opencodecs._czi_reader import CziReader

    arr = np.arange(32 * 48, dtype=np.uint16).reshape(32, 48)
    czi = czi_bytes(arr)
    p = tmp_path / "y.czi"
    p.write_bytes(czi)

    via_path = CziReader(p)[0]
    via_buffer = CziReader(buffer=czi)[0]
    np.testing.assert_array_equal(via_path, via_buffer)


# ---------------------------------------------------------------------------
# JXL over HTTPS
# ---------------------------------------------------------------------------


def test_jxl_open_http(http_server):
    """opencodecs.jxl.open_http should fetch + decode a remote .jxl file."""
    import opencodecs as oc
    if not oc.has_codec("jxl"):
        pytest.skip("JXL codec not built")

    from opencodecs import jxl

    url_for, tmp_path, _ = http_server
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(64, 96, 3), dtype=np.uint8)
    jxl_bytes = jxl.encode(arr, lossless=True)
    (tmp_path / "y.jxl").write_bytes(jxl_bytes)

    with jxl.open_http(url_for("y.jxl")) as r:
        frames = list(r.iter_frames())
    assert len(frames) == 1
    np.testing.assert_array_equal(frames[0], arr)
