"""Benchmark: COG-pattern reads over HTTP Range vs full download.

Runs against a localhost ThreadingHTTPServer for reproducibility. The
wall-clock on loopback is *misleading* — there's no bandwidth
bottleneck — so the real headline metric is **bytes fetched**, not
elapsed time.

Over any non-LAN connection (bandwidth-limited, or RTT > 1 ms), the
range-read path scales with the actual data we touch, while the
full-download path scales with file size. So a 1% bytes-fetched
ratio translates to a ~100× wall-clock win as soon as bandwidth or
latency becomes a real cost.

Run with:

    python bench/bench_tiff_http.py
"""

from __future__ import annotations

import http.server
import io
import socketserver
import statistics
import tempfile
import threading
import time
from pathlib import Path

import numpy as np
import tifffile

import opencodecs as oc
from opencodecs._tiff_codec import TiffStream
from opencodecs._tiff_http import HTTPDataSource


def _serve(directory: Path):
    class H(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            rng = self.headers.get("Range")
            data = (Path(directory) / Path(self.path).name).read_bytes()
            if rng:
                s, e = rng.split("=")[1].split("-")
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
        lambda *a, **kw: H(*a, directory=str(directory), **kw),
    )
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server


def _bench(fn, n=5):
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts) * 1000


def main():
    tmp = Path(tempfile.mkdtemp())
    arr = np.random.default_rng(0).integers(
        0, 4096, size=(8192, 8192), dtype=np.uint16,
    )
    path = tmp / "big_cog.tif"
    tifffile.imwrite(str(path), arr, tile=(256, 256), compression="zstd")
    file_size = path.stat().st_size

    print(f"Test file: {arr.shape} {arr.dtype}  raw={arr.nbytes/1e6:.1f} MB  "
          f"on-disk={file_size/1e6:.1f} MB  (zstd-tiled, 256×256)")

    server = _serve(tmp)
    url = f"http://127.0.0.1:{server.server_address[1]}/big_cog.tif"

    def via_http_range():
        src = HTTPDataSource(url, prefetch_bytes=64 * 1024)
        try:
            with TiffStream(None, read_at=src) as r:
                page = r.page(0)
                rng = np.random.default_rng(0)
                for ti in rng.integers(
                        0, page.tiles_x * page.tiles_y, size=10):
                    offset = int(page.offsets[int(ti)])
                    nbytes = int(page.byte_counts[int(ti)])
                    raw = r._read(offset, nbytes)
                    page._decode_segment(raw)
        finally:
            src.close()

    def via_full_download():
        import urllib.request
        data = urllib.request.urlopen(url).read()
        with oc.get_codec("tiff").open(data) as r:
            page = r.page(0)
            rng = np.random.default_rng(0)
            for ti in rng.integers(
                    0, page.tiles_x * page.tiles_y, size=10):
                offset = int(page.offsets[int(ti)])
                nbytes = int(page.byte_counts[int(ti)])
                raw = r._read(offset, nbytes)
                page._decode_segment(raw)

    http_t = _bench(via_http_range, n=3)
    full_t = _bench(via_full_download, n=3)

    # One more run to capture stats.
    src = HTTPDataSource(url, prefetch_bytes=64 * 1024)
    with TiffStream(None, read_at=src) as r:
        page = r.page(0)
        rng = np.random.default_rng(0)
        for ti in rng.integers(0, page.tiles_x * page.tiles_y, size=10):
            offset = int(page.offsets[int(ti)])
            nbytes = int(page.byte_counts[int(ti)])
            raw = r._read(offset, nbytes)
            page._decode_segment(raw)
    stats = src.stats
    src.close()

    print()
    print("Workload: open + decode 10 random tiles")
    print(f"  HTTP Range:    {http_t:7.1f} ms wall  | "
          f"requests={stats['requests']}  bytes={stats['bytes_fetched']/1e6:.2f} MB "
          f"({stats['bytes_fetched']/file_size*100:.1f}% of file)")
    print(f"  Full download: {full_t:7.1f} ms wall  | "
          f"bytes={file_size/1e6:.1f} MB (100% of file)")
    print()
    print(f"Bytes-fetched ratio: {file_size / stats['bytes_fetched']:.0f}x less data")
    print(
        "On loopback, wall-clock favours the full download because\n"
        "  there's no bandwidth bottleneck. Over any real network the\n"
        "  range path wins by approximately the bytes-fetched ratio."
    )

    server.shutdown()


if __name__ == "__main__":
    main()
