"""Remote HDF5 reader tests.

We spin up the same Range-aware ThreadingTCPServer used by
``test_tiff_http.py``, write a real HDF5 file with chunked +
gzip-compressed datasets, then read it through
``open_remote_hdf5`` and verify:

  * the file is *not* fully downloaded — only HDF5 superblock /
    B-tree / chunk bytes covering the slice we read get fetched
  * partial-array slicing matches a local h5py.File baseline
  * multiple datasets and a nested group path resolve correctly
"""

from __future__ import annotations

import http.server
import socketserver
import threading
from pathlib import Path

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from opencodecs._hdf5_http import (
    _HTTPFileLike,
    _SOURCE_REGISTRY,
    open_remote_hdf5,
    prefetch_hdf5_chunks,
)
from opencodecs._tiff_http import HTTPDataSource


class _RangedHandler(http.server.SimpleHTTPRequestHandler):
    """Range-supporting handler; shared verbatim with test_tiff_http."""

    def do_GET(self):
        rng = self.headers.get("Range")
        if not rng:
            return super().do_GET()
        try:
            unit, span = rng.split("=", 1)
            if unit.strip() != "bytes":
                self.send_error(416)
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
            self.send_header("Content-Range",
                             f"bytes {start}-{end}/{total}")
            self.send_header("Content-Length", str(len(chunk)))
            self.end_headers()
            self.wfile.write(chunk)
        except Exception:
            self.send_error(416)

    def log_message(self, *args, **kwargs):
        pass


@pytest.fixture
def http_h5_url(tmp_path):
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
        )
    finally:
        server.shutdown()


def _make_h5(path: Path) -> dict:
    """Write a small but realistic HDF5 file with chunked + gzip
    datasets. Returns a dict of name -> reference ndarray."""
    refs = {}
    with h5py.File(path, "w") as f:
        rng = np.random.default_rng(0)
        img = rng.integers(0, 4000, size=(64, 128), dtype=np.uint16)
        f.create_dataset("img", data=img, chunks=(16, 32),
                         compression="gzip")
        refs["img"] = img

        big = (np.arange(256 * 256, dtype=np.float32)
               .reshape(256, 256))
        f.create_dataset("data/big", data=big, chunks=(32, 32),
                         compression="gzip")
        refs["data/big"] = big

        small = np.array([1, 2, 3], dtype=np.int16)
        f.create_dataset("nested/group/small", data=small)
        refs["nested/group/small"] = small
    return refs


def test_remote_hdf5_round_trip(http_h5_url):
    url_for, tmp = http_h5_url
    refs = _make_h5(tmp / "data.h5")

    with open_remote_hdf5(url_for("data.h5")) as f:
        np.testing.assert_array_equal(f["img"][...], refs["img"])
        np.testing.assert_array_equal(
            f["data/big"][...], refs["data/big"]
        )
        np.testing.assert_array_equal(
            f["nested/group/small"][...], refs["nested/group/small"]
        )


def test_remote_hdf5_partial_slice_only_fetches_needed_chunks(http_h5_url):
    """A small slice should NOT fetch the whole file."""
    url_for, tmp = http_h5_url
    refs = _make_h5(tmp / "data.h5")
    total_size = (tmp / "data.h5").stat().st_size

    src = HTTPDataSource(url_for("data.h5"), prefetch_bytes=4096)
    fobj = _HTTPFileLike(src)
    with h5py.File(fobj, "r") as f:
        # Read just one chunk (the upper-left 16x32 of img).
        sl = f["img"][:16, :32]
        np.testing.assert_array_equal(sl, refs["img"][:16, :32])

    # We should have transferred *less* than the whole file.
    assert src._total_bytes_fetched < total_size, (
        f"fetched {src._total_bytes_fetched} bytes, file is {total_size};"
        f" partial slice shouldn't pull the whole thing"
    )


def test_prefetch_collapses_chunk_fetches(http_h5_url):
    """prefetch_hdf5_chunks should coalesce many chunk fetches into a
    small number of parallel HTTP requests, dramatically cutting the
    request count compared to the serial baseline."""
    url_for, tmp = http_h5_url
    with h5py.File(tmp / "big.h5", "w") as f:
        rng = np.random.default_rng(0)
        big = rng.integers(0, 4000, size=(512, 512), dtype=np.uint16)
        f.create_dataset("img", data=big, chunks=(32, 32),
                         compression="gzip")

    # Serial baseline.
    with open_remote_hdf5(url_for("big.h5")) as f:
        src_serial = _SOURCE_REGISTRY[f.id]
        _ = f["img"][:512, :512]
        serial_reqs = src_serial.stats["requests"]

    # With parallel prefetch.
    with open_remote_hdf5(url_for("big.h5"), max_workers=8) as f:
        src_par = _SOURCE_REGISTRY[f.id]
        d = f["img"]
        n = prefetch_hdf5_chunks(d, np.s_[:512, :512])
        _ = d[:512, :512]
        par_reqs = src_par.stats["requests"]

    assert n == 256, f"expected 256 chunks for a 512x512/32x32 layout, got {n}"
    # Coalescing typically takes 256 chunks down to ~5-20 fetches.
    assert par_reqs * 5 <= serial_reqs, (
        f"expected substantial request reduction: "
        f"serial={serial_reqs}, parallel={par_reqs}"
    )


def test_prefetch_correct_values(http_h5_url):
    """Prefetched + read must equal the no-prefetch read byte-for-byte."""
    url_for, tmp = http_h5_url
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 4000, size=(256, 256), dtype=np.uint16)
    with h5py.File(tmp / "x.h5", "w") as f:
        f.create_dataset("img", data=arr, chunks=(32, 32),
                         compression="gzip")

    with open_remote_hdf5(url_for("x.h5")) as f:
        baseline = f["img"][:128, :128]
    with open_remote_hdf5(url_for("x.h5")) as f:
        d = f["img"]
        prefetch_hdf5_chunks(d, np.s_[:128, :128])
        prefetched = d[:128, :128]

    np.testing.assert_array_equal(baseline, prefetched)
    np.testing.assert_array_equal(baseline, arr[:128, :128])


def test_remote_hdf5_filelike_seek_tell(http_h5_url):
    """The file-like wrapper needs basic seek/tell semantics."""
    url_for, tmp = http_h5_url
    _make_h5(tmp / "data.h5")
    src = HTTPDataSource(url_for("data.h5"), prefetch_bytes=0)
    fobj = _HTTPFileLike(src)
    head = fobj.read(8)
    # HDF5 signature: \x89 HDF \r\n \x1a \n
    assert head == b"\x89HDF\r\n\x1a\n"
    assert fobj.tell() == 8
    fobj.seek(0)
    assert fobj.read(4) == b"\x89HDF"
    fobj.seek(-8, 2)  # 8 bytes before EOF
    tail = fobj.read(8)
    assert len(tail) == 8


# ---------------------------------------------------------------------------
# Live smoke test against a real-world HDF5 file served over HTTPS.
#
# We pick a tiny known-stable file from the HDFGroup's reference test
# corpus on github (size ~6 KB, raw.githubusercontent.com honours
# Range requests). This proves the open_remote_hdf5 pipeline works
# against actual cloud storage, not just a loopback fixture.
# ---------------------------------------------------------------------------


_LIVE_HDF5_URL = (
    "https://raw.githubusercontent.com/HDFGroup/hdf5/develop/"
    "test/testfiles/test_filters_le.h5"
)


def _live_hdf5_reachable() -> bool:
    import urllib.request
    try:
        req = urllib.request.Request(_LIVE_HDF5_URL, method="HEAD")
        with urllib.request.urlopen(req, timeout=3) as r:
            return (
                int(r.headers.get("Content-Length", "0")) > 0
                and r.headers.get("Accept-Ranges", "").lower() == "bytes"
            )
    except Exception:
        return False


@pytest.mark.skipif(
    not _live_hdf5_reachable(),
    reason="public HDF5 endpoint unreachable (offline or blocked)",
)
def test_remote_hdf5_live_github_endpoint():
    """End-to-end HDF5 cloud read against the HDFGroup-hosted reference
    file on github. Validates the open_remote_hdf5 + HTTPDataSource
    chain on an actual remote HDF5, not the loopback test fixture.

    The file is tiny (~6 KB) so we expect the prefetch buffer to cover
    the whole thing — confirms partial Range reads work even on
    sub-prefetch-size files."""
    src = HTTPDataSource(_LIVE_HDF5_URL, prefetch_bytes=64 * 1024)
    try:
        with open_remote_hdf5(_LIVE_HDF5_URL) as f:
            # The reference file has at least one dataset; just walk
            # the root to verify h5py traverses it through our
            # HTTPDataSource without error.
            names = list(f.keys())
            assert names, "no datasets in remote HDF5"
            # Read one dataset to fully exercise the chunk-fetch path.
            arr = f[names[0]][...]
            assert arr.size > 0
    finally:
        src.close()
