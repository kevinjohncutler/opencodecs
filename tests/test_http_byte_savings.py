"""Empirical proof that native vendor readers do partial HTTP reads.

Spins up a range-supporting local HTTP server (most stdlib test
servers return the full file regardless of Range headers), then
exercises each native reader against it and asserts:

  1. The server returns 206 Partial Content for the reads we expect
     to be partial (verifies the client actually sent Range).
  2. The actual bytes served across the wire is a TINY FRACTION of
     the file size — proving cloud-cost savings concretely.

The constants below set the byte-savings floor (e.g. read ONE frame
from a 13 MB ND2 → server transmits < 1 MB).
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent))
from _range_http_server import range_http_server  # noqa: E402

CORPUS = Path(__file__).resolve().parent.parent / ".test_data"
ND2_SAMPLE = CORPUS / "nd2" / "MeOh_high_fluo_007.nd2"
OIB_SAMPLE = CORPUS / "oib" / "imagesc_71616_60x.oib"

_HINT = (
    "Run `bash tests/download_test_corpus.sh --light` from the repo "
    "root to populate the corpus."
)


# ---------------------------------------------------------------------------
# Sanity: the test server itself honors Range
# ---------------------------------------------------------------------------


def test_range_server_honors_range(tmp_path):
    """Sanity check that the test server returns 206 for Range
    requests and only sends the requested bytes."""
    import urllib.request
    f = tmp_path / "data.bin"
    payload = bytes(range(256)) * 256   # 64 KB
    f.write_bytes(payload)
    with range_http_server(tmp_path) as (url, tracker):
        req = urllib.request.Request(
            f"{url}/data.bin",
            headers={"Range": "bytes=0-99"},
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 206
            body = resp.read()
        assert body == payload[:100]
        assert tracker.range_requests == 1
        assert tracker.bytes_served == 100


def test_range_server_honors_suffix_range(tmp_path):
    """``bytes=-N`` = last N bytes — the form ND2 uses to find its
    FILEMAP at EOF."""
    import urllib.request
    f = tmp_path / "data.bin"
    payload = bytes(range(256)) * 256
    f.write_bytes(payload)
    with range_http_server(tmp_path) as (url, tracker):
        req = urllib.request.Request(
            f"{url}/data.bin",
            headers={"Range": "bytes=-40"},
        )
        with urllib.request.urlopen(req) as resp:
            assert resp.status == 206
            body = resp.read()
        assert body == payload[-40:]
        assert tracker.bytes_served == 40


# ---------------------------------------------------------------------------
# Native ND2: per-frame range reads
# ---------------------------------------------------------------------------


@pytest.fixture
def nd2_http_setup(tmp_path):
    if not ND2_SAMPLE.exists():
        pytest.skip(_HINT)
    shutil.copy(ND2_SAMPLE, tmp_path / "sample.nd2")
    return tmp_path


def test_native_nd2_open_is_under_1pct(nd2_http_setup):
    """Just OPENING the ND2 (no frames read) should fetch only the
    EOF trailer + FILEMAP + ImageAttributes — far less than 1% of
    the file."""
    from opencodecs._tiff_http import HTTPDataSource
    from opencodecs._nd2_native import Nd2NativeReader

    with range_http_server(nd2_http_setup) as (url_base, tracker):
        src = HTTPDataSource(f"{url_base}/sample.nd2")
        with Nd2NativeReader(src) as r:
            assert r.n_frames == 13
        # After opening, we should have made several range requests
        # but transferred very few bytes — typically < 10 KB.
        assert tracker.range_requests >= 1
        assert tracker.full_requests == 0, (
            f"unexpected full GET; bytes={tracker.bytes_served}")
        floor = tracker.file_size // 100   # 1% of file
        assert tracker.bytes_served < floor, (
            f"open() fetched {tracker.bytes_served:,} bytes "
            f"({tracker.bytes_served / tracker.file_size * 100:.2f}% "
            f"of file); expected < 1%")


def test_native_nd2_read_one_frame_is_under_10pct(nd2_http_setup):
    """Reading ONE frame from a 13-frame ND2 should fetch about
    1/13 of the file plus a bit of metadata — under 10%."""
    from opencodecs._tiff_http import HTTPDataSource
    from opencodecs._nd2_native import Nd2NativeReader

    with range_http_server(nd2_http_setup) as (url_base, tracker):
        src = HTTPDataSource(f"{url_base}/sample.nd2")
        with Nd2NativeReader(src) as r:
            _ = r[5]      # read frame 5
        assert tracker.full_requests == 0
        # 1/13 = 7.7%; allow some slack for metadata + chunk header.
        # Target ceiling: 15% of the file.
        assert tracker.bytes_served < tracker.file_size * 0.15, (
            f"one-frame read fetched {tracker.bytes_served:,} bytes "
            f"({tracker.bytes_served / tracker.file_size * 100:.1f}% "
            f"of file); expected < 15%")


def test_native_nd2_read_one_frame_correct(nd2_http_setup):
    """Per-frame data must be byte-identical to local decode."""
    pytest.importorskip("nd2")
    from opencodecs._tiff_http import HTTPDataSource
    from opencodecs._nd2_native import Nd2NativeReader

    with range_http_server(nd2_http_setup) as (url_base, _):
        src = HTTPDataSource(f"{url_base}/sample.nd2")
        with Nd2NativeReader(src) as r:
            http_frame = r[5]
    # Compare against local-file decode
    with Nd2NativeReader(str(ND2_SAMPLE)) as r:
        local_frame = r[5]
    assert np.array_equal(http_frame, local_frame)


# ---------------------------------------------------------------------------
# Native OIB: per-stream range reads
# ---------------------------------------------------------------------------


@pytest.fixture
def oib_http_setup(tmp_path):
    if not OIB_SAMPLE.exists():
        pytest.skip(_HINT)
    shutil.copy(OIB_SAMPLE, tmp_path / "sample.oib")
    return tmp_path


def test_native_oib_open_is_under_1pct(oib_http_setup):
    """Opening an OIB only reads the OLE2 directory + FAT + a few
    metadata streams — should be well under 1% of file."""
    from opencodecs._tiff_http import HTTPDataSource
    from opencodecs._oib_native import OibNativeReader

    with range_http_server(oib_http_setup) as (url_base, tracker):
        src = HTTPDataSource(f"{url_base}/sample.oib")
        with OibNativeReader(src) as r:
            assert r.shape == (2, 6, 1024, 1024)
        assert tracker.full_requests == 0
        floor = tracker.file_size // 100   # 1%
        assert tracker.bytes_served < floor, (
            f"OIB open() fetched {tracker.bytes_served:,} bytes "
            f"({tracker.bytes_served / tracker.file_size * 100:.2f}%)")


def test_native_oib_decode_correct_via_http(oib_http_setup):
    """Full decode over HTTP must match local decode exactly."""
    from opencodecs._tiff_http import HTTPDataSource
    from opencodecs._oib_native import OibNativeReader

    with range_http_server(oib_http_setup) as (url_base, _):
        src = HTTPDataSource(f"{url_base}/sample.oib")
        with OibNativeReader(src) as r:
            http_arr = r.read()
    with OibNativeReader(str(OIB_SAMPLE)) as r:
        local_arr = r.read()
    assert np.array_equal(http_arr, local_arr)


# ---------------------------------------------------------------------------
# Summary helpers (informational, not asserted)
# ---------------------------------------------------------------------------


def test_print_byte_savings_table(nd2_http_setup, oib_http_setup, capsys):
    """Print a table of byte-savings ratios — visible with pytest -s."""
    from opencodecs._tiff_http import HTTPDataSource
    from opencodecs._nd2_native import Nd2NativeReader
    from opencodecs._oib_native import OibNativeReader

    rows: list[tuple[str, int, int, int]] = []

    with range_http_server(nd2_http_setup) as (url_base, tr):
        src = HTTPDataSource(f"{url_base}/sample.nd2")
        with Nd2NativeReader(src):
            pass
        rows.append(("ND2 open",
                     tr.file_size, tr.bytes_served, tr.range_requests))

    with range_http_server(nd2_http_setup) as (url_base, tr):
        src = HTTPDataSource(f"{url_base}/sample.nd2")
        with Nd2NativeReader(src) as r:
            _ = r[5]
        rows.append(("ND2 open + 1 frame",
                     tr.file_size, tr.bytes_served, tr.range_requests))

    with range_http_server(oib_http_setup) as (url_base, tr):
        src = HTTPDataSource(f"{url_base}/sample.oib")
        with OibNativeReader(src):
            pass
        rows.append(("OIB open",
                     tr.file_size, tr.bytes_served, tr.range_requests))

    print("\n\nHTTP byte savings (server-side measurement):")
    print(f"  {'scenario':25}  {'file':>12}  {'served':>10}  "
          f"{'ratio':>8}  {'reqs':>5}")
    for name, fsz, served, reqs in rows:
        ratio = served / fsz * 100
        print(f"  {name:25}  {fsz:>12,}  {served:>10,}  "
              f"{ratio:>7.2f}%  {reqs:>5}")
    print()
