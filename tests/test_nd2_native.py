"""Tests for the native ND2 parser.

The native parser is the *default* backend for Nd2Codec — the nd2
package serves as fallback for legacy or compressed variants.
Native decode is byte-for-byte equivalent to the nd2 package on
raw / uncompressed v3 files, and runs over the same DataSource
abstraction as our TIFF reader (so HTTP range reads work
transparently).
"""

from __future__ import annotations

import http.server
import os
import shutil
import socketserver
import tempfile
import threading
from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc

CORPUS = Path(__file__).resolve().parent.parent / ".test_data"
ND2_SAMPLE = CORPUS / "nd2" / "MeOh_high_fluo_007.nd2"

_HINT = (
    "Run `bash tests/download_test_corpus.sh --light` from the repo "
    "root to populate the ND2 corpus."
)


# ---------------------------------------------------------------------------
# Native parser: structural correctness
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ND2_SAMPLE.exists(), reason=_HINT)
def test_native_nd2_chunkmap():
    """The chunkmap parser finds every named chunk via the
    end-of-file (signature, location) trailer + the FILEMAP body."""
    from opencodecs._nd2_native import Nd2FileParser
    p = Nd2FileParser(str(ND2_SAMPLE))
    # Corpus file has 13 ImageDataSeq frames + metadata chunks.
    assert "ImageAttributes!" in p.chunks
    assert "ImageDataSeq|0!" in p.chunks
    assert "ImageDataSeq|12!" in p.chunks
    assert "ImageDataSeq|13!" not in p.chunks   # exactly 13 frames
    p.close()


@pytest.mark.skipif(not ND2_SAMPLE.exists(), reason=_HINT)
def test_native_nd2_image_attributes():
    from opencodecs._nd2_native import Nd2FileParser
    p = Nd2FileParser(str(ND2_SAMPLE))
    a = p.attributes
    assert a.width == 800
    assert a.height == 600
    assert a.n_channels == 1
    assert a.bits_in_memory == 16
    assert a.sequence_count == 13
    assert a.dtype == np.dtype("<u2")
    p.close()


# ---------------------------------------------------------------------------
# Native parser: decode correctness vs nd2 package
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ND2_SAMPLE.exists(), reason=_HINT)
def test_native_nd2_decode_matches_nd2_package():
    """Full-stack decode (read entire stack) must be byte-identical
    to what the nd2 package returns. This validates: chunkmap walk,
    ImageAttributes parsing, per-frame chunk header resolution, and
    raw-pixel decode."""
    pytest.importorskip("nd2")
    from opencodecs._nd2_native import Nd2NativeReader
    import nd2 as _nd2

    with Nd2NativeReader(str(ND2_SAMPLE)) as r:
        ours = r.read()
    with _nd2.ND2File(str(ND2_SAMPLE)) as f:
        ref = f.asarray()
    assert np.array_equal(ours, ref)


@pytest.mark.skipif(not ND2_SAMPLE.exists(), reason=_HINT)
def test_native_nd2_per_frame_decode_matches_nd2_package():
    """Random-access read_frame(i) must match nd2 package for every i."""
    pytest.importorskip("nd2")
    from opencodecs._nd2_native import Nd2NativeReader
    import nd2 as _nd2

    with Nd2NativeReader(str(ND2_SAMPLE)) as r:
        ours_frames = [r[i] for i in range(r.n_frames)]
    with _nd2.ND2File(str(ND2_SAMPLE)) as f:
        for i, ours in enumerate(ours_frames):
            ref = f.read_frame(i)
            assert np.array_equal(ours, ref), f"diverged at frame {i}"


# ---------------------------------------------------------------------------
# Native parser: HTTP DataSource
# ---------------------------------------------------------------------------


@pytest.fixture
def http_nd2_server(tmp_path):
    """Local HTTPServer serving the ND2 corpus sample."""
    if not ND2_SAMPLE.exists():
        pytest.skip(_HINT)
    served_path = tmp_path / "sample.nd2"
    shutil.copy(ND2_SAMPLE, served_path)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=str(tmp_path), **kw)

    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thr = threading.Thread(target=httpd.serve_forever, daemon=True)
    thr.start()
    try:
        yield f"http://127.0.0.1:{port}/sample.nd2"
    finally:
        httpd.shutdown()


def test_native_nd2_decode_via_http_datasource(http_nd2_server):
    """The native parser accepts an HTTPDataSource directly. The
    resulting array matches a local-file decode byte-for-byte."""
    from opencodecs._tiff_http import HTTPDataSource
    from opencodecs._nd2_native import Nd2NativeReader

    src = HTTPDataSource(http_nd2_server)
    with Nd2NativeReader(src) as r_http:
        http_frame = r_http[5]
    with Nd2NativeReader(str(ND2_SAMPLE)) as r_local:
        local_frame = r_local[5]
    assert np.array_equal(http_frame, local_frame)


def test_native_nd2_codec_open_with_http_datasource(http_nd2_server):
    """oc.get_codec("nd2").open(HTTPDataSource(url)) uses the native
    backend transparently — no temp-file spill."""
    from opencodecs._tiff_http import HTTPDataSource

    src = HTTPDataSource(http_nd2_server)
    codec = oc.get_codec("nd2")
    with codec.open(src) as r:
        # Should be the native reader, not the nd2-package delegate
        from opencodecs._nd2_native import Nd2NativeReader
        assert isinstance(r, Nd2NativeReader)
        assert r.n_frames == 13


# ---------------------------------------------------------------------------
# Codec adapter: native + delegate backends both reachable
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ND2_SAMPLE.exists(), reason=_HINT)
def test_nd2_codec_default_uses_native():
    """Default open() yields the native reader (no backend= override)."""
    from opencodecs._nd2_native import Nd2NativeReader
    with oc.get_codec("nd2").open(str(ND2_SAMPLE)) as r:
        assert isinstance(r, Nd2NativeReader)


@pytest.mark.skipif(not ND2_SAMPLE.exists(), reason=_HINT)
def test_nd2_codec_force_delegate():
    """backend='nd2' opts into the delegate (nd2 package)."""
    pytest.importorskip("nd2")
    from opencodecs._nd2_codec import Nd2Reader
    with oc.get_codec("nd2").open(str(ND2_SAMPLE), backend="nd2") as r:
        assert isinstance(r, Nd2Reader)


@pytest.mark.skipif(not ND2_SAMPLE.exists(), reason=_HINT)
def test_nd2_codec_force_native():
    """backend='native' forces the native parser."""
    from opencodecs._nd2_native import Nd2NativeReader
    with oc.get_codec("nd2").open(str(ND2_SAMPLE), backend="native") as r:
        assert isinstance(r, Nd2NativeReader)


@pytest.mark.skipif(not ND2_SAMPLE.exists(), reason=_HINT)
def test_nd2_codec_lists_native_and_delegate():
    """codec.has_native=True and codec.has_delegate reflects nd2 install."""
    entry = next(c for c in oc.list_codecs() if c["name"] == "nd2")
    assert entry["native"] is True
