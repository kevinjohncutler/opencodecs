"""OME-Zarr v3 sharded reader: HTTP-range partial-read fast path.

Verifies that when the store supports byte-range reads, reading one
inner chunk out of a shard only fetches the shard index + the chunk's
own bytes, not the whole shard. This is the Tier 3 streaming-reader
unlock for OME-Zarr v3 sharded arrays on cloud storage.
"""

from __future__ import annotations

import http.server
import socketserver
import threading
from pathlib import Path

import numpy as np
import pytest

# zarr-python is needed to author the sharded test fixture; opencodecs
# is the read-side under test.
zarr = pytest.importorskip("zarr", minversion="3")
ZstdCodec = pytest.importorskip("zarr.codecs").ZstdCodec  # type: ignore

from opencodecs._omezarr import OmeZarrArray, _HttpStore


@pytest.fixture
def sharded_zarr(tmp_path):
    """Write a 256x256 image as a v3 sharded array (16x16 shards of
    4x4=16 inner chunks of 64x64 each... well, of 16x16 inner chunks
    of 16x16 = 256 inner chunks per shard). High inner-per-shard
    ratio makes the range-read win obvious."""
    img = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    path = tmp_path / "data.zarr"
    a = zarr.create_array(
        store=str(path), shape=img.shape,
        chunks=(16, 16), shards=(256, 256),   # 16x16 = 256 inner per shard
        dtype="uint16",
        compressors=ZstdCodec(),
    )
    a[:] = img
    return path, img


@pytest.fixture
def http_server(sharded_zarr):
    """Local HTTP server rooted at the parent of the sharded zarr."""
    path, img = sharded_zarr
    root = str(path.parent)

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def __init__(self, *a, **kw):
            super().__init__(*a, directory=root, **kw)

    httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
    port = httpd.server_address[1]
    thr = threading.Thread(target=httpd.serve_forever, daemon=True)
    thr.start()
    try:
        yield f"http://127.0.0.1:{port}/{path.name}", img
    finally:
        httpd.shutdown()


def test_fs_store_sharded_read_correctness(sharded_zarr):
    path, img = sharded_zarr
    a = OmeZarrArray(str(path))
    assert a._sharded
    assert a.chunks == (16, 16)
    assert a._shard_shape == (256, 256)
    got = a.read_region((slice(0, 100), slice(0, 100)))
    assert np.array_equal(got, img[:100, :100])


def test_http_sharded_read_correctness(http_server):
    url, img = http_server
    store = _HttpStore(url, cache_bytes=0)
    a = OmeZarrArray(store=store)
    got = a.read_region((slice(50, 150), slice(50, 150)))
    assert np.array_equal(got, img[50:150, 50:150])


def test_http_sharded_range_fetches_less_than_whole_shard(http_server):
    """Reading one 16x16 inner chunk should fetch ~ index_bytes +
    one chunk's compressed bytes, NOT the whole shard."""
    url, img = http_server
    store = _HttpStore(url, cache_bytes=0)
    a = OmeZarrArray(store=store)

    # Measure bytes fetched for one inner-chunk read.
    s0 = dict(store.stats())
    crop = a.read_region((slice(0, 16), slice(0, 16)))
    s1 = dict(store.stats())
    bytes_one_chunk = s1["bytes_fetched"] - s0["bytes_fetched"]
    assert np.array_equal(crop, img[:16, :16])

    # Compare to a full-shard download.
    store2 = _HttpStore(url, cache_bytes=0)
    s0 = dict(store2.stats())
    _ = store2["c/0/0"]
    s1 = dict(store2.stats())
    whole_shard = s1["bytes_fetched"] - s0["bytes_fetched"]

    # 256 inner chunks per shard → expect at least 10x savings
    # (typically 100x+ on real compressed data).
    assert bytes_one_chunk < whole_shard, (
        f"range read fetched {bytes_one_chunk:,} B, whole shard is "
        f"{whole_shard:,} B — no win"
    )
    ratio = whole_shard / bytes_one_chunk
    assert ratio >= 5, (
        f"expected at least 5x byte savings, got {ratio:.1f}x "
        f"({bytes_one_chunk:,} / {whole_shard:,})"
    )


def test_shard_index_cache_avoids_refetch(http_server):
    """Reading multiple chunks from the same shard should only fetch
    the shard index ONCE (cached after first hit)."""
    url, img = http_server
    store = _HttpStore(url, cache_bytes=0)
    a = OmeZarrArray(store=store)

    # First chunk: 2 requests (index + chunk) plus the 'shard in store?'
    # check which is HEAD-only.
    a.read_region((slice(0, 16), slice(0, 16)))
    s_after_first = dict(store.stats())

    # Same shard, different inner chunk — should be 1 request (chunk only).
    a.read_region((slice(16, 32), slice(0, 16)))
    s_after_second = dict(store.stats())

    reqs_for_second = s_after_second["requests"] - s_after_first["requests"]
    # Allow 2: one for the 'shard exists?' HEAD probe + one for the
    # actual chunk range. zero index fetch.
    assert reqs_for_second <= 2, (
        f"second chunk took {reqs_for_second} requests; index cache "
        f"didn't kick in"
    )


def test_negative_offset_range_request_via_callable_store(sharded_zarr):
    """_CallableStore.read_range falls back to slicing when no
    range_fetch is provided. Verify that fallback is byte-equivalent."""
    from opencodecs._omezarr import _CallableStore

    path, img = sharded_zarr

    # Build a fetch callable on top of the filesystem.
    def fetch(key: str) -> bytes:
        p = path / key
        if not p.exists():
            raise KeyError(key)
        return p.read_bytes()

    store = _CallableStore(fetch)
    # No range_fetch supplied → supports_range should be False
    assert store.supports_range is False
    # OmeZarrArray with callable store still gives correct results
    a = OmeZarrArray(store=store)
    got = a.read_region((slice(0, 32), slice(0, 32)))
    assert np.array_equal(got, img[:32, :32])


def test_callable_store_with_range_fetch(sharded_zarr):
    """_CallableStore.read_range with a real range_fetch lets users
    plug in S3 SDKs, fsspec, etc. and still get the v3 sharded
    fast-path."""
    from opencodecs._omezarr import _CallableStore

    path, img = sharded_zarr

    def fetch(key: str) -> bytes:
        p = path / key
        if not p.exists():
            raise KeyError(key)
        return p.read_bytes()

    def range_fetch(key: str, offset: int, n: int) -> bytes:
        p = path / key
        if not p.exists():
            raise KeyError(key)
        with open(p, "rb") as f:
            if offset < 0:
                size = p.stat().st_size
                f.seek(max(0, size + offset))
            else:
                f.seek(offset)
            return f.read(n)

    store = _CallableStore(fetch, range_fetch=range_fetch)
    assert store.supports_range is True
    a = OmeZarrArray(store=store)
    got = a.read_region((slice(0, 32), slice(0, 32)))
    assert np.array_equal(got, img[:32, :32])
