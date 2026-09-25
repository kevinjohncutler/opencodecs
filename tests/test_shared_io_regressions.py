"""Ownership, response validation, and sparse-cache regression coverage."""
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import random
import threading
import urllib.request

import numpy as np
import pytest

from opencodecs.core.io import coerce_data_source
from opencodecs.core.codec import Codec
from opencodecs.core._range_index import RangeIndex
from opencodecs._tiff_http import HTTPDataSource, RangeResponseError


@contextmanager
def serve_response(status=206, content_range='bytes 100-107/256', body=bytes(range(100, 108)),
                   content_length=None, encoding=None):
    requests = []

    class Handler(BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def do_GET(self):
            requests.append(dict(self.headers))
            self.send_response(status)
            if content_range is not None:
                self.send_header('Content-Range', content_range)
            self.send_header('Content-Length', str(len(body) if content_length is None else content_length))
            if encoding:
                self.send_header('Content-Encoding', encoding)
            self.send_header('Connection', 'close')
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/data', requests
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture(params=[False, True], ids=['persistent', 'opener'])
def transport(request):
    return {'opener': urllib.request.build_opener()} if request.param else {}


@pytest.mark.parametrize('response', [
    {'status': 200, 'content_range': None, 'body': bytes(range(256))},
    {'content_range': 'bytes 0-7/256'},
    {'content_range': None},
    {'content_range': 'bytes 100-106/256', 'body': b'x' * 7},
    {'content_length': 7},
    {'body': b'x' * 7, 'content_length': 8},
    {'encoding': 'gzip'},
])
def test_invalid_ranges_never_enter_cache(transport, response):
    with serve_response(**response) as (url, requests):
        with HTTPDataSource(url, prefetch_bytes=0, **transport) as ds:
            with pytest.raises(RangeResponseError):
                ds.read_at(100, 8)
            assert not ds._cache
            assert len(requests) == 1


def test_valid_ranges_are_cached_by_both_transports(transport):
    with serve_response() as (url, requests):
        with HTTPDataSource(url, prefetch_bytes=0, **transport) as ds:
            assert ds.read_at(100, 8) == bytes(range(100, 108))
            assert ds.read_many([(102, 2)]) == [bytes([102, 103])]
            assert len(requests) == 1
            assert requests[0]['Accept-Encoding'] == 'identity'


def test_short_read_at_end_is_valid(transport):
    with serve_response(content_range='bytes 100-103/104', body=b'abcd') as (url, requests):
        with HTTPDataSource(url, prefetch_bytes=0, **transport) as ds:
            assert ds.read_many([(100, 8)]) == [b'abcd']
            assert ds.total_size == 104
            assert (100, 8) not in ds._cache
            assert ds.read_at(102, 2) == b'cd'
            assert len(requests) == 1


def test_eof_response_for_both_transports(transport):
    with serve_response(status=416, content_range='bytes */100', body=b'') as (url, requests):
        with HTTPDataSource(url, prefetch_bytes=0, **transport) as ds:
            assert ds.read_at(100, 8) == b''
            assert ds.total_size == 100
            assert not ds._cache


class MemoryFetch(HTTPDataSource):
    def __init__(self, **kwargs):
        self.fetches = []
        super().__init__('http://unused', prefetch_bytes=0, **kwargs)

    def _range_request(self, offset, n):
        self.fetches.append((offset, n))
        return bytes((i % 251 for i in range(offset, offset + n)))


def test_adaptive_streak_resets_for_distant_and_large_reads():
    for middle in [None, (3000, 20000)]:
        with MemoryFetch() as ds:
            for off in [0, 1024, 2048]:
                ds.read_at(off, 1024)
            if middle:
                ds.read_at(*middle)
            off = 1000000 if middle is None else 23000
            ds.read_at(off, 1024)
            assert ds.fetches[-1] == (off, 1024)


def test_sparse_cache_retains_all_ranges_and_finds_interior_hits():
    with MemoryFetch(adaptive_window=0) as ds:
        for i in range(8000):
            ds.read_at(i * 32, 8)
        before = len(ds.fetches)
        for i in range(8000):
            assert ds.read_at(i * 32 + 2, 3) == bytes((j % 251 for j in range(i * 32 + 2, i * 32 + 5)))
        assert len(ds.fetches) == before


def test_index_matches_brute_force_after_insertions_and_removals():
    rng = random.Random(14)
    index = RangeIndex()
    keys = set()
    for _ in range(3000):
        key = (rng.randrange(10000), rng.randrange(1, 3000))
        keys.add(key)
        index.add(key)
    for _ in range(1000):
        if rng.random() < 0.3:
            key = rng.choice(sorted(keys))
            keys.remove(key)
            index.discard(key)
        start, length = rng.randrange(13000), rng.randrange(1, 3500)
        expected = any(o <= start and start + length <= o + n for o, n in keys)
        found = index.covering(start, length)
        assert (found is not None) == expected
        if found is not None:
            assert found in keys
            assert found[0] <= start and start + length <= sum(found)
    index.clear()
    assert index.covering(0, 1) is None


def test_eviction_removes_index_entries():
    with MemoryFetch(cache_bytes=8, adaptive_window=0) as ds:
        ds.read_at(0, 8)
        ds.read_at(32, 8)
        assert ds._range_index.covering(0, 1) is None
        assert ds.read_at(0, 1) == b'\0'
        assert len(ds.fetches) == 3


def test_source_close_preserves_callers_view_without_copying():
    buf = bytearray(b'abc')
    view = memoryview(buf)
    ds, owns, _ = coerce_data_source(view)
    assert owns
    buf[0] = ord('z')
    assert ds.read_at(0, 3) == b'zbc'
    ds.close()
    ds.close()
    assert view.tobytes() == b'zbc'
    view.release()


def test_buffered_writer_owns_submitted_frames():
    class Capture(Codec):
        name = 'capture'
        can_encode = True
        def encode(self, arr, **opts):
            return arr.copy()
    writer = Capture().writer()
    arr = np.zeros((2, 3), dtype='u1')
    for value in range(3):
        arr.fill(value)
        writer.write_frame(arr)
    arr.fill(99)
    assert writer.close()[:, 0, 0].tolist() == [0, 1, 2]


def test_jxl_adapter_owns_pending_frame_and_rejects_options(monkeypatch):
    from opencodecs import _jxl_codec as jxl
    class Capture:
        def __init__(self, *args, **opts):
            self.frames = []
        def write_frame(self, arr, **opts):
            self.frames.append((arr.copy(), opts))
        def close(self):
            return self.frames
    monkeypatch.setattr(jxl, '_JxlWriter', Capture)
    writer = jxl._JxlStreamWriter()
    arr = np.zeros((2, 3), dtype='u1')
    writer.write_frame(arr)
    arr.fill(1)
    with pytest.raises(TypeError, match='per-frame options'):
        writer.write_frame(arr, unsupported=True)
    writer.write_frame(arr)
    arr.fill(2)
    frames = writer.close()
    assert [int(frame[0, 0]) for frame, _ in frames] == [0, 1]
    assert frames[-1][1] == {'is_last': True}


def test_invalid_prefetch_response_fails_immediately(transport):
    with serve_response(status=200, content_range=None) as (url, requests):
        with pytest.raises(RangeResponseError):
            HTTPDataSource(url, **transport)
        assert len(requests) == 1


def test_whole_file_response_is_rejected_without_reading_body():
    class Response:
        status = 200
        headers = {'Content-Length': str(1 << 40)}
        def read(self, *args):
            pytest.fail('a range source must not consume a whole-file response')
    with MemoryFetch() as ds:
        with pytest.raises(RangeResponseError):
            ds._read_range_response(Response(), 100, 8)


def test_range_content_length_accepts_leading_zeroes(transport):
    with serve_response(content_length='0008') as (url, requests):
        with HTTPDataSource(url, prefetch_bytes=0, **transport) as source:
            assert source.read_at(100, 8) == bytes(range(100, 108))


def test_eager_destination_retries_short_writes_and_preserves_ownership():
    import io
    from opencodecs._none_codec import NoneCodec
    class ShortSink(io.BytesIO):
        def write(self, data):
            return super().write(data[:3])
    sink = ShortSink()
    assert NoneCodec().encode(b'0123456789', dest=sink) is None
    assert sink.getvalue() == b'0123456789'
    assert not sink.closed
    class Stuck:
        def write(self, data):
            return 0
    with pytest.raises(OSError):
        NoneCodec().encode(b'abc', dest=Stuck())
