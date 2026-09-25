"""Bounded independent-piece adapters preserve ordering and source lifetime."""
from __future__ import annotations

import io
import threading
import time

import numpy as np
import pytest


@pytest.mark.parametrize("algorithm", ["RICE_1", "GZIP_1", "GZIP_2", "PLIO_1", "HCOMPRESS_1"])
def test_fits_bounded_heap_matches_astropy(tmp_path, algorithm):
    fits = pytest.importorskip("astropy.io.fits")
    from opencodecs._fits import FitsStream
    values = np.random.default_rng(9).integers(0, 100, (301, 173), dtype="i2")
    path = tmp_path / "tiles.fits"
    fits.CompImageHDU(data=values, compression_type=algorithm, tile_shape=(64, 64)).writeto(path)
    expected = fits.getdata(path)
    encoded = path.read_bytes()
    calls = []
    def read_at(offset, size):
        calls.append((offset, size))
        return encoded[offset:offset + size]
    with FitsStream(None, read_at=read_at) as reader:
        calls.clear()
        got = reader.read(numthreads=4, max_pending_bytes=32 << 10)
        np.testing.assert_array_equal(got, expected)
        assert len(calls) > 2
        assert max(size for _, size in calls) < len(encoded) // 2


def test_fits_bounded_quantized_tiles(tmp_path):
    fits = pytest.importorskip("astropy.io.fits")
    from opencodecs._fits_codec import FitsCodec
    path = tmp_path / "float.fits"
    values = np.random.default_rng(9).normal(100, 10, (131, 173)).astype("f4")
    fits.CompImageHDU(data=values, compression_type="RICE_1", tile_shape=(64, 64)).writeto(path)
    expected = fits.getdata(path)
    got = FitsCodec().decode(path, numthreads=4, max_pending_bytes=32 << 10)
    np.testing.assert_allclose(got, expected, rtol=1e-5, atol=1e-5)


def test_vsi_bounded_region_matches_independent_jpeg(tmp_path):
    imagecodecs = pytest.importorskip("imagecodecs")
    from test_vsi_pyramid import _build_tiled_ets
    from opencodecs._vsi_pyramid import VsiPyramidReader
    path, _ = _build_tiled_ets(tmp_path / "tiles.ets", (173, 131), tile=(64, 64), levels=1)
    encoded = path.read_bytes()
    with VsiPyramidReader(path, numthreads=4, max_pending_bytes=48 << 10) as reader:
        expected = np.zeros((131, 173, 3), dtype="u1")
        for rec in reader.info.records:
            tile = imagecodecs.jpeg_decode(encoded[rec.offset:rec.offset + rec.size])
            y, x = rec.tile_y * 64, rec.tile_x * 64
            h, w = min(64, 131 - y), min(64, 173 - x)
            expected[y:y+h, x:x+w] = tile[:h, :w]
        got = reader._read_region(reader.levels[0], 17, 127, 11, 162)
    np.testing.assert_array_equal(got, expected[17:127, 11:162])


@pytest.mark.parametrize("budget", [None, 1024])
def test_ndtiff_parallel_iterator_close_joins_source_users(budget):
    from opencodecs._ndtiff import NDTiffDataset
    from types import SimpleNamespace
    reader = object.__new__(NDTiffDataset)
    reader.entries = [SimpleNamespace(pixel_compression=0, pixel_nbytes=128, value=i)
                      for i in range(20)]
    lock = threading.Lock()
    active = 0
    def read_entry(entry):
        nonlocal active
        with lock:
            active += 1
        try:
            time.sleep(0.01)
            return np.full((8, 8), entry.value, dtype="u2")
        finally:
            with lock:
                active -= 1
    reader._read_entry = read_entry
    stream = reader.iter_frames_parallel(prefetch=4, max_pending_bytes=budget)
    np.testing.assert_array_equal(next(stream), np.zeros((8, 8), dtype="u2"))
    stream.close()
    assert active == 0


def test_ndtiff_parallel_iterator_orders_uneven_frames():
    from opencodecs._ndtiff import NDTiffDataset
    from types import SimpleNamespace
    reader = object.__new__(NDTiffDataset)
    reader.entries = [SimpleNamespace(pixel_compression=0, pixel_nbytes=2 * (i + 1), value=i)
                      for i in range(13)]
    reader._read_entry = lambda entry: np.full(entry.value + 1, entry.value, dtype="u2")
    stream = reader.iter_frames_parallel(prefetch=3, max_pending_bytes=128)
    for index, frame in enumerate(stream):
        np.testing.assert_array_equal(frame, np.full(index + 1, index, dtype="u2"))
    with pytest.raises(ValueError, match="prefetch"):
        next(reader.iter_frames_parallel(prefetch=0))


def test_dicomweb_multipart_preserves_binary_newline_bytes():
    from opencodecs._dicomweb import _parse_multipart
    payload = b"pixels\r\n\n\r"
    body = b"--bound\r\nContent-Type: application/octet-stream\r\n\r\n" + payload + b"\r\n--bound--\r\n"
    assert _parse_multipart(body, "multipart/related; boundary=bound")[0][1] == payload


def test_dicomweb_bounded_iterator_orders_frames_and_checks_limits(monkeypatch):
    from opencodecs._dicomweb import DicomwebClient
    from test_dicomweb import _build_multipart
    client = DicomwebClient("https://example.invalid")
    def get(url, *, accept, max_bytes=None):
        frame = int(url.rsplit("/", 1)[1])
        payload = np.full((3, 7), frame, dtype="<u2").tobytes()
        body, content_type = _build_multipart([("1.2.840.10008.1.2.1", payload)])
        assert len(body) <= max_bytes
        return body, content_type
    monkeypatch.setattr(client, "_http_get", get)
    frames = client.iter_frames("s", "r", "i", [3, 1, 7], max_frame_bytes=512,
                                max_pending_bytes=8192, numthreads=3,
                                rows=3, columns=7, bits_allocated=16)
    for requested, result in zip([3, 1, 7], frames):
        np.testing.assert_array_equal(result, np.full((3, 7), requested, dtype="u2"))


def test_dicomweb_response_limit_before_decode(monkeypatch):
    from opencodecs._dicomweb import DicomwebClient, DicomwebError
    import urllib.request
    class Response(io.BytesIO):
        headers = {"Content-Type": "multipart/related"}
    monkeypatch.setattr(urllib.request, "urlopen", lambda *args, **kw: Response(b"x" * 1024))
    with pytest.raises(DicomwebError, match="response exceeds"):
        DicomwebClient("https://example.invalid").get_frame("s", "r", "i", max_response_bytes=100)


def test_oir_remote_plane_coalesces_small_record_gaps(monkeypatch):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from opencodecs._oir_codec import OirNativeReader
    from opencodecs._tiff_http import HTTPDataSource
    expected = np.arange(512 * 512, dtype='<u2').reshape(512, 512)
    body = bytearray()
    records = []
    for pixels in (expected[:237], expected[237:474], expected[474:]):
        body.extend(b'\0' * 75)
        records.append({'payload_offset': len(body), 'payload_size': pixels.nbytes})
        body.extend(pixels.tobytes())
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass
        def do_HEAD(self):
            self.send_response(200)
            self.send_header('Content-Length', str(len(body)))
            self.end_headers()
        def do_GET(self):
            start, end = map(int, self.headers['Range'][6:].split('-'))
            end = min(end, len(body) - 1)
            calls.append((start, end + 1))
            self.send_response(206)
            self.send_header('Content-Length', str(end - start + 1))
            self.send_header('Content-Range', f'bytes {start}-{end}/{len(body)}')
            self.end_headers()
            self.wfile.write(body[start:end + 1])
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with HTTPDataSource(f'http://127.0.0.1:{server.server_port}/plane',
                            prefetch_bytes=0, cache_bytes=0, adaptive_window=0) as source:
            monkeypatch.setattr('opencodecs._oir_codec._read_oir_info',
                                lambda ds: {'frame_records': [dict(rec, name='frame') for rec in records]})
            reader = OirNativeReader(source)
            calls.clear()
            np.testing.assert_array_equal(reader[0], expected)
            assert len(calls) == 1
            assert calls[0][1] - calls[0][0] == expected.nbytes + 150
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_oib_bounded_producer_fetch_and_direct_placement():
    import tifffile
    from opencodecs._oib_native import OibFileParser, OibLayout
    from opencodecs.core.pipeline import PipelineStats
    values = np.arange(8 * 32 * 48, dtype='u2').reshape(8, 32, 48)
    payloads = {}
    for index, frame in enumerate(values):
        buffer = io.BytesIO()
        tifffile.imwrite(buffer, frame, compression='deflate')
        payloads[str(index)] = buffer.getvalue()
    producer = threading.get_ident()
    class Compound:
        def get_size(self, name):
            return len(payloads[name])
        def read_stream(self, name):
            assert threading.get_ident() == producer
            return payloads[name]
    parser = object.__new__(OibFileParser)
    parser._ole = Compound()
    parser.layout = OibLayout(48, 32, 1, 8, 1, values.dtype, 'XYZ',
                             {(0, index, 0): str(index) for index in range(8)})
    stats = PipelineStats()
    result = parser.read_all(numthreads=4, max_pending_bytes=32 << 10,
                             pipeline_stats=stats)
    np.testing.assert_array_equal(result, values)
    assert stats.completed == 8
    assert stats.peak_reserved_bytes <= 32 << 10


def test_oib_bounded_fetch_failure_joins_running_decoders():
    from opencodecs._oib_native import OibFileParser, OibLayout
    values = np.zeros((8, 32, 48), dtype='u2')
    started, finished = threading.Event(), threading.Event()
    class Compound:
        def get_size(self, name):
            return 1
        def read_stream(self, name):
            if name == '1':
                assert started.wait(3)
                raise IOError('fixture stream failure')
            return b'x'
    parser = object.__new__(OibFileParser)
    parser._ole = Compound()
    parser.layout = OibLayout(48, 32, 1, 8, 1, values.dtype, 'XYZ',
                             {(0, index, 0): str(index) for index in range(8)})
    def decode(name, payload):
        started.set()
        time.sleep(0.03)
        finished.set()
        return values[0]
    parser._decode_frame_payload = decode
    with pytest.raises(IOError, match='fixture stream failure'):
        parser.read_all(numthreads=4, max_pending_bytes=64 << 10)
    assert finished.is_set()


def test_oib_bounded_matches_independent_corpus_reader():
    from pathlib import Path
    oiffile = pytest.importorskip('oiffile')
    from opencodecs._oib_codec import OibCodec
    path = Path(__file__).resolve().parents[1] / '.test_data/oib/imagesc_71616_60x.oib'
    if not path.is_file():
        pytest.skip('public OIB corpus unavailable')
    expected = oiffile.imread(path)
    got = OibCodec().decode(path, backend='native', numthreads=4,
                            max_pending_bytes=24 << 20)
    np.testing.assert_array_equal(got, expected)
