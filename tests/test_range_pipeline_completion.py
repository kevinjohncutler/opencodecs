"""Selected payload reads, parallel placement, and checksum integrity."""
import json
import struct
import threading
import time

import numpy as np
import pytest

from opencodecs._czi_reader import CziReader, CziError
from opencodecs._czi_writer import CziWriter
from opencodecs._omezarr import OmeZarrArray, _decompress_chunk
from opencodecs.core.checksums import crc32c, strip_crc32c


class CountingSource:
    def __init__(self, data):
        self.data = data
        self.requests = []
        self.closed = False

    def read_at(self, offset, length):
        self.requests.append((offset, length))
        return self.data[offset:offset + length]

    def close(self):
        self.closed = True


def test_czi_selected_ranges(tmp_path):
    path = tmp_path / 'tiles.czi'
    tiles = [np.full((128, 192), i, np.uint16) for i in range(24)]
    with CziWriter(path, compression='none') as writer:
        for tile in tiles:
            writer.write(tile)
    source = CountingSource(path.read_bytes())
    with CziReader(data_source=source, size=len(source.data)) as reader:
        initialization = sum(n for _, n in source.requests)
        assert initialization < len(source.data) // 50
        np.testing.assert_array_equal(reader[17], tiles[17])
        assert sum(n for _, n in source.requests) < len(source.data) // 10
        assert reader.metadata_bytes
        np.testing.assert_array_equal(
            reader.read(n_workers=4, max_pending_bytes=4 * tiles[0].nbytes),
            np.stack(tiles))
    assert not source.closed  # Borrowed source remains caller-owned.


def test_czi_truncated_range(tmp_path):
    path = tmp_path / 'truncated.czi'
    with CziWriter(path, compression='none') as writer:
        writer.write(np.zeros((8, 8), np.uint8))
    source = CountingSource(path.read_bytes()[:64])
    with pytest.raises(CziError, match='truncated'):
        CziReader(data_source=source, size=1000)


def test_czi_http_range_mode_fetches_selected_tiles(tmp_path):
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    path = tmp_path / 'remote.czi'
    tile = np.arange(128 * 192, dtype='u2').reshape(128, 192)
    with CziWriter(path, compression='none') as writer:
        for i in range(24):
            writer.write(tile + i)
    blob = path.read_bytes()
    fetched = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass
        def do_GET(self):
            requested = self.headers.get('Range')
            assert requested is not None, 'range mode must never download the full file'
            lo, hi = map(int, requested.removeprefix('bytes=').split('-'))
            hi = min(hi, len(blob) - 1)
            payload = blob[lo:hi+1]
            fetched.append(len(payload))
            self.send_response(206)
            self.send_header('Content-Range', f'bytes {lo}-{hi}/{len(blob)}')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever)
    thread.start()
    try:
        with CziReader.from_http(f'http://127.0.0.1:{server.server_port}/sample.czi',
                                 range_reads=True) as reader:
            np.testing.assert_array_equal(reader[17], tile + 17)
            assert sum(fetched) < len(blob) // 10
        assert reader._owned_source is None
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_crc32c_validates_payload_and_trailer():
    assert crc32c(b'123456789') == 0xe3069283
    encoded = b'123456789' + struct.pack('<I', 0xe3069283)
    assert bytes(strip_crc32c(encoded)) == b'123456789'
    assert bytes(_decompress_chunk(encoded, [{'name': 'crc32c'}])) == b'123456789'
    for bad in (b'', encoded[:-1], b'X' + encoded[1:]):
        with pytest.raises(ValueError):
            strip_crc32c(bad)


def test_zarr_parallel_missing_ragged_selection():
    expected = np.arange(19 * 23, dtype=np.uint16).reshape(19, 23)
    blobs = {'.zarray': json.dumps(dict(zarr_format=2, shape=[19, 23],
        chunks=[8, 8], dtype='<u2', compressor=None, fill_value=0,
        order='C', filters=None)).encode()}
    for y in range(3):
        for x in range(3):
            chunk = np.zeros((8, 8), dtype='<u2')
            part = expected[y*8:(y+1)*8, x*8:(x+1)*8]
            chunk[:part.shape[0], :part.shape[1]] = part
            blobs[f'{y}.{x}'] = chunk.tobytes()
    del blobs['1.1']
    expected[8:16, 8:16] = 0
    class Store(dict):
        active = peak = 0
        lock = threading.Lock()
        def __getitem__(self, key):
            if key.startswith('.'):
                return super().__getitem__(key)
            with self.lock:
                self.active += 1
                self.peak = max(self.peak, self.active)
            try:
                time.sleep(.002)
                return super().__getitem__(key)
            finally:
                with self.lock:
                    self.active -= 1
    store = Store(blobs)
    reader = OmeZarrArray(store=store, num_workers=4)
    np.testing.assert_array_equal(reader[3:18, 5:22], expected[3:18, 5:22])
    assert 1 < store.peak <= 4
    assert store.active == 0


def test_czi_negative_metadata_cannot_select_header_bytes(tmp_path):
    path = tmp_path / 'invalid_metadata.czi'
    with CziWriter(path, compression='none') as writer:
        writer.write(np.zeros((8, 8), np.uint8))
    blob = bytearray(path.read_bytes())
    offset = blob.index(b'ZISRAWSUBBLOCK')
    struct.pack_into('<i', blob, offset + 32, -16)
    with CziReader(buffer=blob) as reader:
        with pytest.raises(CziError, match='payload range'):
            reader[0]
