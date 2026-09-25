"""Concurrent range batches share one inner pool and join failures before return."""
from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor
import threading
import time
import pytest

from opencodecs._tiff_http import FileDataSource, HTTPDataSource


def make_source(kind, tmp_path, monkeypatch, read):
    if kind == 'http':
        source = HTTPDataSource('http://fixture.invalid/data', prefetch_bytes=0,
                                cache_bytes=0, adaptive_window=0,
                                max_workers=2, coalesce_gap=0)
        monkeypatch.setattr(source, '_range_request', read)
    else:
        path = tmp_path / 'pixels.bin'
        path.write_bytes(bytes(256))
        source = FileDataSource(path, max_workers=2)
        if not source._has_pread:
            source.close()
            pytest.skip('parallel file batches require pread')
        monkeypatch.setattr(source, 'read_at', read)
    return source


@pytest.mark.parametrize('kind', ['http', 'file'])
def test_concurrent_batches_create_only_one_inner_pool(kind, tmp_path, monkeypatch):
    source = make_source(kind, tmp_path, monkeypatch, lambda offset, size: bytes([offset]) * size)
    created = []
    lock = threading.Lock()
    def slow_pool(*args, **kwargs):
        time.sleep(0.02)
        pool = ThreadPoolExecutor(*args, **kwargs)
        with lock:
            created.append(pool)
        return pool
    barrier = threading.Barrier(2)
    def read():
        barrier.wait(timeout=3)
        return source.read_many([(0, 8), (64, 8)])
    try:
        with ThreadPoolExecutor(max_workers=2) as callers:
            monkeypatch.setattr('opencodecs._tiff_http.concurrent.futures.ThreadPoolExecutor', slow_pool)
            jobs = [callers.submit(read) for _ in range(2)]
            for job in jobs:
                assert job.result(timeout=3) == [bytes(8), bytes([64]) * 8]
        assert len(created) == 1
    finally:
        source.close()
        for pool in created:
            pool.shutdown(wait=True)


@pytest.mark.parametrize('kind', ['http', 'file'])
def test_batch_failure_joins_its_other_source_users(kind, tmp_path, monkeypatch):
    started, release, finished, returned = (threading.Event() for _ in range(4))
    errors = []
    def read(offset, size):
        if offset == 0:
            assert started.wait(3)
            raise IOError('fixture range failure')
        started.set()
        try:
            assert release.wait(3)
            return bytes([offset]) * size
        finally:
            finished.set()
    source = make_source(kind, tmp_path, monkeypatch, read)
    def caller():
        try:
            source.read_many([(0, 8), (64, 8)])
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()
    thread = threading.Thread(target=caller)
    thread.start()
    try:
        assert started.wait(3)
        assert not returned.wait(0.05), 'batch returned while another source user was active'
        release.set()
        assert returned.wait(3)
        assert finished.is_set()
        assert len(errors) == 1 and isinstance(errors[0], IOError)
        assert str(errors[0]) == 'fixture range failure'
    finally:
        release.set()
        thread.join(timeout=3)
        source.close()


@pytest.mark.parametrize('owned', [False, True])
def test_background_reader_close_waits_for_blocked_owned_or_borrowed_read(tmp_path, monkeypatch, owned):
    from opencodecs.core.io import BackgroundChunkReader
    started, release, finished, returned = (threading.Event() for _ in range(4))
    path = tmp_path / 'background.bin'
    path.write_bytes(bytes(2048))
    raw = path.open('rb')
    errors = []
    class BlockedFile:
        closed = False
        def fileno(self):
            return raw.fileno()
        def read(self, size):
            started.set()
            try:
                assert release.wait(5)
                return raw.read(size)
            finally:
                finished.set()
        def close(self):
            assert finished.is_set(), 'source closed while its read was active'
            self.closed = True
            raw.close()
    source = BlockedFile()
    if owned:
        monkeypatch.setattr('opencodecs.core.io.open', lambda name, mode: source, raising=False)
    reader = BackgroundChunkReader(path if owned else source, chunk_size=1024, prefetch=1)
    def close():
        try:
            reader.close()
        except BaseException as exc:
            errors.append(exc)
        finally:
            returned.set()
    closer = threading.Thread(target=close)
    try:
        assert started.wait(3)
        closer.start()
        # Exceed the former one-second join timeout to pin the lifetime bug.
        assert not returned.wait(1.1)
        assert not source.closed
        release.set()
        assert returned.wait(3)
        assert finished.is_set()
        assert not reader._thread.is_alive()
        assert source.closed is owned
        assert errors == []
    finally:
        release.set()
        if closer.ident is not None:
            closer.join(timeout=3)
        reader.close()
        raw.close()
