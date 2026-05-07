"""Tests for opencodecs.core.io.BackgroundChunkReader.

This is the substrate underneath JXL's opt-in streaming mode and any
future "decode N chunks while reading the next" wrapper. Even though
the JXL benchmarks showed slurp-then-decode wins on fast storage, the
infrastructure must work correctly — synchronization, EOF sentinels,
exception propagation, backpressure, and resource cleanup are all
catastrophic if broken.
"""

from __future__ import annotations

import io
import os
import threading
import time

import pytest

from opencodecs.core.io import BackgroundChunkReader


# ---------------------------------------------------------------------------
# Basic correctness: read produces byte-identical output to slurp
# ---------------------------------------------------------------------------


def test_chunked_read_byte_identical_to_slurp(tmp_path):
    """Read a file in chunks and verify the concatenation matches the
    file's bytes exactly. The classic correctness test."""
    payload = os.urandom(7 * 1024 * 1024 + 17)  # not chunk-aligned on purpose
    p = tmp_path / "data.bin"
    p.write_bytes(payload)

    with BackgroundChunkReader(p, chunk_size=1024 * 1024) as r:
        chunks = list(r)
    assert b"".join(chunks) == payload


def test_chunked_read_via_iterator(tmp_path):
    """The reader is a proper iterator — `for chunk in r` exhausts it."""
    payload = b"x" * 100_000
    p = tmp_path / "data.bin"
    p.write_bytes(payload)

    out = bytearray()
    with BackgroundChunkReader(p, chunk_size=4096) as r:
        for chunk in r:
            out.extend(chunk)
    assert bytes(out) == payload


def test_chunked_read_smaller_than_chunk_size(tmp_path):
    """File smaller than one chunk should still come back in a single
    chunk, then EOF."""
    payload = b"hello world"
    p = tmp_path / "tiny.bin"
    p.write_bytes(payload)

    with BackgroundChunkReader(p, chunk_size=64 * 1024) as r:
        chunks = list(r)
    assert b"".join(chunks) == payload


def test_chunked_read_empty_file(tmp_path):
    """Empty file → no chunks before EOF, get() returns None."""
    p = tmp_path / "empty.bin"
    p.write_bytes(b"")

    with BackgroundChunkReader(p, chunk_size=4096) as r:
        chunk = r.get()
        assert chunk is None
        # Subsequent calls keep returning None.
        assert r.get() is None
        assert list(r) == []


# ---------------------------------------------------------------------------
# Source variants: path, BytesIO, file object
# ---------------------------------------------------------------------------


def test_chunked_read_from_bytesio():
    payload = os.urandom(64 * 1024)
    bio = io.BytesIO(payload)
    with BackgroundChunkReader(bio, chunk_size=4096) as r:
        chunks = list(r)
    assert b"".join(chunks) == payload


def test_chunked_read_from_open_file(tmp_path):
    payload = os.urandom(64 * 1024)
    p = tmp_path / "data.bin"
    p.write_bytes(payload)

    with open(p, "rb") as f:
        with BackgroundChunkReader(f, chunk_size=4096) as r:
            chunks = list(r)
    assert b"".join(chunks) == payload


def test_non_file_like_source_raises():
    """A plain int isn't a path or a file-like; reader should reject."""
    with pytest.raises(TypeError, match="path or file-like"):
        BackgroundChunkReader(42)


# ---------------------------------------------------------------------------
# Constructor parameter validation
# ---------------------------------------------------------------------------


def test_chunk_size_too_small_raises():
    with pytest.raises(ValueError, match="chunk_size"):
        BackgroundChunkReader(io.BytesIO(b""), chunk_size=512)


def test_prefetch_zero_raises():
    with pytest.raises(ValueError, match="prefetch"):
        BackgroundChunkReader(io.BytesIO(b""), prefetch=0)


# ---------------------------------------------------------------------------
# file_size attribute populated correctly for paths
# ---------------------------------------------------------------------------


def test_file_size_populated_for_path(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"x" * 12345)
    with BackgroundChunkReader(p) as r:
        assert r.file_size == 12345


def test_file_size_populated_for_seekable_file(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"x" * 7777)
    with open(p, "rb") as f:
        with BackgroundChunkReader(f) as r:
            assert r.file_size == 7777


def test_file_size_none_for_non_seekable():
    """A non-seekable stream has no known size."""
    class NonSeekable:
        def __init__(self, data):
            self._data = data
            self._pos = 0

        def read(self, n):
            chunk = self._data[self._pos:self._pos + n]
            self._pos += len(chunk)
            return chunk

    with BackgroundChunkReader(NonSeekable(b"x" * 100)) as r:
        assert r.file_size is None


# ---------------------------------------------------------------------------
# Backpressure: bg thread blocks on full queue, doesn't buffer unbounded
# ---------------------------------------------------------------------------


def test_backpressure_limits_outstanding_chunks(tmp_path):
    """With a slow consumer, the bg thread should sit idle on a full
    queue rather than buffering the entire file. Verify by watching the
    queue size never exceeds prefetch+1 (the +1 is the chunk currently
    in the bg thread's hand)."""
    payload = os.urandom(8 * 1024 * 1024)
    p = tmp_path / "data.bin"
    p.write_bytes(payload)

    with BackgroundChunkReader(p, chunk_size=64 * 1024, prefetch=2) as r:
        # Read one chunk to start the pipeline.
        first = r.get()
        # Give the bg thread time to fill the queue to capacity.
        time.sleep(0.1)
        qsize = r._queue.qsize()
        assert qsize <= r._prefetch, (
            f"queue grew to {qsize}, expected <= prefetch={r._prefetch}")
        # Drain the rest.
        rest = b"".join(iter(r))
    assert first + rest == payload


# ---------------------------------------------------------------------------
# Exception propagation: errors in bg thread reach the consumer
# ---------------------------------------------------------------------------


def test_read_exception_propagates_to_consumer():
    """If file.read() raises in the bg thread, the consumer should see
    that exception, not hang waiting for chunks."""
    class FailingFile:
        def read(self, n):
            raise IOError("simulated read failure")

    with BackgroundChunkReader(FailingFile()) as r:
        with pytest.raises(IOError, match="simulated"):
            r.get()


# ---------------------------------------------------------------------------
# Cleanup: closing while bg thread is busy doesn't deadlock
# ---------------------------------------------------------------------------


def test_close_during_read_terminates_thread(tmp_path):
    """Close mid-read should signal the bg thread and let it exit
    without leaving zombie threads around."""
    payload = os.urandom(16 * 1024 * 1024)
    p = tmp_path / "data.bin"
    p.write_bytes(payload)

    r = BackgroundChunkReader(p, chunk_size=4096, prefetch=1)
    # Read just one chunk, then close abruptly.
    _ = r.get()
    thread = r._thread
    r.close()
    # Bg thread should exit quickly. Allow generous timeout for slow CI.
    thread.join(timeout=2.0)
    assert not thread.is_alive(), "bg thread didn't exit after close()"


def test_double_close_is_safe(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"x" * 1024)
    r = BackgroundChunkReader(p, chunk_size=1024)
    r.close()
    r.close()  # idempotent — must not raise


def test_context_manager_closes_on_exception(tmp_path):
    """Even if the body raises, the bg thread should be cleaned up."""
    p = tmp_path / "data.bin"
    p.write_bytes(b"x" * 1024)
    thread_ref = []

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with BackgroundChunkReader(p) as r:
            thread_ref.append(r._thread)
            raise Boom()

    thread_ref[0].join(timeout=2.0)
    assert not thread_ref[0].is_alive()


# ---------------------------------------------------------------------------
# EOF sentinel: get() returns None deterministically after end-of-file
# ---------------------------------------------------------------------------


def test_get_after_eof_keeps_returning_none(tmp_path):
    p = tmp_path / "data.bin"
    p.write_bytes(b"hi")
    with BackgroundChunkReader(p, chunk_size=4096) as r:
        assert r.get() == b"hi"
        for _ in range(5):
            assert r.get() is None  # repeated EOF reads stay None
