"""How ``CziReader.read`` hands sub-blocks to threads.

``read`` sizes each scheduled task by output bytes rather than by worker
count, because deriving the batch size from the worker count made the read
wait on ``ceil(n/workers)`` sub-blocks even when most workers had already
finished. Two properties of that need holding down, and neither is visible
in the decoded pixels:

**The partition.** Every partition of the same sub-blocks decodes to the
same array, so an output comparison passes whatever the task boundaries
are. That is not hypothetical: the first version of these tests only
compared arrays, and every case in it happened to fall into the serial
branch, so the parallel dispatch it was named after went untested.
``_read_tasks`` is asserted directly instead.

**The worker bound.** When the byte sizing yields more tasks than the
resolved worker count, a shared pool bigger than that count would run all
of them at once. Too many threads first-touching one fresh output measured
slower, which is the whole reason there is a bound. An unbounded pool still
returns correct pixels, so these watch peak concurrency.
"""
from __future__ import annotations

import pathlib
import sys
import threading
import time

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from _czi_fixture import mosaic_czi_bytes  # noqa: E402

from opencodecs import _czi_reader as cz  # noqa: E402
from opencodecs._czi_reader import CziReader, _read_tasks  # noqa: E402

MIB = 1 << 20


def _tiles(n, size, seed=0):
    rs = np.random.RandomState(seed)
    return [rs.randint(0, 4096, (size, size)).astype(np.uint16) for _ in range(n)]


def _mosaic(tiles):
    return mosaic_czi_bytes(
        [(t, (0, i * t.shape[1])) for i, t in enumerate(tiles)],
        compression=6, hilo=True)


# ---------------------------------------------------------------- partition

@pytest.mark.parametrize("n,tile_bytes,budget,expected", [
    # A sub-block at or over the budget gets a task to itself.
    (4, 8 * MIB, 8 * MIB, [range(0, 1), range(1, 2), range(2, 3), range(3, 4)]),
    (2, 32 * MIB, 8 * MIB, [range(0, 1), range(1, 2)]),
    # Smaller ones ride together, and the last task is short.
    (10, 2 * MIB, 8 * MIB, [range(0, 4), range(4, 8), range(8, 10)]),
    (63, 1 * MIB, 8 * MIB, [range(0, 8), range(8, 16), range(16, 24),
                            range(24, 32), range(32, 40), range(40, 48),
                            range(48, 56), range(56, 63)]),
    # Everything fits in one task: the caller takes the serial path.
    (5, 1024, 8 * MIB, [range(0, 5)]),
    (1, 8 * MIB, 8 * MIB, [range(0, 1)]),
    # A zero-byte tile must not divide by zero.
    (3, 0, 8 * MIB, [range(0, 3)]),
])
def test_read_tasks_partition(n, tile_bytes, budget, expected):
    assert _read_tasks(n, tile_bytes, budget) == expected


@pytest.mark.parametrize("n,tile_bytes", [(1, 512), (7, 3 * MIB), (64, 9 * MIB)])
def test_read_tasks_covers_every_index_exactly_once(n, tile_bytes):
    seen = [i for task in _read_tasks(n, tile_bytes) for i in task]
    assert seen == list(range(n))


# ------------------------------------------------------------- worker bound

class _ConcurrencyWatch:
    """Record peak overlap of the decode calls a read makes."""

    def __init__(self, hold=0.05):
        self.hold = hold
        self.peak = 0
        self.calls = 0
        self._live = 0
        self._lock = threading.Lock()

    def wrap(self, real):
        def decode(*args, **kwargs):
            with self._lock:
                self._live += 1
                self.calls += 1
                self.peak = max(self.peak, self._live)
            try:
                # Hold the slot open so genuinely concurrent workers overlap;
                # without it fast tiles can finish before the next one starts
                # and a parallel run would look serial.
                time.sleep(self.hold)
                return real(*args, **kwargs)
            finally:
                with self._lock:
                    self._live -= 1
        return decode


@pytest.fixture
def small_budget(monkeypatch):
    """One task per sub-block for cheap test-sized tiles."""
    monkeypatch.setattr(cz, "_READ_TASK_BYTES", 1024)


def _watched_read(monkeypatch, data, **kw):
    watch = _ConcurrencyWatch()
    monkeypatch.setattr(CziReader, "_decode_one",
                        watch.wrap(CziReader._decode_one), raising=True)
    with CziReader(buffer=data) as r:
        n = len(r.entries)
        tile_bytes = r.entries[0].dtype.itemsize * int(
            np.prod(r.entries[0].stored_shape))
        tasks = _read_tasks(n, tile_bytes, cz._READ_TASK_BYTES)
        out = r.read(**kw)
    return out, watch, len(tasks), n


def test_read_runs_tasks_concurrently(monkeypatch, small_budget):
    """One task per sub-block, enough workers for all of them: they overlap.

    The worker count is explicit because the automatic one is also bounded by
    output bytes, and a test-sized read is far under that floor (see
    ``test_a_tiny_read_stays_serial``).
    """
    tiles = _tiles(8, 32, seed=1)
    out, watch, n_tasks, n = _watched_read(
        monkeypatch, _mosaic(tiles), n_workers=8)
    assert n_tasks == n == 8, "budget should give one task per sub-block"
    assert watch.calls == 8
    assert watch.peak > 1, "tasks ran one after another; dispatch is not parallel"
    assert np.array_equal(out, np.stack(tiles, axis=0))


def test_a_tiny_read_stays_serial(monkeypatch, small_budget):
    """Byte-sized tasks do not override the "too little to thread" floor.

    ``resolve_workers`` caps the count by output bytes as well, so a read
    this small resolves to one worker however many tasks the byte sizing
    produced. Pinned because it surprised me while writing these: the
    partition is per-sub-block here, and the read is still serial.
    """
    tiles = _tiles(8, 32, seed=5)
    out, watch, n_tasks, _ = _watched_read(monkeypatch, _mosaic(tiles))
    assert n_tasks == 8
    assert watch.peak == 1
    assert np.array_equal(out, np.stack(tiles, axis=0))


def test_worker_count_bounds_concurrency(monkeypatch, small_budget):
    """More tasks than workers must not all run at once.

    This is the semaphore's job. The shared pool is far larger than the
    budget, so without it every task starts immediately and the decoded
    pixels are still perfectly correct.
    """
    tiles = _tiles(8, 32, seed=2)
    out, watch, n_tasks, _ = _watched_read(
        monkeypatch, _mosaic(tiles), n_workers=2)
    assert n_tasks == 8, "need more tasks than workers to exercise the bound"
    assert watch.calls == 8
    assert watch.peak > 1, "the gated branch did not run anything in parallel"
    assert watch.peak <= 2, f"worker budget of 2 was exceeded (peak {watch.peak})"
    assert np.array_equal(out, np.stack(tiles, axis=0))


def test_serial_read_never_overlaps(monkeypatch, small_budget):
    tiles = _tiles(4, 32, seed=3)
    out, watch, _, _ = _watched_read(monkeypatch, _mosaic(tiles), n_workers=1)
    assert watch.peak == 1
    assert np.array_equal(out, np.stack(tiles, axis=0))


# ---------------------------------------------------------------- integrity

@pytest.mark.parametrize("n,size", [(8, 32), (3, 64), (17, 16)])
def test_parallel_read_matches_serial(monkeypatch, n, size):
    """Same pixels whatever the partition, across every dispatch branch."""
    tiles = _tiles(n, size, seed=4)
    data = _mosaic(tiles)
    expected = np.stack(tiles, axis=0)
    with CziReader(buffer=data) as r:
        assert np.array_equal(r.read(n_workers=1), expected)
    for budget in (1024, 4096, 1 << 30):   # per-tile, grouped, single task
        monkeypatch.setattr(cz, "_READ_TASK_BYTES", budget)
        with CziReader(buffer=data) as r:
            assert np.array_equal(r.read(), expected), f"budget={budget}"
            # An explicit count is a budget, not an instruction to spend it.
            assert np.array_equal(r.read(n_workers=64), expected)


# ---------------------------------------------------------------- stored pixels

@pytest.mark.parametrize("n", [9, 3])
def test_stored_stack_reads_on_more_than_one_worker(monkeypatch, tmp_path, n):
    """Uncompressed 2 MiB planes read from a file go out one per task, on
    several workers. With the 8 MiB tasks a zstd read uses, nine planes were
    three tasks, under the worker policy's default minimum of four, and the
    read ran on one thread: 0.59x the speed of a reader copying from an
    mmap. Three planes check the minimum itself (``_READ_MIN_TASKS``). Only
    the dispatch shows either; the pixels are right regardless."""
    import os
    from opencodecs.core import parallel
    if (os.cpu_count() or 1) < 2:
        pytest.skip("one CPU: nothing to divide")
    tiles = _tiles(n, 1024, seed=7)
    path = tmp_path / "stored.czi"
    path.write_bytes(mosaic_czi_bytes(
        [(t, (0, i * t.shape[1])) for i, t in enumerate(tiles)], compression=0))
    seen = []
    orig = parallel.resolve_workers

    def spy(*args, **kw):
        workers = orig(*args, **kw)
        seen.append((args[1], workers))
        return workers

    monkeypatch.setattr(parallel, "resolve_workers", spy)
    with CziReader(path) as r:
        out = r.read()
    assert np.array_equal(out, np.stack(tiles, axis=0))
    n_tasks, workers = seen[-1]
    assert n_tasks == n
    assert workers >= 2
