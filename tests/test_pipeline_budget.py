"""Shared pipeline bounds, producer ownership, and cancellation contracts."""
from contextlib import closing
import threading
import time

import numpy as np
import pytest
from opencodecs.core.pipeline import (map_bounded, range_batches, PipelineStats,
                                      WorkerBudget, native_workers)


def test_bytes_include_yielded_result_and_oversized_items_run_alone():
    stats = PipelineStats()
    sizes = [4, 7, 6, 23, 2, 3]
    result = list(map_bounded(lambda x: x, sizes, 4, max_bytes=16,
                              size=lambda x: x, stats=stats))
    assert result == sizes
    assert stats.peak_reserved_bytes == 23
    assert stats.oversized_items == 1
    assert stats.submitted == stats.completed == len(sizes)


@pytest.mark.parametrize('workers', [1, 4])
def test_prepare_owns_reused_pixels_and_metadata_before_next_yield(workers):
    arr = np.zeros(100, dtype='u2')
    metadata = {}
    def source():
        for i in range(50):
            arr.fill(i)
            metadata['frame'] = i
            yield arr, metadata
    def prepare(item):
        return item[0].copy(), dict(item[1])
    def decode(item):
        pixels, axes = item
        assert np.all(pixels == axes['frame'])
        return axes['frame']
    result = list(map_bounded(decode, source(), workers, prepare=prepare,
                              size=lambda _: 400, max_bytes=1600))
    assert result == list(range(50))


def test_early_close_joins_running_workers_and_stops_source():
    consumed = []
    def source():
        for i in range(200):
            consumed.append(i)
            yield i
    with closing(map_bounded(lambda i: i, source(), 3, max_pending=3,
                             max_bytes=2, name='cancel-budget')) as it:
        assert next(it) == 0
        assert len(consumed) <= 3  # two reservations and one borrowed lookahead
    assert not any(t.name.startswith('opencodecs-cancel-budget')
                   for t in threading.enumerate())


def test_worker_and_producer_errors_do_not_leave_workers():
    def source():
        yield 0
        raise RuntimeError('producer failed')
    with pytest.raises(RuntimeError, match='producer failed'):
        list(map_bounded(lambda i: i, source(), 2, name='failure-budget'))
    def worker(i):
        raise RuntimeError('codec failed')
    with pytest.raises(RuntimeError, match='codec failed'):
        list(map_bounded(worker, range(100), 3, name='failure-budget'))
    assert not any(t.name.startswith('opencodecs-failure-budget')
                   for t in threading.enumerate())


def test_shared_budget_and_nested_pipelines_do_not_oversubscribe_or_deadlock():
    from concurrent.futures import ThreadPoolExecutor
    budget = WorkerBudget(2)
    lock = threading.Lock()
    active = 0
    peak = 0
    def inner(i):
        assert native_workers(8) == 1
        return i + 1
    def outer(i):
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(0.002)
            return list(map_bounded(inner, [i], 4, budget=budget))[0]
        finally:
            with lock:
                active -= 1
    def pipeline():
        return list(map_bounded(outer, range(10), 4, budget=budget))
    with ThreadPoolExecutor(3) as ex:
        assert all(v == list(range(1, 11)) for v in ex.map(lambda _: pipeline(), range(3)))
    assert peak <= 2
    assert native_workers(None) is None
    assert native_workers(8) == 8


def test_range_batches_preserve_order_without_splitting_large_items():
    assert list(range_batches([4, 3, 9, 2, 1], lambda n: n,
                              max_bytes=8, max_items=2)) == [(4, 3), (9,), (2, 1)]


@pytest.mark.parametrize('kwargs', [dict(max_bytes=0), dict(max_pending=0),
                                   dict(size=lambda _: -1)])
def test_invalid_budget_is_rejected(kwargs):
    with pytest.raises(ValueError):
        list(map_bounded(lambda i: i, [1], 2, **kwargs))
