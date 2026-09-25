"""Bounded independent-piece scheduling shared by readers and writers.

Adapters supply a conservative byte reservation covering owned input, codec
scratch, and retained output for one task. Caller-owned source and final output
arrays are outside that reservation. A single oversized item runs alone.
"""
from __future__ import annotations

from collections import deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from contextlib import nullcontext
from dataclasses import dataclass
import operator
import threading

_STATE = threading.local()


def in_worker() -> bool:
    return bool(getattr(_STATE, 'depth', 0))


def native_workers(requested=None):
    """Use one native worker inside an outer piece worker; preserve eager policy."""
    return 1 if in_worker() else requested


class WorkerBudget:
    """A shareable limit for simultaneously active outer workers.

    Pass the same instance to concurrent pipelines to share a budget. Nested
    pipelines run inline, so a worker never waits for its own occupied slot.
    Independent callers can retain separate budgets intentionally.
    """
    def __init__(self, workers):
        self.workers = max(1, operator.index(workers))
        self._slots = threading.BoundedSemaphore(self.workers)

    def run(self, fn, item):
        if in_worker():
            return fn(item)
        with self._slots:
            _STATE.depth = 1
            try:
                return fn(item)
            finally:
                _STATE.depth = 0


@dataclass
class PipelineStats:
    """Reservation and scheduling counters, independent of allocator tracing."""
    submitted: int = 0
    completed: int = 0
    peak_pending: int = 0
    peak_reserved_bytes: int = 0
    oversized_items: int = 0


def map_bounded(fn, items, workers=1, *, max_pending=None,
                max_bytes=64 << 20, size=lambda item: 1, prepare=None,
                budget=None, stats=None, name='pipeline', executor=None):
    """Yield results in order with task and byte backpressure.

    ``size(item)`` reserves bytes before ``prepare(item)`` takes ownership.
    Preparation runs on the producer thread before advancing the source, so it
    can snapshot reused acquisition buffers and metadata. Items should be small
    descriptors or borrowed input until prepared. Reservations must include both
    input and result lifetimes, plus codec scratch. Oversized items run alone.

    The current yielded result remains reserved until iteration resumes. Close
    this iterator on early exit (for example with contextlib.closing): queued
    work is canceled and running source users are joined before close returns.
    A supplied executor remains caller-owned and usable after close. Cleanup
    joins only this iterator's submitted work, never unrelated executor jobs.
    The source iterator itself remains caller-owned. Adapter-internal caches,
    native allocator overhead, and the caller's final output need separate limits.
    """
    workers = max(1, operator.index(workers))
    max_bytes = operator.index(max_bytes)
    if max_bytes < 1:
        raise ValueError('max_bytes must be positive')
    if max_pending is None:
        max_pending = workers + 1
    max_pending = operator.index(max_pending)
    if max_pending < 1:
        raise ValueError('max_pending must be positive')
    if budget is None:
        budget = WorkerBudget(workers)
    workers = min(workers, budget.workers)
    if in_worker():
        workers = 1
    stats = PipelineStats() if stats is None else stats

    def cost(item):
        value = operator.index(size(item))
        if value < 0:
            raise ValueError('item byte reservation must not be negative')
        return value

    def account(count, reserved, amount):
        stats.submitted += 1
        stats.peak_pending = max(stats.peak_pending, count)
        stats.peak_reserved_bytes = max(stats.peak_reserved_bytes, reserved)
        if amount > max_bytes:
            stats.oversized_items += 1

    if workers == 1:
        for item in items:
            amount = cost(item)
            owned = item if prepare is None else prepare(item)
            account(1, amount, amount)
            result = budget.run(fn, owned)
            stats.completed += 1
            del owned, item
            yield result
            del result
        return

    pending = deque()
    source = iter(items)
    sentinel = object()
    lookahead = sentinel
    exhausted = False
    reserved = 0
    context = (ThreadPoolExecutor(max_workers=workers,
                                  thread_name_prefix=f'opencodecs-{name}')
               if executor is None else nullcontext(executor))
    with context as pool:
        try:
            while pending or not exhausted:
                while len(pending) < max_pending and not exhausted:
                    if lookahead is sentinel:
                        lookahead = next(source, sentinel)
                        if lookahead is sentinel:
                            exhausted = True
                            break
                        amount = cost(lookahead)
                    # No prepared input may exceed the current reservation.
                    # One borrowed descriptor can wait without advancing source.
                    if pending and reserved + amount > max_bytes:
                        break
                    owned = lookahead if prepare is None else prepare(lookahead)
                    lookahead = sentinel
                    future = pool.submit(budget.run, fn, owned)
                    del owned
                    pending.append((future, amount))
                    del future
                    reserved += amount
                    account(len(pending), reserved, amount)
                    if amount > max_bytes:
                        break
                if not pending:
                    continue
                future, amount_done = pending[0]
                result = future.result()
                pending.popleft()
                del future
                stats.completed += 1
                yield result
                del result
                reserved -= amount_done
        finally:
            for future, _ in pending:
                future.cancel()
            # Caller-owned pools must outlive this iterator, but its own
            # active source users must finish before the source can close.
            # exception() waits without replacing an existing failure or
            # GeneratorExit with a background worker's exception.
            for future, _ in pending:
                try:
                    future.exception()
                except CancelledError:
                    pass
            pending.clear()
            lookahead = sentinel


def range_batches(items, size, *, max_bytes=4 << 20, max_items=64):
    """Group range descriptors without splitting any independently coded piece."""
    if max_bytes < 1 or max_items < 1:
        raise ValueError('batch limits must be positive')
    batch, count = [], 0
    for item in items:
        amount = operator.index(size(item))
        if amount < 0:
            raise ValueError('item size must not be negative')
        if batch and (count + amount > max_bytes or len(batch) >= max_items):
            yield tuple(batch)
            batch, count = [], 0
        batch.append(item)
        count += amount
    if batch:
        yield tuple(batch)


__all__ = ['map_bounded', 'range_batches', 'WorkerBudget', 'PipelineStats',
           'native_workers', 'in_worker']
