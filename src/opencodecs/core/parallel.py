"""Deciding how many threads to spend, and spending them.

Several formats here have the same shape underneath: a file is a
sequence of pieces that decode without reference to each other. TIFF
strips and tiles, HDF5 chunks, DICOM frames, EER frames, ND2 chunks --
what differs between them is only what a piece is. Two decisions do not
differ at all, and were being made again in each codec:

**How many workers.** A fixed count is wrong at both ends, and
measurably so. On a 9 MB TIFF the whole decode is a few milliseconds
and 16 threads lose to 8 on pool overhead; on a 72 MB one, 16 threads
are 12x serial where 8 are only 7x. Sizing the pool by how much output
there actually is gets both, where any single constant gives up one.

**How to hand out the work.** ThreadPoolExecutor.map ignores its
``chunksize`` argument -- that is a process-pool knob -- so the obvious
one-future-per-piece loop gave a 144-tile image 144 futures, each with
its own lock traffic and GIL round trip, wrapping work that for an
uncompressed tile is a reshape and a memcpy. That overhead alone made
threading measure slower than serial. Batching contiguously by worker
fixes it and, as a side effect, keeps each worker writing to a
contiguous band of the output.

The knobs are arguments rather than constants because the crossover is
a property of the format: a TIFF has hundreds of tiles and a DICOM
study has tens of frames, so "too few to bother" is a different number
for each.

**Where the threads come from.** One persistent pool, created on first
use and kept for the life of the process (``shared_pool``). A fresh
ThreadPoolExecutor per call cost 0.2 ms for 2 workers and 1.3 ms for
16, which on a 144-tile TIFF is 7% of the whole read. Each call still
runs at most its own worker count at once: it submits that many batches
and its WorkerBudget gates them.

**Sharing the process.** When several threads call in at once, each
automatic worker count is a share rather than the whole
(``fair_share``). For these helpers the share is of one call's full
width, not of the CPU count: their workers hand the GIL back and forth
between the steps of every piece, and a process has one GIL however
many cores it has. Eight callers reading a tiled TIFF with 16 workers
each measured 0.69x the throughput of the same callers reading serially
on a 64-core host. Native codec threads that never touch the GIL (JPEG
XL, AVIF) divide the CPU count instead (``auto_threads``). A lone call
still gets full width, and an explicit ``numthreads`` is always honored
as given.
"""

from __future__ import annotations

import os
import threading
from contextlib import contextmanager
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

#: Past this, more threads buy nothing on any machine here and start
#: costing pool overhead and memory bandwidth contention.
DEFAULT_MAX_WORKERS = 16

#: Fewer pieces than this and the pool costs more than it saves.
DEFAULT_MIN_ITEMS = 4

#: Don't start a worker for less output than this.
DEFAULT_MIN_BYTES_PER_WORKER = 1 << 20


_INFLIGHT = 0
_INFLIGHT_LOCK = threading.Lock()


@contextmanager
def parallel_call():
    """Mark a call as spending more than one thread while it runs.

    Explicit and automatic counts both register: either way the call is
    using the machine, and later automatic calls should see that.
    """
    global _INFLIGHT
    with _INFLIGHT_LOCK:
        _INFLIGHT += 1
    try:
        yield
    finally:
        with _INFLIGHT_LOCK:
            _INFLIGHT -= 1


def fair_share(total: int | None = None) -> int:
    """This call's share of ``total`` threads (default: the CPU count).

    Divided between the parallel calls already running and this one, so
    a lone call gets all of it and N overlapping calls get about 1/N
    each. A call keeps the share it was given when it started; it is not
    rebalanced as others begin or end.
    """
    if total is None:
        total = os.cpu_count() or 1
    return max(1, total // (_INFLIGHT + 1))


def auto_threads(numthreads: int | None, *, work: int | None = None,
                 per_thread: int = 1,
                 max_threads: int = DEFAULT_MAX_WORKERS) -> int:
    """Thread count for a native codec's own threads (JPEG XL, AVIF).

    An explicit count is returned as given. ``None`` sizes by the work
    (``work // per_thread`` threads, when the work is known) under
    ``max_threads``, the CPU count and this call's fair share; inside an
    outer worker it is 1, since that worker is already one of the
    threads the machine is being divided between.
    """
    if numthreads is not None:
        return int(numthreads)
    from .pipeline import in_worker
    if in_worker():
        return 1
    n = max_threads if not work else max(1, int(work) // max(1, per_thread))
    return max(1, min(n, max_threads, os.cpu_count() or 1, fair_share()))


def shared_pool():
    """The process-wide pool the batch helpers run on, created on first use.

    Sized generously because its threads are only started as work
    arrives; how many run at once is each call's own worker count.
    """
    from .io import get_reader_pool
    return get_reader_pool("shared")


def resolve_workers(numthreads: int | None, n_items: int, *,
                    has_decode_work: bool = True,
                    output_bytes: int | None = None,
                    min_items: int = DEFAULT_MIN_ITEMS,
                    max_workers: int = DEFAULT_MAX_WORKERS,
                    min_bytes_per_worker: int = DEFAULT_MIN_BYTES_PER_WORKER,
                    ) -> int:
    """How many threads to decode ``n_items`` independent pieces with.

    ``None`` means "decide": scale with the CPU count, never exceed the
    number of pieces, stay serial when there is too little to divide, and
    take only this call's share of ``max_workers`` when other parallel
    calls are already running (see "Sharing the process" above). An
    explicit number is honored as given, so a caller can pin it,
    including to 1 for a reproducible serial run.

    ``has_decode_work`` is the one case where an explicit number is
    NOT honored. It is false when the pieces need no decoding -- an
    uncompressed TIFF tile, a raw NRRD slice -- and there the answer is
    1 whatever was asked for. A thread count is a budget, "use up to
    N", not an instruction to spend it, and spending it here cannot
    pay: the work is a memcpy, already memory-bandwidth-bound at about
    5 GB/s on one core. Measured on a 144-tile 3072x3072 uncompressed
    TIFF, the whole read is 1.8 ms serial and 3.6 ms on 4 threads; the
    same file in LZW goes 5.9x faster on 8. Honoring the number
    literally would mean quietly doing what the caller asked, twice as
    slowly.
    """
    from .pipeline import in_worker
    if not has_decode_work or in_worker():
        return 1
    if numthreads is not None:
        n = int(numthreads)
        if n <= 1:
            return 1
        return min(n, max(1, n_items))
    if n_items < min_items:
        return 1
    cap = min(os.cpu_count() or 1, n_items, fair_share(max_workers))
    if output_bytes is not None:
        cap = min(cap, output_bytes // min_bytes_per_worker)
    return max(1, cap)


def run_batched(fn: Callable[[T], None], items: Sequence[T], workers: int, *,
                name: str = "decode", budget=None) -> None:
    """Apply ``fn`` to every item, on ``workers`` threads, in batches.

    ``workers <= 1`` runs inline and creates no pool at all, which is
    what makes it safe to call this unconditionally: the serial path
    stays exactly as cheap as the loop it replaces.

    ``fn`` is expected to write its result somewhere the caller owns,
    typically a slice of a preallocated output array, so nothing is
    collected here and no ordering is imposed beyond the batching.
    Exceptions propagate, and never before every batch has stopped: a
    batch still writing into the output after the caller has its error
    would be a race on memory the caller now thinks is idle.
    """
    if workers <= 1 or len(items) <= 1:
        for it in items:
            fn(it)
        return

    from concurrent.futures import wait
    from .pipeline import WorkerBudget, in_worker
    if in_worker():
        for it in items:
            fn(it)
        return
    budget = WorkerBudget(workers) if budget is None else budget
    workers = min(workers, budget.workers)

    step = (len(items) + workers - 1) // workers
    batches = [items[i:i + step] for i in range(0, len(items), step)]

    def _run_batch(batch) -> None:
        for it in batch:
            fn(it)

    pool = shared_pool()
    with parallel_call():
        futures = [pool.submit(budget.run, _run_batch, b) for b in batches]
        try:
            for f in futures:
                f.result()
        finally:
            for f in futures:
                f.cancel()
            wait(futures)


__all__ = ["resolve_workers", "run_batched", "DEFAULT_MAX_WORKERS",
           "DEFAULT_MIN_ITEMS", "DEFAULT_MIN_BYTES_PER_WORKER", "map_batches",
           "shared_pool", "parallel_call", "fair_share", "auto_threads"]


def map_batches(fn, items, workers: int, *, batch_size: int = 16,
                max_pending: int | None = None, name: str = "codec", budget=None):
    """Yield ordered result batches without consuming the whole input.

    Each future handles a batch, amortizing scheduling for small tiles.
    At most max_pending batches are submitted, plus the batch currently
    yielded to the consumer. Consumption supplies backpressure. A writer
    can emit one result while workers encode the following batches.

    The caller must consume or close the iterator. Early close or an
    exception cancels queued work and waits for this iterator's running
    batches before return. Serial mode uses the same batches and never
    touches a pool.
    """
    from itertools import islice
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    from .pipeline import WorkerBudget, in_worker
    workers = max(1, int(workers))
    budget = WorkerBudget(workers) if budget is None else budget
    workers = min(workers, budget.workers)
    if in_worker():
        workers = 1
    if max_pending is None:
        max_pending = workers + 1
    if max_pending < 1:
        raise ValueError("max_pending must be positive")
    source = iter(items)

    def batches():
        while True:
            batch = tuple(islice(source, batch_size))
            if not batch:
                return
            yield batch

    def apply(batch):
        return tuple(fn(item) for item in batch)

    if workers == 1:
        for batch in batches():
            yield apply(batch)
        return

    from collections import deque
    from concurrent.futures import wait
    pending = deque()
    source_batches = batches()
    pool = shared_pool()
    with parallel_call():
        try:
            for batch in islice(source_batches, max_pending):
                pending.append(pool.submit(budget.run, apply, batch))
            # Do not retain the last input batch in the generator frame.
            batch = None
            while pending:
                future = pending.popleft()
                result = future.result()
                del future
                yield result
                del result
                batch = next(source_batches, None)
                if batch is not None:
                    pending.append(pool.submit(budget.run, apply, batch))
                batch = None
        finally:
            for future in pending:
                future.cancel()
            wait(pending)
