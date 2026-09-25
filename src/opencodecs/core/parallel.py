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
"""

from __future__ import annotations

import os
from typing import Callable, Sequence, TypeVar

T = TypeVar("T")

#: Past this, more threads buy nothing on any machine here and start
#: costing pool overhead and memory bandwidth contention.
DEFAULT_MAX_WORKERS = 16

#: Fewer pieces than this and the pool costs more than it saves.
DEFAULT_MIN_ITEMS = 4

#: Don't start a worker for less output than this.
DEFAULT_MIN_BYTES_PER_WORKER = 1 << 20


def resolve_workers(numthreads: int | None, n_items: int, *,
                    has_decode_work: bool = True,
                    output_bytes: int | None = None,
                    min_items: int = DEFAULT_MIN_ITEMS,
                    max_workers: int = DEFAULT_MAX_WORKERS,
                    min_bytes_per_worker: int = DEFAULT_MIN_BYTES_PER_WORKER,
                    ) -> int:
    """How many threads to decode ``n_items`` independent pieces with.

    ``None`` means "decide": scale with the CPU count, never exceed the
    number of pieces, and stay serial when there is too little to
    divide. An explicit number is honored as given, so a caller can pin
    it, including to 1 for a reproducible serial run.

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
    cap = min(os.cpu_count() or 1, n_items, max_workers)
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
    Exceptions propagate: the results iterator is drained rather than
    dropped, or a worker's failure would be swallowed and the caller
    would get a half-filled array with no error.
    """
    if workers <= 1 or len(items) <= 1:
        for it in items:
            fn(it)
        return

    from concurrent.futures import ThreadPoolExecutor
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

    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix=f"opencodecs-{name}") as ex:
        for _ in ex.map(lambda batch: budget.run(_run_batch, batch), batches):
            pass


__all__ = ["resolve_workers", "run_batched", "DEFAULT_MAX_WORKERS",
           "DEFAULT_MIN_ITEMS", "DEFAULT_MIN_BYTES_PER_WORKER", "map_batches"]


def map_batches(fn, items, workers: int, *, batch_size: int = 16,
                max_pending: int | None = None, name: str = "codec", budget=None):
    """Yield ordered result batches without consuming the whole input.

    Each future handles a batch, amortizing scheduling for small tiles.
    At most max_pending batches are submitted, plus the batch currently
    yielded to the consumer. Consumption supplies backpressure. A writer
    can emit one result while workers encode the following batches.

    The caller must consume or close the iterator. Early close or an
    exception cancels queued work and joins running workers before return.
    Serial mode uses the same batches and never creates a pool.
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
    from concurrent.futures import ThreadPoolExecutor
    pending = deque()
    source_batches = batches()
    with ThreadPoolExecutor(max_workers=workers,
                            thread_name_prefix=f"opencodecs-{name}") as pool:
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
