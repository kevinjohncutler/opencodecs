"""One definition of what a performance guard is allowed to assert.

These tests exist to catch a thread pool being removed, not to certify
a speedup figure. That distinction decides the threshold, and getting
it wrong is how the same failure arrived twice: a bar of 1.5x taken
from an 8-core workstation, which a 4-core CI runner missed at 1.50x
for DICOM and 1.42x for zfp. In both cases the parallelism was working
and the number was simply not achievable on that hardware.

So the bar is what a SMALL, LOADED machine can still clear while a
removed pool cannot: a working fan-out reaches at least ~1.4x on four
cores, a removed one gives 1.0x or worse. 1.2x sits between them with
room on both sides.

Where a claim really is machine-independent, assert it where it is:
test_decode_releases_gil.py pins openjpeg to one internal thread, so
the frame fan-out is the only variable and shows 7.4x. That is the test
protecting the GIL fix. The per-format guards only have to notice the
pool disappearing.
"""

from __future__ import annotations

import os
import time

import pytest

#: Below this, a speedup cannot be demonstrated at all and the guard
#: would be measuring noise.
MIN_CORES = 4

#: See the module docstring for why this is not the measured figure.
MIN_SPEEDUP = 1.2

needs_cores = pytest.mark.skipif(
    (os.cpu_count() or 1) < MIN_CORES,
    reason=f"needs >= {MIN_CORES} cores to tell parallel from serial")


def best_of(fn, rounds: int = 3) -> float:
    """Fastest of ``rounds`` runs, in seconds.

    Fastest rather than mean: on a shared runner the slow tail is other
    tenants, and the question here is whether the work CAN go faster,
    not what it averages under someone else's load.
    """
    best = float("inf")
    for _ in range(rounds):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def assert_faster(serial: float, parallel: float, what: str) -> float:
    """Assert a real speedup, and say the number when it fails."""
    got = serial / parallel
    assert got > MIN_SPEEDUP, (
        f"{what}: {got:.2f}x, below the {MIN_SPEEDUP}x floor -- the "
        f"parallel path looks gone (serial {serial * 1e3:.1f} ms, "
        f"parallel {parallel * 1e3:.1f} ms)")
    return got
