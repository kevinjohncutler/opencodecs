"""A codec that holds the GIL through its decode is silently serial.

Every correctness test passes either way, so nothing else in this suite
can tell the difference. What it costs is that any caller decoding
frames on a thread pool -- multi-frame DICOM, a tile fan-out, a batch
of files -- pays the pool's overhead to take turns.

Five codecs did this and none of them looked different from the ones
that did not: jpeg2k, mozjpeg, charls, zfp and openjph, all measuring
0.94x to 1.12x where their neighbours reached 3.7x to 3.9x on the same
four threads. openjph is the one worth remembering: its .pxd already
declared the shim functions ``nogil``, which means "safe to call
without the GIL" and not "called without the GIL", so reading the
declaration rather than measuring would have cleared it.

The threshold is deliberately far below the ~7x these actually reach.
This is a regression guard against a nogil block being dropped, not a
benchmark, and it has to hold on a loaded CI runner with fewer cores
than a workstation.
"""

from __future__ import annotations

import os
import threading
import time

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.perf

CORES = os.cpu_count() or 1
needs_cores = pytest.mark.skipif(
    CORES < 4, reason="needs at least 4 cores to distinguish serial from not")

N = 16


def _scaling(decode, blob, threads: int) -> float:
    """Best-of-3 speedup of ``threads`` threads over one, same total work."""
    def once(n: int) -> float:
        def worker():
            for _ in range(N // n):
                decode(blob)
        ts = [threading.Thread(target=worker) for _ in range(n)]
        t = time.perf_counter()
        for x in ts:
            x.start()
        for x in ts:
            x.join()
        return time.perf_counter() - t

    once(1)                                   # warm
    serial = min(once(1) for _ in range(3))
    parallel = min(once(threads) for _ in range(3))
    return serial / parallel


@pytest.fixture(scope="module")
def rgb():
    return np.random.default_rng(0).integers(0, 255, (512, 512, 3)).astype("u1")


@pytest.fixture(scope="module")
def gray16():
    return np.random.default_rng(1).integers(0, 4000, (512, 512)).astype("u2")


@pytest.fixture(scope="module")
def volume():
    """zfp is a float-array codec, not an image one."""
    return np.random.default_rng(2).normal(0, 1, (64, 128, 128)).astype("f4")


@needs_cores
@pytest.mark.parametrize("name,fixture", [
    ("jpeg2k", "gray16"),
    ("jpegls", "gray16"),
    ("mozjpeg", "rgb"),
    ("htj2k", "rgb"),
    ("zfp", "volume"),
    ("jpeg", "rgb"),
    ("webp", "rgb"),
])
def test_decode_scales_with_threads(name, fixture, request):
    try:
        codec = oc.get_codec(name)
    except Exception as exc:                              # noqa: BLE001
        pytest.skip(f"{name} not built here: {exc}")
    arr = request.getfixturevalue(fixture)
    blob = codec.encode(arr)

    threads = min(4, CORES)
    got = _scaling(codec.decode, blob, threads)
    assert got > 1.5, (
        f"{name} decode on {threads} threads is {got:.2f}x serial; "
        f"the GIL is being held across the decode call")


@needs_cores
def test_jpeg2k_scales_with_openjpeg_threads_off(gray16):
    """openjpeg threads internally, which masks a held GIL.

    With its own T1 threads on, jpeg2k measured 2.2x whether or not the
    GIL was released, because the baseline was already parallel. Pinning
    openjpeg to one thread is what makes the frame-level fan-out
    visible: 1.00x before the fix, 7.4x after.
    """
    codec = oc.get_codec("jpeg2k")
    blob = codec.encode(gray16)
    threads = min(4, CORES)
    got = _scaling(lambda b: codec.decode(b, numthreads=1), blob, threads)
    assert got > 1.5, f"jpeg2k with openjpeg pinned to 1 thread: {got:.2f}x"
