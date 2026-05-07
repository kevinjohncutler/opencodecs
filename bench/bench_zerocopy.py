"""Benchmark: zero-copy memoryview path vs bytes() coercion.

The codecs now accept any buffer-protocol object directly. For mmap-
backed files or numpy arrays, this skips one full bytes() copy of the
input. Measures whether that copy was actually material on real data.
"""

from __future__ import annotations

import gc
import mmap
import os
import statistics
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opencodecs.codecs._zstd import encode as zstd_encode, decode as zstd_decode
from opencodecs.codecs._lz4 import encode as lz4_encode, decode as lz4_decode
from opencodecs.codecs._brotli import encode as brotli_encode, decode as brotli_decode
from opencodecs.codecs._deflate import encode as deflate_encode, decode as deflate_decode


def _stable(fn, n=10):
    times = []
    for _ in range(n):
        gc.collect(); gc.disable()
        try:
            t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
        finally: gc.enable()
    return statistics.median(times)


def main() -> None:
    rng = np.random.default_rng(0)

    # Generate compressed payloads of different sizes
    sizes = [(1 * 1024 * 1024, "1 MB"), (16 * 1024 * 1024, "16 MB"),
             (128 * 1024 * 1024, "128 MB")]

    for size_bytes, label in sizes:
        raw = rng.integers(0, 256, size_bytes, dtype=np.uint8).tobytes()
        zstd_payload = zstd_encode(raw)

        # Write the compressed payload to a temp file, mmap it
        fd, path = tempfile.mkstemp()
        os.write(fd, zstd_payload); os.close(fd)
        f = open(path, "rb")
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

        try:
            # 1) bytes already in memory (the previous fast path)
            t_bytes = _stable(lambda: zstd_decode(zstd_payload), n=20)
            # 2) memoryview of the bytes (also already memory)
            mv_bytes = memoryview(zstd_payload)
            t_mv_bytes = _stable(lambda: zstd_decode(mv_bytes), n=20)
            # 3) mmap directly (this used to incur a bytes() copy via
            #    bytes(mmap_object) before our zero-copy patch)
            t_mmap = _stable(lambda: zstd_decode(mm), n=20)

            print(f"{label}  decode  bytes={t_bytes*1e3:6.2f} ms  "
                  f"memoryview={t_mv_bytes*1e3:6.2f} ms  "
                  f"mmap={t_mmap*1e3:6.2f} ms")
        finally:
            mm.close(); f.close(); os.unlink(path)


if __name__ == "__main__":
    main()
