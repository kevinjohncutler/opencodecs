"""Benchmark: parallel-pread tiff reader vs stock tifffile.

Tests both local-disk and NAS-mounted files, and varies the number of
worker threads to find the optimal point. Drops OS page cache between
runs (best-effort) so we measure true I/O.
"""

from __future__ import annotations

import gc
import os
import shutil
import statistics
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opencodecs.tiff_reader import imread as oc_imread


def _drop_caches() -> None:
    """Best-effort drop of the macOS unified buffer cache. On Linux this
    is a no-op without root; benchmarks below already average enough runs
    to expose perf differences regardless."""
    if sys.platform == "darwin":
        os.system("/usr/sbin/purge >/dev/null 2>&1")


def _stable_time(fn, n: int = 5, drop_cache: bool = False) -> tuple[float, float]:
    """Median + min runtime over ``n`` runs."""
    times = []
    for _ in range(n):
        if drop_cache:
            _drop_caches()
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        finally:
            gc.enable()
    return statistics.median(times), min(times)


def make_tiled_tiff(path: str, arr: np.ndarray, tile: int = 256,
                    compression: str = "zstd") -> None:
    tifffile.imwrite(path, arr, compression=compression, tile=(tile, tile))


def bench_path(label: str, path: str, *, drop_cache: bool):
    print(f"\n=== {label} ({os.path.getsize(path)/1e6:.1f} MB) ===")
    # warmup
    _ = tifffile.imread(path)

    t_stock_med, t_stock_min = _stable_time(
        lambda: tifffile.imread(path), n=5, drop_cache=drop_cache)
    print(f"stock tifffile:               {t_stock_med*1e3:7.2f} ms (min {t_stock_min*1e3:6.2f})")

    for nw in (4, 8, 16, 32):
        t_med, t_min = _stable_time(
            lambda nw=nw: oc_imread(path, n_workers=nw),
            n=5, drop_cache=drop_cache)
        speedup = t_stock_med / t_med if t_med > 0 else 0
        print(f"opencodecs n_workers={nw:<3d}      {t_med*1e3:7.2f} ms (min {t_min*1e3:6.2f})  speedup={speedup:.2f}x")


def main() -> None:
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 65536, (8192, 8192), dtype=np.uint16)

    # Local disk (tmp)
    fd, local_path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    make_tiled_tiff(local_path, arr, tile=256)

    # NAS path (HiprDrive). Copy the local file there.
    nas_dir = Path("/Volumes/HiprDrive/_oc_bench_tmp")
    nas_dir.mkdir(exist_ok=True)
    nas_path = str(nas_dir / "bench.tif")
    shutil.copyfile(local_path, nas_path)

    try:
        bench_path("LOCAL DISK (warm cache)",   local_path, drop_cache=False)
        bench_path("LOCAL DISK (cold cache)",   local_path, drop_cache=True)
        bench_path("NAS (warm cache)",          nas_path,   drop_cache=False)
        bench_path("NAS (cold cache)",          nas_path,   drop_cache=True)
    finally:
        os.unlink(local_path)
        try:
            os.unlink(nas_path)
            nas_dir.rmdir()
        except OSError:
            pass


if __name__ == "__main__":
    main()
