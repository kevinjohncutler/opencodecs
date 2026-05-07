"""Benchmark: opencodecs native CziReader vs czifile vs aicspylibczi.

Measures full-file decode wall time on a real lab CZI (zstd1 compression).
Tests both warm and cold cache, and on NAS-mounted vs local-disk paths.
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

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opencodecs._czi_reader import CziReader

import czifile

try:
    import aicspylibczi
    HAVE_AICS = True
except ImportError:
    HAVE_AICS = False


def _drop_caches() -> None:
    if sys.platform == "darwin":
        os.system("/usr/sbin/purge >/dev/null 2>&1")


def _stable_time(fn, n: int = 5, drop_cache: bool = False) -> tuple[float, float]:
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


def read_native(path: str) -> np.ndarray:
    with CziReader(path) as r:
        return r.read()


def read_native_serial(path: str) -> np.ndarray:
    with CziReader(path) as r:
        return r.read(n_workers=1)


def read_czifile(path: str) -> np.ndarray:
    with czifile.CziFile(path) as cz:
        arrs = [sb.data() for sb in cz.subblocks()]
        return np.stack([np.squeeze(a) for a in arrs], axis=0)


def read_aicspylibczi(path: str) -> np.ndarray:
    cz = aicspylibczi.CziFile(path)
    arr, _shape = cz.read_image()
    return arr


def bench_path(label: str, path: str, *, drop_cache: bool):
    print(f"\n=== {label} ({os.path.getsize(path)/1e6:.1f} MB) ===")

    # warmup
    _ = read_native(path)

    cases = [
        ("opencodecs native (parallel)", read_native),
        ("opencodecs native (serial)",   read_native_serial),
        ("czifile",                      read_czifile),
    ]
    if HAVE_AICS:
        cases.append(("aicspylibczi",    read_aicspylibczi))

    times = {}
    for name, fn in cases:
        med, mn = _stable_time(lambda fn=fn: fn(path), n=5, drop_cache=drop_cache)
        times[name] = med
        print(f"  {name:32s}: {med*1e3:7.1f} ms  (min {mn*1e3:6.1f})")

    base = times["czifile"]
    print(f"  --- speedups vs czifile ---")
    for name, t in times.items():
        if name == "czifile": continue
        print(f"  {name:32s}: {base/t:5.2f}x")


def main() -> None:
    fp_nas = (
        "/Volumes/HiprDrive/2024_02_02_GNE_synthetic_community/"
        "2024_02_02_GNEPanelTest_slide1_B1_GNE0001_cellmix01_200nMENC_"
        "20nMCOMP_quarterpower_fov_4_561.czi"
    )
    if not Path(fp_nas).is_file():
        print(f"NAS file not found: {fp_nas}")
        return

    # Copy NAS file to local /tmp for warm-disk comparison.
    fd, fp_local = tempfile.mkstemp(suffix=".czi")
    os.close(fd)
    shutil.copyfile(fp_nas, fp_local)
    print(f"copied {os.path.getsize(fp_local)/1e6:.1f} MB to {fp_local}")

    try:
        bench_path("LOCAL DISK (warm cache)",  fp_local, drop_cache=False)
        bench_path("LOCAL DISK (cold cache)",  fp_local, drop_cache=True)
        bench_path("NAS (warm cache)",         fp_nas,   drop_cache=False)
        bench_path("NAS (cold cache)",         fp_nas,   drop_cache=True)
    finally:
        os.unlink(fp_local)


if __name__ == "__main__":
    main()
