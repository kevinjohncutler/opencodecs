"""Benchmark: stock tifffile vs opencodecs-patched tifffile.

Tests decode (and write) speed for several TIFF compression schemes.
Measures both single-threaded read and the parallel-decode path tifffile
uses for tiled/striped files.
"""

from __future__ import annotations

import gc
import os
import sys
import tempfile
import time
import statistics

import numpy as np
import tifffile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import opencodecs.tifffile_patch as patch


def _stable_time(fn, n: int = 5) -> float:
    """Return median runtime over ``n`` runs."""
    times = []
    for _ in range(n):
        gc.collect()
        gc.disable()
        try:
            t0 = time.perf_counter()
            fn()
            times.append(time.perf_counter() - t0)
        finally:
            gc.enable()
    return statistics.median(times)


def make_tiff(arr: np.ndarray, compression: str, tile=None, **kwargs) -> str:
    fd, p = tempfile.mkstemp(suffix=".tif")
    os.close(fd)
    if tile is not None:
        tifffile.imwrite(p, arr, compression=compression, tile=tile, **kwargs)
    else:
        tifffile.imwrite(p, arr, compression=compression, **kwargs)
    return p


def bench_one(label: str, arr: np.ndarray, compression: str, *, tile=None,
              write_kwargs=None):
    write_kwargs = write_kwargs or {}
    path = make_tiff(arr, compression, tile=tile, **write_kwargs)
    size = os.path.getsize(path)
    try:
        # Stock
        patch.uninstall()
        # warmup
        _ = tifffile.imread(path)
        t_stock = _stable_time(lambda: tifffile.imread(path))

        # Patched
        patch.install()
        try:
            _ = tifffile.imread(path)
            t_patched = _stable_time(lambda: tifffile.imread(path))
        finally:
            patch.uninstall()

        speedup = t_stock / t_patched if t_patched else float("nan")
        print(f"{label:24s}  size={size/1e6:5.1f} MB  "
              f"stock={t_stock*1e3:7.2f} ms   "
              f"patched={t_patched*1e3:7.2f} ms   "
              f"speedup={speedup:5.2f}x")
        return t_stock, t_patched
    finally:
        os.unlink(path)


def main() -> None:
    rng = np.random.default_rng(0)

    # Mixed test images; vary size and entropy to exercise codec hot-path
    cases = []
    natural = (rng.normal(128, 30, (2048, 2048)).clip(0, 255)).astype(np.uint8)
    cases.append(("uint8 2K natural-ish", natural, "zstd"))
    cases.append(("uint8 2K natural-ish", natural, "deflate"))

    grayscale16 = rng.integers(0, 65536, (2048, 2048), dtype=np.uint16)
    cases.append(("uint16 2K random",     grayscale16, "zstd"))
    cases.append(("uint16 2K random",     grayscale16, "deflate"))

    # Big stack — tiled tiff with multiple frames
    stack = rng.integers(0, 65536, (32, 512, 512), dtype=np.uint16)
    cases.append(("uint16 32x512x512 stack",  stack, "zstd"))
    cases.append(("uint16 32x512x512 stack",  stack, "deflate"))

    # Highly compressible (constant)
    constant = np.full((2048, 2048), 100, dtype=np.uint16)
    cases.append(("uint16 2K constant",   constant, "zstd"))


    print()
    print(f"{'case':32s}  {'size':<13s} {'stock':<14s} {'patched':<14s} {'speedup':<9s}")
    print("-" * 86)
    results = []
    for label, arr, comp in cases:
        full_label = f"{label} {comp}"
        results.append(bench_one(full_label, arr, comp))

    # Tiled cases — many small tiles -> per-call dispatch overhead matters
    big = rng.integers(0, 65536, (8192, 8192), dtype=np.uint16)
    for tile in (256, 512):
        for comp in ("zstd", "deflate"):
            label = f"uint16 8K tile={tile} {comp}"
            results.append(bench_one(label, big, comp, tile=(tile, tile)))
    print()

    # Summary
    speedups = [s / p for s, p in results if p > 0]
    print(f"summary  median speedup: {statistics.median(speedups):.2f}x   "
          f"range: {min(speedups):.2f}x .. {max(speedups):.2f}x")


if __name__ == "__main__":
    main()
