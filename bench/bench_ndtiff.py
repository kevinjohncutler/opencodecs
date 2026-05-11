"""Benchmark: opencodecs.NDTiffDataset vs ndstorage.

Three workloads:
  A) Index parse on synthetic 100K-record index (open-time)
  B) Random-access read of 100 frames (warm cache)
  C) Iterate + discard all 6001 frames (matched semantics)

Pass an NDTiff folder via env var `OC_NDTIFF_PATH` to use a real
acquisition; otherwise the script only runs (A) on synthetic data.

Run with:
    OC_NDTIFF_PATH=/path/to/Acq_Cam-1 python bench/bench_ndtiff.py
"""

from __future__ import annotations

import io
import json
import os
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

try:
    import ndstorage
    _HAS_NDSTORAGE = True
except ImportError:
    _HAS_NDSTORAGE = False

from opencodecs._ndtiff import NDTiffDataset
from opencodecs.codecs._ndtiff import (
    parse_ndtiff_index,
    PIXEL_TYPE_SIXTEEN_BIT,
)


def bench(fn, n=3):
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def silence_ndstorage(fn):
    """Wrap a function so ndstorage's `Reading index...` progress is muted."""
    def w(*a, **kw):
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            return fn(*a, **kw)
        finally:
            sys.stdout = old
    return w


# ---------------------------------------------------------------------------
# A) Index parse on synthetic 100K records (no disk needed)
# ---------------------------------------------------------------------------


def make_fake_index(n_records: int = 100_000) -> bytes:
    """Build a fake NDTiff.index buffer with N records. Offsets roll
    over to NDTiffStack_<i>.tif at ~4 GB to mimic the real layout."""
    pixel_bytes = 4000 * 400 * 2  # 3.2 MB per frame
    file_capacity = (4 << 30) - 4 * pixel_bytes
    buf = bytearray()
    for i in range(n_records):
        ax = json.dumps({"z": i % 1000, "c": i // 1000}).encode("utf-8")
        file_idx = (i * pixel_bytes) // file_capacity
        fn = (f"NDTiffStack_{file_idx}.tif" if file_idx
              else "NDTiffStack.tif").encode("utf-8")
        local_off = (i * pixel_bytes) % file_capacity
        buf += struct.pack("<I", len(ax)) + ax
        buf += struct.pack("<I", len(fn)) + fn
        buf += struct.pack(
            "<IIIIIIII",
            local_off, 4000, 400, PIXEL_TYPE_SIXTEEN_BIT, 0,
            local_off + pixel_bytes, 200, 0,
        )
    return bytes(buf)


def workload_a_index_parse():
    print("=" * 88)
    print("Workload A: Index parse on synthetic 100K-record index")
    print(f"{'parser':50s} {'time':>12s} {'us/record':>12s}")
    big = make_fake_index(100_000)

    def oc():
        parse_ndtiff_index(big)
    t_oc = bench(oc, n=3)
    print(f"{'opencodecs (Cython)':50s} {t_oc:>10.2f} ms "
          f"{1e6 * t_oc / 100_000:>10.2f} us")

    if _HAS_NDSTORAGE:
        from ndstorage.ndtiff_index import read_ndtiff_index

        @silence_ndstorage
        def nd():
            read_ndtiff_index(big, verbose=False)
        t_nd = bench(nd, n=3)
        print(f"{'ndstorage (pure Python)':50s} {t_nd:>10.2f} ms "
              f"{1e6 * t_nd / 100_000:>10.2f} us")
        print(f"  → opencodecs is {t_nd / t_oc:.1f}× faster")
    else:
        print("(ndstorage not installed; skipping comparison)")


# ---------------------------------------------------------------------------
# B + C) Real-data workloads
# ---------------------------------------------------------------------------


def workload_bc_on_disk(path: str):
    if not Path(path).is_dir():
        print(f"\n(skipping disk workloads: {path} not a directory)")
        return

    ds_oc = NDTiffDataset(path)
    n_frames = len(ds_oc)
    print()
    print("=" * 88)
    print(f"Real NDTiff dataset: {path}")
    print(f"  {n_frames} frames, shape={ds_oc.shape}, dtype={ds_oc.dtype}, "
          f"axes={sorted(ds_oc.axes_names)}")

    ds_nd = None
    if _HAS_NDSTORAGE:
        ds_nd = silence_ndstorage(ndstorage.NDTiffDataset)(path)

    # Build a random-axes sample
    import random
    random.seed(0)
    zs = sorted({e.axes.get("z") for e in ds_oc.entries
                 if e.axes.get("z") is not None})
    sample = random.sample(zs, min(100, len(zs)))

    # B) 100 random frames
    print("\nWorkload B: 100 random frames (warm cache)")
    print(f"{'mode':50s} {'time':>12s}")

    def oc_seq():
        for z in sample:
            ds_oc.read_frame(z=z)

    def oc_par():
        ds_oc.read_many(keys=[{"z": z} for z in sample])

    t_oc_seq = bench(oc_seq, n=3)
    t_oc_par = bench(oc_par, n=3)
    print(f"{'opencodecs sequential':50s} {t_oc_seq:>10.2f} ms")
    print(f"{'opencodecs read_many (parallel)':50s} {t_oc_par:>10.2f} ms")

    if ds_nd is not None:
        def nd_seq():
            for z in sample:
                ds_nd.read_image(z=z)
        t_nd_seq = bench(nd_seq, n=3)
        print(f"{'ndstorage sequential':50s} {t_nd_seq:>10.2f} ms "
              f"({t_nd_seq / t_oc_seq:.2f}× of oc-seq, "
              f"{t_nd_seq / t_oc_par:.2f}× of oc-par)")

    # C) Iterate-and-discard all frames
    print("\nWorkload C: iterate + discard all frames (matched semantics)")
    print(f"{'mode':50s} {'time':>12s} {'GB/s':>10s}")
    all_keys = [{"z": z} for z in zs]
    raw_gb = len(zs) * (ds_oc.shape[-1] * ds_oc.shape[-2] *
                        ds_oc.dtype.itemsize) / 1e9

    def oc_iter():
        for _ in ds_oc.iter_frames_parallel(all_keys, prefetch=32):
            pass

    t_oc_iter = bench(oc_iter, n=2)
    gbps = raw_gb / (t_oc_iter / 1000)
    print(f"{'opencodecs iter_frames_parallel(prefetch=32)':50s} "
          f"{t_oc_iter:>10.2f} ms {gbps:>8.2f}")

    if ds_nd is not None:
        def nd_iter():
            for z in zs:
                ds_nd.read_image(z=z)
        t_nd_iter = bench(nd_iter, n=2)
        gbps_nd = raw_gb / (t_nd_iter / 1000)
        print(f"{'ndstorage sequential':50s} "
              f"{t_nd_iter:>10.2f} ms {gbps_nd:>8.2f}")
        print(f"  → opencodecs is {t_nd_iter / t_oc_iter:.2f}× faster")

    ds_oc.close()
    if ds_nd is not None:
        ds_nd.close()


if __name__ == "__main__":
    workload_a_index_parse()
    path = os.environ.get("OC_NDTIFF_PATH")
    if path:
        workload_bc_on_disk(path)
    else:
        print("\nSet OC_NDTIFF_PATH=/path/to/Acq to run on-disk workloads.")
