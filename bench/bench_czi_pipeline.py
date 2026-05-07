"""Pipeline benchmark: read many CZI files in sequence.

The lab's actual workflow opens hundreds of CZIs per FOV per day. A
single-file benchmark answers "is the codec fast?" — this one answers
"does the persistent thread pool actually amortise across files?"

If the pool is being torn down per-call, we'd see a flat per-file cost
matching the single-file benchmark plus some overhead. If amortisation
works, the per-file cost should be lower than the single-file number
when reads are back-to-back.

Compares opencodecs against czifile and (when available) aicspylibczi
on a real glob of files from the HiprDrive archive.
"""

from __future__ import annotations

import gc
import os
import statistics
import sys
import time
from glob import glob
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import opencodecs as oc

import czifile

try:
    import aicspylibczi
    HAVE_AICS = True
except ImportError:
    HAVE_AICS = False


def _stable_pipeline(read_fn, files, n_passes: int = 3):
    """Read every file in `files` once per pass; report median of pass totals
    plus the per-file median across all reads."""
    pass_totals = []
    per_file_times = []
    # warmup
    for fp in files:
        read_fn(fp)
    for _ in range(n_passes):
        gc.collect()
        gc.disable()
        t0 = time.perf_counter()
        try:
            for fp in files:
                t_file = time.perf_counter()
                read_fn(fp)
                per_file_times.append(time.perf_counter() - t_file)
            pass_totals.append(time.perf_counter() - t0)
        finally:
            gc.enable()
    return statistics.median(pass_totals), statistics.median(per_file_times)


def main() -> None:
    candidates = sorted(
        glob("/Volumes/HiprDrive/2024_02_02_GNE_synthetic_community/*.czi")
    )[:8]
    if not candidates:
        print("no CZI files found at /Volumes/HiprDrive/2024_02_02_*/")
        return

    total_bytes = sum(os.path.getsize(f) for f in candidates)
    print(f"pipeline: {len(candidates)} CZI files, "
          f"total {total_bytes/1e6:.0f} MB")
    print()

    cases = [
        ("opencodecs (persistent pool)", lambda fp: oc.read(fp)),
    ]

    def via_czifile(fp):
        with czifile.CziFile(fp) as cz:
            return np.stack(
                [np.squeeze(sb.data()) for sb in cz.subblocks()], axis=0,
            )

    cases.append(("czifile", via_czifile))

    if HAVE_AICS:
        cases.append(
            ("aicspylibczi", lambda fp: aicspylibczi.CziFile(fp).read_image()[0])
        )

    print(f"{'reader':<32s}  {'pass total':>12s}  {'per-file':>12s}  {'GB/s':>6s}")
    print("-" * 72)
    for name, fn in cases:
        total, per_file = _stable_pipeline(fn, candidates, n_passes=3)
        gbps = total_bytes / total / 1e9
        print(f"{name:<32s}  {total*1e3:>10.1f} ms  {per_file*1e3:>10.1f} ms  {gbps:>5.2f}")
    print()
    print(f"single-file expected (from previous benchmarks): "
          f"opencodecs ~15 ms warm")


if __name__ == "__main__":
    main()
