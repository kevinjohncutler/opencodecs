"""Diagnose the NAS-vs-local CZI gap.

Hypotheses to test:
  H1: mmap on SMB demand-pages 4 KB at a time -> many RPC round trips
      vs one bulk read(). If true, slurp-then-bytes should be faster
      than mmap on NAS.
  H2: madvise(SEQUENTIAL) hurts on SMB (prefetch decisions are wrong).
  H3: Per-worker thread context-switching is hurting on NAS.
  H4: Variance is just sampling noise — try more reps.
"""

from __future__ import annotations

import gc
import mmap
import os
import statistics
import struct
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from opencodecs._czi_reader import CziReader, _pixel_type_dtype
from opencodecs.codecs._zstd import decode as zstd_decode
from opencodecs.codecs._bytetools import byteshuffle_decode
import aicspylibczi


def _t(fn, n=10):
    times = []
    for _ in range(n):
        gc.collect(); gc.disable()
        try:
            t0 = time.perf_counter(); fn(); times.append(time.perf_counter() - t0)
        finally: gc.enable()
    return statistics.median(times), min(times), max(times)


def time_just_io_mmap(path: str) -> float:
    """Time JUST the mmap read of all sub-block bytes (no decode)."""
    with CziReader(path) as r:
        # touch every sub-block's bytes once
        out = []
        for entry in r.entries:
            view, sz = r._pixel_data_view(entry)
            # force the OS to page in by reading first/last byte of each
            _ = bytes(view[:1])
            _ = bytes(view[-1:])
            out.append(sz)
        return sum(out)


def time_just_io_slurp(path: str) -> int:
    """Read whole file into bytes via one read()."""
    with open(path, "rb") as f:
        return len(f.read())


def time_decode_from_bytes(path: str, n_workers: int) -> np.ndarray:
    """Same as CziReader.read but with the file slurped into bytes
    instead of mmap."""
    from concurrent.futures import ThreadPoolExecutor
    with open(path, "rb") as f:
        data = f.read()
    mv = memoryview(data)

    # Re-open to get the directory parsed by our existing code; replace
    # _mmap with our memoryview for the data accesses.
    r = CziReader(path)
    try:
        # Hack: monkey-patch the data source to use our slurped memoryview
        # by wrapping the data fetch. We can't easily replace mmap, but
        # we can replicate the decode loop using `mv` directly.
        first = r.entries[0]
        out = np.empty((len(r.entries), *first.stored_shape), dtype=first.dtype)

        def _decode_slurp(i):
            entry = r.entries[i]
            sb_off = entry.file_position
            meta_size, _att, data_size = struct.unpack_from(
                "<iiq", mv, sb_off + 32)
            entry_storage = entry.storage_size
            pad = max(240 - entry_storage, 0)
            data_off = sb_off + 32 + 16 + entry_storage + pad + meta_size
            view = mv[data_off:data_off + data_size]
            dtype, samples = _pixel_type_dtype(entry.pixel_type)
            n_pixels = 1
            for s in entry.stored_shape:
                n_pixels *= s
            if entry.compression == 6:
                header_size = view[0]
                hilo = False
                pos = 1
                while pos < header_size:
                    ct = view[pos]; pos += 1
                    if ct == 1:
                        hilo = (view[pos] & 1) != 0; pos += 1
                    else:
                        break
                raw = zstd_decode(view[header_size:])
                if hilo:
                    n = len(raw) // dtype.itemsize
                    raw = byteshuffle_decode(raw, dtype.itemsize, n)
                arr = np.frombuffer(raw, dtype=dtype, count=n_pixels)
            elif entry.compression == 0:
                arr = np.frombuffer(view, dtype=dtype, count=n_pixels).copy()
            else:
                raise RuntimeError(entry.compression)
            out[i] = arr.reshape(entry.stored_shape)

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_decode_slurp, range(len(r.entries))))
        return out
    finally:
        r.close()


def main() -> None:
    fp_nas = (
        "/Volumes/HiprDrive/2024_02_02_GNE_synthetic_community/"
        "2024_02_02_GNEPanelTest_slide1_B1_GNE0001_cellmix01_200nMENC_"
        "20nMCOMP_quarterpower_fov_4_561.czi"
    )
    if not Path(fp_nas).is_file():
        print(f"NAS file not found: {fp_nas}")
        return

    # warm caches
    for _ in range(2):
        with CziReader(fp_nas) as r:
            r.read()

    print(f"file: {os.path.getsize(fp_nas)/1e6:.1f} MB on NAS\n")

    # I/O cost: mmap touch vs slurp
    med, mn, mx = _t(lambda: time_just_io_mmap(fp_nas), n=5)
    print(f"mmap touch all sub-blocks:   {med*1e3:6.1f} ms (min {mn*1e3:.1f}, max {mx*1e3:.1f})")
    med, mn, mx = _t(lambda: time_just_io_slurp(fp_nas), n=5)
    print(f"slurp whole file via read(): {med*1e3:6.1f} ms (min {mn*1e3:.1f}, max {mx*1e3:.1f})")

    print()
    # Full decode comparison
    print("=== full decode (8 workers, 15 reps) ===")
    med, mn, mx = _t(lambda: CziReader(fp_nas).__enter__().read(n_workers=8), n=15)
    print(f"opencodecs (mmap)         : {med*1e3:6.1f} ms (min {mn*1e3:.1f}, max {mx*1e3:.1f})")
    med, mn, mx = _t(lambda: time_decode_from_bytes(fp_nas, 8), n=15)
    print(f"opencodecs (slurp+bytes)  : {med*1e3:6.1f} ms (min {mn*1e3:.1f}, max {mx*1e3:.1f})")
    med, mn, mx = _t(
        lambda: aicspylibczi.CziFile(fp_nas).read_image()[0], n=15)
    print(f"aicspylibczi               : {med*1e3:6.1f} ms (min {mn*1e3:.1f}, max {mx*1e3:.1f})")

    # Worker count sweep on NAS for opencodecs (mmap path)
    print("\n=== opencodecs (mmap) worker sweep ===")
    for nw in (2, 4, 6, 8, 12, 16):
        med, mn, mx = _t(
            lambda nw=nw: CziReader(fp_nas).__enter__().read(n_workers=nw), n=10)
        print(f"  n_workers={nw:<3d}: {med*1e3:6.1f} ms (min {mn*1e3:.1f})")


if __name__ == "__main__":
    main()
