"""Unified opencodecs benchmark harness.

Three tiers selected via --fast / --medium / --slow:

  --fast   ≤5 min total. Synthetic small fixtures. Smoke-check that
           nothing regressed. Run after every non-trivial change.
  --medium ~30 min. Realistic small-lab sizes (1-2 GB synthetic).
           Run before releases.
  --slow   1+ hour. Large-file workloads (10-50 GB). Run on the
           threadripper or any beefy machine when investigating
           perf claims.

Noise robustness:
  * Each workload runs N times (default 7 fast, 5 medium, 3 slow)
    plus 1 untimed warmup.
  * We report median + min + max + IQR of the wall-clock samples.
  * If iqr/median > 0.25 the result is flagged "NOISY" in the
    summary — don't trust ratios derived from noisy runs.
  * Outliers > 3*MAD from median are dropped before computing the
    median when n >= 5 (Hampel filter).

Output:
  bench/results/<hostname>_<arch>/
      <timestamp>.json     full run, machine-readable
      history.jsonl        append-only one-line-per-bench history
      latest.md            most recent run's summary table

Regression detection:
  After a run, compare each median to the rolling median of the last
  five runs on the same machine. Flags any workload with current
  median > 1.2 × rolling median (configurable via --regression-threshold).
  Exits with code 1 if any regression flagged (for CI gating).

Usage::

  python bench/run_benchmarks.py --fast
  python bench/run_benchmarks.py --slow --filter tiff
  python bench/run_benchmarks.py --fast --no-regression-check
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import platform
import shutil
import socket
import statistics
import struct
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import numpy as np

# ---------------------------------------------------------------------------
# Workload registry
# ---------------------------------------------------------------------------


WORKLOADS: dict[str, dict] = {}


def workload(name: str, tier: str, *, group: str = "misc"):
    """Register a workload. ``tier`` ∈ {fast, medium, slow}."""

    def deco(fn: Callable):
        WORKLOADS[name] = dict(fn=fn, tier=tier, group=group)
        return fn

    return deco


# ---------------------------------------------------------------------------
# Reference libraries (all optional)
# ---------------------------------------------------------------------------


def _try_import(name: str):
    try:
        return __import__(name)
    except ImportError:
        return None


_tifffile = _try_import("tifffile")
_ndstorage = _try_import("ndstorage")
_czifile = _try_import("czifile")
_imagecodecs = _try_import("imagecodecs")


# ---------------------------------------------------------------------------
# Timing primitives
# ---------------------------------------------------------------------------


def _hampel_filter(values: list[float], n_sigmas: float = 3.0) -> list[float]:
    """Drop outliers > n_sigmas MAD from the median. No-op for n < 5."""
    if len(values) < 5:
        return values
    med = statistics.median(values)
    deviations = [abs(v - med) for v in values]
    mad = statistics.median(deviations)
    if mad == 0:
        return values  # all identical — nothing to filter
    k = 1.4826  # consistency factor for normal distribution
    sigma = k * mad
    return [v for v in values if abs(v - med) <= n_sigmas * sigma]


def _time_fn(fn: Callable, n: int, warmup: int = 1) -> dict:
    """Run ``fn()`` N times with warmup. Returns timing statistics in ms."""
    for _ in range(warmup):
        try:
            fn()
        except Exception:
            pass  # warmup errors won't kill the bench

    samples_ms: list[float] = []
    for _ in range(n):
        gc.collect()  # one GC pause per run so it doesn't poison timing
        t0 = time.perf_counter()
        fn()
        samples_ms.append((time.perf_counter() - t0) * 1000)

    filtered = _hampel_filter(samples_ms)
    if not filtered:
        filtered = samples_ms  # don't return empty
    med = statistics.median(filtered)
    # IQR requires sorted samples; use the filtered set then sort.
    sorted_f = sorted(filtered)
    iqr = 0.0
    if len(sorted_f) >= 4:
        q1 = sorted_f[len(sorted_f) // 4]
        q3 = sorted_f[3 * len(sorted_f) // 4]
        iqr = q3 - q1
    return {
        "median_ms": med,
        "min_ms": min(filtered),
        "max_ms": max(filtered),
        "iqr_ms": iqr,
        # Sub-millisecond medians are below the timing floor of
        # perf_counter on most kernels; flag them as inherently noisy
        # so the caller doesn't take the ratio at face value.
        "noisy": (med < 0.5) or (med > 0 and (iqr / med) > 0.25),
        "n_runs_total": n,
        "n_runs_after_filter": len(filtered),
        "all_samples_ms": samples_ms,
    }


# ---------------------------------------------------------------------------
# Workload helpers
# ---------------------------------------------------------------------------


def _silence_stdout(fn: Callable) -> Callable:
    """Wrap a function so stdout is muted while it runs (for ndstorage)."""

    def w(*a, **kw):
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            return fn(*a, **kw)
        finally:
            sys.stdout = old

    return w


# ---------------------------------------------------------------------------
# Fast-tier workloads
# ---------------------------------------------------------------------------


@workload("tiff_random_tile_read", tier="fast", group="tiff")
def bench_tiff_random_tile_read():
    """Decode 100 random tiles from a 4K×4K u16 tiled TIFF (in-memory bytes).

    This is the COG / OME-TIFF selling-point workload: tile-aware reads
    fetching only the bytes for the requested tiles. opencodecs's
    TiffStream vs tifffile (which has no tile-level API — degrades to
    full page decode + slice).
    """
    import opencodecs as oc
    if not oc.has_codec("tiff") or _tifffile is None:
        return {"skipped": "tiff codec or tifffile not available"}

    arr = np.arange(4096 * 4096, dtype=np.uint16).reshape(4096, 4096)
    buf = io.BytesIO()
    _tifffile.imwrite(buf, arr, compression=None, tile=(256, 256))
    data = buf.getvalue()

    oc_codec = oc.get_codec("tiff")

    def oc_fn():
        with oc_codec.open(data) as r:
            page = r.page(0)
            rng = np.random.default_rng(0)
            for ti in rng.integers(0, page.tiles_x * page.tiles_y, size=100):
                offset = int(page.offsets[int(ti)])
                nbytes = int(page.byte_counts[int(ti)])
                raw = r._read(offset, nbytes)
                page._decode_segment(raw)

    def tf_fn():
        # tifffile has no public per-tile decode; we measure its full
        # page decode which is the closest equivalent the user has
        # access to.
        with _tifffile.TiffFile(io.BytesIO(data)) as tf:
            tf.pages[0].asarray()

    oc_t = _time_fn(oc_fn, n=7)
    tf_t = _time_fn(tf_fn, n=7)
    return {
        "opencodecs": oc_t,
        "reference": {"tifffile_full_page": tf_t},
        "speedup_vs_tifffile": tf_t["median_ms"] / oc_t["median_ms"],
        "notes": "tifffile has no public tile-level API; comparison is full page decode (the workaround a user would write)",
    }


def _build_pyramid_tiff(base_size: int = 4096) -> bytes:
    """Build a multi-level pyramid TIFF in memory."""
    arr = np.arange(base_size * base_size,
                    dtype=np.uint16).reshape(base_size, base_size)
    buf = io.BytesIO()
    with _tifffile.TiffWriter(buf) as tw:
        tw.write(arr, tile=(256, 256), compression=None, subfiletype=0)
        cur = arr
        # 4 overview levels.
        for _ in range(4):
            h2 = (cur.shape[0] // 2) * 2
            w2 = (cur.shape[1] // 2) * 2
            ds = cur[:h2, :w2].reshape(h2 // 2, 2, w2 // 2, 2).mean(axis=(1, 3))
            ds = ds.astype(arr.dtype)
            tw.write(ds, tile=(256, 256), compression=None, subfiletype=1)
            cur = ds
    return buf.getvalue()


@workload("tiff_pyramid_crop_from_fullres", tier="fast", group="tiff_pyramid")
def bench_tiff_pyramid_crop_fullres():
    """Crop a 1024×1024 region from the full-res level of a 4K pyramid.

    The COG selling point: only tiles overlapping the bbox get
    decoded. tifffile's closest equivalent is decoding the whole
    page then slicing.
    """
    import opencodecs as oc
    from opencodecs._tiff_pyramid import TiffPyramidReader
    if not oc.has_codec("tiff") or _tifffile is None:
        return {"skipped": "tiff codec or tifffile not available"}

    data = _build_pyramid_tiff(base_size=4096)

    def oc_fn():
        with TiffPyramidReader(data) as p:
            p.read_region(level=0, y=(1000, 2024), x=(1000, 2024))

    def tf_fn():
        with _tifffile.TiffFile(io.BytesIO(data)) as tf:
            page = tf.pages[0]
            full = page.asarray()
            _ = full[1000:2024, 1000:2024]

    oc_t = _time_fn(oc_fn, n=7)
    tf_t = _time_fn(tf_fn, n=7)
    return {
        "opencodecs": oc_t,
        "reference": {"tifffile_full_then_slice": tf_t},
        "speedup_vs_tifffile": tf_t["median_ms"] / oc_t["median_ms"],
    }


@workload("ndtiff_index_parse_synthetic_10k", tier="fast", group="ndtiff")
def bench_ndtiff_index_parse():
    """Parse a synthetic 10K-record NDTiff index."""
    if _ndstorage is None:
        return {"skipped": "ndstorage not installed"}
    from opencodecs.codecs._ndtiff import parse_ndtiff_index
    from ndstorage.ndtiff_index import read_ndtiff_index

    pixel_bytes = 4000 * 400 * 2
    buf = bytearray()
    for i in range(10_000):
        ax = json.dumps({"z": i}).encode()
        fn = b"NDTiffStack.tif"
        buf += struct.pack("<I", len(ax)) + ax
        buf += struct.pack("<I", len(fn)) + fn
        off = i * pixel_bytes % ((4 << 30) - 4 * pixel_bytes)
        buf += struct.pack("<IIIIIIII",
                           off, 4000, 400, 1, 0,
                           off + pixel_bytes, 200, 0)
    data = bytes(buf)

    oc_t = _time_fn(lambda: parse_ndtiff_index(data), n=7)
    nd_t = _time_fn(_silence_stdout(
        lambda: read_ndtiff_index(data, verbose=False)), n=5)
    return {
        "opencodecs": oc_t,
        "reference": {"ndstorage": nd_t},
        "speedup_vs_ndstorage": nd_t["median_ms"] / oc_t["median_ms"],
    }


@workload("ndtiff_random_frame_read", tier="fast", group="ndtiff")
def bench_ndtiff_random_frame_read():
    """100 random frames from a synthetic 200-frame NDTiff (uncompressed)."""
    if _ndstorage is None:
        return {"skipped": "ndstorage not installed"}
    from opencodecs._ndtiff_writer import NDTiffWriter
    from opencodecs._ndtiff import NDTiffDataset

    H, W = 200, 400
    N = 200
    tmp = tempfile.mkdtemp()
    try:
        with NDTiffWriter(tmp) as w:
            for i in range(N):
                # Generate u16 frame; values stay in range by modding the
                # full grid against (uint16 max // 2) so each frame is
                # distinguishable but never overflows.
                base = (i * 137) & 0x7FFF   # ensure < 32768
                a = ((np.arange(H * W, dtype=np.int32) + base) & 0xFFFF
                     ).astype(np.uint16).reshape(H, W)
                w.write_frame({"z": i}, a)

        rng = np.random.default_rng(0)
        zs = rng.integers(0, N, size=100)

        # Open BOTH datasets once outside the timed loop — we're
        # measuring per-frame read speed, not open() overhead.
        # (Open-time is a separate bench: ``ndtiff_index_parse_*``.)
        ds_oc = NDTiffDataset(tmp)
        ds_nd = _silence_stdout(_ndstorage.NDTiffDataset)(tmp)

        def oc_fn():
            for z in zs:
                ds_oc.read_frame(z=int(z))

        def nd_fn():
            for z in zs:
                ds_nd.read_image(z=int(z))

        try:
            oc_t = _time_fn(oc_fn, n=5)
            nd_t = _time_fn(nd_fn, n=5)
        finally:
            try: ds_oc.close()
            except Exception: pass
            try: ds_nd.close()
            except Exception: pass
        return {
            "opencodecs": oc_t,
            "reference": {"ndstorage": nd_t},
            "speedup_vs_ndstorage": nd_t["median_ms"] / oc_t["median_ms"],
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


@workload("ndtiff_write_50_frames", tier="fast", group="ndtiff")
def bench_ndtiff_write():
    """Write 50 frames of 400×400 u16 (16 MB total) — fast-tier scale."""
    if _ndstorage is None:
        return {"skipped": "ndstorage not installed"}
    from opencodecs._ndtiff_writer import NDTiffWriter

    H, W, N = 400, 400, 50
    frames = [np.random.default_rng(i).integers(0, 4000, size=(H, W),
                                                dtype=np.uint16)
              for i in range(N)]

    def oc_fn():
        tmp = tempfile.mkdtemp()
        try:
            with NDTiffWriter(tmp) as w:
                w.write_many(({"z": i}, a, None) for i, a in enumerate(frames))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def nd_fn():
        tmp = tempfile.mkdtemp()
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            from ndstorage.ndtiff_file import SingleNDTiffWriter
            w = SingleNDTiffWriter(tmp, "NDTiffStack.tif", {"PT": "u16"})
            for i, a in enumerate(frames):
                w.write_image(frozenset({"z": i}.items()), a, {"z_um": i * 0.5})
            w.finished_writing()
        finally:
            sys.stdout = old
            shutil.rmtree(tmp, ignore_errors=True)

    oc_t = _time_fn(oc_fn, n=5)
    nd_t = _time_fn(nd_fn, n=5)
    return {
        "opencodecs": oc_t,
        "reference": {"ndstorage": nd_t},
        "speedup_vs_ndstorage": nd_t["median_ms"] / oc_t["median_ms"],
        "bytes_written": N * H * W * 2,
    }


@workload("ndtiff_write_compressed_zstd", tier="fast", group="ndtiff_compress")
def bench_ndtiff_write_compressed():
    """Write 50 frames with zstd-on-the-fly compression."""
    from opencodecs._ndtiff_writer import NDTiffWriter

    H, W, N = 400, 400, 50
    # Use correlated data so compression actually wins
    rng = np.random.default_rng(0)
    yy, xx = np.indices((H, W))
    frames = [(500 + 0.5 * yy + 0.3 * xx + rng.normal(0, 5, (H, W))
               + i * 0.1).astype(np.uint16) for i in range(N)]

    def oc_fn_none():
        tmp = tempfile.mkdtemp()
        try:
            with NDTiffWriter(tmp, compression="none") as w:
                w.write_many(({"z": i}, a, None) for i, a in enumerate(frames))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def oc_fn_zstd():
        tmp = tempfile.mkdtemp()
        try:
            with NDTiffWriter(tmp, compression="zstd",
                              compression_level=1) as w:
                w.write_many(({"z": i}, a, None) for i, a in enumerate(frames))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def oc_fn_zstd_parallel():
        tmp = tempfile.mkdtemp()
        try:
            with NDTiffWriter(tmp, compression="zstd",
                              compression_level=1) as w:
                # n_workers=None → auto = min(cpu_count, 8)
                w.write_many((({"z": i}, a, None)
                              for i, a in enumerate(frames)),
                             n_workers=None)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    none_t = _time_fn(oc_fn_none, n=5)
    zstd_t = _time_fn(oc_fn_zstd, n=5)
    zstd_par_t = _time_fn(oc_fn_zstd_parallel, n=5)
    return {
        "opencodecs_none": none_t,
        "opencodecs_zstd_l1": zstd_t,
        "opencodecs_zstd_l1_parallel": zstd_par_t,
        "encode_overhead": zstd_t["median_ms"] / none_t["median_ms"],
        "parallel_encode_overhead": zstd_par_t["median_ms"] / none_t["median_ms"],
        "parallel_speedup": zstd_t["median_ms"] / zstd_par_t["median_ms"],
        "bytes_written_raw": N * H * W * 2,
    }


@workload("ndtiff_write_compressed_zstd_large", tier="fast",
          group="ndtiff_compress")
def bench_ndtiff_write_compressed_large():
    """Same as ndtiff_write_compressed_zstd but with realistic
    2048×2048 frames. Here each encode is ~3-5 ms — well above thread
    overhead — so the parallel pipeline should land a real speedup."""
    from opencodecs._ndtiff_writer import NDTiffWriter

    H, W, N = 2048, 2048, 8
    rng = np.random.default_rng(0)
    yy, xx = np.indices((H, W))
    frames = [(500 + 0.5 * yy + 0.3 * xx
               + rng.normal(0, 5, (H, W)) + i * 0.1).astype(np.uint16)
              for i in range(N)]

    def oc_fn_zstd_serial():
        tmp = tempfile.mkdtemp()
        try:
            with NDTiffWriter(tmp, compression="zstd",
                              compression_level=1) as w:
                w.write_many((({"z": i}, a, None)
                              for i, a in enumerate(frames)),
                             n_workers=1)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def oc_fn_zstd_parallel():
        tmp = tempfile.mkdtemp()
        try:
            with NDTiffWriter(tmp, compression="zstd",
                              compression_level=1) as w:
                w.write_many((({"z": i}, a, None)
                              for i, a in enumerate(frames)),
                             n_workers=None)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    serial_t = _time_fn(oc_fn_zstd_serial, n=5)
    par_t = _time_fn(oc_fn_zstd_parallel, n=5)
    return {
        "opencodecs": serial_t,
        "opencodecs_zstd_l1_parallel": par_t,
        "speedup_vs_serial": serial_t["median_ms"] / par_t["median_ms"],
        "bytes_written_raw": N * H * W * 2,
    }


@workload("tier1_codecs_roundtrip_10mb", tier="fast", group="codecs")
def bench_tier1_codecs():
    """Round-trip a 10 MB u16 ndarray through each Tier 1 codec."""
    import opencodecs as oc
    arr = np.random.default_rng(0).integers(
        0, 4000, size=(2048, 2560), dtype=np.uint16)
    raw = arr.tobytes()
    raw_mb = len(raw) / 1e6
    results: dict[str, Any] = {"size_mb": raw_mb}

    for codec_name, encode_kw, decode_kw in [
        ("zstd",   dict(level=3),              {}),
        ("lz4",    {},                          {}),
        ("brotli", dict(level=4),              {}),
        ("blosc2", dict(level=3),              {}),
        ("deflate", dict(level=3),             {}),
        ("bitshuffle", dict(itemsize=2),       dict(itemsize=2)),
    ]:
        if not oc.has_codec(codec_name):
            results[codec_name] = {"skipped": "not built"}
            continue

        def enc():
            oc.write(None, raw, format=codec_name, **encode_kw)
        t = _time_fn(enc, n=5)

        # For ndarray-aware codecs:
        results[codec_name] = {
            "encode": t,
            "mb_per_s": raw_mb / (t["median_ms"] / 1000),
        }

    # b2nd: ndarray-aware
    if oc.has_codec("b2nd"):
        def enc():
            oc.write(None, arr, format="b2nd",
                     compressor="zstd", shuffle="bit")
        t = _time_fn(enc, n=5)
        results["b2nd"] = {"encode": t,
                           "mb_per_s": raw_mb / (t["median_ms"] / 1000)}

    # pcodec: ndarray-aware
    if oc.has_codec("pcodec"):
        def enc():
            oc.write(None, arr, format="pcodec")
        t = _time_fn(enc, n=5)
        results["pcodec"] = {"encode": t,
                             "mb_per_s": raw_mb / (t["median_ms"] / 1000)}

    return results


@workload("tiff_write_64mb", tier="fast", group="tiff_write")
def bench_tiff_write_64mb():
    """Write a 4K×4K u16 image (~32 MB) as tiled+zstd TIFF — opencodecs
    vs tifffile.

    Fast-tier scale; main signal is per-write latency rather than
    throughput. The medium-tier ``tiff_write_1gb`` exercises sustained
    throughput.
    """
    if _tifffile is None:
        return {"skipped": "tifffile not installed"}
    from opencodecs._tiff_writer import TiffWriter
    import opencodecs as oc
    if not oc.has_codec("tiff"):
        return {"skipped": "tiff codec not built"}

    arr = np.random.default_rng(0).integers(
        0, 4000, size=(2048, 2048), dtype=np.uint16,
    )

    def oc_fn():
        buf = io.BytesIO()
        with TiffWriter(buf) as w:
            w.write_page(arr, tile=(256, 256),
                         compression="zstd", compression_level=1)

    def oc_fn_none():
        buf = io.BytesIO()
        with TiffWriter(buf) as w:
            w.write_page(arr, tile=(256, 256), compression="none")

    def tf_fn():
        buf = io.BytesIO()
        _tifffile.imwrite(buf, arr, tile=(256, 256), compression="zstd")

    oc_t = _time_fn(oc_fn, n=5)
    oc_none_t = _time_fn(oc_fn_none, n=5)
    tf_t = _time_fn(tf_fn, n=5)
    raw_mb = arr.nbytes / 1e6
    return {
        "opencodecs_zstd_l1": oc_t,
        "opencodecs_none": oc_none_t,
        "reference": {"tifffile_zstd": tf_t},
        "speedup_vs_tifffile_zstd": tf_t["median_ms"] / oc_t["median_ms"],
        "compress_overhead": oc_t["median_ms"] / oc_none_t["median_ms"],
        "raw_mb": raw_mb,
        "mb_per_s_zstd": raw_mb / (oc_t["median_ms"] / 1000),
        "mb_per_s_none": raw_mb / (oc_none_t["median_ms"] / 1000),
    }


# ---------------------------------------------------------------------------
# Medium-tier workloads (placeholders for the next session)
# ---------------------------------------------------------------------------


@workload("tiff_write_1gb", tier="medium", group="tiff_write")
def bench_tiff_write_1gb():
    """Write an 8K×8K u16 (~128 MB) image as tiled TIFF — sustained
    throughput vs tifffile. Repeated to land near 1 GB total written."""
    if _tifffile is None:
        return {"skipped": "tifffile not installed"}
    from opencodecs._tiff_writer import TiffWriter

    arr = np.random.default_rng(0).integers(
        0, 4000, size=(8192, 8192), dtype=np.uint16,
    )

    def oc_fn():
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
        try:
            with TiffWriter(tmp) as w:
                w.write_page(arr, tile=(256, 256), compression="none")
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def oc_fn_zstd():
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
        try:
            with TiffWriter(tmp) as w:
                w.write_page(arr, tile=(256, 256),
                             compression="zstd", compression_level=1)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    def tf_fn():
        tmp = tempfile.NamedTemporaryFile(suffix=".tif", delete=False).name
        try:
            _tifffile.imwrite(tmp, arr, tile=(256, 256), compression=None)
        finally:
            try:
                os.unlink(tmp)
            except OSError:
                pass

    oc_t = _time_fn(oc_fn, n=3)
    oc_zstd_t = _time_fn(oc_fn_zstd, n=3)
    tf_t = _time_fn(tf_fn, n=3)
    return {
        "opencodecs_none": oc_t,
        "opencodecs_zstd_l1": oc_zstd_t,
        "reference": {"tifffile_none": tf_t},
        "speedup_vs_tifffile_none": tf_t["median_ms"] / oc_t["median_ms"],
        "bytes_written": int(arr.nbytes),
    }


@workload("tiff_full_decode_1gb", tier="medium", group="tiff")
def bench_tiff_full_decode_1gb():
    """Decode a 1 GB tiled u16 TIFF — opencodecs vs tifffile."""
    if _tifffile is None:
        return {"skipped": "tifffile not installed"}
    import opencodecs as oc
    if not oc.has_codec("tiff"):
        return {"skipped": "tiff codec not built"}

    # 8K × 8K u16 ~= 128 MB on disk, x8 = 1 GB via multiple pages.
    arr = np.random.default_rng(0).integers(
        0, 4000, size=(8192, 8192), dtype=np.uint16)
    buf = io.BytesIO()
    _tifffile.imwrite(buf, arr, compression=None, tile=(256, 256))
    data = buf.getvalue()

    def oc_fn():
        with oc.get_codec("tiff").open(data) as r:
            r.page(0).asarray()

    def tf_fn():
        _tifffile.imread(io.BytesIO(data))

    oc_t = _time_fn(oc_fn, n=5)
    tf_t = _time_fn(tf_fn, n=5)
    return {
        "opencodecs": oc_t,
        "reference": {"tifffile": tf_t},
        "speedup_vs_tifffile": tf_t["median_ms"] / oc_t["median_ms"],
        "bytes_decoded": arr.nbytes,
    }


@workload("ndtiff_write_1gb", tier="medium", group="ndtiff")
def bench_ndtiff_write_1gb():
    """Write 250 frames × 400×4000 u16 ≈ 800 MB."""
    if _ndstorage is None:
        return {"skipped": "ndstorage not installed"}
    from opencodecs._ndtiff_writer import NDTiffWriter

    H, W, N = 400, 4000, 250
    rng = np.random.default_rng(0)
    frames = [rng.integers(0, 4000, size=(H, W), dtype=np.uint16)
              for _ in range(N)]

    def oc_fn():
        tmp = tempfile.mkdtemp()
        try:
            with NDTiffWriter(tmp) as w:
                w.write_many(({"z": i}, a, None) for i, a in enumerate(frames))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def nd_fn():
        tmp = tempfile.mkdtemp()
        old = sys.stdout
        sys.stdout = io.StringIO()
        try:
            from ndstorage.ndtiff_file import SingleNDTiffWriter
            w = SingleNDTiffWriter(tmp, "NDTiffStack.tif", {"PT": "u16"})
            for i, a in enumerate(frames):
                w.write_image(frozenset({"z": i}.items()), a,
                              {"z_um": i * 0.5})
            w.finished_writing()
        finally:
            sys.stdout = old
            shutil.rmtree(tmp, ignore_errors=True)

    oc_t = _time_fn(oc_fn, n=3)
    nd_t = _time_fn(nd_fn, n=3)
    return {
        "opencodecs": oc_t,
        "reference": {"ndstorage": nd_t},
        "speedup_vs_ndstorage": nd_t["median_ms"] / oc_t["median_ms"],
        "bytes_written": N * H * W * 2,
    }


# ---------------------------------------------------------------------------
# Slow-tier workloads
# ---------------------------------------------------------------------------


@workload("ndtiff_write_10gb", tier="slow", group="ndtiff")
def bench_ndtiff_write_10gb():
    """Write 800 frames × 1024×4000 u16 ≈ 6.5 GB. For threadripper/SSD."""
    from opencodecs._ndtiff_writer import NDTiffWriter

    H, W, N = 1024, 4000, 800
    rng = np.random.default_rng(0)

    def oc_fn():
        tmp = tempfile.mkdtemp()
        try:
            with NDTiffWriter(tmp) as w:
                for i in range(N):
                    a = rng.integers(0, 4000, size=(H, W), dtype=np.uint16)
                    w.write_frame({"z": i}, a)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    oc_t = _time_fn(oc_fn, n=2)
    return {
        "opencodecs": oc_t,
        "bytes_written": N * H * W * 2,
        "gb_per_s": (N * H * W * 2 / 1e9) / (oc_t["median_ms"] / 1000),
    }


# ---------------------------------------------------------------------------
# System info
# ---------------------------------------------------------------------------


def _system_info() -> dict:
    info = {
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor() or "(unknown)",
        "python": platform.python_version(),
        "cpu_count": os.cpu_count() or -1,
    }
    # CPU model on Linux
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except (FileNotFoundError, PermissionError):
        pass
    # macOS CPU
    if sys.platform == "darwin":
        try:
            info["cpu_model"] = subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except Exception:
            pass
    # RAM
    try:
        import resource
        info["ram_kb_max"] = resource.getrusage(
            resource.RUSAGE_SELF).ru_maxrss
    except ImportError:
        pass
    # opencodecs version + git commit
    try:
        import opencodecs as oc
        info["opencodecs_version"] = getattr(oc, "__version__", "unknown")
    except Exception:
        info["opencodecs_version"] = "unknown"
    try:
        rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=Path(__file__).parent.parent,
        ).strip()
        info["git_rev"] = rev
    except Exception:
        info["git_rev"] = "unknown"
    # Reference library versions
    info["ref_versions"] = {
        "tifffile": getattr(_tifffile, "__version__", None) if _tifffile else None,
        "ndstorage": getattr(_ndstorage, "__version__", None) if _ndstorage else None,
        "czifile":   getattr(_czifile, "__version__", None) if _czifile else None,
        "imagecodecs": getattr(_imagecodecs, "__version__", None) if _imagecodecs else None,
    }
    return info


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def _results_dir(host: str, machine: str) -> Path:
    base = Path(__file__).parent / "results"
    d = base / f"{host}_{machine}"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _format_markdown(results: dict, sysinfo: dict) -> str:
    """Build a Markdown summary for this run."""
    lines = []
    lines.append(f"# opencodecs bench — {sysinfo['hostname']} ({sysinfo['machine']})")
    lines.append("")
    lines.append(f"- Run at: `{results['timestamp']}`")
    lines.append(f"- opencodecs: `{sysinfo.get('opencodecs_version', '?')}` "
                 f"(git: `{sysinfo.get('git_rev', '?')}`)")
    lines.append(f"- Python: {sysinfo['python']}, "
                 f"CPU: {sysinfo.get('cpu_model', sysinfo['processor'])} "
                 f"× {sysinfo['cpu_count']}")
    rv = sysinfo.get("ref_versions") or {}
    refs = ", ".join(f"{k} {v}" for k, v in rv.items() if v)
    if refs:
        lines.append(f"- Reference libraries: {refs}")
    lines.append("")
    lines.append(f"## Workloads (tier: {results['tier']})")
    lines.append("")
    lines.append("| Workload | median (ms) | min | max | IQR | noisy | ratio |")
    lines.append("|---|---:|---:|---:|---:|:-:|---:|")
    for name, r in results["workloads"].items():
        if "skipped" in r:
            lines.append(f"| {name} | skipped: {r['skipped']} | — | — | — | — | — |")
            continue
        oc = r.get("opencodecs") or r.get("opencodecs_zstd_l1") \
            or r.get("opencodecs_none")
        if not oc:
            # codec multi-row workload
            for codec, sub in r.items():
                if isinstance(sub, dict) and "encode" in sub:
                    enc = sub["encode"]
                    flag = "⚠️" if enc.get("noisy") else ""
                    mbps = sub.get("mb_per_s", 0)
                    lines.append(
                        f"| {name}/{codec} | {enc['median_ms']:.2f} | "
                        f"{enc['min_ms']:.2f} | {enc['max_ms']:.2f} | "
                        f"{enc['iqr_ms']:.2f} | {flag} | "
                        f"{mbps:.0f} MB/s |"
                    )
            continue
        ratio_key = next(
            (k for k in r if k.startswith("speedup_vs_") or k == "encode_overhead"),
            None,
        )
        ratio_str = f"{r[ratio_key]:.2f}×" if ratio_key else "—"
        if ratio_key == "encode_overhead":
            ratio_str = f"{r[ratio_key]:.2f}× of uncompressed"
        flag = "⚠️" if oc.get("noisy") else ""
        lines.append(
            f"| {name} | {oc['median_ms']:.2f} | {oc['min_ms']:.2f} | "
            f"{oc['max_ms']:.2f} | {oc['iqr_ms']:.2f} | {flag} | {ratio_str} |"
        )
    lines.append("")
    return "\n".join(lines)


def _detect_regressions(
    current: dict, host_dir: Path,
    threshold: float = 1.20,
    history_window: int = 5,
) -> list[dict]:
    """Compare current run's medians to rolling median of last N runs."""
    history_file = host_dir / "history.jsonl"
    if not history_file.exists():
        return []
    history: list[dict] = []
    with history_file.open() as f:
        for line in f:
            try:
                history.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if not history:
        return []

    regressions = []
    # Group history by workload name; pick the last N entries per workload.
    by_name: dict[str, list[float]] = {}
    for h in history:
        by_name.setdefault(h["name"], []).append(h["median_ms"])
    for name, hist_medians in by_name.items():
        recent = hist_medians[-history_window:]
        if len(recent) < 2:
            continue
        rolling_median = statistics.median(recent)
        cur = current["workloads"].get(name)
        if not cur:
            continue
        cur_med = (cur.get("opencodecs") or {}).get("median_ms")
        if cur_med is None:
            continue
        if cur_med > rolling_median * threshold:
            regressions.append({
                "workload": name,
                "current_ms": cur_med,
                "rolling_median_ms": rolling_median,
                "ratio": cur_med / rolling_median,
            })
    return regressions


def _append_history(current: dict, host_dir: Path) -> None:
    """Append one line per workload to the rolling history."""
    history_file = host_dir / "history.jsonl"
    with history_file.open("a") as f:
        ts = current["timestamp"]
        rev = current.get("git_rev", "unknown")
        for name, r in current["workloads"].items():
            oc = r.get("opencodecs") or r.get("opencodecs_zstd_l1")
            if not oc:
                continue
            f.write(json.dumps({
                "timestamp": ts,
                "git_rev": rev,
                "name": name,
                "median_ms": oc["median_ms"],
                "min_ms": oc["min_ms"],
                "max_ms": oc["max_ms"],
                "noisy": oc.get("noisy", False),
            }) + "\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fast", action="store_true",
        help="Fast tier — small fixtures, ≤5 min",
    )
    parser.add_argument(
        "--medium", action="store_true",
        help="Medium tier — realistic small-lab sizes, ~30 min",
    )
    parser.add_argument(
        "--slow", action="store_true",
        help="Slow tier — 10+ GB workloads, 1+ hour",
    )
    parser.add_argument(
        "--filter", default=None,
        help="Substring filter on workload name",
    )
    parser.add_argument(
        "--no-regression-check", action="store_true",
        help="Skip the regression check against history",
    )
    parser.add_argument(
        "--regression-threshold", type=float, default=1.20,
        help="Median-time ratio threshold for regression flag (default 1.20)",
    )
    args = parser.parse_args()

    tiers = set()
    if args.fast:
        tiers.add("fast")
    if args.medium:
        tiers.add("medium")
    if args.slow:
        tiers.add("slow")
    if not tiers:
        tiers = {"fast"}

    sysinfo = _system_info()
    host_dir = _results_dir(sysinfo["hostname"], sysinfo["machine"])
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    print(f"opencodecs bench — host={sysinfo['hostname']} "
          f"machine={sysinfo['machine']} tiers={sorted(tiers)}")
    print(f"opencodecs={sysinfo.get('opencodecs_version','?')} "
          f"git={sysinfo.get('git_rev','?')}")
    print()

    workloads_to_run = [
        (name, meta) for name, meta in WORKLOADS.items()
        if meta["tier"] in tiers
        and (args.filter is None or args.filter in name)
    ]

    results: dict[str, dict] = {}
    for name, meta in workloads_to_run:
        print(f"→ {name} ({meta['tier']}, {meta['group']}) ...", flush=True)
        t0 = time.perf_counter()
        try:
            r = meta["fn"]()
        except Exception as exc:
            r = {"error": f"{type(exc).__name__}: {exc}"}
        dt = time.perf_counter() - t0
        results[name] = r
        # Brief one-line summary
        if "skipped" in r:
            print(f"   skipped: {r['skipped']}")
        elif "error" in r:
            print(f"   ERROR: {r['error']}")
        else:
            oc = r.get("opencodecs") or r.get("opencodecs_zstd_l1") or {}
            if oc:
                noise_flag = " NOISY" if oc.get("noisy") else ""
                ratio_key = next(
                    (k for k in r if k.startswith("speedup_vs_")
                     or k == "encode_overhead"),
                    None,
                )
                ratio_str = (f"  ratio={r[ratio_key]:.2f}×"
                             if ratio_key else "")
                print(f"   opencodecs median={oc['median_ms']:.2f} ms "
                      f"(min={oc['min_ms']:.2f}, "
                      f"iqr={oc['iqr_ms']:.2f}){ratio_str}{noise_flag}")
            else:
                # Multi-codec workload
                cdc_summaries = []
                for cdc, sub in r.items():
                    if isinstance(sub, dict) and "encode" in sub:
                        cdc_summaries.append(
                            f"{cdc}={sub['encode']['median_ms']:.1f}ms")
                if cdc_summaries:
                    print(f"   {', '.join(cdc_summaries)}")
        print(f"   ({dt:.1f}s wall)")

    full = {
        "timestamp": timestamp,
        "git_rev": sysinfo.get("git_rev"),
        "tier": ",".join(sorted(tiers)),
        "system": sysinfo,
        "workloads": results,
    }

    # Persist results
    json_path = host_dir / f"{timestamp}.json"
    with json_path.open("w") as f:
        json.dump(full, f, indent=2, default=str)
    md = _format_markdown(full, sysinfo)
    (host_dir / "latest.md").write_text(md)

    # Append-only history
    _append_history(full, host_dir)

    print()
    print(f"Results: {json_path}")
    print(f"Summary: {host_dir / 'latest.md'}")

    # Regression check
    if not args.no_regression_check:
        regressions = _detect_regressions(
            full, host_dir, threshold=args.regression_threshold)
        if regressions:
            print()
            print(
                f"⚠️  {len(regressions)} regression(s) detected vs rolling "
                f"median of last 5 runs (threshold "
                f"{args.regression_threshold:.2f}×):"
            )
            for r in regressions:
                print(
                    f"   {r['workload']}: {r['current_ms']:.2f} ms vs "
                    f"rolling {r['rolling_median_ms']:.2f} ms "
                    f"({r['ratio']:.2f}×)"
                )
            return 1
        else:
            print()
            print("No regressions vs last 5 runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
