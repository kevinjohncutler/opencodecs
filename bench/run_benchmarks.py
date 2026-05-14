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


# Windows default console codepage is cp1252 — the per-workload progress
# lines below use a few non-ASCII characters (arrows, ×). Force the
# script's stdout to UTF-8 so we don't crash with UnicodeEncodeError
# under PowerShell / cmd. errors="replace" keeps us alive if the
# terminal still can't render a glyph.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # pragma: no cover - non-text streams (e.g. piped to a buffer)
    pass


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


# ---------------------------------------------------------------------------
# Direct head-to-head: opencodecs vs imagecodecs (raw codec calls)
# ---------------------------------------------------------------------------
#
# Both packages wrap the same underlying C libraries (libzstd, libdeflate,
# libjxl, …), but they differ in binding strategy: opencodecs aims for
# zero-copy buffer-protocol dispatch and minimal Python-side overhead;
# imagecodecs uses a more conservative bytes() coercion path for safety.
# These benches measure whether the difference is observable in practice.
#
# One workload per (codec, op) pair — each renders as a single row in
# the Markdown summary, with `speedup_vs_imagecodecs` driving the ratio
# column. Payloads are sized so encode + decode each take 2-20 ms on
# a modern CPU (above the timing floor, below the patience threshold).


def _h2h_byte_stream(name: str, oc_encode, oc_decode, ic_encode, ic_decode,
                     *, data, encode_kw=None, decode_kw=None):
    """One round of head-to-head for a byte-stream codec.

    Returns a dict with side-by-side encode + decode timings. Each
    library decodes its own encoded payload (mirrors real-world use
    and avoids cross-decode interop bugs polluting timing).
    """
    encode_kw = encode_kw or {}
    decode_kw = decode_kw or {}
    oc_payload = oc_encode(data, **encode_kw)
    ic_payload = ic_encode(data, **encode_kw)
    raw_mb = len(data) / 1e6

    def oc_enc(): oc_encode(data, **encode_kw)
    def ic_enc(): ic_encode(data, **encode_kw)
    def oc_dec(): oc_decode(oc_payload, **decode_kw)
    def ic_dec(): ic_decode(ic_payload, **decode_kw)

    oc_e = _time_fn(oc_enc, n=7)
    ic_e = _time_fn(ic_enc, n=7)
    oc_d = _time_fn(oc_dec, n=7)
    ic_d = _time_fn(ic_dec, n=7)
    return {
        "raw_mb": raw_mb,
        "encoded_mb_oc": len(oc_payload) / 1e6,
        "encoded_mb_ic": len(ic_payload) / 1e6,
        "opencodecs_encode_ms": oc_e["median_ms"],
        "imagecodecs_encode_ms": ic_e["median_ms"],
        "speedup_vs_imagecodecs_encode":
            ic_e["median_ms"] / oc_e["median_ms"],
        "opencodecs_decode_ms": oc_d["median_ms"],
        "imagecodecs_decode_ms": ic_d["median_ms"],
        "speedup_vs_imagecodecs_decode":
            ic_d["median_ms"] / oc_d["median_ms"],
        "opencodecs_encode_mb_per_s":
            raw_mb / (oc_e["median_ms"] / 1000),
        "imagecodecs_encode_mb_per_s":
            raw_mb / (ic_e["median_ms"] / 1000),
        "opencodecs_decode_mb_per_s":
            raw_mb / (oc_d["median_ms"] / 1000),
        "imagecodecs_decode_mb_per_s":
            raw_mb / (ic_d["median_ms"] / 1000),
        "opencodecs": oc_e,    # canonical key for the renderer
        "reference": {"imagecodecs_encode": ic_e,
                      "imagecodecs_decode": ic_d,
                      "opencodecs_decode": oc_d},
        "speedup_vs_imagecodecs":     # average of encode+decode
            ((ic_e["median_ms"] + ic_d["median_ms"]) /
             (oc_e["median_ms"] + oc_d["median_ms"])),
    }


def _h2h_image(name: str, oc_encode, oc_decode, ic_encode, ic_decode,
               *, arr, encode_kw=None, decode_kw=None):
    """Head-to-head for an image-format codec (ndarray ↔ bytes)."""
    encode_kw = encode_kw or {}
    decode_kw = decode_kw or {}
    oc_payload = oc_encode(arr, **encode_kw)
    ic_payload = ic_encode(arr, **encode_kw)
    raw_mb = arr.nbytes / 1e6

    def oc_enc(): oc_encode(arr, **encode_kw)
    def ic_enc(): ic_encode(arr, **encode_kw)
    def oc_dec(): oc_decode(oc_payload, **decode_kw)
    def ic_dec(): ic_decode(ic_payload, **decode_kw)

    oc_e = _time_fn(oc_enc, n=5)
    ic_e = _time_fn(ic_enc, n=5)
    oc_d = _time_fn(oc_dec, n=5)
    ic_d = _time_fn(ic_dec, n=5)
    return {
        "raw_mb": raw_mb,
        "encoded_mb_oc": len(oc_payload) / 1e6,
        "encoded_mb_ic": len(ic_payload) / 1e6,
        "opencodecs_encode_ms": oc_e["median_ms"],
        "imagecodecs_encode_ms": ic_e["median_ms"],
        "speedup_vs_imagecodecs_encode":
            ic_e["median_ms"] / oc_e["median_ms"],
        "opencodecs_decode_ms": oc_d["median_ms"],
        "imagecodecs_decode_ms": ic_d["median_ms"],
        "speedup_vs_imagecodecs_decode":
            ic_d["median_ms"] / oc_d["median_ms"],
        "opencodecs_encode_mb_per_s":
            raw_mb / (oc_e["median_ms"] / 1000),
        "imagecodecs_encode_mb_per_s":
            raw_mb / (ic_e["median_ms"] / 1000),
        "opencodecs_decode_mb_per_s":
            raw_mb / (oc_d["median_ms"] / 1000),
        "imagecodecs_decode_mb_per_s":
            raw_mb / (ic_d["median_ms"] / 1000),
        "opencodecs": oc_e,
        "reference": {"imagecodecs_encode": ic_e,
                      "imagecodecs_decode": ic_d,
                      "opencodecs_decode": oc_d},
        "speedup_vs_imagecodecs":
            ((ic_e["median_ms"] + ic_d["median_ms"]) /
             (oc_e["median_ms"] + oc_d["median_ms"])),
    }


@workload("h2h_zstd_10mb", tier="fast", group="h2h_byte_stream")
def bench_h2h_zstd():
    """opencodecs.zstd vs imagecodecs.zstd on 10 MB compressible bytes."""
    if _imagecodecs is None:
        return {"skipped": "imagecodecs not installed"}
    from opencodecs.codecs._zstd import encode as oc_enc, decode as oc_dec
    data = np.random.default_rng(0).integers(
        0, 4000, size=(5 * 1024 * 1024,), dtype=np.uint16
    ).tobytes()
    return _h2h_byte_stream(
        "zstd", oc_enc, oc_dec,
        _imagecodecs.zstd_encode, _imagecodecs.zstd_decode,
        data=data, encode_kw=dict(level=3),
    )


@workload("h2h_deflate_10mb", tier="fast", group="h2h_byte_stream")
def bench_h2h_deflate():
    if _imagecodecs is None:
        return {"skipped": "imagecodecs not installed"}
    from opencodecs.codecs._deflate import encode as oc_enc, decode as oc_dec
    data = np.random.default_rng(0).integers(
        0, 4000, size=(5 * 1024 * 1024,), dtype=np.uint16
    ).tobytes()
    return _h2h_byte_stream(
        "deflate", oc_enc, oc_dec,
        _imagecodecs.zlib_encode, _imagecodecs.zlib_decode,
        data=data, encode_kw=dict(level=6),
    )


@workload("h2h_lz4_10mb", tier="fast", group="h2h_byte_stream")
def bench_h2h_lz4():
    if _imagecodecs is None:
        return {"skipped": "imagecodecs not installed"}
    from opencodecs.codecs._lz4 import encode as oc_enc, decode as oc_dec
    data = np.random.default_rng(0).integers(
        0, 4000, size=(5 * 1024 * 1024,), dtype=np.uint16
    ).tobytes()
    return _h2h_byte_stream(
        "lz4", oc_enc, oc_dec,
        _imagecodecs.lz4f_encode, _imagecodecs.lz4f_decode,
        data=data,
    )


@workload("h2h_brotli_10mb", tier="fast", group="h2h_byte_stream")
def bench_h2h_brotli():
    if _imagecodecs is None:
        return {"skipped": "imagecodecs not installed"}
    from opencodecs.codecs._brotli import encode as oc_enc, decode as oc_dec
    # Brotli at level 11 is slow; level 4 is the realistic fast-tier level.
    data = np.random.default_rng(0).integers(
        0, 4000, size=(5 * 1024 * 1024,), dtype=np.uint16
    ).tobytes()
    return _h2h_byte_stream(
        "brotli", oc_enc, oc_dec,
        _imagecodecs.brotli_encode, _imagecodecs.brotli_decode,
        data=data, encode_kw=dict(level=4),
    )


@workload("h2h_blosc2_10mb", tier="fast", group="h2h_byte_stream")
def bench_h2h_blosc2():
    if _imagecodecs is None:
        return {"skipped": "imagecodecs not installed"}
    from opencodecs.codecs._blosc2 import encode as oc_enc, decode as oc_dec
    data = np.random.default_rng(0).integers(
        0, 4000, size=(5 * 1024 * 1024,), dtype=np.uint16
    ).tobytes()
    return _h2h_byte_stream(
        "blosc2", oc_enc, oc_dec,
        _imagecodecs.blosc2_encode, _imagecodecs.blosc2_decode,
        data=data, encode_kw=dict(level=3),
    )


@workload("h2h_jpeg_4mp_rgb", tier="fast", group="h2h_image")
def bench_h2h_jpeg():
    if _imagecodecs is None:
        return {"skipped": "imagecodecs not installed"}
    from opencodecs.codecs._jpeg import encode as oc_enc, decode as oc_dec
    arr = np.random.default_rng(0).integers(
        0, 256, size=(2048, 2048, 3), dtype=np.uint8,
    )
    return _h2h_image(
        "jpeg", oc_enc, oc_dec,
        _imagecodecs.jpeg_encode, _imagecodecs.jpeg_decode,
        arr=arr, encode_kw=dict(level=85),
    )


@workload("h2h_png_4mp_rgb", tier="fast", group="h2h_image")
def bench_h2h_png():
    if _imagecodecs is None:
        return {"skipped": "imagecodecs not installed"}
    from opencodecs.codecs._png import encode as oc_enc, decode as oc_dec
    # PNG at full level on random data is slow + huge — use 2K×2K
    # at moderate compression so encode lands ~30 ms.
    arr = np.random.default_rng(0).integers(
        0, 256, size=(2048, 2048, 3), dtype=np.uint8,
    )
    return _h2h_image(
        "png", oc_enc, oc_dec,
        _imagecodecs.png_encode, _imagecodecs.png_decode,
        arr=arr, encode_kw=dict(level=4),
    )


@workload("h2h_webp_4mp_rgb", tier="fast", group="h2h_image")
def bench_h2h_webp():
    """WebP head-to-head at matching mode + quality.

    imagecodecs.webp_encode defaults to ``lossless=True`` while
    opencodecs defaults to ``lossless=False`` at quality 75. Comparing
    the defaults gives bitstreams 5× different in size — meaningless
    as a codec comparison. Force both into lossy q=75 (the WebP
    real-world default) so the bench measures the encoder/decoder, not
    the format choice.
    """
    if _imagecodecs is None:
        return {"skipped": "imagecodecs not installed"}
    from opencodecs.codecs._webp import encode as oc_enc, decode as oc_dec
    arr = np.random.default_rng(0).integers(
        0, 256, size=(2048, 2048, 3), dtype=np.uint8,
    )
    return _h2h_image(
        "webp", oc_enc, oc_dec,
        _imagecodecs.webp_encode, _imagecodecs.webp_decode,
        arr=arr, encode_kw=dict(level=75, lossless=False),
    )


@workload("h2h_jpeg2k_4mp_u16", tier="fast", group="h2h_image")
def bench_h2h_jpeg2k():
    if _imagecodecs is None:
        return {"skipped": "imagecodecs not installed"}
    from opencodecs.codecs._jpeg2k import encode as oc_enc, decode as oc_dec
    arr = np.random.default_rng(0).integers(
        0, 4000, size=(2048, 2048), dtype=np.uint16,
    )
    return _h2h_image(
        "jpeg2k", oc_enc, oc_dec,
        _imagecodecs.jpeg2k_encode, _imagecodecs.jpeg2k_decode,
        arr=arr,
    )


@workload("h2h_qoi_4mp_rgb", tier="fast", group="h2h_image")
def bench_h2h_qoi():
    if _imagecodecs is None:
        return {"skipped": "imagecodecs not installed"}
    from opencodecs.codecs._qoi import encode as oc_enc, decode as oc_dec
    arr = np.random.default_rng(0).integers(
        0, 256, size=(2048, 2048, 3), dtype=np.uint8,
    )
    return _h2h_image(
        "qoi", oc_enc, oc_dec,
        _imagecodecs.qoi_encode, _imagecodecs.qoi_decode,
        arr=arr,
    )


@workload("h2h_lerc_4mp_u16", tier="fast", group="h2h_image")
def bench_h2h_lerc():
    """LERC head-to-head — run each library in a separate Python
    subprocess to avoid the libLerc symbol clash that crashes the
    second-loaded copy.

    opencodecs and imagecodecs each statically link their own libLerc.
    When both extension modules are loaded in the same process, the
    duplicated symbols (`Lerc_Encode`, `Lerc_Decode`, internal globals)
    resolve to the wrong implementation on the second-loaded copy and
    we abort with SIGABRT. The fix is to time each library in its own
    Python interpreter.
    """
    if _imagecodecs is None:
        return {"skipped": "imagecodecs not installed"}

    # Generate the source array once and serialize it so both
    # subprocesses see the same input bytes.
    raw_mb = 8.0      # 2048×2048 u16 = 8 MiB
    arr = np.random.default_rng(0).integers(
        0, 4000, size=(2048, 2048), dtype=np.uint16,
    )
    payload = arr.tobytes()
    shape = arr.shape

    enc_script = """
import sys, time, statistics, numpy as np
shape = tuple({shape})
arr = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint16).reshape(shape)
{import_line}
samples = []
# Warmup
encode_fn(arr)
for _ in range(5):
    t0 = time.perf_counter()
    enc = encode_fn(arr)
    samples.append((time.perf_counter() - t0) * 1000)
print('SAMPLES', *samples, 'BYTES', len(enc))
"""
    dec_script = """
import sys, time, statistics, numpy as np
shape = tuple({shape})
arr = np.frombuffer(sys.stdin.buffer.read(), dtype=np.uint16).reshape(shape)
{import_line}
enc = encode_fn(arr)
samples = []
decode_fn(enc)
for _ in range(5):
    t0 = time.perf_counter()
    decode_fn(enc)
    samples.append((time.perf_counter() - t0) * 1000)
print('SAMPLES', *samples)
"""

    def _run_subprocess(script: str, import_line: str) -> dict:
        import subprocess as sp
        py = sys.executable
        proc = sp.run(
            [py, "-c", script.format(shape=list(shape), import_line=import_line)],
            input=payload,
            capture_output=True, timeout=120,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"LERC subprocess failed: rc={proc.returncode}\n"
                f"stderr={proc.stderr.decode('utf-8', 'replace')[:500]}"
            )
        out = proc.stdout.decode().strip().split()
        i = out.index("SAMPLES") + 1
        samples = []
        for tok in out[i:]:
            try:
                samples.append(float(tok))
            except ValueError:
                break
        filtered = _hampel_filter(samples)
        if not filtered:
            filtered = samples
        sorted_f = sorted(filtered)
        med = statistics.median(filtered)
        iqr = (sorted_f[3 * len(sorted_f) // 4]
               - sorted_f[len(sorted_f) // 4]) if len(sorted_f) >= 4 else 0.0
        return {
            "median_ms": med, "min_ms": min(filtered), "max_ms": max(filtered),
            "iqr_ms": iqr, "noisy": (med < 0.5) or (med > 0 and iqr/med > 0.25),
            "n_runs_total": len(samples),
            "n_runs_after_filter": len(filtered),
            "all_samples_ms": samples,
        }

    OC = ("from opencodecs.codecs._lerc import encode as encode_fn, "
          "decode as decode_fn")
    IC = ("import imagecodecs; "
          "encode_fn = imagecodecs.lerc_encode; "
          "decode_fn = imagecodecs.lerc_decode")

    oc_e = _run_subprocess(enc_script, OC)
    ic_e = _run_subprocess(enc_script, IC)
    oc_d = _run_subprocess(dec_script, OC)
    ic_d = _run_subprocess(dec_script, IC)
    return {
        "raw_mb": raw_mb,
        "opencodecs_encode_ms": oc_e["median_ms"],
        "imagecodecs_encode_ms": ic_e["median_ms"],
        "speedup_vs_imagecodecs_encode":
            ic_e["median_ms"] / oc_e["median_ms"],
        "opencodecs_decode_ms": oc_d["median_ms"],
        "imagecodecs_decode_ms": ic_d["median_ms"],
        "speedup_vs_imagecodecs_decode":
            ic_d["median_ms"] / oc_d["median_ms"],
        "opencodecs_encode_mb_per_s": raw_mb / (oc_e["median_ms"] / 1000),
        "imagecodecs_encode_mb_per_s": raw_mb / (ic_e["median_ms"] / 1000),
        "opencodecs_decode_mb_per_s": raw_mb / (oc_d["median_ms"] / 1000),
        "imagecodecs_decode_mb_per_s": raw_mb / (ic_d["median_ms"] / 1000),
        "opencodecs": oc_e,
        "reference": {"imagecodecs_encode": ic_e,
                      "imagecodecs_decode": ic_d,
                      "opencodecs_decode": oc_d},
        "speedup_vs_imagecodecs":
            ((ic_e["median_ms"] + ic_d["median_ms"]) /
             (oc_e["median_ms"] + oc_d["median_ms"])),
    }


@workload("h2h_jxl_4mp_rgb", tier="fast", group="h2h_image")
def bench_h2h_jxl():
    if _imagecodecs is None:
        return {"skipped": "imagecodecs not installed"}
    from opencodecs.codecs._jxl import encode as oc_enc, decode as oc_dec
    arr = np.random.default_rng(0).integers(
        0, 256, size=(2048, 2048, 3), dtype=np.uint8,
    )
    return _h2h_image(
        "jxl", oc_enc, oc_dec,
        _imagecodecs.jpegxl_encode, _imagecodecs.jpegxl_decode,
        arr=arr,
    )


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

    def oc_fn_serial():
        buf = io.BytesIO()
        with TiffWriter(buf) as w:
            w.write_page(arr, tile=(256, 256),
                         compression="zstd", compression_level=1,
                         n_workers=1)

    def oc_fn_parallel():
        buf = io.BytesIO()
        with TiffWriter(buf) as w:
            w.write_page(arr, tile=(256, 256),
                         compression="zstd", compression_level=1,
                         n_workers=None)   # auto = min(cpu, 8)

    def oc_fn_none():
        buf = io.BytesIO()
        with TiffWriter(buf) as w:
            w.write_page(arr, tile=(256, 256), compression="none")

    def tf_fn():
        buf = io.BytesIO()
        _tifffile.imwrite(buf, arr, tile=(256, 256), compression="zstd")

    oc_t = _time_fn(oc_fn_parallel, n=5)
    oc_serial_t = _time_fn(oc_fn_serial, n=5)
    oc_none_t = _time_fn(oc_fn_none, n=5)
    tf_t = _time_fn(tf_fn, n=5)
    raw_mb = arr.nbytes / 1e6
    return {
        "opencodecs": oc_t,    # canonical key — the recommended default
        "opencodecs_zstd_l1_serial": oc_serial_t,
        "opencodecs_none": oc_none_t,
        "reference": {"tifffile_zstd": tf_t},
        "speedup_vs_tifffile": tf_t["median_ms"] / oc_t["median_ms"],
        "parallel_speedup_over_serial":
            oc_serial_t["median_ms"] / oc_t["median_ms"],
        "compress_overhead": oc_t["median_ms"] / oc_none_t["median_ms"],
        "raw_mb": raw_mb,
        "mb_per_s_zstd_parallel": raw_mb / (oc_t["median_ms"] / 1000),
        "mb_per_s_zstd_serial": raw_mb / (oc_serial_t["median_ms"] / 1000),
        "mb_per_s_none": raw_mb / (oc_none_t["median_ms"] / 1000),
    }


@workload("hdf5_parallel_read_compressed", tier="fast", group="hdf5")
def bench_hdf5_parallel_read():
    """Parallel-decompress HDF5 read vs ``h5py``.

    Build a 512 MB u16 chunked + gzip-4 dataset, then time:
      * h5py's vanilla ``ds[...]`` (serialized by libhdf5's library lock)
      * opencodecs's :meth:`HdfReader.read_parallel` (raw-chunk read +
        N-worker decompress).

    The win comes from moving the deflate decode out of libhdf5's lock.
    """
    try:
        import h5py  # noqa: F401
    except ImportError:
        return {"skipped": "h5py not installed"}
    import os
    import tempfile
    from opencodecs._hdf5_codec import HdfReader

    arr = np.random.default_rng(0).integers(
        0, 1000, size=(256, 1024, 1024), dtype=np.uint16,
    )
    fd, path = tempfile.mkstemp(suffix=".h5")
    os.close(fd)
    try:
        with h5py.File(path, "w") as f:
            f.create_dataset(
                "img", data=arr, chunks=(8, 256, 256),
                compression="gzip", compression_opts=4,
            )
        on_disk = os.path.getsize(path)

        def h5py_serial():
            with h5py.File(path, "r") as f:
                _ = f["img"][:]

        def oc_parallel():
            r = HdfReader(path)
            try:
                _ = r.read_parallel(n_workers=None)
            finally:
                r.close()

        oc_p = _time_fn(oc_parallel, n=3)
        h5_t = _time_fn(h5py_serial, n=3)
        raw_mb = arr.nbytes / 1e6
        return {
            "raw_mb": raw_mb,
            "on_disk_mb": on_disk / 1e6,
            "opencodecs": oc_p,
            "reference": {"h5py": h5_t},
            "speedup_vs_h5py": h5_t["median_ms"] / oc_p["median_ms"],
            "mb_per_s_parallel": raw_mb / (oc_p["median_ms"] / 1000),
            "mb_per_s_h5py": raw_mb / (h5_t["median_ms"] / 1000),
        }
    finally:
        os.remove(path)


@workload("omezarr_write_512mb_zstd", tier="fast", group="omezarr_write")
def bench_omezarr_write():
    """Write a 512 MB u16 array as Zarr v2 + zstd chunks.

    Compares opencodecs serial (workers=1), parallel (workers=auto),
    and ``zarr-python`` defaults. Confirms the parallel-chunk-encode
    win documented in feedback_perf_patterns.
    """
    try:
        import zarr  # noqa: F401
        from numcodecs import Zstd
    except ImportError:
        return {"skipped": "zarr-python not installed"}
    from opencodecs._omezarr_writer import write_zarr_array
    import tempfile

    arr = np.random.default_rng(0).integers(
        0, 1000, size=(2048, 2048, 64), dtype=np.uint16,
    )
    chunks = (256, 256, 32)

    def oc_serial():
        with tempfile.TemporaryDirectory() as td:
            write_zarr_array(td + "/a", arr, chunks=chunks,
                             compressor="zstd", compression_level=3,
                             workers=1, zarr_format=2)

    def oc_parallel():
        with tempfile.TemporaryDirectory() as td:
            write_zarr_array(td + "/a", arr, chunks=chunks,
                             compressor="zstd", compression_level=3,
                             workers=None, zarr_format=2)

    def zarr_py():
        import zarr
        from numcodecs import Zstd
        with tempfile.TemporaryDirectory() as td:
            z = zarr.open(td + "/a", mode="w", shape=arr.shape,
                          chunks=chunks, dtype=arr.dtype,
                          compressor=Zstd(level=3), zarr_format=2)
            z[:] = arr

    oc_p = _time_fn(oc_parallel, n=3)
    oc_s = _time_fn(oc_serial, n=3)
    zp = _time_fn(zarr_py, n=3)
    raw_mb = arr.nbytes / 1e6
    return {
        "raw_mb": raw_mb,
        "opencodecs": oc_p,
        "opencodecs_serial": oc_s,
        "reference": {"zarr_python": zp},
        "speedup_vs_zarr_python": zp["median_ms"] / oc_p["median_ms"],
        "parallel_speedup_over_serial":
            oc_s["median_ms"] / oc_p["median_ms"],
        "mb_per_s_parallel": raw_mb / (oc_p["median_ms"] / 1000),
        "mb_per_s_serial": raw_mb / (oc_s["median_ms"] / 1000),
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

    # NDTiff write at 1GB is heavily bimodal — page-cache fills mid-run
    # and the OS switches into a synchronous write-back mode. With low
    # sample count, median picks one mode or the other at random and
    # masks the code's true throughput. We run more samples and report
    # the ratio from min_ms (the kernel's "free path" speed, which is
    # what the user actually sees on every run that isn't bounded by
    # write-back saturation).
    oc_t = _time_fn(oc_fn, n=9)
    nd_t = _time_fn(nd_fn, n=9)
    return {
        "opencodecs": oc_t,
        "reference": {"ndstorage": nd_t},
        "speedup_vs_ndstorage": nd_t["min_ms"] / oc_t["min_ms"],
        "speedup_vs_ndstorage_median": nd_t["median_ms"] / oc_t["median_ms"],
        "bytes_written": N * H * W * 2,
        "_note": "ratio from min_ms; median is bimodal under OS write-back",
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
        # Head-to-head workloads carry both encode + decode timings
        # against imagecodecs. Render them as two rows (one per op)
        # so the ratio column reads naturally per direction.
        if "speedup_vs_imagecodecs_encode" in r:
            ref = r.get("reference", {})
            ic_enc = ref.get("imagecodecs_encode", {})
            oc_enc = r["opencodecs"]
            oc_dec = ref.get("opencodecs_decode", {})
            ic_dec = ref.get("imagecodecs_decode", {})
            enc_ratio = r["speedup_vs_imagecodecs_encode"]
            dec_ratio = r["speedup_vs_imagecodecs_decode"]
            flag_e = "⚠️" if oc_enc.get("noisy") else ""
            flag_d = "⚠️" if oc_dec.get("noisy") else ""
            lines.append(
                f"| {name}/encode | {oc_enc['median_ms']:.2f} | "
                f"{oc_enc['min_ms']:.2f} | {oc_enc['max_ms']:.2f} | "
                f"{oc_enc['iqr_ms']:.2f} | {flag_e} | {enc_ratio:.2f}× vs ic |"
            )
            if oc_dec:
                lines.append(
                    f"| {name}/decode | {oc_dec['median_ms']:.2f} | "
                    f"{oc_dec['min_ms']:.2f} | {oc_dec['max_ms']:.2f} | "
                    f"{oc_dec['iqr_ms']:.2f} | {flag_d} | {dec_ratio:.2f}× vs ic |"
                )
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
