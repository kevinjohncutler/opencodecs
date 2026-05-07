#!/usr/bin/env python
"""Benchmark imagecodecs vs opencodecs JPEG XL on local disk and NAS.

Times encode + decode round-trips for synthetic-but-realistic images
(structured pattern + noise so JXL compresses non-trivially) at three
sizes, against two storage locations. Reports best-of-3 timings and
encoded size.

Cold-cache reads are forced via opencodecs.parallel.open_uncached
(F_NOCACHE on Darwin / POSIX_FADV_DONTNEED on Linux). Without this, the
kernel page cache makes a second read of the same file return in
microseconds even from a NAS, which would defeat the comparison.

Run: python bench/bench_jxl.py
"""

from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

import imagecodecs
import opencodecs.jxl as ocjxl
from opencodecs.parallel import read_bytes as _uncached_read_bytes


# ---------------------------------------------------------------------------
# Test data generation
# ---------------------------------------------------------------------------


def synth_rgb(h: int, w: int, seed: int = 0) -> np.ndarray:
    """A structured pattern + noise; compresses to a meaningful JXL size.

    Uses a sum of low-freq sinusoids per channel + per-pixel noise so the
    image has both smooth gradients (compression-friendly) and texture
    (non-trivial entropy). Close enough to natural-image statistics for
    JXL distance/effort to behave like they would on a photo.
    """
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    arr = np.stack(
        [
            127 + 100 * np.sin(yy / 80) * np.cos(xx / 90),
            127 + 100 * np.sin(yy / 50 + 1.0) * np.cos(xx / 70),
            127 + 100 * np.sin(yy / 30 + 2.0) * np.cos(xx / 50),
        ],
        axis=-1,
    )
    arr += rng.standard_normal(arr.shape).astype(np.float32) * 8
    return np.clip(arr, 0, 255).astype(np.uint8)


def synth_gray16(h: int, w: int, seed: int = 0) -> np.ndarray:
    """Microscopy-like uint16 grayscale: bright punctate features on a
    low-mean background. Compresses similarly to fluorescence images."""
    rng = np.random.default_rng(seed)
    base = (rng.poisson(lam=80, size=(h, w))).astype(np.float32)
    n_spots = (h * w) // 400
    sy = rng.integers(0, h, n_spots)
    sx = rng.integers(0, w, n_spots)
    intensity = rng.integers(2000, 60000, n_spots).astype(np.float32)
    base[sy, sx] = intensity
    base = np.clip(base, 0, 65535)
    return base.astype(np.uint16)


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_call(fn: Callable, repeats: int = 3) -> float:
    """Best-of-N elapsed seconds. Calls gc.collect between to reduce noise."""
    best = float("inf")
    for _ in range(repeats):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        elapsed = time.perf_counter() - t0
        if elapsed < best:
            best = elapsed
    return best


def _drop_cache(path: Path) -> None:
    """Force the next read to hit the underlying storage, not the page cache."""
    # F_NOCACHE / POSIX_FADV_DONTNEED on a temporary fd evicts that file
    # from the buffer cache for the next read (best-effort; on macOS this
    # works for the next read, on Linux it's a hint).
    try:
        _uncached_read_bytes(path, uncached=True)
    except FileNotFoundError:
        pass


# ---------------------------------------------------------------------------
# Bench cases
# ---------------------------------------------------------------------------


def _format_row(label, encode_ic, encode_oc, decode_ic, decode_oc, size_kb):
    """Format one comparison row."""
    return (
        f"{label:<32} | "
        f"{encode_ic*1000:>7.1f} | {encode_oc*1000:>7.1f} | "
        f"{encode_ic/encode_oc:>5.2f}x | "
        f"{decode_ic*1000:>7.1f} | {decode_oc*1000:>7.1f} | "
        f"{decode_ic/decode_oc:>5.2f}x | "
        f"{size_kb:>7.1f}"
    )


def _bench_one(name: str, arr: np.ndarray, dest_dir: Path, *, lossless: bool = True,
               threads: int = 8, effort: int = 1) -> dict:
    """Encode `arr` to dest_dir/<name>.jxl and decode it back, both with
    imagecodecs and opencodecs.

    Both codecs are pinned to `threads` worker threads and `effort=1`
    (the speed-over-size setting we actually use in production).
    Defaults differ between codecs (ic=1 thread, oc=all CPUs; both
    default-effort=5), so explicit pinning is essential for a fair compare.

    Decode uses cold-cache reads (F_NOCACHE / POSIX_FADV_DONTNEED) so we
    measure the actual storage path each time.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    p_ic = dest_dir / f"{name}.ic.jxl"
    p_oc = dest_dir / f"{name}.oc.jxl"

    # ---- encode ----
    def enc_ic():
        data = imagecodecs.jpegxl_encode(
            arr, lossless=lossless, effort=effort, numthreads=threads)
        p_ic.write_bytes(data)

    def enc_oc():
        ocjxl.write(p_oc, arr, lossless=lossless, effort=effort, numthreads=threads)

    t_enc_ic = _time_call(enc_ic)
    t_enc_oc = _time_call(enc_oc)

    sz_ic = p_ic.stat().st_size
    sz_oc = p_oc.stat().st_size
    assert sz_ic > 0 and sz_oc > 0

    # ---- decode (cold cache, equal threads) ----
    def dec_ic():
        data = _uncached_read_bytes(p_ic, uncached=True)
        out = imagecodecs.jpegxl_decode(data, numthreads=threads)
        assert out.shape == arr.shape

    def dec_oc():
        data = _uncached_read_bytes(p_oc, uncached=True)
        out = ocjxl.decode(data, numthreads=threads)
        assert out.shape == arr.shape

    t_dec_ic = _time_call(dec_ic)
    t_dec_oc = _time_call(dec_oc)

    p_ic.unlink(missing_ok=True)
    p_oc.unlink(missing_ok=True)

    return {
        "encode_ic": t_enc_ic,
        "encode_oc": t_enc_oc,
        "decode_ic": t_dec_ic,
        "decode_oc": t_dec_oc,
        "size_kb": sz_oc / 1024.0,
    }


def _bench_streaming_reduce(name: str, stack: np.ndarray, dest_dir: Path,
                            *, threads: int = 8, effort: int = 1) -> dict:
    """Decode-and-reduce comparison: max-projection over a multi-frame stack.

    imagecodecs path: jpegxl_decode -> full (T, Y, X[, C]) array -> reduce.
    opencodecs path: iter_frames + per-frame max-update (no full stack
    in memory).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    p = dest_dir / f"{name}.jxl"
    ocjxl.write(p, stack, lossless=True, animation=True, effort=effort, numthreads=threads)

    def reduce_ic():
        data = _uncached_read_bytes(p, uncached=True)
        full = imagecodecs.jpegxl_decode(data, numthreads=threads)
        return full.max(axis=0)

    def reduce_oc():
        data = _uncached_read_bytes(p, uncached=True)
        acc = None
        for frame in ocjxl.iter_frames(data, numthreads=threads):
            if acc is None:
                acc = frame.copy()
            else:
                np.maximum(acc, frame, out=acc)
        return acc

    t_ic = _time_call(reduce_ic)
    t_oc = _time_call(reduce_oc)
    size_kb = p.stat().st_size / 1024.0
    p.unlink(missing_ok=True)

    return {
        "reduce_ic": t_ic,
        "reduce_oc": t_oc,
        "size_kb": size_kb,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _bench_parallel_batch(n_files: int, h: int, w: int, dest_dir: Path,
                          *, threads: int = 8, effort: int = 1) -> dict:
    """Bench reading N JXL files cold-cache:
    - imagecodecs: serial loop, decode_threads=`threads` per file.
    - opencodecs:  parallel pread+decode via opencodecs.parallel.read_files,
                   N concurrent file workers, decode_threads=1 (oversubscribe-aware).

    This is the case where opencodecs's parallel I/O design pays off — many
    independent JXL files (e.g., a tile pyramid, a stack of frames stored
    one-per-file). With cold-cache reads from NAS, the I/O+decode pipeline
    overlaps across workers.
    """
    import opencodecs.parallel as ocp

    dest_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_files):
        p = dest_dir / f"batch_{i:03d}.jxl"
        arr = synth_rgb(h, w, seed=i)
        ocjxl.write(p, arr, lossless=True, effort=effort, numthreads=threads)
        paths.append(p)

    def read_serial_ic():
        out = []
        for p in paths:
            data = _uncached_read_bytes(p, uncached=True)
            out.append(imagecodecs.jpegxl_decode(data, numthreads=threads))
        return out

    def read_parallel_oc():
        return ocp.read_files(
            paths, n_workers=min(threads, n_files),
            uncached=True, decode_threads=1,
        )

    t_ic = _time_call(read_serial_ic)
    t_oc = _time_call(read_parallel_oc)
    total_kb = sum(p.stat().st_size for p in paths) / 1024.0
    for p in paths:
        p.unlink(missing_ok=True)

    return {
        "serial_ic": t_ic,
        "parallel_oc": t_oc,
        "total_kb": total_kb,
        "n": n_files,
    }


def _default_paths() -> tuple[Path, Path]:
    """Return (local_dir, nas_dir) appropriate for this OS."""
    home = Path.home()
    if sys.platform == "darwin":
        return Path("/tmp/opencodecs_bench"), Path("/Volumes/HiprDrive/imagecodecs/opencodecs/bench_data")
    if sys.platform.startswith("linux"):
        return Path("/tmp/opencodecs_bench"), home / "HiprDrive/imagecodecs/opencodecs/bench_data"
    if sys.platform == "win32":
        return home / "AppData/Local/Temp/opencodecs_bench", Path("Z:/imagecodecs/opencodecs/bench_data")
    return Path("/tmp/opencodecs_bench"), home / "bench_data"


def main():
    LOCAL, NAS = _default_paths()

    print(f"# imagecodecs vs opencodecs JPEG XL — {time.strftime('%Y-%m-%d %H:%M')}")
    print(f"# host: {os.uname().nodename}  python: {sys.version.split()[0]}")
    print(f"# imagecodecs: {imagecodecs.__version__}  "
          f"libjxl: {ocjxl.libjxl_version()}")
    print(f"# settings: lossless=True, effort=1 (speed > size), threads=8")
    print(f"# local dir: {LOCAL}")
    print(f"# nas dir:   {NAS}")
    print()

    # Build test images once
    tests = {
        "small_rgb_512":    synth_rgb(512, 512),
        "med_rgb_2048":     synth_rgb(2048, 2048),
        "large_rgb_4096":   synth_rgb(4096, 4096),
        "fluor_u16_1024":   synth_gray16(1024, 1024),
        "fluor_u16_2048":   synth_gray16(2048, 2048),
    }

    print("## Single-image round-trip (lossless), best-of-3, ms")
    print()
    header = (
        f"{'case':<32} | "
        f"{'enc-ic':>7} | {'enc-oc':>7} | {'spd':>5} | "
        f"{'dec-ic':>7} | {'dec-oc':>7} | {'spd':>5} | "
        f"{'KB':>7}"
    )
    sep = "-" * len(header)

    for label, dest_dir in [("LOCAL (APFS)", LOCAL), ("NAS (smbfs)", NAS)]:
        print(f"### {label}")
        print(header)
        print(sep)
        for name, arr in tests.items():
            try:
                r = _bench_one(name, arr, dest_dir, lossless=True)
                print(_format_row(
                    f"{name} {arr.shape}{arr.dtype}",
                    r["encode_ic"], r["encode_oc"],
                    r["decode_ic"], r["decode_oc"],
                    r["size_kb"],
                ))
            except Exception as e:
                print(f"{name}: SKIP ({type(e).__name__}: {e})")
        print()

    print("## Parallel batch decode: N independent .jxl files, cold cache")
    print("# imagecodecs: serial loop, decode_threads=8 each.")
    print("# opencodecs:  read_files() with 8 file-workers, 1 decode-thread")
    print("#              each (overlapped pread + decode).")
    print()
    bh = f"{'case':<32} | {'ic-serial':>9} | {'oc-parallel':>11} | {'spd':>5} | {'total KB':>9}"
    print(bh)
    print("-" * len(bh))
    for label, dest_dir in [("LOCAL", LOCAL), ("NAS", NAS)]:
        for n_files, dim in [(8, 512), (16, 1024)]:
            try:
                r = _bench_parallel_batch(n_files, dim, dim, dest_dir)
                print(f"{label} {n_files:>2}x({dim},{dim},3)u8        | "
                      f"{r['serial_ic']*1000:>9.1f} | "
                      f"{r['parallel_oc']*1000:>11.1f} | "
                      f"{r['serial_ic']/r['parallel_oc']:>5.2f}x | "
                      f"{r['total_kb']:>9.1f}")
            except Exception as e:
                print(f"{label} {n_files}x({dim},{dim},3)u8: SKIP ({type(e).__name__}: {e})")
    print()

    print("## Multi-frame stack: decode + max-projection")
    print("# Compares full-stack-decode-then-reduce (imagecodecs) vs")
    print("# streaming iter_frames + per-frame max-update (opencodecs).")
    print("# opencodecs avoids materializing the full (T, Y, X) stack.")
    print()
    sub_header = f"{'case':<32} | {'ic':>7} | {'oc':>7} | {'spd':>5} | {'KB':>7}"
    print(sub_header)
    print("-" * len(sub_header))
    for label, dest_dir in [("LOCAL", LOCAL), ("NAS", NAS)]:
        # 8-frame uint16 stack — typical scientific multi-channel acquisition
        stack = np.stack([synth_gray16(1024, 1024, seed=i) for i in range(8)], axis=0)
        try:
            r = _bench_streaming_reduce(f"{label}_stack8", stack, dest_dir)
            print(f"{label} 8x(1024,1024)u16            | "
                  f"{r['reduce_ic']*1000:>7.1f} | "
                  f"{r['reduce_oc']*1000:>7.1f} | "
                  f"{r['reduce_ic']/r['reduce_oc']:>5.2f}x | "
                  f"{r['size_kb']:>7.1f}")
        except Exception as e:
            print(f"{label} 8x(1024,1024)u16: SKIP ({type(e).__name__}: {e})")


if __name__ == "__main__":
    main()
