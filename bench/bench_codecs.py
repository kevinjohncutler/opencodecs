#!/usr/bin/env python
"""Unified head-to-head perf bench across every codec we ship.

Captures encode + decode wall-clock for ``opencodecs`` against
``imagecodecs`` on a workload that's *meaningful for that codec*,
not synthetic noise — random bytes are the worst case for any
compressor, so we use natural-image data for image codecs and
smooth float fields for scientific compressors.

Two modes:

* **default** — runs every measurement, prints a table with
  oc/ic ratios, and optionally emits JSON.
* **--check** — additionally enforces that each ratio stays
  within +30% of the value recorded in ``bench/perf_baseline.json``.
  Designed for catching regressions like "filter_sum split got
  reverted" or "libdeflate stopped linking" (both would move
  ratios by 2-3×, not 30%) without false-alarming on normal
  build-to-build jitter.

The baseline file is a captured *setpoint*, not a contract.
When a real perf change lands intentionally, re-record it with
``--update-baseline``.

Per-architecture baselines live in
``bench/perf_baseline.<machine>.json``. ``arm64`` / ``x86_64`` are
captured separately because the compiler autovectorises to
different widths (NEON 128b / SSE2 128b / AVX2 256b).

Usage::

    python bench/bench_codecs.py                  # measure + print
    python bench/bench_codecs.py --check          # enforce setpoints
    python bench/bench_codecs.py --json           # machine-readable
    python bench/bench_codecs.py --update-baseline
    python bench/bench_codecs.py --only png,deflate
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

try:
    import imagecodecs as ic
except ImportError:
    print("imagecodecs not installed — bench requires it as the reference")
    sys.exit(2)

import opencodecs as oc


# ---------------------------------------------------------------------------
# Workloads
# ---------------------------------------------------------------------------


def _load_kodak() -> np.ndarray | None:
    for p in (Path(".test_data/png/kodak24/kodim01.png"),
              Path(__file__).resolve().parent.parent
                  / ".test_data" / "png" / "kodak24" / "kodim01.png"):
        if p.exists():
            return oc.get_codec("png").decode(p.read_bytes())
    return None


def _smooth_field_3d(shape=(32, 32, 32)) -> np.ndarray:
    z, y, x = np.mgrid[0:shape[0], 0:shape[1], 0:shape[2]].astype(np.float32)
    z /= shape[0]; y /= shape[1]; x /= shape[2]
    base = (np.sin(4 * np.pi * x)
            + np.cos(3 * np.pi * y)
            + np.sin(2 * np.pi * z) * 0.5).astype(np.float32)
    noise = np.random.default_rng(0).standard_normal(base.shape).astype(np.float32) * 0.05
    return base + noise


def _rand_bytes(n: int) -> bytes:
    return np.random.default_rng(0).integers(0, 256, size=n, dtype=np.uint8).tobytes()


def _kodak_bytes() -> bytes:
    """Flattened Kodak photo as bytes — natural-image byte distribution,
    not random noise, so the byte compressors get a fair shake."""
    img = _load_kodak()
    return img.tobytes() if img is not None else _rand_bytes(2_000_000)


# A "Workload" is everything needed to time encode + decode on one
# codec: the prep step (build input + reference encoded blob), the
# encode call, and the decode call. Returning closures keeps each
# measurement self-contained and reusable.
@dataclass
class Workload:
    key: str                                 # "png/kodak_photo"
    encode_oc: Callable[[], object]
    encode_ic: Callable[[], object]
    decode_oc: Callable[[], object]
    decode_ic: Callable[[], object]


def _try_prep(label: str, fn: Callable[[], Workload | None],
              skipped: list[str]) -> Workload | None:
    """Run a prep closure that returns a Workload. If anything goes
    wrong (codec backend missing, encoder not built, upstream library
    SIGABRT we caught via subprocess wrapping, ...), record the skip
    and return None instead of aborting the whole bench."""
    try:
        return fn()
    except Exception as e:
        skipped.append(f"{label}: {type(e).__name__}: {e}")
        return None


def _build_workloads(rng: np.random.Generator,
                      skipped: list[str] | None = None) -> list[Workload]:
    if skipped is None:
        skipped = []
    workloads: list[Workload] = []
    kodak = _load_kodak()
    field3d = _smooth_field_3d()

    # ---- Image codecs on a Kodak photo (when available) -------------
    if kodak is not None:
        # Lossless image codecs — ic encodes the reference blob.
        for name in ("png", "qoi", "jpegls", "jpeg2k"):
            def prep(name=name):
                if not oc.has_codec(name):
                    raise RuntimeError(f"opencodecs lacks codec {name}")
                ic_enc = getattr(ic, f"{name}_encode", None)
                ic_dec = getattr(ic, f"{name}_decode", None)
                if not (ic_enc and ic_dec):
                    raise RuntimeError(f"imagecodecs lacks {name}_encode/decode")
                blob_ic = ic_enc(kodak)
                return Workload(
                    key=f"{name}/kodak_photo",
                    encode_oc=lambda c=name, im=kodak: oc.get_codec(c).encode(im),
                    encode_ic=lambda f=ic_enc, im=kodak: f(im),
                    decode_oc=lambda c=name, b=blob_ic: oc.get_codec(c).decode(b),
                    decode_ic=lambda f=ic_dec, b=blob_ic: f(b),
                )
            w = _try_prep(f"{name}/kodak_photo", prep, skipped)
            if w is not None: workloads.append(w)

        # Lossy image codecs — oc encodes the reference blob so we
        # measure ic-side decode of our wire format.
        for name, ic_dec_name in [
            ("jpeg", "jpeg_decode"),
            ("webp", "webp_decode"),
            ("avif", "avif_decode"),
        ]:
            def prep(name=name, ic_dec_name=ic_dec_name):
                if not oc.has_codec(name):
                    raise RuntimeError(f"opencodecs lacks codec {name}")
                ic_enc = getattr(ic, f"{name}_encode", None)
                ic_dec = getattr(ic, ic_dec_name, None)
                if not (ic_enc and ic_dec):
                    raise RuntimeError(f"imagecodecs lacks {name}")
                blob_oc = oc.get_codec(name).encode(kodak)
                return Workload(
                    key=f"{name}/kodak_photo",
                    encode_oc=lambda c=name, im=kodak: oc.get_codec(c).encode(im),
                    encode_ic=lambda f=ic_enc, im=kodak: f(im),
                    decode_oc=lambda c=name, b=blob_oc: oc.get_codec(c).decode(b),
                    decode_ic=lambda f=ic_dec, b=blob_oc: f(b),
                )
            w = _try_prep(f"{name}/kodak_photo", prep, skipped)
            if w is not None: workloads.append(w)

        # JPEG XL — ic exposes as jpegxl_*, oc as jxl.
        def prep_jxl():
            if not oc.has_codec("jxl") or not hasattr(ic, "jpegxl_encode"):
                raise RuntimeError("jxl not available on both sides")
            blob_oc = oc.get_codec("jxl").encode(kodak)
            return Workload(
                key="jxl/kodak_photo",
                encode_oc=lambda im=kodak: oc.get_codec("jxl").encode(im),
                encode_ic=lambda im=kodak: ic.jpegxl_encode(im),
                decode_oc=lambda b=blob_oc: oc.get_codec("jxl").decode(b),
                decode_ic=lambda b=blob_oc: ic.jpegxl_decode(b),
            )
        w = _try_prep("jxl/kodak_photo", prep_jxl, skipped)
        if w is not None: workloads.append(w)

    # ---- General-purpose byte compressors on natural-image bytes ----
    nat = _kodak_bytes()
    for name in ("deflate", "zstd", "brotli", "lzma", "bz2", "snappy"):
        def prep(name=name):
            if not oc.has_codec(name):
                raise RuntimeError(f"opencodecs lacks codec {name}")
            ic_enc = getattr(ic, f"{name}_encode", None)
            ic_dec = getattr(ic, f"{name}_decode", None)
            if not (ic_enc and ic_dec):
                raise RuntimeError(f"imagecodecs lacks {name}")
            blob_ic = ic_enc(nat)
            return Workload(
                key=f"{name}/kodak_bytes",
                encode_oc=lambda c=name, b=nat: oc.get_codec(c).encode(b),
                encode_ic=lambda f=ic_enc, b=nat: f(b),
                decode_oc=lambda c=name, b=blob_ic: oc.get_codec(c).decode(b),
                decode_ic=lambda f=ic_dec, b=blob_ic: f(b),
            )
        w = _try_prep(f"{name}/kodak_bytes", prep, skipped)
        if w is not None: workloads.append(w)

    # ---- LZ4: oc uses frame format -> ic.lz4f_*
    def prep_lz4():
        if not oc.has_codec("lz4") or not hasattr(ic, "lz4f_encode"):
            raise RuntimeError("lz4f not available on both sides")
        blob_oc = oc.get_codec("lz4").encode(nat)
        return Workload(
            key="lz4/kodak_bytes",
            encode_oc=lambda b=nat: oc.get_codec("lz4").encode(b),
            encode_ic=lambda b=nat: ic.lz4f_encode(b),
            decode_oc=lambda b=blob_oc: oc.get_codec("lz4").decode(b),
            decode_ic=lambda b=blob_oc: ic.lz4f_decode(b),
        )
    w = _try_prep("lz4/kodak_bytes", prep_lz4, skipped)
    if w is not None: workloads.append(w)

    # ---- Scientific float compressors on a smooth field -------------
    def prep_zfp():
        if not oc.has_codec("zfp") or not hasattr(ic, "zfp_encode"):
            raise RuntimeError("zfp not available on both sides")
        blob_ic = ic.zfp_encode(field3d)
        return Workload(
            key="zfp/smooth_f32_3d",
            encode_oc=lambda x=field3d: oc.get_codec("zfp").encode(x),
            encode_ic=lambda x=field3d: ic.zfp_encode(x),
            decode_oc=lambda b=blob_ic: oc.get_codec("zfp").decode(b),
            decode_ic=lambda b=blob_ic: ic.zfp_decode(b),
        )
    w = _try_prep("zfp/smooth_f32_3d", prep_zfp, skipped)
    if w is not None: workloads.append(w)

    # LERC is intentionally skipped from this head-to-head: both
    # opencodecs and imagecodecs vendor their own LERC build, and
    # whichever resolves dylib symbols second ends up calling into
    # the first's allocator. ic.lerc_encode then SIGABRTs on certain
    # inputs in a way that's not the bench's bug to chase. Run LERC
    # benches via a separate process if needed.

    def prep_blosc2():
        if not oc.has_codec("blosc2") or not hasattr(ic, "blosc2_encode"):
            raise RuntimeError("blosc2 not available on both sides")
        blob_ic = ic.blosc2_encode(nat)
        return Workload(
            key="blosc2/kodak_bytes",
            encode_oc=lambda b=nat: oc.get_codec("blosc2").encode(b),
            encode_ic=lambda b=nat: ic.blosc2_encode(b),
            decode_oc=lambda b=blob_ic: oc.get_codec("blosc2").decode(b),
            decode_ic=lambda b=blob_ic: ic.blosc2_decode(b),
        )
    w = _try_prep("blosc2/kodak_bytes", prep_blosc2, skipped)
    if w is not None: workloads.append(w)

    return workloads


# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------


def _time_min(fn: Callable[[], object], n: int) -> float:
    fn()                                  # warmup, drops first-call jitter
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        elapsed = (time.perf_counter_ns() - t0) / 1e6
        if elapsed < best:
            best = elapsed
    return best


def _bench_one(w: Workload, n: int) -> dict:
    enc_oc = _time_min(w.encode_oc, n)
    enc_ic = _time_min(w.encode_ic, n)
    dec_oc = _time_min(w.decode_oc, n)
    dec_ic = _time_min(w.decode_ic, n)
    return {
        "key": w.key,
        "encode_oc_ms": enc_oc,
        "encode_ic_ms": enc_ic,
        "encode_ratio": enc_oc / enc_ic,
        "decode_oc_ms": dec_oc,
        "decode_ic_ms": dec_ic,
        "decode_ratio": dec_oc / dec_ic,
    }


# ---------------------------------------------------------------------------
# Baseline I/O
# ---------------------------------------------------------------------------


def _machine_tag() -> str:
    """A short tag used to namespace the baseline file. Distinguishes
    arm64 from x86_64 so the per-architecture autovectorisation
    differences don't trip --check."""
    m = platform.machine().lower()
    if m in ("arm64", "aarch64"):
        return "arm64"
    if m in ("amd64", "x86_64"):
        return "x86_64"
    return m


def _baseline_path() -> Path:
    return Path(__file__).resolve().parent / f"perf_baseline.{_machine_tag()}.json"


# How much regression we tolerate before failing --check. Loose enough
# that compiler-version drift and CI-runner noise don't cause flakes,
# tight enough that the kinds of regressions we care about (whole
# fast-paths going dark, wrong library getting linked) trip the check
# every time. Measured baseline ratios usually drift by 2-5% across
# clean rebuilds; we allow 30% for safety.
DEFAULT_CEILING_DRIFT = 0.30


def _check_against_baseline(results: list[dict], drift: float) -> list[str]:
    path = _baseline_path()
    if not path.exists():
        return [f"no baseline found at {path} — record one with --update-baseline"]
    baseline = json.loads(path.read_text())
    by_key = {r["key"]: r for r in results}
    failures: list[str] = []
    for entry in baseline["ratios"]:
        key = entry["key"]
        current = by_key.get(key)
        if current is None:
            failures.append(f"  {key}: missing from current run")
            continue
        for side in ("encode_ratio", "decode_ratio"):
            base = entry.get(side)
            cur = current[side]
            if base is None:
                continue
            ceiling = base * (1.0 + drift)
            if cur > ceiling:
                failures.append(
                    f"  {key} {side[:6]}: {cur:.3f} > "
                    f"baseline {base:.3f} * (1+{drift:.0%}) = {ceiling:.3f}"
                )
    return failures


def _write_baseline(results: list[dict]) -> Path:
    path = _baseline_path()
    payload = {
        "machine": _machine_tag(),
        "machine_full": platform.platform(),
        "python": sys.version.split()[0],
        "ratios": [
            {
                "key": r["key"],
                "encode_ratio": r["encode_ratio"],
                "decode_ratio": r["decode_ratio"],
            }
            for r in results
        ],
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")
    return path


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                     help="enforce regression ceilings against baseline")
    ap.add_argument("--update-baseline", action="store_true",
                     help="overwrite the baseline file with the current run")
    ap.add_argument("--json", action="store_true",
                     help="emit machine-readable results")
    ap.add_argument("--only", default="",
                     help="comma-separated codec name filter (e.g. 'png,deflate')")
    ap.add_argument("--drift", type=float, default=DEFAULT_CEILING_DRIFT,
                     help=f"per-ratio drift allowance (default {DEFAULT_CEILING_DRIFT})")
    ap.add_argument("-n", type=int, default=5,
                     help="repeats per measurement (min reported)")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    skipped: list[str] = []
    workloads = _build_workloads(rng, skipped)
    if args.only:
        allowed = {s.strip() for s in args.only.split(",") if s.strip()}
        workloads = [w for w in workloads if w.key.split("/")[0] in allowed]

    results = [_bench_one(w, args.n) for w in workloads]

    if skipped and not args.json:
        print(f"# skipped {len(skipped)} workload(s):")
        for s in skipped:
            print(f"#   {s}")

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"# machine={_machine_tag()}")
        print(f"{'workload':<28s} {'enc r':>7s}  {'dec r':>7s}"
              f"  {'enc oc':>9s} {'enc ic':>9s}  {'dec oc':>9s} {'dec ic':>9s}")
        for r in results:
            print(f"  {r['key']:<26s} "
                  f"{r['encode_ratio']:>7.3f}  {r['decode_ratio']:>7.3f}  "
                  f"{r['encode_oc_ms']:>7.2f}ms {r['encode_ic_ms']:>7.2f}ms  "
                  f"{r['decode_oc_ms']:>7.2f}ms {r['decode_ic_ms']:>7.2f}ms")

    if args.update_baseline:
        path = _write_baseline(results)
        print(f"\nbaseline written to {path}")
        return 0

    if args.check:
        failures = _check_against_baseline(results, args.drift)
        if failures:
            print("\nFAIL: perf regressions detected:", file=sys.stderr)
            for f in failures:
                print(f, file=sys.stderr)
            print(
                "\nIf intentional, re-record with: "
                "`python bench/bench_codecs.py --update-baseline`",
                file=sys.stderr,
            )
            return 1
        print(f"\nOK: all ratios within +{int(args.drift * 100)}% of baseline.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
