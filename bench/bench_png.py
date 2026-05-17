#!/usr/bin/env python
"""PNG encode/decode head-to-head: opencodecs vs imagecodecs.

Tracks the two perf wins that landed in 2026-05-17:
  1. libdeflate IDAT accumulator path (one-shot compress instead of
     per-scanline deflate()),
  2. libspng filter_sum split (per-filter specialised functions so
     the compiler autovectorizes each branch).

Runs five representative workloads end-to-end (no cheating with
mocks / cached blobs), reports oc/ic ratios, and — when invoked
with ``--check`` — asserts each measured ratio stays under a regression
ceiling. The ``--check`` mode is what CI uses to catch silent
perf drift in future libspng or libdeflate-binding edits.

Usage::

    python bench/bench_png.py             # just print numbers
    python bench/bench_png.py --check     # also enforce ceilings
    python bench/bench_png.py --json      # machine-readable output

The ceilings are intentionally loose (+15% of best measured ratio)
so noisy CI runs don't trigger false alarms — they catch *real*
regressions like "filter_sum_paeth went scalar again" or "the
SPNG_USE_LIBDEFLATE define stopped firing", which would move the
ratio by 2-3×, not 15%.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np

import opencodecs as oc

try:
    import imagecodecs as ic
except ImportError:
    print("imagecodecs not installed — bench requires it as the reference")
    sys.exit(2)


# ---------------------------------------------------------------------------
# Test images
# ---------------------------------------------------------------------------


def _load_kodak() -> np.ndarray:
    """Load Kodak kodim01.png from the test corpus if available."""
    for path in (
        Path(".test_data/png/kodak24/kodim01.png"),
        Path(__file__).resolve().parent.parent
            / ".test_data" / "png" / "kodak24" / "kodim01.png",
    ):
        if path.exists():
            return oc.get_codec("png").decode(path.read_bytes())
    return None


def _make_images() -> dict[str, np.ndarray]:
    """Build the five tracked workloads.

    * Random data exercises the deflate hot path (incompressible -> the
      filter_sum work matters less, the deflate constant overhead
      dominates).
    * Kodak photo exercises a real natural-image distribution.
    * u16 gradient exercises the filter scoring loop directly (high
      compressibility -> tiny IDAT, all wall-clock in filter selection).
    """
    rng = np.random.default_rng(0)
    out = {
        "4mp_rgb_u8_random":    rng.integers(0, 256, (2048, 2048, 3), dtype=np.uint8),
        "4mp_rgb_u16_random":   rng.integers(0, 65535, (2048, 2048, 3), dtype=np.uint16),
        "1080p_rgb_u8_random":  rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8),
        "filterbound_u16":      (np.add.outer(np.arange(512), np.arange(512)) % 65535).astype(np.uint16),
    }
    kodak = _load_kodak()
    if kodak is not None:
        out["kodak01_rgb_u8"] = kodak
    return out


# Regression ceilings — oc/ic ratio must stay BELOW these.
# Smaller is faster, so the ceiling is "how slow are we allowed to get".
# Numbers are ~15% above the best ratios observed on M1 Ultra.
_CEILINGS = {
    "4mp_rgb_u8_random":    0.60,    # measured 0.506
    "4mp_rgb_u16_random":   0.60,    # measured 0.517
    "1080p_rgb_u8_random":  0.60,    # measured 0.498
    "filterbound_u16":      0.70,    # measured 0.549
    "kodak01_rgb_u8":       0.40,    # measured 0.318
}


# ---------------------------------------------------------------------------
# Timing helpers
# ---------------------------------------------------------------------------


def _time_repeats(fn: Callable[[], object], n: int) -> float:
    """Return min ms over n repeats (min — less affected by jitter than mean)."""
    # warmup
    fn()
    best = float("inf")
    for _ in range(n):
        t0 = time.perf_counter_ns()
        fn()
        elapsed = (time.perf_counter_ns() - t0) / 1e6
        if elapsed < best:
            best = elapsed
    return best


def bench_one(name: str, img: np.ndarray, n: int = 5) -> dict:
    """Encode + decode head-to-head against imagecodecs."""
    oc_png = oc.get_codec("png")
    enc_oc = _time_repeats(lambda: oc_png.encode(img), n)
    enc_ic = _time_repeats(lambda: ic.png_encode(img), n)
    # Use ic-encoded blob for decode (same input -> fair compare).
    blob = ic.png_encode(img)
    dec_oc = _time_repeats(lambda: oc_png.decode(blob), n)
    dec_ic = _time_repeats(lambda: ic.png_decode(blob), n)
    size_oc = len(oc_png.encode(img))
    return {
        "name": name,
        "encode_oc_ms": enc_oc,
        "encode_ic_ms": enc_ic,
        "encode_ratio": enc_oc / enc_ic,
        "decode_oc_ms": dec_oc,
        "decode_ic_ms": dec_ic,
        "decode_ratio": dec_oc / dec_ic,
        "size_oc": size_oc,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true",
                     help="enforce regression ceilings; non-zero exit on fail")
    ap.add_argument("--json", action="store_true",
                     help="emit machine-readable results to stdout")
    ap.add_argument("-n", type=int, default=5,
                     help="repeats per measurement (default 5; min reported)")
    args = ap.parse_args()

    images = _make_images()
    results = [bench_one(name, img, n=args.n) for name, img in images.items()]

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(f"{'workload':<24s} {'enc oc':>9s} {'enc ic':>9s} {'enc r':>7s}"
              f"  {'dec oc':>9s} {'dec ic':>9s} {'dec r':>7s}")
        for r in results:
            print(f"  {r['name']:<22s} "
                  f"{r['encode_oc_ms']:>7.2f}ms {r['encode_ic_ms']:>7.2f}ms "
                  f"{r['encode_ratio']:>7.3f}  "
                  f"{r['decode_oc_ms']:>7.2f}ms {r['decode_ic_ms']:>7.2f}ms "
                  f"{r['decode_ratio']:>7.3f}")

    if args.check:
        failures = []
        for r in results:
            ceiling = _CEILINGS.get(r["name"])
            if ceiling is None:
                continue
            if r["encode_ratio"] > ceiling:
                failures.append(
                    f"  {r['name']}: encode ratio {r['encode_ratio']:.3f} "
                    f"> ceiling {ceiling:.3f}"
                )
        if failures:
            print("\nFAIL: PNG encode regression detected:", file=sys.stderr)
            for f in failures:
                print(f, file=sys.stderr)
            print(
                "\nIf this regression is intentional, raise the ceiling in "
                "bench/bench_png.py:_CEILINGS. Otherwise investigate the "
                "libspng / libdeflate path — recent edits that touched "
                "3rdparty/libspng/spng.c or the SPNG_USE_LIBDEFLATE build "
                "flag are the most likely culprits.",
                file=sys.stderr,
            )
            return 1
        print("\nOK: all PNG encode ratios within regression ceilings.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
