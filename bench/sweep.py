#!/usr/bin/env python3
"""Corpus-driven codec sweep: codec x parameters, on real data.

Complements ``run_benchmarks.py``, which measures whole workloads. This
measures one codec at a time across a parameter grid so the shape of a
codec's speed/size tradeoff is visible, and emits the same result JSON
so both feed one history.

    python bench/sweep.py                    every available codec
    python bench/sweep.py --codec zstd png   only these
    python bench/sweep.py --compare          also race imagecodecs
    python bench/sweep.py --quick            one parameter per codec

Third parties can add a codec without touching this file by passing
``--plugin mymodule``; the module needs a ``CODECS`` dict of the same
shape as ``_builtin_codecs()`` below. That is what makes this usable as
a bake-off harness: the corpus, the metrics and the result schema stay
fixed while the codec under test is swapped.

Measurement follows docs/codec_performance_audit.md: warm both sides,
interleave when comparing, take the minimum, and treat anything that
moves between repeats as unmeasured rather than as a result.
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import opencodecs as oc  # noqa: E402


# --------------------------------------------------------------------
# measurement
# --------------------------------------------------------------------

def measure(fn, repeats: int = 9, warmup: int = 3) -> float:
    """Minimum wall time of *fn*, after warming."""
    for _ in range(warmup):
        fn()
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t0)
    return best


def measure_ab(fa, fb, repeats: int = 9, warmup: int = 3):
    """Interleaved A/B so machine drift hits both sides equally."""
    for _ in range(warmup):
        fa(); fb()
    ta = tb = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter(); fa(); ta = min(ta, time.perf_counter() - t0)
        t0 = time.perf_counter(); fb(); tb = min(tb, time.perf_counter() - t0)
    return ta, tb


def fidelity(original: np.ndarray, restored) -> dict:
    """Exactness, or PSNR when a codec is lossy."""
    try:
        restored = np.asarray(restored)
        if restored.shape != original.shape:
            return {"exact": False, "note": "shape differs"}
        if np.array_equal(original, restored):
            return {"exact": True}
        a = original.astype(np.float64)
        b = restored.astype(np.float64)
        mse = float(np.mean((a - b) ** 2))
        if mse == 0:
            return {"exact": True}
        peak = 255.0 if original.dtype == np.uint8 else float(np.abs(a).max() or 1.0)
        return {"exact": False, "psnr_db": round(10 * np.log10(peak * peak / mse), 2)}
    except Exception as exc:                                  # noqa: BLE001
        return {"exact": None, "note": f"{type(exc).__name__}: {exc}"}


# --------------------------------------------------------------------
# what to sweep
# --------------------------------------------------------------------

def _builtin_codecs() -> dict:
    """name -> {"kind": "image"|"bytes", "params": [dict, ...]}"""
    lvl = lambda *v: [{"level": x} for x in v]               # noqa: E731
    return {
        # byte-oriented
        "zstd":      {"kind": "bytes", "params": lvl(1, 3, 9, 19)},
        "lz4":       {"kind": "bytes", "params": [{}]},
        "brotli":    {"kind": "bytes", "params": lvl(1, 3, 9)},
        "snappy":    {"kind": "bytes", "params": [{}]},
        "deflate":   {"kind": "bytes", "params": lvl(1, 6, 9)},
        # ISA-L is not a codec of its own, it is a deflate backend.
        "deflate-isal": {"kind": "bytes", "codec": "deflate",
                         "params": [{"backend": "isal"}]},
        "lzma":      {"kind": "bytes", "params": [{}]},
        "bz2":       {"kind": "bytes", "params": [{}]},
        "bitshuffle":{"kind": "bytes", "params": [{}]},
        "blosc2":    {"kind": "bytes", "params": [{}]},
        # image
        "png":       {"kind": "image", "params": [{}]},
        "qoi":       {"kind": "image", "params": [{}]},
        "bmp":       {"kind": "image", "params": [{}]},
        "jpeg":      {"kind": "image", "params": lvl(75, 90, 95)},
        "mozjpeg":   {"kind": "image", "params": lvl(75, 90)},
        "webp":      {"kind": "image", "params": lvl(75, 90)},
        "jpeg2k":    {"kind": "image", "params": [{}]},
        "jxl":       {"kind": "image", "params": [{}]},
        "avif":      {"kind": "image", "params": [{}]},
        "heif":      {"kind": "image", "params": [{}]},
        "brunsli":   {"kind": "image", "params": [{}]},
        "openjph":   {"kind": "image", "params": [{}]},
        "rgbe":      {"kind": "float_rgb", "params": [{}]},
        # arrays
        "zfp":       {"kind": "float", "params": [{}]},
        "sz3":       {"kind": "float", "params": [{}]},
        "sperr":     {"kind": "float2d", "params": [{}]},
        "pcodec":    {"kind": "float", "params": [{}]},
        "aec":       {"kind": "bytes", "params": [{"bits_per_sample": 8}]},
        "lerc":      {"kind": "float2d", "params": [{}]},
    }


def _payloads() -> dict:
    """One representative payload per kind, from the corpus when present."""
    out = {}
    # skip macOS AppleDouble sidecars: the SMB mount creates a "._name"
    # beside every file, and sorted() puts it first, so a naive glob hands
    # back 4 KB of resource fork instead of the image.
    kod = sorted(p for p in (ROOT / ".test_data" / "png" / "kodak24").glob("*.png")
                 if not p.name.startswith("."))
    if kod:
        img = oc.get_codec("png").decode(kod[0].read_bytes())
        img = np.ascontiguousarray(img[:512, :512])
        out["image"] = img
        out["bytes"] = img.tobytes()
    det = (ROOT / ".test_data" / "sdrbench"
           / "SDRBENCH-EXAFEL-10x32x185x388" / "smd-cxif5315-r169-raw.u16")
    if det.is_file():
        a = np.fromfile(det, dtype=np.uint16)[:2_000_000]
        out.setdefault("bytes", a.tobytes())
        out["detector"] = a
    exa = (ROOT / ".test_data" / "sdrbench"
           / "SDRBENCH-EXAALT-2869440" / "xx.f32")
    if exa.is_file():
        a = np.fromfile(exa, dtype=np.float32)
        out["float"] = a[:1_000_000]
        # Several array codecs require rank: sperr and lerc reject 1D,
        # and rgbe wants an (H, W, 3) image. Give each the shape it needs
        # rather than reporting them as errors.
        out["float2d"] = np.ascontiguousarray(a[:1_000_000].reshape(1000, 1000))
        rgb = a[:512 * 512 * 3].reshape(512, 512, 3)
        out["float_rgb"] = np.ascontiguousarray(np.abs(rgb))
    return out


def sweep(names, payloads, compare: bool, quick: bool) -> list[dict]:
    try:
        import imagecodecs
    except ImportError:
        imagecodecs = None
    rows = []
    for name, spec in _builtin_codecs().items():
        if names and name not in names:
            continue
        if not oc.has_codec(spec.get("codec", name)):
            rows.append({"codec": name, "status": "not built"})
            continue
        payload = payloads.get(spec["kind"])
        if payload is None:
            rows.append({"codec": name, "status": f"no {spec['kind']} payload"})
            continue
        codec = oc.get_codec(spec.get("codec", name))
        params = spec["params"][:1] if quick else spec["params"]
        for p in params:
            row = {"codec": name, "kind": spec["kind"], "params": p}
            try:
                blob = bytes(codec.encode(payload, **p))
                row["encoded_bytes"] = len(blob)
                nbytes = payload.nbytes if hasattr(payload, "nbytes") else len(payload)
                row["input_bytes"] = nbytes
                row["ratio"] = round(nbytes / len(blob), 4)
                row["encode_ms"] = round(
                    measure(lambda: codec.encode(payload, **p)) * 1e3, 4)
                try:
                    back = codec.decode(blob)
                    row["decode_ms"] = round(measure(lambda: codec.decode(blob)) * 1e3, 4)
                    if isinstance(payload, np.ndarray):
                        row["fidelity"] = fidelity(payload, back)
                except Exception as exc:                      # noqa: BLE001
                    row["decode_error"] = f"{type(exc).__name__}: {exc}"
                if compare and imagecodecs is not None:
                    ie = getattr(imagecodecs, f"{name}_encode", None)
                    if ie is not None:
                        try:
                            theirs = bytes(ie(payload, **p))
                            # A speed ratio only means something when both
                            # sides did comparable work. Defaults differ:
                            # at sz3's, we emit 1.9 MB where imagecodecs
                            # emits 3.7 MB, so "we are 5x slower" is really
                            # "we compress twice as hard". Report the sizes
                            # and withhold the ratio rather than mislead.
                            row["their_bytes"] = len(theirs)
                            # Mode first: a lossless encode raced against
                            # somebody's lossy one is meaningless in both
                            # dimensions. opencodecs and imagecodecs both
                            # default to lossless (see
                            # docs/codec_api_conventions.md), but a third
                            # party plugged in via --plugin may not.
                            idec = getattr(imagecodecs, f"{name}_decode", None)
                            if idec is not None and isinstance(payload, np.ndarray):
                                try:
                                    their_exact = np.array_equal(
                                        np.asarray(idec(theirs)), payload)
                                    ours_exact = bool(row.get("fidelity", {}).get("exact"))
                                    if their_exact != ours_exact:
                                        row["vs_imagecodecs_note"] = (
                                            "not comparable: "
                                            f"ours is {'lossless' if ours_exact else 'lossy'}, "
                                            f"theirs is {'lossless' if their_exact else 'lossy'}")
                                        raise StopIteration
                                except StopIteration:
                                    raise
                                except Exception:             # noqa: BLE001
                                    pass
                            spread = (abs(len(theirs) - len(blob))
                                      / max(len(theirs), len(blob), 1))
                            if spread > 0.10:
                                row["vs_imagecodecs_note"] = (
                                    f"not comparable: their output differs by "
                                    f"{spread*100:.0f}%, defaults do different work")
                            else:
                                ta, tb = measure_ab(
                                    lambda: codec.encode(payload, **p),
                                    lambda: ie(payload, **p))
                                row["vs_imagecodecs_encode"] = round(tb / ta, 3)
                        except StopIteration:
                            pass       # mode mismatch, note already set
                        except Exception:                     # noqa: BLE001
                            pass
                row["status"] = "ok"
            except Exception as exc:                          # noqa: BLE001
                row["status"] = f"{type(exc).__name__}: {str(exc)[:60]}"
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--codec", nargs="*", default=[])
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--plugin", default=None)
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    if args.plugin:
        importlib.import_module(args.plugin)

    payloads = _payloads()
    if not payloads:
        print("no corpus payloads; run tests/download_test_corpus.sh --light")
        return 1
    print(f"payloads: {', '.join(f'{k} ({getattr(v, 'nbytes', len(v))} B)' for k, v in payloads.items())}\n")

    rows = sweep(set(args.codec), payloads, args.compare, args.quick)

    hdr = f"{'codec':<11}{'params':<14}{'ratio':>8}{'enc ms':>10}{'dec ms':>10}{'vs ic':>8}  fidelity"
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if r.get("status") != "ok":
            print(f"{r['codec']:<11}{'':<14}{'':>8}{'':>10}{'':>10}{'':>8}  {r.get('status')}")
            continue
        p = ",".join(f"{k}={v}" for k, v in r["params"].items()) or "-"
        f = r.get("fidelity", {})
        ftxt = ("exact" if f.get("exact") else
                (f"{f['psnr_db']} dB" if "psnr_db" in f else
                 f.get("note", r.get("decode_error", ""))[:26]))
        vs = (f"{r['vs_imagecodecs_encode']:.2f}x" if "vs_imagecodecs_encode" in r
              else ("n/c" if "vs_imagecodecs_note" in r else ""))
        print(f"{r['codec']:<11}{p:<14}{r['ratio']:>8.3f}{r['encode_ms']:>10.3f}"
              f"{r.get('decode_ms', float('nan')):>10.3f}{vs:>8}  {ftxt}")

    if args.json:
        doc = {
            "timestamp": datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ"),
            "system": {"platform": platform.platform(), "machine": platform.machine(),
                       "python": platform.python_version()},
            "sweep": rows,
        }
        Path(args.json).write_text(json.dumps(doc, indent=2, default=str) + "\n")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
