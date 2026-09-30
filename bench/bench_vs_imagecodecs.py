"""opencodecs against imagecodecs, codec by codec: the README table.

Every row is one codec operation with the same settings on both sides.
Each sample is a fresh process that imports one library and times one
case, and the driver alternates the two libraries in a shuffled order
every round, so neither gets a warm cache or a quiet moment the other
did not. (Loading both in one process is also unsafe: each statically
links its own libLerc, and the second copy aborts.)

Decode rows decode one bitstream, written once by imagecodecs, and both
libraries must return identical arrays or bytes. Encode rows time each
library's own encoder at matching settings; each result is decoded
again by the same library and, where the setting is lossless, must equal
the input. Output sizes are recorded, so a faster encoder that writes a
much larger file shows up in the results.

Usage (the package under test comes from the environment or PYTHONPATH):

  python bench/bench_vs_imagecodecs.py run --rounds 5 --data /tmp/ocic --out mac.json
  python bench/bench_vs_imagecodecs.py table mac=mac.json linux=linux.json windows=win.json
  python bench/bench_vs_imagecodecs.py once CASE LIB DATADIR   (internal)
"""
from __future__ import annotations

import hashlib
import json
import os
import pathlib
import platform
import random
import statistics
import subprocess
import sys
import time

import numpy as np

HERE = pathlib.Path(__file__).resolve()


# ---------------------------------------------------------------------------
# Inputs: one natural-looking image, and views of it, so compressors see
# realistic redundancy rather than random bytes.
# ---------------------------------------------------------------------------

def natural(h: int, w: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = 20000 + 8000 * np.sin(x / 97.0) * np.cos(y / 131.0)
    return (base + rng.normal(0, 300, (h, w))).clip(0, 65535).astype(np.uint16)


def inputs() -> dict:
    u16 = natural(4096, 4096)                                   # 32 MB
    g = u16[:2048, :2048]
    rgb8 = np.stack([g >> 8, (g >> 4) & 255, g & 255], -1).astype(np.uint8)
    rgb8[..., 2] = (rgb8[..., 0].astype(np.int32) * 3 // 4 + 20).astype(np.uint8)
    rng = np.random.default_rng(9)
    return {
        "u16": u16,
        "u16_small": u16[:1024].copy(),                          # 8 MB, slow codecs
        "u8": (u16 >> 8).astype(np.uint8),                       # 16 MB
        "g16": g.copy(),                                         # 2048 x 2048 uint16
        "rgb8": rgb8,                                            # 2048 x 2048 x 3
        "f32": (g.astype(np.float32) / 65535.0) * 3.0 - 1.0,
        "bc1": rng.integers(0, 256, 512 * 512 * 8, dtype=np.uint8).tobytes(),
        "bc7": rng.integers(0, 256, 512 * 512 * 16, dtype=np.uint8).tobytes(),
    }


def digest(obj) -> str:
    if isinstance(obj, np.ndarray):
        obj = np.ascontiguousarray(obj)
        return hashlib.sha1(obj.tobytes()).hexdigest()[:16] + str(obj.dtype) + str(obj.shape)
    return hashlib.sha1(bytes(obj)).hexdigest()[:16]


# ---------------------------------------------------------------------------
# Cases. Each maps a library to (encode, decode) callables at matching
# settings; "data" names the input. Byte-stream codecs take the input's
# bytes. A decode row decodes imagecodecs' encoding of the input.
# ---------------------------------------------------------------------------

def _oc(name):
    import opencodecs
    return opencodecs.get_codec(name)


def _ic():
    import imagecodecs
    return imagecodecs


def codec_pairs(lib: str) -> dict:
    """name -> (group, label, settings, data key, encode(x), decode(b, x))."""
    ic = _ic() if lib == "ic" else None
    P = {}

    def add(name, group, label, settings, data, oc_enc, oc_dec, ic_enc, ic_dec):
        if lib == "oc":
            P[name] = (group, label, settings, data, oc_enc, oc_dec)
        else:
            P[name] = (group, label, settings, data, ic_enc, ic_dec)

    def by(x):
        return x.tobytes() if isinstance(x, np.ndarray) else x

    # -- general-purpose compressors (bytes) --------------------------------
    for key, data, oc_kw, ic_kw, settings in [
        ("zstd", "u16", dict(level=3), dict(level=3), "level 3"),
        ("deflate", "u16", dict(level=6), dict(level=6), "zlib stream, level 6"),
        ("lz4", "u16", {}, {}, "frame format, default level"),
        ("brotli", "u16", dict(level=4), dict(level=4), "level 4"),
        ("blosc2", "u16", dict(level=5, compressor="zstd", typesize=2, shuffle=True),
         dict(level=5, compressor="zstd", typesize=2, shuffle=1), "zstd, level 5, byte shuffle"),
        ("lzma", "u16_small", dict(level=6), dict(level=6), "level 6"),
        ("bz2", "u16_small", dict(level=9), dict(level=9), "level 9"),
        ("snappy", "u16", {}, {}, ""),
    ]:
        icname = {"deflate": "zlib", "lz4": "lz4f"}.get(key, key)
        add(key, "Compression", key if key != "deflate" else "deflate", settings, data,
            (lambda x, k=key, kw=oc_kw: _oc(k).encode(by(x), **kw)),
            (lambda b, x, k=key: _oc(k).decode(b)),
            (lambda x, n=icname, kw=ic_kw: getattr(ic, f"{n}_encode")(by(x), **kw)),
            (lambda b, x, n=icname: getattr(ic, f"{n}_decode")(b)))

    # -- TIFF compression (bytes) -------------------------------------------
    def oc_tiff():
        from opencodecs.codecs import _tiff
        return _tiff
    add("lzw", "TIFF compression", "LZW", "TIFF flavor", "u8",
        lambda x: oc_tiff().lzw_encode(by(x)), lambda b, x: oc_tiff().lzw_decode(b, x.nbytes),
        lambda x: ic.lzw_encode(by(x)), lambda b, x: ic.lzw_decode(b))
    add("packbits", "TIFF compression", "PackBits", "", "u8",
        None, lambda b, x: oc_tiff().packbits_decode(b, x.nbytes),
        lambda x: ic.packbits_encode(by(x)), lambda b, x: ic.packbits_decode(b))

    # -- filters and predictors (arrays) ------------------------------------
    add("delta_u16", "Filters", "delta", "uint16, distance 1", "u16",
        None, lambda b, x: _oc("delta").decode(b, dtype=np.uint16, shape=x.shape),
        lambda x: ic.delta_encode(x), lambda b, x: ic.delta_decode(np.frombuffer(b, np.uint16).reshape(x.shape)))
    add("xor_u16", "Filters", "XOR", "uint16, distance 1", "u16",
        None, lambda b, x: _oc("xor").decode(b, dtype=np.uint16, shape=x.shape),
        lambda x: ic.xor_encode(x), lambda b, x: ic.xor_decode(np.frombuffer(b, np.uint16).reshape(x.shape)))
    add("bitshuffle", "Filters", "bitshuffle", "2-byte items", "u16",
        lambda x: _oc("bitshuffle").encode(by(x), itemsize=2),
        lambda b, x: _oc("bitshuffle").decode(b, itemsize=2),
        lambda x: ic.bitshuffle_encode(by(x), itemsize=2),
        lambda b, x: ic.bitshuffle_decode(b, itemsize=2))
    add("packints12", "Filters", "packed integers", "12-bit into uint16", "u16",
        None, lambda b, x: _oc("packints").decode(b, dtype=np.uint16, bitspersample=12, n_elements=x.size),
        lambda x: ic.packints_encode((x >> 4).ravel(), 12),
        lambda b, x: ic.packints_decode(b, np.uint16, 12))

    # -- image formats (arrays) ---------------------------------------------
    for key, icname, data, oc_kw, ic_kw, settings in [
        ("png", "png", "rgb8", dict(level=6), dict(level=6), "RGB uint8, level 6"),
        ("jpeg", "jpeg", "rgb8", dict(level=90), dict(level=90), "RGB uint8, quality 90"),
        ("webp", "webp", "rgb8", dict(level=75, lossless=False, method=4),
         dict(level=75, lossless=False, method=4), "RGB uint8, lossy quality 75, method 4"),
        ("qoi", "qoi", "rgb8", {}, {}, "RGB uint8"),
        ("jpeg2k", "jpeg2k", "g16", dict(lossless=True, codec="jp2"),
         dict(reversible=True, codecformat="jp2"), "uint16, lossless"),
        ("jpegls", "jpegls", "g16", {}, {}, "uint16, lossless"),
        ("jxl_lossy", "jpegxl", "rgb8", dict(distance=1.0, effort=5, lossless=False),
         dict(distance=1.0, effort=5, lossless=False), "RGB uint8, distance 1, effort 5"),
        ("jxl_lossless", "jpegxl", "g16", dict(lossless=True, effort=3),
         dict(lossless=True, effort=3), "uint16, lossless, effort 3"),
        ("lerc", "lerc", "g16", dict(max_z_error=0.0), dict(level=0.0), "uint16, lossless"),
        ("zfp", "zfp", "f32", dict(mode="reversible"), dict(), "float32, reversible"),
    ]:
        ockey = {"jxl_lossy": "jxl", "jxl_lossless": "jxl"}.get(key, key)
        label = {"jxl_lossy": "JPEG XL", "jxl_lossless": "JPEG XL", "jpeg2k": "JPEG 2000",
                 "jpegls": "JPEG-LS", "png": "PNG", "jpeg": "JPEG", "webp": "WebP",
                 "qoi": "QOI", "lerc": "LERC", "zfp": "ZFP"}[key]
        add(key, "Image formats", label, settings, data,
            (lambda x, k=ockey, kw=oc_kw: _oc(k).encode(x, **kw)),
            (lambda b, x, k=ockey: _oc(k).decode(b)),
            (lambda x, n=icname, kw=ic_kw: getattr(ic, f"{n}_encode")(x, **kw)),
            (lambda b, x, n=icname: getattr(ic, f"{n}_decode")(b)))

    # -- GPU texture blocks (decode only) -----------------------------------
    for key, fmt in (("bc1", 1), ("bc7", 7)):
        add(key, "Image formats", key.upper(), "2048 x 2048, decode to RGBA", key,
            None, (lambda b, x, f=key: _oc("bcn").decode(b, format=f, width=2048, height=2048)),
            None, (lambda b, x, f=fmt: ic.bcn_decode(b, f, shape=(2048, 2048, 4))))
    return P


LOSSLESS = {"zstd", "deflate", "lz4", "brotli", "blosc2", "lzma", "bz2", "snappy", "lzw",
            "packbits", "bitshuffle", "png", "qoi", "jpeg2k", "jpegls", "jxl_lossless",
            "lerc", "zfp"}
# Lossy decodes whose two builds round some values differently (by at most
# one, measured); the table marks them instead of failing them.
ROUNDING = {"jxl_lossy:decode": "both use libjxl 0.12.0; on Linux about a third of the lossy "
                                "output values differ by 1, never more"}
# Whose decode input is not an encoding of the input array itself.
RAW_INPUT = {"bc1", "bc7"}


def case_names() -> list[str]:
    """Every (codec, operation) the table measures, in table order."""
    out = []
    for name, (_, _, _, _, enc, _) in codec_pairs("oc").items():
        if enc is not None:
            out.append(f"{name}:encode")
        out.append(f"{name}:decode")
    return out


# ---------------------------------------------------------------------------
# One sample
# ---------------------------------------------------------------------------

def _time(fn, min_s=0.4, min_n=5, max_n=40):
    """Median ms, samples, and process CPU time over wall time (above
    about 1.3, the call ran on more than one thread)."""
    fn()                                                         # warm up
    ts, total = [], 0.0
    c0 = time.process_time()
    while (len(ts) < min_n or total < min_s) and len(ts) < max_n:
        t = time.perf_counter(); fn(); dt = time.perf_counter() - t
        ts.append(dt); total += dt
    return statistics.median(ts) * 1e3, len(ts), (time.process_time() - c0) / total


def prepare(data: pathlib.Path) -> None:
    """Write every decode row's input with imagecodecs, in its own process."""
    data.mkdir(parents=True, exist_ok=True)
    X = inputs()
    for name, (_, _, _, key, enc, _) in codec_pairs("ic").items():
        path = data / f"{name}.bin"
        if path.exists():
            continue
        blob = X[key] if name in RAW_INPUT else enc(X[key])
        tmp = path.with_suffix(f".{os.getpid()}")
        tmp.write_bytes(bytes(blob))
        os.replace(tmp, path)


def once(case: str, lib: str, data: pathlib.Path) -> dict:
    name, op = case.split(":")
    X = inputs()
    group, label, settings, key, enc, dec = codec_pairs(lib)[name]
    x = X[key]
    rec = {"case": case, "lib": lib}
    if op == "decode":
        blob = (data / f"{name}.bin").read_bytes()
        out = dec(blob, x)
        rec["ms"], rec["n"], rec["cpu"] = _time(lambda: dec(blob, x))
        rec["check"] = digest(np.asarray(out) if not isinstance(out, (bytes, bytearray, memoryview)) else out)
    else:
        blob = bytes(enc(x))
        rec["ms"], rec["n"], rec["cpu"] = _time(lambda: enc(x))
        rec["bytes"] = len(blob)
        back = dec(blob, x)
        if name in LOSSLESS:
            want = x.tobytes() if isinstance(x, np.ndarray) else bytes(x)
            got = np.asarray(back).tobytes() if not isinstance(back, (bytes, bytearray, memoryview)) else bytes(back)
            rec["check"] = "lossless" if got == want else "ROUND TRIP DIFFERS"
        else:
            rec["check"] = "lossy"
    return rec


# ---------------------------------------------------------------------------
# Driver and table
# ---------------------------------------------------------------------------

def versions() -> dict:
    code = ("import json, platform, opencodecs, imagecodecs; print(json.dumps({'opencodecs': "
            "opencodecs.__version__, 'imagecodecs': imagecodecs.__version__, 'python': "
            "platform.python_version(), 'machine': platform.machine(), 'system': platform.system()}))")
    return json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True,
                                     text=True, check=True).stdout)


def run(rounds: int, data: pathlib.Path, out: pathlib.Path, cases: list[str]) -> None:
    subprocess.run([sys.executable, str(HERE), "prepare", str(data)], check=True)
    meta = versions()
    print(meta, flush=True)
    rows = []
    jobs = [(c, lib) for c in cases for lib in ("oc", "ic")]
    for r in range(rounds):
        random.Random(7000 + r).shuffle(jobs)
        for c, lib in jobs:
            p = subprocess.run([sys.executable, str(HERE), "once", c, lib, str(data)],
                               capture_output=True, text=True)
            if p.returncode:
                rows.append({"case": c, "lib": lib, "error": p.stderr.strip().splitlines()[-1:]})
                continue
            rows.append(json.loads(p.stdout.strip().splitlines()[-1]))
        out.write_text(json.dumps({"meta": meta, "rows": rows}, indent=1))
        print(f"round {r + 1}/{rounds}", flush=True)
    print("RUN_DONE", flush=True)


def summarize(path: pathlib.Path) -> tuple[dict, dict]:
    d = json.loads(path.read_text())
    by = {}
    for r in d["rows"]:
        by.setdefault(r["case"], {}).setdefault(r["lib"], []).append(r)
    res = {}
    for case, libs in by.items():
        entry = {}
        for lib in ("oc", "ic"):
            rs = [x for x in libs.get(lib, []) if "ms" in x]
            if not rs:
                entry["error"] = (libs.get(lib) or [{}])[0].get("error", "missing")
                break
            entry[lib] = statistics.median(x["ms"] for x in rs)
            entry[f"{lib}_check"] = sorted({x["check"] for x in rs})
            entry[f"{lib}_cpu"] = statistics.median(x.get("cpu", 1.0) for x in rs)
            if "bytes" in rs[0]:
                entry[f"{lib}_bytes"] = rs[0]["bytes"]
        if "error" not in entry:
            entry["speedup"] = entry["ic"] / entry["oc"]
            if case.endswith(":decode"):
                entry["same"] = entry["oc_check"] == entry["ic_check"] and len(entry["oc_check"]) == 1
            else:
                entry["same"] = all(c in ("lossless", "lossy") for c in entry["oc_check"] + entry["ic_check"])
        res[case] = entry
    return d["meta"], res


def table(hosts: list[tuple[str, pathlib.Path]]) -> None:
    data = {h: summarize(p) for h, p in hosts}
    pairs = codec_pairs("oc")
    names = [h for h, _ in hosts]
    print("| Codec | Settings | Operation | " + " | ".join(names) + " |")
    print("|---|---|---|" + "---:|" * len(names))
    notes, group = [], None
    for case in case_names():
        name, op = case.split(":")
        g, label, settings, *_ = pairs[name]
        if g != group:
            group = g
            print(f"| **{g}** | | |" + " |" * len(names))
        cells = []
        for h in names:
            e = data[h][1].get(case)
            if not e or "error" in e:
                cells.append("n/a")
                notes.append(f"{h} {case}: {e.get('error') if e else 'not run'}")
                continue
            cell = f"{e['speedup']:.2f}x" if e["speedup"] < 10 else f"{e['speedup']:.0f}x"
            threads = [lib for lib in ("oc", "ic") if e[f"{lib}_cpu"] > 1.3]
            if threads:
                cell += " (" + {("oc",): "opencodecs threaded", ("ic",): "imagecodecs threaded",
                                ("oc", "ic"): "both threaded"}[tuple(threads)] + ")"
            if not e["same"] and case in ROUNDING:
                cell += " *"
            elif not e["same"]:
                cell += " (output differs)"
                notes.append(f"{h} {case}: checks oc={e['oc_check']} ic={e['ic_check']}")
            if "oc_bytes" in e and abs(e["oc_bytes"] / e["ic_bytes"] - 1) > 0.02:
                notes.append(f"{h} {case}: size oc {e['oc_bytes']} ic {e['ic_bytes']} "
                             f"({e['oc_bytes'] / e['ic_bytes']:.3f}x)")
            cells.append(cell)
        print(f"| {label} | {settings} | {op} | " + " | ".join(cells) + " |")
    print()
    for h in names:
        print(h, data[h][0])
    for case, why in ROUNDING.items():
        print(f"* {case}: {why}")
    print("\nnotes:" if notes else "\nnotes: none")
    for n in notes:
        print(" ", n)
    print("\nmedian ms, opencodecs / imagecodecs:")
    for case in case_names():
        print(f"  {case:24s} " + "  ".join(
            f"{h} {data[h][1][case]['oc']:8.2f} / {data[h][1][case]['ic']:8.2f}"
            if case in data[h][1] and "oc" in data[h][1][case] else f"{h} n/a" for h in names))


if __name__ == "__main__":
    cmd = sys.argv[1]
    if cmd == "once":
        print(json.dumps(once(sys.argv[2], sys.argv[3], pathlib.Path(sys.argv[4]))))
    elif cmd == "prepare":
        prepare(pathlib.Path(sys.argv[2]))
    elif cmd == "run":
        import argparse
        ap = argparse.ArgumentParser()
        ap.add_argument("cmd")
        ap.add_argument("--rounds", type=int, default=5)
        ap.add_argument("--data", required=True)
        ap.add_argument("--out", required=True)
        ap.add_argument("--cases", default="")
        a = ap.parse_args()
        run(a.rounds, pathlib.Path(a.data), pathlib.Path(a.out),
            a.cases.split(",") if a.cases else case_names())
    elif cmd == "table":
        table([(s.split("=", 1)[0], pathlib.Path(s.split("=", 1)[1])) for s in sys.argv[2:]])
    else:
        raise SystemExit(__doc__)
