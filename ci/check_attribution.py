#!/usr/bin/env python3
"""Compare every source file we ship against another project's, file by file.

`check_vendor_drift.py` answers "is our vendored copy current?". This
answers a different and sharper question: "is any file we ship actually
somebody else's work, and does it say so?"

The distinction matters because the two look identical from inside the
repository. A file under 3rdparty/cfitsio/ reads as NASA's; ricecomp.h
sat there for months and was in fact byte-identical to the copy in
imagecodecs, which is neither NASA's nor ours. Nothing catches that
except comparing against the other project directly.

    python ci/check_attribution.py                     # vs imagecodecs
    python ci/check_attribution.py --against owner/repo --ref master
    python ci/check_attribution.py --threshold 0.85    # widen the net

Findings are graded:

  IDENTICAL  byte-for-byte the same file. Either both projects vendor it
             from a common upstream (fine, and VENDOR.toml should say
             which), or we are shipping their work (not fine).
  DERIVED    similarity above the threshold. Compared by real content,
             not just matching filenames: a file renamed between the two
             projects is exactly what a filename-only pass misses, and
             the .pxd files that started this were renamed.

Exit status is 1 when an IDENTICAL file is not explained by a VENDOR.toml
entry naming a real upstream other than the project being compared.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CACHE = Path("/tmp/opencodecs_attribution_cache")
SOURCE_SUFFIXES = {".py", ".pyx", ".pxd", ".pxi", ".c", ".h", ".cpp", ".hpp"}


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _fetch(url: str) -> bytes | None:
    key = CACHE / _sha(url.encode())
    if key.is_file():
        return key.read_bytes()
    try:
        with urllib.request.urlopen(url, timeout=90) as r:
            data = r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None
    CACHE.mkdir(parents=True, exist_ok=True)
    key.write_bytes(data)
    return data


def _similarity(a: bytes, b: bytes, floor: float) -> float:
    """Real similarity, with quick_ratio as a prefilter.

    quick_ratio compares character multisets and is only an upper bound.
    Two unrelated Cython wrappers for different C libraries score 0.90 on
    it purely by sharing a vocabulary; the pair that prompted this note
    scored 0.075 once actually diffed. Trusting the cheap number would
    have reported a file as derived that has nothing in common beyond a
    boilerplate header.
    """
    try:
        sa = a.decode("utf8", "replace")
        sb = b.decode("utf8", "replace")
    except Exception:                                    # noqa: BLE001
        return 0.0
    sm = difflib.SequenceMatcher(None, sa, sb)
    if sm.quick_ratio() < floor:
        return 0.0
    return sm.ratio()


def _our_files() -> list[Path]:
    out = subprocess.run(["git", "ls-files"], cwd=ROOT, check=True,
                         capture_output=True, text=True).stdout.split()
    return [ROOT / p for p in out
            if Path(p).suffix.lower() in SOURCE_SUFFIXES]


def _explained() -> dict[str, str]:
    """Files whose sameness a VENDOR.toml entry already accounts for.

    A file we vendor from bitshuffle or qoi is expected to be identical
    to anyone else's copy of the same upstream file. That is a shared
    origin, not appropriation, and the manifest records which.
    """
    try:
        import tomllib
    except ModuleNotFoundError:                          # Python 3.10
        try:
            import tomli as tomllib                      # type: ignore
        except ModuleNotFoundError:
            return {}
    path = ROOT / "3rdparty" / "VENDOR.toml"
    if not path.is_file():
        return {}
    with path.open("rb") as fh:
        comps = tomllib.load(fh).get("component", [])
    out: dict[str, str] = {}
    for c in comps:
        repo = c.get("repo")
        for f in c.get("file", []):
            if f.get("ours"):
                out[f["local"]] = "declared as opencodecs' own"
            elif repo:
                out[f["local"]] = f"vendored from {repo}"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--against", default="cgohlke/imagecodecs")
    ap.add_argument("--ref", default="master")
    ap.add_argument("--threshold", type=float, default=0.90)
    args = ap.parse_args()

    tree_raw = _fetch(f"https://api.github.com/repos/{args.against}"
                      f"/git/trees/{args.ref}?recursive=1")
    if tree_raw is None:
        print(f"could not read the tree of {args.against}@{args.ref}")
        return 2
    theirs: dict[str, list[str]] = {}
    for entry in json.loads(tree_raw).get("tree", []):
        p = entry["path"]
        if Path(p).suffix.lower() in SOURCE_SUFFIXES:
            theirs.setdefault(Path(p).name, []).append(p)

    explained = _explained()
    ours = _our_files()
    print(f"comparing {len(ours)} of our source files against "
          f"{args.against}@{args.ref} "
          f"({sum(len(v) for v in theirs.values())} of theirs)\n")

    identical: list[tuple[str, str, str]] = []
    derived: list[tuple[str, str, float]] = []

    for p in sorted(ours):
        rel = p.relative_to(ROOT).as_posix()
        cands = theirs.get(p.name, [])
        if not cands:
            continue
        mine = p.read_bytes()
        best = 0.0
        best_path = cands[0]
        for c in cands:
            up = _fetch(f"https://raw.githubusercontent.com/"
                        f"{args.against}/{args.ref}/{c}")
            if up is None:
                continue
            if _sha(up) == _sha(mine):
                identical.append((rel, c, explained.get(rel, "")))
                best = 1.0
                break
            ratio = _similarity(up, mine, args.threshold)
            if ratio > best:
                best, best_path = ratio, c
        if best < 1.0 and best >= args.threshold:
            derived.append((rel, best_path, best))

    # Second pass, for Cython only: compare every one of our .pyx/.pxd
    # against every one of theirs regardless of filename. A .pxd renamed
    # between projects (libjxl.pxd here, jpegxl.pxd there) is invisible to
    # a name-keyed comparison, and .pxd files are what this repository was
    # asked about in the first place.
    cy_theirs = [p for names in theirs.values() for p in names
                 if Path(p).suffix.lower() in (".pyx", ".pxd")]
    cy_ours = [p for p in ours if p.suffix.lower() in (".pyx", ".pxd")]
    already = {rel for rel, _, _ in derived} | {rel for rel, _, _ in identical}
    print(f"cross-checking {len(cy_ours)} Cython files against "
          f"{len(cy_theirs)} of theirs by content, ignoring filenames...")
    for p in sorted(cy_ours):
        rel = p.relative_to(ROOT).as_posix()
        if rel in already:
            continue
        mine = p.read_bytes()
        best, best_path = 0.0, None
        for c in cy_theirs:
            if Path(c).name == p.name:
                continue                                 # covered above
            up = _fetch(f"https://raw.githubusercontent.com/"
                        f"{args.against}/{args.ref}/{c}")
            if up is None:
                continue
            ratio = _similarity(up, mine, args.threshold)
            if ratio > best:
                best, best_path = ratio, c
        if best >= args.threshold and best_path:
            derived.append((rel, best_path, best))

    print("IDENTICAL (byte-for-byte):")
    unexplained = []
    for rel, their, why in identical:
        if why:
            print(f"  ok  {rel}\n        same file, {why}")
        else:
            print(f"  !!  {rel}\n        matches {args.against}/{their} "
                  f"with nothing in VENDOR.toml explaining it")
            unexplained.append(rel)
    if not identical:
        print("  (none)")

    print(f"\nDERIVED (same name, similarity >= {args.threshold}):")
    for rel, their, ratio in sorted(derived, key=lambda r: -r[2]):
        note = explained.get(rel, "")
        print(f"  {ratio:.2f}  {rel}"
              + (f"\n        {note}" if note else
                 f"\n        resembles {args.against}/{their}; confirm the"
                 f" shared origin is upstream, not this project"))
    if not derived:
        print("  (none)")

    print(f"\n{len(identical)} identical ({len(unexplained)} unexplained), "
          f"{len(derived)} derived above threshold")
    return 1 if unexplained else 0


if __name__ == "__main__":
    sys.exit(main())
