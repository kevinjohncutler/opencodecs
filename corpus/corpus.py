#!/usr/bin/env python3
"""Reference-corpus tool: fetch, verify and report coverage.

Nothing is redistributed. ``manifest.toml`` records where each file
lives upstream, what it is licensed under, and which codecs it
exercises; this fetches from origin into the gitignored ``.test_data/``
and can tell you which codecs still have no real data behind them.

    python corpus/corpus.py list                  what is in the manifest
    python corpus/corpus.py coverage              codecs with and without data
    python corpus/corpus.py fetch [ID ...]        download (all, or by id)
    python corpus/corpus.py verify                re-hash what is on disk
    python corpus/corpus.py freeze                write checksums back

Checksums are recorded on first fetch rather than hand-entered, because
an unverified digest is worse than none: it looks like provenance and
is not. Once frozen, a changed upstream file is an error rather than a
silent difference in results.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = Path(__file__).resolve().parent / "manifest.toml"
CHECKSUMS = Path(__file__).resolve().parent / "checksums.json"


def _load_manifest() -> list[dict]:
    try:
        import tomllib
    except ModuleNotFoundError:                      # Python 3.10
        try:
            import tomli as tomllib                  # type: ignore
        except ModuleNotFoundError:
            sys.exit("corpus: needs Python 3.11+ or `pip install tomli`")
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh).get("dataset", [])


def _checksums() -> dict:
    return json.loads(CHECKSUMS.read_text()) if CHECKSUMS.is_file() else {}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _compiled_codecs() -> set[str]:
    d = ROOT / "src" / "opencodecs" / "codecs"
    return {p.name[1:-4] for p in d.glob("_*.pyx") if not p.name.startswith("._")}


def cmd_list(args) -> int:
    for ds in _load_manifest():
        have = sum((ROOT / f["path"]).is_file() for f in ds["file"])
        mark = "ok " if have == len(ds["file"]) else ("part" if have else "  -")
        print(f"  [{mark}] {ds['id']:<24} {ds['license']:<14} "
              f"{have}/{len(ds['file'])} files  {','.join(ds.get('codecs', []))}")
    return 0


def cmd_coverage(args) -> int:
    compiled = _compiled_codecs()
    covered: dict[str, list[str]] = {}
    for ds in _load_manifest():
        for c in ds.get("codecs", []):
            covered.setdefault(c, []).append(ds["id"])
    have = sorted(c for c in compiled if c in covered)
    missing = sorted(compiled - set(covered))
    print(f"{len(have)} of {len(compiled)} compiled codecs have real data\n")
    print("covered:")
    for c in have:
        print(f"  {c:<14} {', '.join(covered[c])}")
    print("\nNO real-data coverage:")
    for c in missing:
        print(f"  {c}")
    # entries naming a codec we do not ship are a manifest typo
    stray = sorted(set(covered) - compiled)
    if stray:
        print("\nmanifest names codecs that are not compiled extensions "
              "(container formats or typos):")
        print("  " + ", ".join(stray))
    return 0


def cmd_fetch(args) -> int:
    sums = _checksums()
    wanted = set(args.ids)
    for ds in _load_manifest():
        if wanted and ds["id"] not in wanted:
            continue
        for f in ds["file"]:
            dest = ROOT / f["path"]
            if dest.is_file() and dest.stat().st_size:
                print(f"  [skip] {f['path']}")
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            print(f"  [get ] {f['url']}")
            tmp = dest.with_suffix(dest.suffix + ".part")
            try:
                urllib.request.urlretrieve(f["url"], tmp)
            except Exception as exc:                 # noqa: BLE001
                print(f"  [FAIL] {ds['id']}: {type(exc).__name__}: {exc}")
                tmp.unlink(missing_ok=True)
                continue
            tmp.replace(dest)
            got = _sha256(dest)
            want = sums.get(f["path"])
            if want and got != want:
                print(f"  [HASH] {f['path']} changed upstream!\n"
                      f"         expected {want}\n         got      {got}")
    return 0


def cmd_verify(args) -> int:
    sums = _checksums()
    if not sums:
        print("no checksums recorded yet; run `corpus.py freeze`")
        return 1
    bad = missing = 0
    for ds in _load_manifest():
        for f in ds["file"]:
            dest = ROOT / f["path"]
            if not dest.is_file():
                missing += 1
                continue
            want = sums.get(f["path"])
            if want is None:
                continue
            if _sha256(dest) != want:
                bad += 1
                print(f"  MISMATCH {f['path']}")
    print(f"{bad} mismatched, {missing} not downloaded")
    return 1 if bad else 0


def cmd_freeze(args) -> int:
    sums = _checksums()
    added = 0
    for ds in _load_manifest():
        for f in ds["file"]:
            dest = ROOT / f["path"]
            if dest.is_file() and f["path"] not in sums:
                sums[f["path"]] = _sha256(dest)
                added += 1
    CHECKSUMS.write_text(json.dumps(sums, indent=2, sort_keys=True) + "\n")
    print(f"recorded {added} new checksum(s); {len(sums)} total")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    sub.add_parser("coverage").set_defaults(fn=cmd_coverage)
    p = sub.add_parser("fetch"); p.add_argument("ids", nargs="*"); p.set_defaults(fn=cmd_fetch)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("freeze").set_defaults(fn=cmd_freeze)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
