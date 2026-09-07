#!/usr/bin/env python3
"""Keep capabilities.toml honest about what each codec can actually do.

opencodecs promises more than "it decodes": formats that stream, that
serve over HTTP with range requests, that decode a tile at a time, that
open at multiple resolutions. Those promises are per format, and some of
them are impossible for some formats. Without a record, "not done yet"
and "cannot be done" are indistinguishable from outside, and a promise
nobody re-checks becomes a wrong docstring -- which is exactly what
happened to _sdr_from_hdr_kernel, whose comment claimed a vectorized
loop the compiler had never produced.

So this does not read the file and believe it. It re-derives every
mechanical field from the code and fails when the two disagree:

    python ci/check_capabilities.py verify   compare file against code
    python ci/check_capabilities.py sync     rewrite the derived fields
    python ci/check_capabilities.py report   what is done, what is not

`feasible` is the one field the code cannot derive, because it is a
judgement about the format rather than a fact about the tree. It is
carried through untouched, and `report` lists the codecs where it is
still "unassessed" -- that list is the worklist.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "capabilities.toml"

# Mechanically derivable: each is a fact about the tree, not an opinion.
DERIVED = ("multi_frame", "chunked", "streaming_decode", "parallel_decode",
           "range_reads", "pyramid", "http")


def _load():
    try:
        import tomllib
    except ModuleNotFoundError:                      # Python 3.10
        try:
            import tomli as tomllib                  # type: ignore
        except ModuleNotFoundError:
            sys.exit("check_capabilities: needs Python 3.11+ or `pip install tomli`")
    if not MANIFEST.is_file():
        sys.exit(f"check_capabilities: {MANIFEST} does not exist; run `sync`")
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh).get("codec", [])


def _grep(pattern: str) -> set[str]:
    r = subprocess.run(["git", "grep", "-l", pattern, "--", "src/opencodecs"],
                       cwd=ROOT, capture_output=True, text=True)
    return set(r.stdout.split())


def derive() -> dict[str, dict]:
    """Re-read the capabilities out of the code."""
    sys.path.insert(0, str(ROOT / "src"))
    import opencodecs as oc

    src = ROOT / "src/opencodecs"
    range_capable = _grep("open_read_at\\|read_at(")
    pyramid_files = _grep("PyramidReader")
    http_files = _grep("HTTPDataSource")

    def files_for(name: str) -> set[str]:
        return {str(p.relative_to(ROOT)) for p in src.rglob(f"*{name}*")
                if p.suffix in (".py", ".pyx")}

    out = {}
    for info in oc.list_codecs():
        name = info["name"]
        c = oc.get_codec(name)
        mine = files_for(name)
        out[name] = {
            "multi_frame": bool(getattr(c, "multi_frame", False)),
            "chunked": bool(getattr(c, "chunked", False)),
            "streaming_decode": bool(getattr(c, "streaming_decode", False)),
            "parallel_decode": bool(getattr(c, "parallel_decode", False)),
            "range_reads": bool(mine & range_capable),
            "pyramid": bool(mine & pyramid_files),
            "http": bool(mine & http_files),
        }
    return out


def cmd_verify(args) -> int:
    recorded = {c["name"]: c for c in _load()}
    actual = derive()
    bad = 0

    missing = sorted(set(actual) - set(recorded))
    extra = sorted(set(recorded) - set(actual))
    for n in missing:
        print(f"  UNRECORDED  {n} is registered but not in capabilities.toml")
        bad += 1
    for n in extra:
        print(f"  STALE       {n} is in capabilities.toml but not registered")
        bad += 1

    for name in sorted(set(recorded) & set(actual)):
        for field in DERIVED:
            want = actual[name][field]
            got = recorded[name].get(field)
            if got != want:
                print(f"  DRIFTED     {name}.{field}: file says {got}, "
                      f"code says {want}")
                bad += 1

    print(f"{len(actual)} codecs; {bad} discrepancy(ies)")
    return 1 if bad else 0


def cmd_sync(args) -> int:
    """Rewrite the derived fields, carrying `feasible` and notes across."""
    actual = derive()
    old = {c["name"]: c for c in (_load() if MANIFEST.is_file() else [])}

    head = MANIFEST.read_text().split("schema = 1", 1)[0] if MANIFEST.is_file() \
        else ""
    lines = [head.rstrip("\n"), "", "schema = 1", ""] if head else \
        ["schema = 1", ""]
    for name in sorted(actual):
        lines.append("[[codec]]")
        lines.append(f'name = "{name}"')
        for field in DERIVED:
            lines.append(f"{field} = {str(actual[name][field]).lower()}")
        prev = old.get(name, {})
        lines.append(f'feasible = "{prev.get("feasible", "unassessed")}"')
        if prev.get("note"):
            note = prev["note"].replace('"', '\\"')
            lines.append(f'note = "{note}"')
        lines.append("")
    MANIFEST.write_text("\n".join(lines))
    print(f"synced {len(actual)} codecs into {MANIFEST.name}")
    return 0


def cmd_report(args) -> int:
    recorded = _load()
    n = len(recorded)
    print(f"{n} codecs\n")
    print(f"{'capability':18s} {'have':>5s}  what it buys a caller")
    print("-" * 74)
    blurb = {
        "streaming_decode": "decode without holding the whole file",
        "chunked": "fetch tile N without decoding 0..N-1",
        "range_reads": "open over HTTP without downloading it",
        "http": "a data source that speaks range requests",
        "multi_frame": "stacks and animations",
        "pyramid": "open at a resolution that fits the screen",
        "parallel_decode": "more than one core on one image",
    }
    for field in ("streaming_decode", "chunked", "range_reads", "http",
                  "multi_frame", "pyramid", "parallel_decode"):
        have = sum(1 for c in recorded if c.get(field))
        print(f"{field:18s} {have:3d}/{n}  {blurb[field]}")

    un = [c["name"] for c in recorded if c.get("feasible") == "unassessed"]
    print(f"\nfeasibility not yet judged for {len(un)} codec(s).")
    print("That is the worklist: for each, is the missing capability")
    print("impossible for the format, or merely not built yet?")
    if args.verbose and un:
        for i in range(0, len(un), 8):
            print("  " + " ".join(un[i:i + 8]))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(required=True, dest="cmd")
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("sync").set_defaults(fn=cmd_sync)
    r = sub.add_parser("report")
    r.add_argument("-v", "--verbose", action="store_true")
    r.set_defaults(fn=cmd_report)
    args = p.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
