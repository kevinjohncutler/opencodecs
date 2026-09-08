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

`feasible` and `gaps` are the fields the code cannot derive, because
they are judgments about the format rather than facts about the tree.
They are carried through untouched by `sync`, and `report` adds them up:
for each capability, how many codecs have it, and how many could. That
second number is the worklist, and keeping it separate from "the format
cannot do this" is the entire point of the file.
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
# Same fields, ordered the way a caller cares about them.
DERIVED_ORDER = ("streaming_decode", "chunked", "range_reads", "http",
                 "multi_frame", "pyramid", "parallel_decode")


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
    # Reaching storage by offset. `read_at=` matters as much as
    # `read_at(`: a reader that takes the callable and threads it into a
    # shared stream (_eer_reader does exactly this) is doing range reads
    # without ever spelling the call itself, and the narrower pattern
    # recorded that as a `false`.
    range_capable = _grep("open_read_at\\|read_at(\\|read_at=\\|read_many(")
    pyramid_files = _grep("PyramidReader")

    # `http` is meant to say "fetches only the bytes it needs", not
    # merely "accepts a URL". jxl.py names HTTPDataSource only in a
    # docstring, to point callers at the formats that do have it, and
    # gets its own bytes from a single whole-file GET -- so the bare
    # name marked it `true` for a capability its docstring denies.
    #
    # The exclusion is aimed at that specific shape rather than at
    # "no offsets in this file", because a reader can perfectly well
    # delegate its seeking: _oib_codec hands the source to _ole2, which
    # does the offset arithmetic, and requiring the call site to be in
    # the codec's own file would record oib as a false.
    #
    # The other direction of the same delegation problem: core's
    # open_read_at() turns an http(s) URL into an HTTPDataSource itself,
    # so every reader that opens through it already serves range
    # requests without naming the class. dicom, mrc and nrrd all do, and
    # all three were recorded as `false` for a capability the README
    # documents them as having.
    whole_file_get = _grep("http_fetch_all")
    http_files = {f for f in _grep("HTTPDataSource")
                  if f in range_capable or f not in whole_file_get}
    http_files |= _grep("open_read_at")

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

        # The judgment fields cannot be re-derived, but they can still
        # contradict themselves, and a contradiction is how a worklist
        # turns into decoration. A gap has to name a real capability
        # that is really missing, and the verdict has to agree with
        # whether anything is listed.
        rec = recorded[name]
        verdict = rec.get("feasible")
        gaps = rec.get("gaps") or []
        if verdict not in ("done", "gap", "no", "unassessed"):
            print(f"  BAD         {name}.feasible: {verdict!r} is not one "
                  f"of done/gap/no/unassessed")
            bad += 1
        if verdict == "gap" and not gaps:
            print(f"  BAD         {name}: feasible=gap but nothing listed")
            bad += 1
        if verdict in ("done", "no") and gaps:
            print(f"  BAD         {name}: feasible={verdict} but lists "
                  f"gaps {gaps}")
            bad += 1
        if verdict != "unassessed" and not rec.get("note"):
            print(f"  BAD         {name}: judged {verdict!r} with no reason")
            bad += 1
        for g in gaps:
            if g not in DERIVED:
                print(f"  BAD         {name}: {g!r} is not a capability")
                bad += 1
            elif actual[name][g]:
                print(f"  BAD         {name}: lists {g} as a gap, but the "
                      f"code already has it")
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
        if prev.get("gaps"):
            inner = ", ".join(f'"{g}"' for g in prev["gaps"])
            lines.append(f"gaps = [{inner}]")
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
    blurb = {
        "streaming_decode": "decode without holding the whole file",
        "chunked": "fetch tile N without decoding 0..N-1",
        "range_reads": "reaches storage by offset, not whole-file",
        "http": "fetches only the bytes it needs over HTTP",
        "multi_frame": "stacks and animations",
        "pyramid": "open at a resolution that fits the screen",
        "parallel_decode": "more than one core on ONE image",
    }
    print(f"{'capability':18s} {'have':>5s} {'+can':>5s}  "
          f"what it buys a caller")
    print("-" * 74)
    for field in DERIVED_ORDER:
        have = sum(1 for c in recorded if c.get(field))
        can = sum(1 for c in recorded if field in (c.get("gaps") or []))
        print(f"{field:18s} {have:3d}/{n} {'+' + str(can):>5s}  {blurb[field]}")
    print("\n'have' is built. '+can' is feasible and not built: the work "
          "this file exists\nto keep visible, rather than letting it blur "
          "into what the format cannot do.")

    un = [c["name"] for c in recorded if c.get("feasible") == "unassessed"]
    if un:
        print(f"\nfeasibility not yet judged for {len(un)} codec(s):")
        for i in range(0, len(un), 8):
            print("  " + " ".join(un[i:i + 8]))

    gapped = [c for c in recorded if c.get("gaps")]
    done = [c["name"] for c in recorded if c.get("feasible") == "done"]
    shut = [c["name"] for c in recorded if c.get("feasible") == "no"]
    print(f"\n{len(gapped)} codec(s) with feasible work left, "
          f"{sum(len(c['gaps']) for c in gapped)} capability(ies) in total.")
    print(f"{len(done)} complete; {len(shut)} where nothing further is "
          f"available (see each note for why).")
    if args.verbose:
        print()
        for c in sorted(gapped, key=lambda c: -len(c["gaps"])):
            print(f"  {c['name']:10s} {', '.join(c['gaps'])}")
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
