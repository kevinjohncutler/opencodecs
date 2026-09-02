#!/usr/bin/env python3
"""Tell us when vendored source has drifted from its upstream.

Vendoring is the right call for most of what is under ``3rdparty/``:
single-header libraries, a format frozen in 1991, three files carved out
of a large autotools project, and our own clean-room code. What vendoring
does not give you is a signal when upstream fixes something. PLIO shipped
a decoder that died with SIGBUS on malformed input long after upstream
cfitsio had added the bounds checks that prevent it, and nothing in the
repository was in a position to notice.

This closes that gap without moving everything to build-time fetching:

    python ci/check_vendor_drift.py verify     offline: has anyone edited
                                               a vendored file since it
                                               was recorded?
    python ci/check_vendor_drift.py check      online: does our copy still
                                               match upstream at the
                                               pinned ref, and is there a
                                               newer release?
    python ci/check_vendor_drift.py identify   online: which upstream tag
                                               does each file match? Use
                                               to pin a component whose
                                               ref is a moving branch.
    python ci/check_vendor_drift.py freeze     record current local hashes

``verify`` needs no network and runs in the test suite.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "3rdparty" / "VENDOR.toml"
HASHES = ROOT / "3rdparty" / "vendor_hashes.json"
RAW = "https://raw.githubusercontent.com/{repo}/{ref}/{path}"
TAGS = "https://api.github.com/repos/{repo}/tags?per_page=30"


def _load():
    try:
        import tomllib
    except ModuleNotFoundError:                      # Python 3.10
        try:
            import tomli as tomllib                  # type: ignore
        except ModuleNotFoundError:
            sys.exit("check_vendor_drift: needs Python 3.11+ or `pip install tomli`")
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh).get("component", [])


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fetch(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            return r.read()
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None


def _hashes() -> dict:
    return json.loads(HASHES.read_text()) if HASHES.is_file() else {}


def _tracked_files(comps) -> set[str]:
    return {f["local"] for c in comps for f in c.get("file", [])}


# --------------------------------------------------------------------


def cmd_verify(args) -> int:
    """Offline: local files still match what was recorded, and every
    vendored file is accounted for by the manifest."""
    comps = _load()
    recorded = _hashes()
    bad = missing = 0

    for c in comps:
        for f in c.get("file", []):
            p = ROOT / f["local"]
            if not p.is_file():
                print(f"  MISSING  {f['local']}")
                missing += 1
                continue
            want = recorded.get(f["local"])
            if want is None:
                print(f"  UNRECORDED  {f['local']} (run `freeze`)")
                missing += 1
            elif _sha(p.read_bytes()) != want:
                print(f"  CHANGED  {f['local']} differs from the recorded hash")
                bad += 1

    # A vendored source file nobody declared is the failure mode this is
    # really for: it has no recorded origin, so no one can tell later
    # whether it is ours, current, or three years stale.
    tracked = _tracked_files(comps)
    declared_dirs = {c["name"] for c in comps}
    undeclared = []
    for d in sorted((ROOT / "3rdparty").iterdir()):
        if not d.is_dir() or d.name.startswith("."):
            continue
        if d.name not in declared_dirs:
            undeclared.append(f"{d.name}/ (whole directory)")
            continue
        for p in sorted(d.iterdir()):
            if p.suffix.lower() not in (".c", ".h", ".cpp", ".hpp"):
                continue
            if p.name.startswith("._"):
                continue
            rel = str(p.relative_to(ROOT))
            comp = next(c for c in comps if c["name"] == d.name)
            if comp.get("upstream") == "none":
                continue                      # our own code, nothing to track
            if rel not in tracked:
                undeclared.append(rel)
    for u in undeclared:
        print(f"  UNDECLARED  {u} has no entry in 3rdparty/VENDOR.toml")

    total = bad + missing + len(undeclared)
    print(f"{len(tracked)} tracked file(s); {bad} changed, {missing} missing/unrecorded, "
          f"{len(undeclared)} undeclared")
    return 1 if total else 0


def cmd_check(args) -> int:
    """Online: compare against upstream at the pinned ref, and look for
    a newer release."""
    comps = _load()
    drifted = 0
    for c in comps:
        repo, ref = c.get("repo"), c.get("ref")
        if not repo:
            print(f"[skip] {c['name']}: {c.get('note', 'no upstream')[:60]}")
            continue
        print(f"[{c['name']}] {repo} @ {ref}")

        newest = None
        tags = _fetch(TAGS.format(repo=repo))
        if tags:
            try:
                names = [t["name"] for t in json.loads(tags)]
                newest = names[0] if names else None
                if newest and newest != ref:
                    print(f"    newer upstream tag available: {newest}")
            except Exception:                        # noqa: BLE001
                pass

        for f in c.get("file", []):
            if f.get("ours"):
                continue                      # our file in their directory
            local = ROOT / f["local"]
            if not local.is_file():
                print(f"    MISSING {f['local']}")
                drifted += 1
                continue
            # A file may pin its own ref when it legitimately tracks a
            # different release from the rest of its component.
            fref = f.get("ref", ref)
            up = _fetch(RAW.format(repo=repo, ref=fref, path=f["upstream"]))
            if up is None:
                print(f"    ? {f['upstream']}: could not fetch")
                continue
            same = _sha(up) == _sha(local.read_bytes())
            mod = f.get("modified", False)
            at = "" if fref == ref else f" @ {fref}"
            if same:
                print(f"    ok  {f['upstream']}{at}")
            elif mod:
                # Expected: we carry a deliberate change. Still worth
                # knowing whether upstream moved underneath it.
                print(f"    ~   {f['upstream']}: differs (declared: "
                      f"{f.get('note', 'modified')[:60]})")
            else:
                print(f"    !!  {f['upstream']}: differs from upstream at {ref} "
                      f"but is not declared modified")
                drifted += 1

            if newest and newest != ref:
                newer = _fetch(RAW.format(repo=repo, ref=newest, path=f["upstream"]))
                if newer is not None and _sha(newer) != _sha(up):
                    print(f"        upstream CHANGED this file between "
                          f"{ref} and {newest} — review before assuming ours is current")
                    drifted += 1
    print(f"\n{drifted} item(s) need attention")
    return 1 if drifted else 0


def cmd_identify(args) -> int:
    """Online: find which upstream tag each vendored file matches.

    For components pinned to a moving branch, this is how you turn `ref =
    "master"` into a real version without guessing.
    """
    for c in _load():
        repo = c.get("repo")
        if not repo:
            continue
        tags = _fetch(TAGS.format(repo=repo))
        names = [t["name"] for t in json.loads(tags)] if tags else []
        print(f"[{c['name']}] {repo}: {len(names)} recent tag(s)")
        for f in c.get("file", []):
            if f.get("ours"):
                continue
            local = ROOT / f["local"]
            if not local.is_file():
                continue
            mine = _sha(local.read_bytes())
            hit = None
            for t in names[:args.limit]:
                up = _fetch(RAW.format(repo=repo, ref=t, path=f["upstream"]))
                if up is not None and _sha(up) == mine:
                    hit = t
                    break
            print(f"    {f['upstream']}: "
                  + (f"matches {hit}" if hit
                     else f"no exact match in the {min(len(names), args.limit)} "
                          f"most recent tags"))
    return 0


def cmd_freeze(args) -> int:
    recorded = _hashes()
    n = 0
    for c in _load():
        for f in c.get("file", []):
            p = ROOT / f["local"]
            if p.is_file():
                h = _sha(p.read_bytes())
                if recorded.get(f["local"]) != h:
                    recorded[f["local"]] = h
                    n += 1
    HASHES.write_text(json.dumps(recorded, indent=2, sort_keys=True) + "\n")
    print(f"recorded {n} hash(es); {len(recorded)} total")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("verify").set_defaults(fn=cmd_verify)
    sub.add_parser("check").set_defaults(fn=cmd_check)
    p = sub.add_parser("identify")
    p.add_argument("--limit", type=int, default=12,
                   help="how many recent tags to test (default 12)")
    p.set_defaults(fn=cmd_identify)
    sub.add_parser("freeze").set_defaults(fn=cmd_freeze)
    args = ap.parse_args()
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
