"""The corpus manifest must stay in step with the download script.

Two sources of truth for the same URLs will drift, and the failure mode
is quiet: a dataset gets added to one and the coverage report silently
under-reports, or the fetcher silently skips it. Until the shell script
becomes a thin wrapper over the manifest, this test is what keeps them
honest.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "corpus" / "manifest.toml"
SCRIPT = ROOT / "tests" / "download_test_corpus.sh"

SHELL_VARS = {
    "$ZARR_URL": "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr",
}


def _load():
    try:
        import tomllib
    except ModuleNotFoundError:
        tomllib = pytest.importorskip("tomli")
    with MANIFEST.open("rb") as fh:
        return tomllib.load(fh)


def _expand(url: str) -> str:
    for k, v in SHELL_VARS.items():
        url = url.replace(k, v)
    return url


def test_manifest_parses_and_has_required_fields():
    doc = _load()
    assert doc.get("schema") == 1
    datasets = doc["dataset"]
    assert datasets, "manifest has no datasets"
    seen = set()
    for ds in datasets:
        for field in ("id", "name", "license", "codecs", "file"):
            assert field in ds, f"{ds.get('id', '?')} missing {field}"
        assert ds["id"] not in seen, f"duplicate id {ds['id']}"
        seen.add(ds["id"])
        assert ds["file"], f"{ds['id']} lists no files"
        for f in ds["file"]:
            assert f["url"].startswith(("http://", "https://")), f["url"]
            assert f["path"].startswith(".test_data/"), f["path"]


def test_every_download_script_url_is_in_the_manifest():
    script = SCRIPT.read_text()
    urls = set()
    for line in script.splitlines():
        if re.match(r"\s*[A-Z_]+=", line):
            continue                      # a variable definition, not a fetch
        for m in re.finditer(r'"(https?://[^"]+)"', line):
            urls.add(_expand(m.group(1)))
    listed = {_expand(f["url"]) for ds in _load()["dataset"] for f in ds["file"]}
    # the script builds some URLs in loops, so compare on the stem
    missing = {u for u in urls
               if u not in listed and not any(u.split("${")[0] in l for l in listed)}
    assert not missing, (
        "these URLs are fetched by the script but absent from "
        f"corpus/manifest.toml: {sorted(missing)}")


def test_manifest_codecs_name_real_extensions():
    """A codec named in the manifest that we do not ship is a typo.

    Checked against the live registry rather than a hand-kept allowlist
    of container formats. The allowlist version failed the moment a new
    pure-Python reader was added, which is a maintenance burden with no
    safety benefit: a typo is not registered either, so the registry
    catches exactly what the allowlist caught and nothing more.
    """
    import opencodecs as oc

    # Directory formats are read through a class rather than the byte
    # codec registry, because there is no single blob to hand decode():
    # the array is a tree of metadata and chunk files. They cannot be
    # registered codecs, so they are named here.
    DIRECTORY_FORMATS = {"omezarr", "n5", "imaris", "dicom"}

    known = {c["name"] for c in oc.list_codecs()} | DIRECTORY_FORMATS
    for c in oc.list_codecs():
        known.update(c.get("aliases", ()) or ())
    # Extensions that fail to build on this host still register (the
    # optional-backend stub pattern), so this set does not vary with
    # which codec libraries happen to be installed.
    built = {p.name[1:-4] for p in (ROOT / "src" / "opencodecs" / "codecs").glob("_*.pyx")
             if not p.name.startswith("._")}
    for ds in _load()["dataset"]:
        for c in ds["codecs"]:
            assert c in known or c in built, (
                f"{ds['id']} lists codec {c!r}, which no registered codec "
                f"provides; check for a typo")
