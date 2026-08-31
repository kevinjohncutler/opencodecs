"""Consistency checks on the NAS shadow-copy loader's extension list.

``opencodecs/codecs/__init__.py`` copies each compiled extension off a
network mount into a local cache before importing it, because macOS
Gatekeeper blocks a ``dlopen`` of a binary on a quarantined SMB volume
and, even unquarantined, a cold ``dlopen`` over SMB can stall for a long
time. The loader only does that for names listed in ``_EXTENSIONS``.

An extension missing from that tuple therefore imports straight off the
mount and can wedge inside dyld with no error message. That has happened
twice: once to ``_uhdr`` (the ``_ultrahdr`` rename left the new name
unlisted) and once to ``_plio``, which hung the FITS PLIO round-trip
test. These tests make the next occurrence a red test instead of a hang.
"""

from __future__ import annotations

import pathlib
import re

import pytest

_CODECS_DIR = pathlib.Path(__file__).resolve().parents[1] / "src" / "opencodecs" / "codecs"


def _listed_extensions() -> list[str]:
    src = (_CODECS_DIR / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r"_EXTENSIONS\s*=\s*\((.*?)\n\)", src, re.S)
    assert m, "could not locate the _EXTENSIONS tuple"
    return re.findall(r'"(_[A-Za-z0-9_]+)"', m.group(1))


def _built_extensions() -> list[str]:
    # Skip macOS AppleDouble sidecars (``._foo.pyx``), which the SMB mount
    # this repo lives on creates next to every real file.
    return sorted(
        p.name.split(".")[0]
        for p in _CODECS_DIR.glob("*.pyx")
        if not p.name.startswith(".")
    )


@pytest.mark.skipif(not _CODECS_DIR.is_dir(), reason="source tree not available")
def test_every_compiled_extension_is_shadow_loaded():
    """A .pyx with no _EXTENSIONS entry loads straight off the mount."""
    missing = sorted(set(_built_extensions()) - set(_listed_extensions()))
    assert not missing, (
        "these extensions are compiled but missing from _EXTENSIONS in "
        f"opencodecs/codecs/__init__.py, so they bypass the shadow-copy "
        f"loader and can hang on a network mount: {missing}"
    )


@pytest.mark.skipif(not _CODECS_DIR.is_dir(), reason="source tree not available")
def test_no_stale_extension_names_listed():
    """A listed name with no .pyx is a leftover from a rename."""
    stale = sorted(set(_listed_extensions()) - set(_built_extensions()))
    assert not stale, (
        "these names are listed in _EXTENSIONS but have no .pyx source, so "
        f"they are leftovers from a rename and only ever load stale build/ "
        f"artifacts: {stale}"
    )


@pytest.mark.skipif(not _CODECS_DIR.is_dir(), reason="source tree not available")
def test_extension_list_has_no_duplicates():
    listed = _listed_extensions()
    dupes = sorted({n for n in listed if listed.count(n) > 1})
    assert not dupes, f"duplicate _EXTENSIONS entries: {dupes}"
