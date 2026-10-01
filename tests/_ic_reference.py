"""The imagecodecs release the compatibility tests compare against.

The codec behavior those tests pin is imagecodecs 2026.8.16's: its
parameter meanings, defaults and output layouts. Older releases differ in
some of them, and some crash on the edge cases the tests probe (a 0-d
array into delta_encode, an index into avif_decode), so a test file that
compares against imagecodecs is skipped when an older one is installed.
Without imagecodecs the files still run what does not need it.
"""
from __future__ import annotations

import importlib.metadata

import pytest
from packaging.version import Version

IMAGECODECS_REFERENCE = "2026.8.16"


def _installed() -> str | None:
    try:
        return importlib.metadata.version("imagecodecs")
    except importlib.metadata.PackageNotFoundError:
        return None


_version = _installed()

skip_if_old_imagecodecs = pytest.mark.skipif(
    _version is not None and Version(_version) < Version(IMAGECODECS_REFERENCE),
    reason=f"compares against imagecodecs {IMAGECODECS_REFERENCE} or later; "
           f"{_version} is installed")
