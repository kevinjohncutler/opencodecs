"""Whether tifffile will run codecs that opencodecs.tifffile_patch installs.

tifffile 2026.8.23 and later check each codec function it looks up and
raise RuntimeError when its module is neither ``imagecodecs`` nor
``tifffile``, so reads and writes through the patch fail there. Tests that
go through tifffile with the patch active are skipped on such a tifffile;
tests that call the patch's functions directly still run.
"""
from __future__ import annotations

import io

import numpy as np
import pytest


def tifffile_accepts_patched_codecs() -> bool:
    try:
        import tifffile
        from opencodecs import tifffile_patch as patch
    except ImportError:
        return False
    try:
        with patch.patched():
            buf = io.BytesIO()
            tifffile.imwrite(buf, np.zeros((8, 8), np.uint8), compression="zlib")
            buf.seek(0)
            tifffile.imread(buf)
    except RuntimeError:
        return False
    return True


requires_patchable_tifffile = pytest.mark.skipif(
    not tifffile_accepts_patched_codecs(),
    reason="this tifffile refuses codecs from modules other than imagecodecs "
           "and tifffile, so opencodecs.tifffile_patch cannot take effect")
