"""A multi-page TIFF reads into one preallocated stack.

Pages of one shape and dtype are decoded into slices of a single output
instead of separately and then stacked, a second copy of everything. A
stack stored as plain pixels, end to end in the file, is read as one span
(``_read_stored_stack_into``). These tests pin both paths to tifffile's
answer, and check that each is taken when it should be and not otherwise:
reading one span where the pages are not contiguous would return the wrong
pixels without raising.
"""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

import opencodecs as oc
from opencodecs import _tiff_codec


def _stack(n=5, h=37, w=29, dtype=np.uint16, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, np.iinfo(dtype).max, (n, h, w), dtype=dtype)


def _spy(monkeypatch):
    calls = []
    cls = next(c for c in vars(_tiff_codec).values()
               if isinstance(c, type) and hasattr(c, "_read_stored_stack_into"))
    orig = cls._read_stored_stack_into

    def wrapped(self, pages, out, numthreads):
        took = orig(self, pages, out, numthreads)
        calls.append(took)
        return took

    monkeypatch.setattr(cls, "_read_stored_stack_into", wrapped)
    return calls


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_contiguous_uncompressed_stack_is_one_span(tmp_path, monkeypatch, dtype):
    data = _stack(dtype=np.uint16).astype(dtype)
    path = tmp_path / "stack.tif"
    tifffile.imwrite(path, data, photometric="minisblack", contiguous=True)
    calls = _spy(monkeypatch)
    got = oc.read(path, format="tiff")
    np.testing.assert_array_equal(got, tifffile.imread(path))
    assert calls == [True]


def test_multi_strip_pages_are_one_span(tmp_path, monkeypatch):
    data = _stack(n=4, h=64, w=33)
    path = tmp_path / "strips.tif"
    tifffile.imwrite(path, data, photometric="minisblack", rowsperstrip=7, contiguous=True)
    calls = _spy(monkeypatch)
    np.testing.assert_array_equal(oc.read(path, format="tiff"), data)
    assert calls == [True]


def test_pages_not_end_to_end_take_the_page_path(tmp_path, monkeypatch):
    # Separate series: each page's data follows its own IFD, so the pixel
    # bytes are not one span and must be read page by page.
    data = _stack(n=3)
    path = tmp_path / "separate.tif"
    with tifffile.TiffWriter(path) as w:
        for page in data:
            w.write(page, photometric="minisblack", contiguous=False)
    calls = _spy(monkeypatch)
    np.testing.assert_array_equal(oc.read(path, format="tiff"), data)
    assert calls == [False]


@pytest.mark.parametrize("compression", ["zlib", "lzw", "zstd"])
def test_compressed_stack_decodes_into_slices(tmp_path, monkeypatch, compression):
    data = _stack(n=4, h=50, w=41)
    path = tmp_path / f"{compression}.tif"
    tifffile.imwrite(path, data, photometric="minisblack", compression=compression)
    calls = _spy(monkeypatch)
    np.testing.assert_array_equal(oc.read(path, format="tiff"), data)
    assert calls == [False]


def test_pages_of_different_shapes_still_stack_or_raise_as_before(tmp_path):
    path = tmp_path / "mixed.tif"
    with tifffile.TiffWriter(path) as w:
        w.write(np.zeros((8, 9), np.uint16), photometric="minisblack")
        w.write(np.ones((8, 9), np.uint16), photometric="minisblack")
    np.testing.assert_array_equal(oc.read(path, format="tiff"),
                                  np.stack([np.zeros((8, 9)), np.ones((8, 9))]))


def test_asarray_out_rejects_a_wrong_destination(tmp_path):
    path = tmp_path / "one.tif"
    tifffile.imwrite(path, np.arange(12, dtype=np.uint16).reshape(3, 4))
    with oc.get_codec("tiff").open(path) as r:
        page = r.page(0)
        with pytest.raises(ValueError):
            page.asarray(out=np.empty((4, 3), np.uint16))
        with pytest.raises(ValueError):
            page.asarray(out=np.empty((3, 4), np.int32))
        out = np.empty((3, 4), np.uint16)
        assert page.asarray(out=out) is out
        np.testing.assert_array_equal(out, np.arange(12).reshape(3, 4))
