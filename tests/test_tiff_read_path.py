"""How TIFF gets its bytes, and how it interprets them.

Two changes this file exists to hold in place.

The reader maps local files instead of read()ing each segment into a
fresh bytes object. That removes a copy, and it introduces a hazard
worth testing rather than trusting: an array that views the mapping is
valid only while the reader is open, and handing one to a caller is a
use-after-free that surfaces as garbage pixels much later.

And `_bytes_to_array` used to decide whether to byte-swap by comparing
numpy's byte-order characters, where native is "=" and explicit little
is "<". On a little-endian machine reading a little-endian file -- the
common case by a wide margin -- that read as a mismatch and paid a full
swapping copy per tile that changed nothing. Removing it must not
remove the swap where one is genuinely needed, which is what most of
these tests check.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
import tifffile

import opencodecs as oc
from opencodecs._tiff_codec import (
    _byteorder_differs, _resolved_byteorder,
)


def _img(h=200, w=300, dtype=np.uint16, seed=0):
    rng = np.random.default_rng(seed)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return rng.integers(info.min // 2 or 0, info.max // 2,
                            size=(h, w)).astype(dtype)
    return (rng.random((h, w)) * 1000).astype(dtype)


# ---- byte order -----


def test_resolved_byteorder_maps_native_to_a_real_order():
    native = _resolved_byteorder(np.dtype("u2"))
    assert native in ("<", ">")
    assert _resolved_byteorder(np.dtype("<u2")) == "<"
    assert _resolved_byteorder(np.dtype(">u2")) == ">"
    # Single-byte types report "|"; it must resolve, not leak through.
    assert _resolved_byteorder(np.dtype("u1")) in ("<", ">")


def test_byteorder_differs_only_when_a_swap_is_needed():
    assert _byteorder_differs(np.dtype("<u2"), np.dtype(">u2")) is True
    assert _byteorder_differs(np.dtype(">u2"), np.dtype("<u2")) is True
    assert _byteorder_differs(np.dtype("<u2"), np.dtype("<u2")) is False
    # The case that used to cost a copy per tile: explicit vs native,
    # on a machine where they are the same order.
    same = np.dtype("u2").newbyteorder(_resolved_byteorder(np.dtype("u2")))
    assert _byteorder_differs(same, np.dtype("u2")) is False
    # One-byte types never need swapping, whatever they claim.
    assert _byteorder_differs(np.dtype("u1"), np.dtype(">u2")) is False


@pytest.mark.parametrize("byteorder", ["<", ">"])
@pytest.mark.parametrize("dtype", [np.uint16, np.int16, np.uint32,
                                   np.int32, np.float32, np.float64])
@pytest.mark.parametrize("compression", [None, "zlib", "zstd", "lzw"])
def test_both_byte_orders_survive_every_byte_stream_codec(
        byteorder, dtype, compression, tmp_path):
    """The swap decision is made once, in a path every codec shares.

    So it has to be right for each of them, in both orders. A file
    written big-endian and read on a little-endian machine that skips
    the swap returns numbers that are wrong by orders of magnitude
    while the shape and dtype look perfect.
    """
    arr = _img(dtype=dtype)
    buf = io.BytesIO()
    try:
        tifffile.imwrite(buf, arr, compression=compression,
                         byteorder=byteorder)
    except Exception as exc:                                  # noqa: BLE001
        pytest.skip(f"tifffile cannot write {compression}/{byteorder}: {exc}")
    # "<" and ">" are legal in a numpy byteorder and illegal in a
    # Windows filename, so name the file after what they mean. This
    # failed 96 times on Windows and nowhere else.
    tag = "le" if byteorder == "<" else "be"
    p = tmp_path / f"order_{tag}_{np.dtype(dtype).name}.tif"
    p.write_bytes(buf.getvalue())
    with oc.get_codec("tiff").open(str(p)) as r:
        np.testing.assert_array_equal(r.read(), arr)


def test_big_endian_tiled_multi_tile(tmp_path):
    """Tiles go through _bytes_to_array once each, so a swap that is
    applied per segment has more chances to be wrong."""
    arr = _img(512, 512, np.uint16)
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, byteorder=">", tile=(128, 128),
                     compression="zlib")
    p = tmp_path / "be_tiled.tif"
    p.write_bytes(buf.getvalue())
    for nt in (1, 8):
        with oc.get_codec("tiff").open(str(p), numthreads=nt) as r:
            np.testing.assert_array_equal(r.read(), arr)


# ---- the mapped read path -----


def test_local_files_are_mapped(tmp_path):
    arr = _img()
    p = tmp_path / "m.tif"
    with oc.TiffWriter(str(p)) as w:
        w.write_page(arr, compression="zstd")
    with oc.get_codec("tiff").open(str(p)) as r:
        assert r._mmap is not None, "local file was not mapped"
        # The mapping also gives parse_ifd_chain its buffer fast path.
        assert getattr(r._read, "_buf", None) is not None


def test_bytes_and_file_objects_do_not_map(tmp_path):
    arr = _img()
    p = tmp_path / "b.tif"
    with oc.TiffWriter(str(p)) as w:
        w.write_page(arr, compression="zstd")
    raw = p.read_bytes()
    with oc.get_codec("tiff").open(raw) as r:
        assert r._mmap is None
        np.testing.assert_array_equal(r.read(), arr)
    with p.open("rb") as fh, oc.get_codec("tiff").open(fh) as r:
        assert r._mmap is None
        np.testing.assert_array_equal(r.read(), arr)


@pytest.mark.parametrize("compression", ["none", "zstd", "deflate", "lzw"])
@pytest.mark.parametrize("tile", [None, (128, 128)])
def test_decoded_pixels_outlive_the_reader(tmp_path, compression, tile):
    """The whole point of the copy in the single-segment fast path.

    ascontiguousarray returns its input untouched when that input is
    already contiguous, and for a mapped uncompressed page that input
    is a window into the file. Closing the reader would then pull the
    memory out from under the array the caller is holding. Reading the
    values after close() is what catches it.
    """
    arr = _img(256, 256, np.uint16)
    p = tmp_path / f"o_{compression}_{tile}.tif"
    with oc.TiffWriter(str(p)) as w:
        w.write_page(arr, compression=compression,
                     **({"tile": tile} if tile else {}))
    with oc.get_codec("tiff").open(str(p)) as r:
        got = r.read()
    # Reader closed; the array must still be ours.
    assert got.base is None or not isinstance(got.base, memoryview)
    np.testing.assert_array_equal(got, arr)
    assert int(got.sum()) == int(arr.sum())


def test_close_is_idempotent_and_safe(tmp_path):
    arr = _img()
    p = tmp_path / "c.tif"
    with oc.TiffWriter(str(p)) as w:
        w.write_page(arr, compression="zstd")
    r = oc.get_codec("tiff").open(str(p))
    r.read()
    r.close()
    r.close()          # must not raise on a second call
    assert r._mmap is None


def test_falls_back_when_mapping_is_refused(tmp_path, monkeypatch):
    """Not every filesystem will map. The reader has to still work.

    Some network mounts refuse mmap, and an empty file cannot be
    mapped at all -- neither should be a decode failure.
    """
    import mmap as mmap_mod
    from opencodecs import _tiff_codec

    arr = _img()
    p = tmp_path / "nomap.tif"
    with oc.TiffWriter(str(p)) as w:
        w.write_page(arr, compression="zstd")

    def refuse(*a, **k):
        raise OSError("mapping refused")

    monkeypatch.setattr(_tiff_codec.mmap, "mmap", refuse)
    with oc.get_codec("tiff").open(str(p)) as r:
        assert r._mmap is None
        np.testing.assert_array_equal(r.read(), arr)
