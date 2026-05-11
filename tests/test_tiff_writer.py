"""TiffWriter round-trip tests.

Verifies that files produced by ``opencodecs.TiffWriter`` decode
cleanly via:

* opencodecs.TiffStream (our own reader)
* tifffile (the reference reader)

across all standard dtypes, layouts (strip vs tile), compressions,
predictors, and multi-page chains.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._tiff_codec import TiffStream
from opencodecs._tiff_writer import TiffWriter, imwrite as tiff_imwrite

tifffile = pytest.importorskip("tifffile")


def _need_tiff():
    if not oc.has_codec("tiff"):
        pytest.skip("native TIFF reader not built")


# ---------------------------------------------------------------------------
# Header / minimal round-trip
# ---------------------------------------------------------------------------


def test_writer_emits_valid_classic_tiff_header(tmp_path):
    p = tmp_path / "smoke.tif"
    arr = np.arange(64, dtype=np.uint16).reshape(8, 8)
    with TiffWriter(p) as w:
        w.write_page(arr)
    with open(p, "rb") as f:
        head = f.read(8)
    assert head[:2] == b"II"
    assert head[2:4] == b"\x2a\x00"           # magic 42 LE
    # First IFD offset must be non-zero (patched by write_page).
    import struct
    first_ifd = struct.unpack("<I", head[4:8])[0]
    assert first_ifd > 0


@pytest.mark.parametrize("dtype", [
    np.uint8, np.int8,
    np.uint16, np.int16,
    np.uint32, np.int32,
    np.uint64, np.int64,
    np.float32, np.float64,
])
def test_writer_dtype_roundtrip(tmp_path, dtype):
    _need_tiff()
    p = tmp_path / f"dt_{np.dtype(dtype).name}.tif"
    rng = np.random.default_rng(7)
    if np.issubdtype(dtype, np.floating):
        arr = rng.standard_normal((37, 53)).astype(dtype)
    else:
        # Generate random bytes and view as target dtype to avoid
        # numpy's int64-only random-int range for u64/i64.
        raw = rng.integers(0, 256, size=(37, 53, np.dtype(dtype).itemsize),
                           dtype=np.uint8)
        arr = raw.tobytes()
        arr = np.frombuffer(arr, dtype=dtype).reshape(37, 53).copy()

    with TiffWriter(p) as w:
        w.write_page(arr)

    # Our reader.
    with TiffStream(str(p)) as r:
        back = r.read()
    np.testing.assert_array_equal(back, arr)

    # tifffile parity.
    back_tf = tifffile.imread(str(p))
    np.testing.assert_array_equal(back_tf, arr)


# ---------------------------------------------------------------------------
# Strip / tile layouts
# ---------------------------------------------------------------------------


def test_writer_default_strips(tmp_path):
    _need_tiff()
    p = tmp_path / "strips.tif"
    arr = np.arange(64 * 80, dtype=np.uint16).reshape(64, 80)
    with TiffWriter(p) as w:
        info = w.write_page(arr)
    # default rps picks ~8KB strips; should produce multiple strips
    # for this image but not one strip per row.
    assert 1 < info["n_segments"] < 64
    with TiffStream(str(p)) as r:
        back = r.read()
        page = r.page(0)
    assert not page.is_tiled
    np.testing.assert_array_equal(back, arr)


def test_writer_rows_per_strip_explicit(tmp_path):
    _need_tiff()
    p = tmp_path / "rps.tif"
    arr = np.arange(100 * 50, dtype=np.uint16).reshape(100, 50)
    with TiffWriter(p) as w:
        info = w.write_page(arr, rows_per_strip=10)
    assert info["n_segments"] == 10
    with TiffStream(str(p)) as r:
        page = r.page(0)
        assert page.tile_height == 10
        back = r.read()
    np.testing.assert_array_equal(back, arr)
    np.testing.assert_array_equal(tifffile.imread(str(p)), arr)


def test_writer_tiled(tmp_path):
    _need_tiff()
    p = tmp_path / "tiled.tif"
    arr = np.arange(300 * 400, dtype=np.uint16).reshape(300, 400)
    with TiffWriter(p) as w:
        info = w.write_page(arr, tile=(128, 128))
    # (300/128, 400/128) = (3, 4) → 12 tiles
    assert info["n_segments"] == 12
    with TiffStream(str(p)) as r:
        page = r.page(0)
        assert page.is_tiled
        assert page.tile_width == 128 and page.tile_height == 128
        back = r.read()
    np.testing.assert_array_equal(back, arr)
    np.testing.assert_array_equal(tifffile.imread(str(p)), arr)


def test_writer_tile_dims_must_be_multiple_of_16(tmp_path):
    p = tmp_path / "bad.tif"
    arr = np.zeros((64, 64), dtype=np.uint8)
    with TiffWriter(p) as w:
        with pytest.raises(Exception):
            w.write_page(arr, tile=(17, 32))


def test_writer_rgb(tmp_path):
    _need_tiff()
    p = tmp_path / "rgb.tif"
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, size=(60, 90, 3), dtype=np.uint8)
    with TiffWriter(p) as w:
        w.write_page(arr)
    with TiffStream(str(p)) as r:
        page = r.page(0)
        assert page.samples_per_pixel == 3
        assert page.photometric == 2  # RGB
        back = r.read()
    np.testing.assert_array_equal(back, arr)
    np.testing.assert_array_equal(tifffile.imread(str(p)), arr)


# ---------------------------------------------------------------------------
# Compression
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compression", ["deflate", "zstd"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int32, np.float32])
def test_writer_byte_stream_compression_roundtrip(tmp_path, compression, dtype):
    _need_tiff()
    p = tmp_path / f"{compression}_{np.dtype(dtype).name}.tif"
    rng = np.random.default_rng(1)
    if np.issubdtype(dtype, np.floating):
        arr = rng.standard_normal((48, 64)).astype(dtype)
    else:
        info = np.iinfo(dtype)
        arr = rng.integers(
            max(info.min, -(1 << 30)),
            min(info.max, (1 << 30)),
            size=(48, 64),
        ).astype(dtype)
    with TiffWriter(p) as w:
        w.write_page(arr, compression=compression)
    with TiffStream(str(p)) as r:
        back = r.read()
    np.testing.assert_array_equal(back, arr)
    np.testing.assert_array_equal(tifffile.imread(str(p)), arr)


@pytest.mark.parametrize("compression", ["deflate", "zstd"])
def test_writer_compression_reduces_size(tmp_path, compression):
    p_raw = tmp_path / "raw.tif"
    p_cmp = tmp_path / f"{compression}.tif"
    # Highly compressible content: constant + low-variance noise.
    rng = np.random.default_rng(2)
    arr = (np.ones((256, 256), dtype=np.uint16) * 1000 +
           rng.integers(0, 4, size=(256, 256), dtype=np.uint16))
    with TiffWriter(p_raw) as w:
        w.write_page(arr)
    with TiffWriter(p_cmp) as w:
        w.write_page(arr, compression=compression)
    assert p_cmp.stat().st_size < p_raw.stat().st_size


@pytest.mark.parametrize("n_workers", [1, 2, 4])
def test_writer_parallel_encode_byte_identical(tmp_path, n_workers):
    """write_page with N parallel encoder threads must produce a file
    byte-identical to the serial path. Encodes run in parallel but
    the writer drains them in submission order — only scheduling
    changes, not the on-disk layout."""
    _need_tiff()
    rng = np.random.default_rng(42)
    arr = rng.integers(0, 4000, size=(1024, 1024), dtype=np.uint16)

    serial = tmp_path / "serial.tif"
    with TiffWriter(serial) as w:
        w.write_page(arr, tile=(256, 256), compression="zstd",
                     compression_level=1, n_workers=1)

    par = tmp_path / f"par_{n_workers}.tif"
    with TiffWriter(par) as w:
        w.write_page(arr, tile=(256, 256), compression="zstd",
                     compression_level=1, n_workers=n_workers)

    assert serial.read_bytes() == par.read_bytes()
    # Sanity: still readable
    with TiffStream(str(par)) as r:
        back = r.read()
    np.testing.assert_array_equal(back, arr)


def test_writer_horizontal_predictor_roundtrip(tmp_path):
    _need_tiff()
    p = tmp_path / "pred2.tif"
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 10000, size=(80, 120), dtype=np.uint16)
    with TiffWriter(p) as w:
        w.write_page(arr, compression="deflate", predictor=2)
    with TiffStream(str(p)) as r:
        page = r.page(0)
        assert page.predictor == 2
        back = r.read()
    np.testing.assert_array_equal(back, arr)
    np.testing.assert_array_equal(tifffile.imread(str(p)), arr)


def test_writer_predictor_shrinks_smooth_data(tmp_path):
    """Predictor 2 should compress better than predictor 1 on a
    smoothly varying image."""
    p1 = tmp_path / "p1.tif"
    p2 = tmp_path / "p2.tif"
    y, x = np.mgrid[0:256, 0:256]
    arr = (y * 8 + x * 4).astype(np.uint16)
    with TiffWriter(p1) as w:
        w.write_page(arr, compression="deflate", predictor=1)
    with TiffWriter(p2) as w:
        w.write_page(arr, compression="deflate", predictor=2)
    assert p2.stat().st_size < p1.stat().st_size


# ---------------------------------------------------------------------------
# Multi-page (pyramid-style)
# ---------------------------------------------------------------------------


def test_writer_multi_page(tmp_path):
    _need_tiff()
    p = tmp_path / "multi.tif"
    rng = np.random.default_rng(4)
    pages = [rng.integers(0, 255, size=(40, 50), dtype=np.uint8) for _ in range(5)]
    with TiffWriter(p) as w:
        for arr in pages:
            w.write_page(arr)
    with TiffStream(str(p)) as r:
        assert r.n_frames == 5
        for i, arr in enumerate(pages):
            np.testing.assert_array_equal(r[i], arr)
    # tifffile parity
    with tifffile.TiffFile(str(p)) as tf:
        assert len(tf.pages) == 5
        for i, arr in enumerate(pages):
            np.testing.assert_array_equal(tf.pages[i].asarray(), arr)


def test_writer_pyramid_convenience(tmp_path):
    _need_tiff()
    p = tmp_path / "pyr.tif"
    base = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    levels = [base, base[::2, ::2].copy(), base[::4, ::4].copy()]
    with TiffWriter(p) as w:
        w.write_pyramid(levels, tile=(64, 64), compression="zstd")
    with TiffStream(str(p)) as r:
        assert r.n_frames == 3
        for i, lvl in enumerate(levels):
            np.testing.assert_array_equal(r[i], lvl)
        # First page is the base; later pages flag NewSubfileType = 1.
        from opencodecs._tiff_writer import TAG_NEW_SUBFILE_TYPE
        assert r.page(1).tags.get(TAG_NEW_SUBFILE_TYPE) is not None
        # Tag values are (tag_type, count, value).
        assert int(r.page(1).tags[TAG_NEW_SUBFILE_TYPE][2]) == 1


# ---------------------------------------------------------------------------
# Convenience imwrite
# ---------------------------------------------------------------------------


def test_module_level_imwrite(tmp_path):
    _need_tiff()
    p = tmp_path / "one_shot.tif"
    arr = np.arange(40 * 60, dtype=np.uint8).reshape(40, 60)
    tiff_imwrite(p, arr, compression="zstd")
    with TiffStream(str(p)) as r:
        back = r.read()
    np.testing.assert_array_equal(back, arr)


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_writer_metadata_tag(tmp_path):
    _need_tiff()
    p = tmp_path / "md.tif"
    arr = np.zeros((8, 8), dtype=np.uint8)
    with TiffWriter(p) as w:
        w.write_page(arr, metadata="hello world", software="opencodecs-test")
    with tifffile.TiffFile(str(p)) as tf:
        page = tf.pages[0]
        assert page.tags["ImageDescription"].value == "hello world"
        assert page.tags["Software"].value == "opencodecs-test"


def test_writer_big_endian_roundtrip(tmp_path):
    _need_tiff()
    p = tmp_path / "be.tif"
    arr = np.arange(32 * 32, dtype=np.uint16).reshape(32, 32)
    with TiffWriter(p, byte_order=">") as w:
        w.write_page(arr)
    with TiffStream(str(p)) as r:
        back = r.read()
    np.testing.assert_array_equal(back, arr)
    np.testing.assert_array_equal(tifffile.imread(str(p)), arr)


# ---------------------------------------------------------------------------
# SubIFD-based pyramid write (bioformats / OME-TIFF convention)
# ---------------------------------------------------------------------------


def test_writer_pyramid_subifds_basic(tmp_path):
    """Write a 3-level pyramid in SubIFD layout, read back via our
    own SubIFD-aware reader."""
    _need_tiff()
    from opencodecs._tiff_pyramid import TiffPyramidReader
    p = tmp_path / "pyr_subifds.tif"
    base = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    levels = [base, base[::2, ::2].copy(), base[::4, ::4].copy()]
    with TiffWriter(p) as w:
        infos = w.write_pyramid(levels, tile=(64, 64), subifds=True)
    assert len(infos) == 3
    with TiffPyramidReader(str(p)) as r:
        assert r.n_levels == 3
        for i, lvl in enumerate(levels):
            assert r.level(i).shape == lvl.shape
            back = r.read_region(i, y=(0, lvl.shape[0]), x=(0, lvl.shape[1]))
            np.testing.assert_array_equal(back, lvl)


def test_writer_pyramid_subifds_top_level_chain_is_just_main(tmp_path):
    """SubIFD layout means only ONE top-level IFD — the sub-resolutions
    aren't reachable via the next-IFD chain. Confirms our writer
    actually produces the bioformats layout, not the COG layout."""
    _need_tiff()
    p = tmp_path / "subifd_chain.tif"
    levels = [
        np.arange(128 * 128, dtype=np.uint16).reshape(128, 128),
        np.arange(64 * 64, dtype=np.uint16).reshape(64, 64),
        np.arange(32 * 32, dtype=np.uint16).reshape(32, 32),
    ]
    with TiffWriter(p) as w:
        w.write_pyramid(levels, tile=(32, 32), subifds=True)
    # Walk just the top-level chain via TiffStream — should see 1, not 3.
    with TiffStream(str(p)) as r:
        assert r.n_frames == 1
        page = r.page(0)
        # And that one page should carry tag 330 with 2 sub-IFD offsets.
        from opencodecs._tiff_codec import TAG_SUB_IFDS
        sub_entry = page.tags.get(TAG_SUB_IFDS)
        assert sub_entry is not None
        # (dtype, count, value) — value can be int (count==1) or tuple.
        v = sub_entry[2]
        if isinstance(v, int):
            v = (v,)
        assert len(v) == 2, f"expected 2 sub-IFD offsets, got {v!r}"


def test_writer_pyramid_subifds_round_trips_through_tifffile(tmp_path):
    """The bioformats convention is what tifffile recognizes as
    ``series[0].levels``. After we write subifds=True, tifffile should
    expose 3 levels in series 0."""
    _need_tiff()
    p = tmp_path / "subifd_tff.tif"
    base = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)
    levels = [base, base[::2, ::2].copy(), base[::4, ::4].copy()]
    with TiffWriter(p) as w:
        w.write_pyramid(levels, tile=(32, 32), subifds=True)
    with tifffile.TiffFile(str(p)) as tf:
        # series 0 should expose 3 levels (the main + 2 sub-IFDs).
        s = tf.series[0]
        assert len(s.levels) == 3, (
            f"tifffile sees {len(s.levels)} levels; expected 3 — our "
            f"SubIFDs aren't being recognized as a pyramid"
        )
        for i, lvl in enumerate(levels):
            tf_lvl = np.asarray(s.levels[i].asarray())
            np.testing.assert_array_equal(tf_lvl, lvl)
