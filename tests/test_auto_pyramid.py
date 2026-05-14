"""Auto-pyramid tests — opt-in convenience wrappers.

Verifies that:
* :func:`make_pyramid_levels` builds correct levels, stops at min_size,
  preserves dtype, handles 2D/3D, supports custom axes.
* :func:`write_omezarr_pyramid_auto` produces a valid OME-NGFF pyramid
  readable by zarr-python with level 0 byte-equal to input.
* :meth:`TiffWriter.write_pyramid_auto` produces a SubIFD-style
  pyramid readable by tifffile.
* Single-level (no downsample needed) inputs produce 1-level pyramids
  without surprise size bloat.
"""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest

from opencodecs._pyramid_build import make_pyramid_levels


# ---------------------------------------------------------------------------
# make_pyramid_levels — building blocks
# ---------------------------------------------------------------------------


def test_pyramid_default_stops_at_min_size():
    """Default min_size=512 caps levels conservatively."""
    arr = np.zeros((2048, 2048), dtype=np.uint16)
    lv = make_pyramid_levels(arr)
    # 2048 → 1024 → 512 (next would be 256 < min_size).
    assert [a.shape for a in lv] == [(2048, 2048), (1024, 1024), (512, 512)]


def test_pyramid_explicit_levels_overrides_min_size():
    arr = np.zeros((2048, 2048), dtype=np.uint16)
    lv = make_pyramid_levels(arr, levels=5)
    assert len(lv) == 5
    assert lv[-1].shape == (128, 128)


def test_pyramid_small_input_single_level():
    """An input already smaller than min_size yields just 1 level."""
    arr = np.zeros((300, 300), dtype=np.uint8)
    lv = make_pyramid_levels(arr)
    assert len(lv) == 1
    assert lv[0].shape == arr.shape


def test_pyramid_dtype_preserved():
    for dtype in (np.uint8, np.uint16, np.uint32, np.int16, np.float32, np.float64):
        arr = np.random.default_rng(0).integers(0, 100, (1024, 1024)).astype(dtype)
        lv = make_pyramid_levels(arr)
        assert all(a.dtype == arr.dtype for a in lv), \
            f"dtype changed for {dtype}: got {[a.dtype for a in lv]}"


def test_pyramid_zyx_only_downsamples_last_two():
    """For a 3D array, default downsamples Y, X (not Z)."""
    arr = np.zeros((8, 1024, 1024), dtype=np.uint16)
    lv = make_pyramid_levels(arr)
    assert all(a.shape[0] == 8 for a in lv)
    assert lv[0].shape == (8, 1024, 1024)
    assert lv[1].shape == (8, 512, 512)


def test_pyramid_axes_zyx_includes_depth():
    """Passing axes='zyx' downsamples all 3 axes."""
    arr = np.zeros((8, 1024, 1024), dtype=np.uint16)
    lv = make_pyramid_levels(arr, axes="zyx", min_size=4)
    assert lv[1].shape == (4, 512, 512)


def test_pyramid_level0_is_input_unchanged():
    arr = np.random.default_rng(0).integers(0, 1000, (256, 256), dtype=np.uint16)
    lv = make_pyramid_levels(arr, levels=3)
    # Level 0 IS the input (not a copy) — saves memory + ensures byte-equal.
    assert lv[0] is arr


def test_pyramid_mean_pool_within_1ulp():
    """Level 1 is approximately a 2x2 mean of level 0 within ±1 from
    round-to-nearest rounding (per-axis vs full-block ordering can
    differ by 1)."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 10000, (1024, 1024), dtype=np.uint16)
    lv = make_pyramid_levels(arr, levels=2)
    full_block_mean = arr.reshape(512, 2, 512, 2).mean(axis=(1, 3))
    diff = np.abs(lv[1].astype(np.int32) - full_block_mean.astype(np.int32))
    assert diff.max() <= 1


def test_pyramid_rejects_1d_input():
    with pytest.raises(ValueError, match="at least 2D"):
        make_pyramid_levels(np.zeros(100))


def test_pyramid_rejects_unsupported_downsample():
    with pytest.raises(ValueError, match="downsample=2"):
        make_pyramid_levels(np.zeros((64, 64)), downsample=4)


# ---------------------------------------------------------------------------
# write_omezarr_pyramid_auto
# ---------------------------------------------------------------------------


def test_omezarr_auto_pyramid_writes_level_zero_byte_equal(tmp_path):
    zarr = pytest.importorskip("zarr")
    from opencodecs._omezarr_writer import write_omezarr_pyramid_auto

    arr = np.random.default_rng(0).integers(0, 1000, (1024, 1024), dtype=np.uint16)
    write_omezarr_pyramid_auto(tmp_path / "pyr", arr, zarr_format=2)

    # zarr-python reads each level.
    levels_on_disk = sorted(
        int(p.name) for p in (tmp_path / "pyr").iterdir() if p.name.isdigit()
    )
    assert levels_on_disk == [0, 1]   # 1024 → 512 (stops at min_size=512)
    z0 = zarr.open(str(tmp_path / "pyr" / "0"), mode="r")
    np.testing.assert_array_equal(z0[:], arr)


def test_omezarr_auto_pyramid_explicit_levels(tmp_path):
    """pyramid_levels=N forces depth even past min_size."""
    zarr = pytest.importorskip("zarr")
    from opencodecs._omezarr_writer import write_omezarr_pyramid_auto

    arr = np.zeros((2048, 2048), dtype=np.uint8)
    write_omezarr_pyramid_auto(
        tmp_path / "pyr", arr, pyramid_levels=4, zarr_format=2,
    )
    levels_on_disk = sorted(
        int(p.name) for p in (tmp_path / "pyr").iterdir() if p.name.isdigit()
    )
    assert levels_on_disk == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# TiffWriter.write_pyramid_auto
# ---------------------------------------------------------------------------


def test_tiff_pyramid_auto_subifd_layout(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    from opencodecs._tiff_writer import TiffWriter

    arr = np.random.default_rng(0).integers(0, 1000, (2048, 2048), dtype=np.uint16)
    out_path = tmp_path / "pyr.tif"
    with TiffWriter(str(out_path)) as w:
        infos = w.write_pyramid_auto(
            arr, tile=(256, 256),
            compression="zstd", compression_level=3,
            subifds=True,
        )
    # 3 levels: 2048, 1024, 512.
    assert len(infos) == 3

    tf = tifffile.TiffFile(str(out_path))
    # subifds=True → only 1 top-level page.
    assert len(tf.pages) == 1
    main = tf.pages[0]
    assert main.shape == (2048, 2048)
    # SubIFDs hold the reduced-resolution levels.
    if hasattr(main, "pages"):
        sub_shapes = [p.shape for p in main.pages]
        assert sub_shapes == [(1024, 1024), (512, 512)]


def test_tiff_pyramid_auto_cog_layout(tmp_path):
    """``subifds=False`` (default) writes a COG-style pyramid where each
    level is a separate top-level IFD."""
    tifffile = pytest.importorskip("tifffile")
    from opencodecs._tiff_writer import TiffWriter

    arr = np.zeros((2048, 2048), dtype=np.uint16)
    out_path = tmp_path / "pyr.tif"
    with TiffWriter(str(out_path)) as w:
        infos = w.write_pyramid_auto(arr, subifds=False)
    assert len(infos) == 3

    tf = tifffile.TiffFile(str(out_path))
    assert len(tf.pages) == 3
    assert [p.shape for p in tf.pages] == [(2048, 2048), (1024, 1024), (512, 512)]


def test_tiff_pyramid_auto_size_overhead_reasonable(tmp_path):
    """Pyramid should add roughly +33% (geometric series for 2D) on
    top of the full-res page — verify we're not accidentally writing
    a 10x-larger file."""
    from opencodecs._tiff_writer import TiffWriter

    arr = np.random.default_rng(0).integers(0, 1000, (2048, 2048), dtype=np.uint16)

    pyr_path = tmp_path / "pyr.tif"
    with TiffWriter(str(pyr_path)) as w:
        w.write_pyramid_auto(
            arr, tile=(256, 256),
            compression="zstd", compression_level=3, subifds=True,
        )

    plain_path = tmp_path / "plain.tif"
    with TiffWriter(str(plain_path)) as w:
        w.write_page(arr, tile=(256, 256),
                     compression="zstd", compression_level=3)

    ratio = pyr_path.stat().st_size / plain_path.stat().st_size
    # Realistic compressed pyramid is 1.2-1.5x full-res depending on
    # content. Anything >2x is a regression.
    assert 1.0 < ratio < 1.6, f"unexpected pyramid size overhead: {ratio:.2f}x"
