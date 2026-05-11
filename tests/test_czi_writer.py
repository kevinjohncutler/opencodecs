"""Public CziWriter / CziPyramidWriter round-trip + cross-validation.

These tests use *three* independent readers:

* opencodecs's own CziReader / CziPyramidReader
* ``czifile`` (Christoph Gohlke's reference Python implementation)
* ``pylibCZIrw`` (Zeiss's libCZI C++ bindings)

If our writer produces wrong bytes, all three should disagree with us
or with each other. When all three plus our reader agree on every
sub-block's metadata and the decoded pixels, the output is correct.
"""

from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc
from opencodecs import CziWriter, CziPyramidWriter, CziPyramidReader

pytestmark = pytest.mark.skipif(
    not oc.has_codec("czi"),
    reason="czi codec requires native zstd + bytetools extensions",
)


# ---------------------------------------------------------------------------
# CziWriter (single-resolution)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("compression", ["none", "zstd", "zstdhdr"])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
def test_writer_round_trips_through_opencodecs(tmp_path, compression, dtype):
    """Write via CziWriter, read via opencodecs CziReader — pixels match."""
    if dtype is np.float32:
        arr = np.random.default_rng(0).standard_normal((48, 64)).astype(dtype)
    else:
        info = np.iinfo(dtype)
        arr = np.random.default_rng(0).integers(
            0, info.max + 1, size=(48, 64),
        ).astype(dtype)
    out = tmp_path / "single.czi"
    with CziWriter(out, compression=compression) as w:
        w.write(arr)
    with oc.get_codec("czi").open(str(out)) as r:
        back = r.read()
    back = np.squeeze(back)
    np.testing.assert_array_equal(back, arr)


def test_writer_round_trips_through_czifile(tmp_path):
    """The same CZI bytes must decode correctly via czifile."""
    czifile = pytest.importorskip("czifile")
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(40, 80), dtype=np.uint8)
    out = tmp_path / "single_cf.czi"
    with CziWriter(out, compression="none") as w:
        w.write(arr)
    with czifile.CziFile(str(out)) as f:
        back = np.squeeze(f.asarray())
    np.testing.assert_array_equal(back, arr)


def test_writer_round_trips_through_pylibCZIrw(tmp_path):
    """And via Zeiss's own libCZI Python wrapper."""
    pylibCZIrw = pytest.importorskip("pylibCZIrw")
    from pylibCZIrw import czi as pyczi
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
    out = tmp_path / "single_pl.czi"
    with CziWriter(out, compression="none") as w:
        w.write(arr)
    with pyczi.open_czi(str(out)) as r:
        back = r.read(plane={"C": 0, "T": 0, "Z": 0})
    if back.ndim == 3 and back.shape[2] == 1:
        back = back[..., 0]
    np.testing.assert_array_equal(back, arr)


# ---------------------------------------------------------------------------
# CziPyramidWriter — the main event
# ---------------------------------------------------------------------------


@pytest.fixture
def _pyramid_levels():
    rng = np.random.default_rng(42)
    base = rng.integers(0, 256, size=(128, 128), dtype=np.uint8)
    return [base, base[::2, ::2].copy(), base[::4, ::4].copy()]


def test_pyramid_writer_round_trips_through_pyramid_reader(
    tmp_path, _pyramid_levels,
):
    """opencodecs writer → opencodecs pyramid reader: every level reads
    back identically."""
    out = tmp_path / "pyr.czi"
    with CziPyramidWriter(out) as w:
        w.write_pyramid(_pyramid_levels)
    czi = oc.get_codec("czi").open(str(out))
    p = CziPyramidReader(czi)
    try:
        assert p.n_levels == len(_pyramid_levels)
        for i, lvl in enumerate(_pyramid_levels):
            out_lvl = p.read_region(
                i, y=(0, lvl.shape[0]), x=(0, lvl.shape[1]),
            )
            np.testing.assert_array_equal(out_lvl, lvl)
    finally:
        p.close()


def test_pyramid_writer_layout_validated_by_czifile(
    tmp_path, _pyramid_levels,
):
    """The pyramid bytes must round-trip through czifile too — each
    sub-block's pyramid_type / stored_shape / shape must match what
    czifile sees."""
    czifile = pytest.importorskip("czifile")
    out = tmp_path / "pyr_cf.czi"
    with CziPyramidWriter(out) as w:
        w.write_pyramid(_pyramid_levels)
    expected_stored = [(128, 128), (64, 64), (32, 32)]
    with czifile.CziFile(str(out)) as f:
        directory = list(f.subblock_directory)
        assert len(directory) == 3
        for i, (e, exp_stored) in enumerate(
            zip(directory, expected_stored)
        ):
            # Logical shape is always 128x128 (level-0 extent).
            assert e.shape == (128, 128, 1), f"level {i}: shape {e.shape}"
            # Stored shape shrinks with each level.
            assert e.stored_shape == (*exp_stored, 1), (
                f"level {i}: stored {e.stored_shape}"
            )
            assert e.is_pyramid == (i > 0)
            assert e.pyramid_type == (0 if i == 0 else 2)


def test_pyramid_writer_decodes_via_pylibCZIrw_all_levels(
    tmp_path, _pyramid_levels,
):
    """Every pyramid level must round-trip pixel-equal through Zeiss's
    own libCZI reader. pylibCZIrw exposes per-level access via
    ``zoom=`` (1.0 = level 0 / full-res, 0.5 = level 1, etc.). If our
    pyramid layout is wrong libCZI would return blurred / interpolated
    pixels rather than the original stored values.
    """
    pytest.importorskip("pylibCZIrw")
    from pylibCZIrw import czi as pyczi
    out = tmp_path / "pyr_pl_all.czi"
    with CziPyramidWriter(out) as w:
        w.write_pyramid(_pyramid_levels)
    with pyczi.open_czi(str(out)) as r:
        for i, lvl in enumerate(_pyramid_levels):
            zoom = 1.0 / (2 ** i)
            back = r.read(zoom=zoom)
            if back.ndim == 3 and back.shape[2] == 1:
                back = back[..., 0]
            np.testing.assert_array_equal(
                back, lvl,
                err_msg=f"pylibCZIrw zoom={zoom} (level {i}) "
                        f"returned pixels that don't match the "
                        f"source level",
            )


def test_pyramid_writer_decodes_via_czifile_all_levels(
    tmp_path, _pyramid_levels,
):
    """Every pyramid level must round-trip pixel-equal through czifile's
    per-subblock decode path. czifile is the reference Python CZI
    reader; if our pyramid layout is wrong it'd either fail to decode
    or hand back wrong pixels."""
    czifile = pytest.importorskip("czifile")
    out = tmp_path / "pyr_cf_all.czi"
    with CziPyramidWriter(out) as w:
        w.write_pyramid(_pyramid_levels)
    with czifile.CziFile(str(out)) as f:
        sbs = list(f.subblocks())
        assert len(sbs) == len(_pyramid_levels)
        for i, (sb, lvl) in enumerate(zip(sbs, _pyramid_levels)):
            decoded = np.squeeze(sb.data())
            np.testing.assert_array_equal(
                decoded, lvl,
                err_msg=f"czifile sub-block {i} pixel-mismatch with "
                        f"source level {i}",
            )


@pytest.mark.parametrize("compression", ["none", "zstd", "zstdhdr"])
def test_pyramid_writer_compressed_round_trip(
    tmp_path, _pyramid_levels, compression,
):
    """All three CZI compression modes round-trip through the
    pyramid reader."""
    out = tmp_path / f"pyr_{compression}.czi"
    with CziPyramidWriter(out, compression=compression) as w:
        w.write_pyramid(_pyramid_levels)
    czi = oc.get_codec("czi").open(str(out))
    p = CziPyramidReader(czi)
    try:
        for i, lvl in enumerate(_pyramid_levels):
            out_lvl = p.read_region(
                i, y=(0, lvl.shape[0]), x=(0, lvl.shape[1]),
            )
            np.testing.assert_array_equal(out_lvl, lvl)
    finally:
        p.close()


def test_pyramid_writer_higher_bit_depth(tmp_path):
    """u16 and f32 pyramid levels round-trip correctly through both
    opencodecs and czifile."""
    czifile = pytest.importorskip("czifile")
    rng = np.random.default_rng(3)
    base = rng.integers(0, 4000, size=(64, 64), dtype=np.uint16)
    half = base[::2, ::2].copy()
    out = tmp_path / "u16.czi"
    with CziPyramidWriter(out) as w:
        w.write_pyramid([base, half])
    # opencodecs side
    czi = oc.get_codec("czi").open(str(out))
    p = CziPyramidReader(czi)
    try:
        np.testing.assert_array_equal(
            p.read_region(0, y=(0, 64), x=(0, 64)), base,
        )
        np.testing.assert_array_equal(
            p.read_region(1, y=(0, 32), x=(0, 32)), half,
        )
    finally:
        p.close()
    # czifile side
    with czifile.CziFile(str(out)) as f:
        assert f.subblock_directory[0].shape == (64, 64, 1)
        assert f.subblock_directory[1].stored_shape == (32, 32, 1)


def test_pyramid_writer_streaming_via_write_level(
    tmp_path, _pyramid_levels,
):
    """write_level() can be called incrementally; close() finalizes."""
    out = tmp_path / "stream.czi"
    w = CziPyramidWriter(out)
    try:
        for lvl in _pyramid_levels:
            w.write_level(lvl)
    finally:
        w.close()
    czi = oc.get_codec("czi").open(str(out))
    p = CziPyramidReader(czi)
    try:
        for i, lvl in enumerate(_pyramid_levels):
            np.testing.assert_array_equal(
                p.read_region(i, y=(0, lvl.shape[0]), x=(0, lvl.shape[1])),
                lvl,
            )
    finally:
        p.close()


def test_pyramid_writer_rejects_mixed_dtypes(tmp_path):
    """All levels must share dtype; we raise rather than silently
    casting."""
    base = np.zeros((64, 64), dtype=np.uint8)
    half = np.zeros((32, 32), dtype=np.uint16)
    out = tmp_path / "bad.czi"
    w = CziPyramidWriter(out)
    w.write_level(base)
    w.write_level(half)
    with pytest.raises(Exception):
        w.close()


def test_writer_rejects_3d_input(tmp_path):
    out = tmp_path / "bad.czi"
    with pytest.raises(Exception):
        with CziWriter(out) as w:
            w.write(np.zeros((4, 16, 16), dtype=np.uint8))
