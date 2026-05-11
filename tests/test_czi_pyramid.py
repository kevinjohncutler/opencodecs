"""CZI pyramid reader tests.

CZI pyramids are encoded implicitly: each sub-block carries a logical
``shape`` (full-resolution coordinate extent) and a ``stored_shape``
(actual pixel grid). Sub-blocks with ``shape != stored_shape`` are
downscaled, and the ratio is the pyramid level's scale factor.

We don't have a real multiscale CZI in the corpus, so the pyramid
tests exercise the level-grouping logic on synthesized
``CziSubBlockEntry`` lists. The non-pyramid path is exercised end-to-
end via the existing CZI fixture so we know the new code didn't
regress the flat-file case.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._czi_reader import (
    CziPyramidReader, CziReader, CziSubBlockEntry,
)

pytestmark = pytest.mark.skipif(
    not oc.has_codec("czi"),
    reason="czi codec requires native zstd + bytetools extensions",
)


# ---------------------------------------------------------------------------
# Entry-level property tests — no real CZI needed
# ---------------------------------------------------------------------------


def _mk_entry(shape, stored_shape, *, dims=("Y", "X", "S"), start=None,
              pyramid_type=0):
    if start is None:
        start = tuple(0 for _ in shape)
    return CziSubBlockEntry(
        file_position=0,
        pixel_type=0,         # u8
        compression=0,
        dimensions_count=len(dims),
        dims=tuple(dims),
        shape=tuple(shape),
        stored_shape=tuple(stored_shape),
        start=tuple(start),
        mosaic_index=-1,
        scene_index=-1,
        storage_size=0,
        pyramid_type=pyramid_type,
    )


def test_entry_is_pyramid_flag():
    full = _mk_entry((512, 512, 1), (512, 512, 1))
    half = _mk_entry((512, 512, 1), (256, 256, 1))
    assert not full.is_pyramid
    assert half.is_pyramid


def test_entry_scale_factors():
    half = _mk_entry((512, 512, 1), (256, 256, 1))
    quarter = _mk_entry((512, 512, 1), (128, 128, 1))
    assert half.scale_factors == (2.0, 2.0, 1.0)
    assert quarter.scale_factors == (4.0, 4.0, 1.0)


# ---------------------------------------------------------------------------
# Non-pyramid CZI via existing fixture — make sure new field doesn't break it
# ---------------------------------------------------------------------------


def test_non_pyramidal_czi_reports_flat(tmp_path):
    """Ordinary single-resolution CZI should report is_pyramidal=False
    and have a single scale level."""
    from _czi_fixture import czi_bytes  # type: ignore[import-not-found]
    arr = np.random.RandomState(3).randint(0, 256, (32, 48), dtype=np.uint8)
    data = czi_bytes(arr, compression=0)
    with oc.get_codec("czi").open(data) as r:
        assert r.is_pyramidal is False
        assert r.scale_factors_per_level() == [(1.0, 1.0)]
        # All entries belong to level 0 (only level).
        assert len(r.entries_at_level(0)) == len(r.entries)


def test_czi_pyramid_reader_wraps_flat_file(tmp_path):
    """CziPyramidReader on a flat CZI exposes one level."""
    from _czi_fixture import czi_bytes  # type: ignore[import-not-found]
    arr = np.random.RandomState(4).randint(0, 256, (32, 48), dtype=np.uint8)
    data = czi_bytes(arr, compression=0)
    czi = oc.get_codec("czi").open(data)
    p = CziPyramidReader(czi)
    try:
        assert p.n_levels == 1
        L = p.level(0)
        assert L.downscale == (1, 1)
        assert L.shape == (32, 48)
        # Full read should round-trip
        out = p.read_region(0, y=(0, 32), x=(0, 48))
        np.testing.assert_array_equal(out, arr)
    finally:
        p.close()


# ---------------------------------------------------------------------------
# Synthetic pyramid: directly construct entries to test grouping logic
# ---------------------------------------------------------------------------


class _StubCzi:
    """Stand-in for CziReader that returns canned entries — lets us
    test pyramid grouping without forging a real ZISRAW byte stream.
    """

    def __init__(self, entries):
        self.entries = list(entries)
        self.is_pyramidal = any(e.is_pyramid for e in entries)
        # The first entry's dims drive the level-grouping axes lookup;
        # mirror CziReader's behaviour.
        self._ref = entries[0] if entries else None

    def scale_factors_per_level(self, axes=("Y", "X")):
        return CziReader.scale_factors_per_level(self, axes=axes)

    def entries_at_level(self, level: int = 0, *, axes=("Y", "X")):
        return CziReader.entries_at_level(self, level, axes=axes)

    def close(self):
        pass

    def _decode_one(self, entry):
        # Return synthetic pixels: a unique value per entry so each
        # tile is identifiable in assembled output.
        h, w = entry.stored_shape[0], entry.stored_shape[1]
        # Use the file_position as the tag so callers can assert which
        # tile landed where.
        return np.full((h, w), entry.file_position, dtype=np.uint8)


def test_scale_factors_per_level_finds_3_levels():
    entries = []
    # Level 0: 4 full-res tiles
    for ty in (0, 256):
        for tx in (0, 256):
            entries.append(_mk_entry(
                shape=(256, 256, 1),
                stored_shape=(256, 256, 1),
                start=(ty, tx, 0),
            ))
    # Level 1: 1 half-res tile covering the same logical extent
    entries.append(_mk_entry(
        shape=(512, 512, 1),
        stored_shape=(256, 256, 1),
        start=(0, 0, 0),
    ))
    # Level 2: 1 quarter-res tile
    entries.append(_mk_entry(
        shape=(512, 512, 1),
        stored_shape=(128, 128, 1),
        start=(0, 0, 0),
    ))
    czi = _StubCzi(entries)
    scales = czi.scale_factors_per_level()
    assert scales == [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)]
    assert len(czi.entries_at_level(0)) == 4
    assert len(czi.entries_at_level(1)) == 1
    assert len(czi.entries_at_level(2)) == 1


def test_synthetic_pyramid_reader_reads_correct_tile():
    """A synthetic 2-tile single-level pyramid should reconstruct
    correctly via read_region."""
    e0 = _mk_entry(
        shape=(64, 64, 1), stored_shape=(64, 64, 1), start=(0, 0, 0),
    )
    e0.file_position = 100   # Tag value for the stub decode
    e1 = _mk_entry(
        shape=(64, 64, 1), stored_shape=(64, 64, 1), start=(0, 64, 0),
    )
    e1.file_position = 200
    czi = _StubCzi([e0, e1])
    p = CziPyramidReader(czi)
    try:
        assert p.n_levels == 1
        assert p.level(0).shape == (64, 128)   # 2 tiles side-by-side
        # Full read should contain both tile-tag values
        out = p.read_region(0, y=(0, 64), x=(0, 128))
        assert out.shape == (64, 128)
        assert np.all(out[:, :64] == 100)
        assert np.all(out[:, 64:] == 200)
        # Cropped read across the seam
        crop = p.read_region(0, y=(20, 40), x=(50, 80))
        # Left half (50..64) = tile 0; right half (64..80) = tile 1.
        assert crop.shape == (20, 30)
        assert np.all(crop[:, :14] == 100)
        assert np.all(crop[:, 14:] == 200)
    finally:
        p.close()


def test_real_pyramid_fixture_round_trips_through_reader(tmp_path):
    """End-to-end: build a 3-level pyramid CZI fixture using the same
    binary layout ZEN writes, then read it back through our
    CziPyramidReader. The decoded pixels at every level must match the
    source arrays.

    This is the validation the synthetic-stub tests couldn't reach —
    here we put real CZI bytes through the full parse + decode path.
    """
    from _czi_fixture import pyramid_czi_bytes  # type: ignore[import-not-found]

    rng = np.random.RandomState(5)
    base = rng.randint(0, 256, (128, 128), dtype=np.uint8)
    half = base[::2, ::2].copy()
    quarter = base[::4, ::4].copy()
    levels = [base, half, quarter]

    data = pyramid_czi_bytes(levels, compression=0)

    czi = oc.get_codec("czi").open(data)
    assert czi.is_pyramidal is True
    assert czi.scale_factors_per_level() == [(1.0, 1.0), (2.0, 2.0), (4.0, 4.0)]

    p = CziPyramidReader(czi)
    try:
        assert p.n_levels == 3
        for i, lvl in enumerate(levels):
            assert p.level(i).shape == lvl.shape, (
                f"level {i}: pyramid shape {p.level(i).shape} != "
                f"source {lvl.shape}"
            )
            # Full read of each level
            out = p.read_region(
                i, y=(0, lvl.shape[0]), x=(0, lvl.shape[1]),
            )
            np.testing.assert_array_equal(out, lvl)
        # Crop from full-resolution
        crop = p.read_region(0, y=(10, 50), x=(20, 80))
        np.testing.assert_array_equal(crop, base[10:50, 20:80])
        # best_level_for picks the right level
        assert p.best_level_for(max_pixels_y=100) == 1
        assert p.best_level_for(max_pixels_y=40) == 2
    finally:
        p.close()


@pytest.mark.parametrize("compression", [0, 5, 6])
def test_real_pyramid_fixture_compressed(tmp_path, compression):
    """Pyramid fixture round-trips through compressed code paths too
    (uncompressed, raw zstd, ZSTDHDR — every supported CZI compression)."""
    from _czi_fixture import pyramid_czi_bytes  # type: ignore[import-not-found]

    rng = np.random.RandomState(6)
    base = rng.randint(0, 256, (64, 64), dtype=np.uint8)
    half = base[::2, ::2].copy()
    levels = [base, half]
    data = pyramid_czi_bytes(levels, compression=compression)

    czi = oc.get_codec("czi").open(data)
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


def test_pyramid_fixture_validated_by_czifile(tmp_path):
    """Cross-validate the fixture's pyramid layout. ``czifile`` (the
    independently-maintained reference Python CZI reader) must see the
    same pyramid_type / is_pyramid / shape / stored_shape values our
    reader does. If our fixture writes the bytes wrong, czifile would
    see a different layout — this test catches that.
    """
    czifile = pytest.importorskip("czifile")
    from _czi_fixture import pyramid_czi_bytes  # type: ignore[import-not-found]
    import io

    rng = np.random.RandomState(7)
    base = rng.randint(0, 256, (96, 96), dtype=np.uint8)
    half = base[::2, ::2].copy()
    quarter = base[::4, ::4].copy()
    levels = [base, half, quarter]
    raw = pyramid_czi_bytes(levels, compression=0)

    expected = [
        # (dims, shape, stored_shape, is_pyramid, pyramid_type)
        (("Y", "X", "S"), (96, 96, 1), (96, 96, 1), False, 0),
        (("Y", "X", "S"), (96, 96, 1), (48, 48, 1), True, 2),
        (("Y", "X", "S"), (96, 96, 1), (24, 24, 1), True, 2),
    ]
    with czifile.CziFile(io.BytesIO(raw)) as f:
        directory = list(f.subblock_directory)
        assert len(directory) == 3
        for e, exp in zip(directory, expected):
            dims, shape, stored, is_pyr, ptype = exp
            assert e.dims == dims
            assert e.shape == shape
            assert e.stored_shape == stored
            assert e.is_pyramid is is_pyr
            assert e.pyramid_type == ptype

    # Our own reader must agree with czifile.
    czi = oc.get_codec("czi").open(raw)
    assert czi.scale_factors_per_level() == [
        (1.0, 1.0), (2.0, 2.0), (4.0, 4.0),
    ]
    for ours, exp in zip(czi.entries, expected):
        _dims, _shape, _stored, is_pyr, ptype = exp
        assert ours.is_pyramid is is_pyr
        assert ours.pyramid_type == ptype


def test_pyramid_fixture_round_trips_through_pylibCZIrw(tmp_path):
    """A second independent decoder. ``pylibCZIrw`` is Zeiss's own
    libCZI Python wrapper — if our fixture bytes are wrong, this is
    where it would surface as a decode error or wrong pixel data.
    """
    pylibCZIrw = pytest.importorskip("pylibCZIrw")
    from _czi_fixture import pyramid_czi_bytes  # type: ignore[import-not-found]
    from pylibCZIrw import czi as pyczi

    rng = np.random.RandomState(8)
    base = rng.randint(0, 256, (64, 64), dtype=np.uint8)
    half = base[::2, ::2].copy()
    levels = [base, half]
    raw = pyramid_czi_bytes(levels, compression=0)

    # pylibCZIrw needs a file path; spill to disk.
    p = tmp_path / "fixture.czi"
    p.write_bytes(raw)

    with pyczi.open_czi(str(p)) as r:
        # Full-resolution read via the default plane.
        out = r.read(plane={"C": 0, "T": 0, "Z": 0})
        # libCZI returns (H, W, C). Squeeze the trailing dim.
        if out.ndim == 3 and out.shape[2] == 1:
            out = out[..., 0]
        np.testing.assert_array_equal(out, base)


def test_synthetic_pyramid_multi_level():
    """Two levels: full-res 4 tiles + 1 half-res tile. The pyramid
    reader should expose both."""
    entries = []
    for ty in (0, 32):
        for tx in (0, 32):
            entries.append(_mk_entry(
                shape=(32, 32, 1),
                stored_shape=(32, 32, 1),
                start=(ty, tx, 0),
            ))
    entries[0].file_position = 10
    entries[1].file_position = 20
    entries[2].file_position = 30
    entries[3].file_position = 40
    # Half-res — single tile covering 64x64 logical / 32x32 stored
    half = _mk_entry(
        shape=(64, 64, 1), stored_shape=(32, 32, 1), start=(0, 0, 0),
    )
    half.file_position = 99
    entries.append(half)
    czi = _StubCzi(entries)
    p = CziPyramidReader(czi)
    try:
        assert p.n_levels == 2
        assert p.level(0).downscale == (1, 1)
        assert p.level(1).downscale == (2, 2)
        # Half-res read returns the single 32x32 tile
        out = p.read_region(1)
        assert out.shape == (32, 32)
        assert np.all(out == 99)
        # best_level_for: 50x50 envelope → only half-res (32x32) fits
        assert p.best_level_for(max_pixels_y=50, max_pixels_x=50) == 1
    finally:
        p.close()
