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
