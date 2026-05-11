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

import os

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


_REAL_PYRAMID_CZI = (
    "/Volumes/HiprDrive/2024-08-21_microarray/24-08-21_array_scan_raw.czi"
)

# OME-curated, publicly downloadable pyramid CZI:
# https://downloads.openmicroscopy.org/images/Zeiss-CZI/zenodo-10577186/
# (Originally from https://zenodo.org/records/10577186, CC-BY-4.0, ~505 MB,
# Axioscan slide scanner output processed in ZEN.) Cached locally under
# tests/.test_data so the test is fast and doesn't depend on the network.
_OME_PYRAMID_CZI = ".test_data/ome_zeiss_sample.czi"


@pytest.mark.skipif(
    not os.path.exists(_OME_PYRAMID_CZI),
    reason=(
        "OME public pyramid CZI sample not downloaded. To enable this "
        "test, run: curl -L -o tests/.test_data/ome_zeiss_sample.czi "
        "https://downloads.openmicroscopy.org/images/Zeiss-CZI/"
        "zenodo-10577186/2023_11_30__RecognizedCode-27.czi"
    ),
)
def test_pyramid_metadata_matches_czifile_on_public_ome_sample():
    """End-to-end metadata validation against a *publicly downloadable*
    ZEN-produced pyramid CZI from the OME sample collection.

    Unlike the lab-CZI test below (gated on a private NAS path), this
    one is reproducible by anyone — the same .czi file is available at
    OME's public download mirror. 481 sub-blocks across 6 pyramid
    levels (1x, 2x, 4x, 8x, 16x, 32x), JPEG-XR compressed.

    czifile is the reference Python CZI reader. For every sub-block in
    the file, our reader must agree with czifile on shape,
    stored_shape, is_pyramid, and pyramid_type.
    """
    czifile = pytest.importorskip("czifile")
    oc_r = oc.get_codec("czi").open(_OME_PYRAMID_CZI)
    try:
        with czifile.CziFile(_OME_PYRAMID_CZI) as cf:
            cf_by_pos = {e.file_position: e for e in cf.subblock_directory}
        assert len(oc_r.entries) == len(cf_by_pos)
        mismatches = []
        for oc_e in oc_r.entries:
            cf_e = cf_by_pos[oc_e.file_position]
            if (oc_e.shape != cf_e.shape
                    or oc_e.stored_shape != cf_e.stored_shape
                    or oc_e.is_pyramid != cf_e.is_pyramid
                    or oc_e.pyramid_type != cf_e.pyramid_type):
                mismatches.append((oc_e.file_position, oc_e, cf_e))
        assert not mismatches, (
            f"{len(mismatches)} sub-blocks disagree with czifile "
            f"(first: oc={mismatches[0][1]} vs cf={mismatches[0][2]})"
        )
        # 6 pyramid levels: 1, 2, 4, 8, 16, 32x
        assert oc_r.scale_factors_per_level() == [
            (1.0, 1.0), (2.0, 2.0), (4.0, 4.0),
            (8.0, 8.0), (16.0, 16.0), (32.0, 32.0),
        ]
    finally:
        oc_r.close()


@pytest.mark.skipif(
    not os.path.exists(_REAL_PYRAMID_CZI),
    reason="real ZEN-produced pyramid CZI not available (NAS mount)",
)
def test_pyramid_metadata_matches_czifile_on_real_zen_file():
    """Validate our pyramid metadata interpretation against a real
    ZEN-microscope-produced pyramid CZI.

    czifile is the canonical Python CZI reader. For every sub-block in
    the real 51 GB file (6288 sub-blocks across 7 pyramid levels), the
    pyramid-related fields opencodecs decodes must match the values
    czifile decodes from the same bytes. If we get the pyramid layout
    wrong on synthetic fixtures we might miss something; if we get it
    wrong on a real ZEN file the lab-data pipeline breaks. This is the
    gold-standard validation.
    """
    czifile = pytest.importorskip("czifile")
    oc_r = oc.get_codec("czi").open(_REAL_PYRAMID_CZI)
    try:
        with czifile.CziFile(_REAL_PYRAMID_CZI) as cf_r:
            cf_entries = list(cf_r.subblock_directory)
        # Map by file_position so we compare the same sub-block
        # regardless of internal iteration order.
        cf_by_pos = {e.file_position: e for e in cf_entries}
        assert len(oc_r.entries) == len(cf_entries)

        mismatches = []
        for oc_e in oc_r.entries:
            cf_e = cf_by_pos[oc_e.file_position]
            if (oc_e.shape != cf_e.shape
                    or oc_e.stored_shape != cf_e.stored_shape
                    or oc_e.is_pyramid != cf_e.is_pyramid
                    or oc_e.pyramid_type != cf_e.pyramid_type):
                mismatches.append((oc_e.file_position, oc_e, cf_e))
        assert not mismatches, (
            f"{len(mismatches)} sub-blocks disagree on pyramid metadata "
            f"(first: oc={mismatches[0][1]} vs cf={mismatches[0][2]})"
        )
        # Confirm our scale-factor-per-level enumeration matches: every
        # distinct (sy, sx) in the directory should be enumerated.
        cf_factors = set()
        for e in cf_entries:
            # czifile shape includes leading T/C/Z; the spatial axes are
            # the last two before the optional S axis.
            spatial_dims = [
                (s, ss) for d, s, ss in zip(e.dims, e.shape, e.stored_shape)
                if d in ("Y", "X")
            ]
            if len(spatial_dims) >= 2:
                cf_factors.add(tuple(
                    (s / ss) if ss > 0 else 1.0
                    for s, ss in spatial_dims
                ))
        oc_factors = set(oc_r.scale_factors_per_level())
        assert cf_factors == oc_factors, (
            f"scale-factor enumeration mismatch: cf={sorted(cf_factors)} "
            f"vs oc={sorted(oc_factors)}"
        )
    finally:
        oc_r.close()


@pytest.mark.skipif(
    not os.path.exists(_REAL_PYRAMID_CZI),
    reason="real ZEN-produced pyramid CZI not available (NAS mount)",
)
def test_pyramid_pixel_decode_matches_czifile_on_real_zen_file():
    """One sub-block per pyramid level decoded through both opencodecs
    and czifile; pixels must match.

    Bgr24 sub-blocks have a channel-order convention difference: our
    reader returns the file-storage order (B-G-R); czifile reorders to
    R-G-B. We assert pixel-equality after reversing channels — both
    are valid representations of the same data.
    """
    czifile = pytest.importorskip("czifile")
    oc_r = oc.get_codec("czi").open(_REAL_PYRAMID_CZI)
    try:
        # Sample one sub-block per available level
        levels = list(range(min(oc_r.n_levels if hasattr(oc_r, "n_levels")
                                else len(oc_r.scale_factors_per_level()),
                                3)))
        with czifile.CziFile(_REAL_PYRAMID_CZI) as cf_r:
            for lvl in levels:
                oc_e = oc_r.entries_at_level(lvl)[0]
                oc_pixels = np.squeeze(oc_r._decode_one(oc_e))
                cf_pixels = None
                for sb in cf_r.subblocks():
                    if sb.directory_entry.file_position == oc_e.file_position:
                        cf_pixels = np.squeeze(sb.data())
                        break
                assert cf_pixels is not None, (
                    f"czifile couldn't find sub-block at file_position "
                    f"{oc_e.file_position}"
                )
                # Bgr24 channel-order convention: oc returns BGR (file order),
                # czifile returns RGB. Compare after reversing the last axis.
                if oc_pixels.ndim == 3 and oc_pixels.shape[-1] == 3:
                    np.testing.assert_array_equal(
                        oc_pixels[..., ::-1], cf_pixels,
                        err_msg=f"level {lvl}: pixel mismatch after BGR↔RGB",
                    )
                else:
                    np.testing.assert_array_equal(
                        oc_pixels, cf_pixels,
                        err_msg=f"level {lvl}: pixel mismatch",
                    )
    finally:
        oc_r.close()


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
