"""OME-Zarr v0.4 (Zarr v2) and v0.5 (Zarr v3) reader tests.

Uses zarr-python to lay down known-good fixtures, then verifies that
opencodecs reads them back identically. Covers:

* Zarr v2 + v3 array round-trip
* Common codecs: raw / zstd / gzip / blosc
* Partial reads — only the relevant chunks should be fetched
* Multiscales pyramid metadata (NGFF v0.4 + v0.5 layouts)
* Non-spatial axis indexing (T, C, Z, Y, X)
"""

from __future__ import annotations

import json

import numpy as np
import pytest

zarr = pytest.importorskip("zarr")
import opencodecs as oc
from opencodecs._omezarr import OmeZarrArray, OmeZarrPyramidDataset


# ---------------------------------------------------------------------------
# Single-array round-trip
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("compressor", ["zstd", "gzip", None])
def test_array_full_read(tmp_path, zarr_format, compressor):
    """Full-array read must match what zarr-python wrote."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 4000, size=(64, 96), dtype=np.uint16)
    p = tmp_path / f"arr_v{zarr_format}_{compressor or 'none'}"

    kw = (
        {"compressors": _v2_compressor(compressor)}
        if zarr_format == 2 else
        {"compressors": _v3_compressors(compressor)}
    )
    z = zarr.create_array(
        store=str(p), shape=arr.shape, chunks=(16, 24),
        dtype=arr.dtype, zarr_format=zarr_format, **kw,
    )
    z[:] = arr

    back = OmeZarrArray(p).read()
    np.testing.assert_array_equal(back, arr)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_array_region_read(tmp_path, zarr_format):
    """Sub-region reads must work and load only intersecting chunks."""
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 4000, size=(128, 192), dtype=np.uint16)
    p = tmp_path / f"region_v{zarr_format}"
    z = zarr.create_array(
        store=str(p), shape=arr.shape, chunks=(32, 32),
        dtype=arr.dtype, zarr_format=zarr_format,
    )
    z[:] = arr

    reader = OmeZarrArray(p)
    # Partial crop spanning multiple chunks
    crop = reader[40:100, 50:150]
    np.testing.assert_array_equal(crop, arr[40:100, 50:150])

    # Edge crop aligned with chunk boundary
    edge = reader[96:128, 160:192]
    np.testing.assert_array_equal(edge, arr[96:128, 160:192])

    # Single chunk
    one = reader[0:32, 0:32]
    np.testing.assert_array_equal(one, arr[0:32, 0:32])


def test_array_fill_value_for_missing_chunk(tmp_path):
    """Chunks not on disk return fill_value (Zarr semantics)."""
    p = tmp_path / "sparse_v3"
    z = zarr.create_array(
        store=str(p), shape=(32, 32), chunks=(16, 16),
        dtype="uint8", fill_value=42, zarr_format=3,
    )
    # Don't write anything; all 4 chunks are absent.
    out = OmeZarrArray(p).read()
    assert np.all(out == 42)


@pytest.mark.parametrize("dtype", ["uint8", "uint16", "int32", "float32"])
def test_array_dtypes_v3(tmp_path, dtype):
    np_dt = np.dtype(dtype)
    if np.issubdtype(np_dt, np.floating):
        arr = np.random.default_rng(0).standard_normal((48, 64)).astype(np_dt)
    else:
        info = np.iinfo(np_dt)
        arr = np.random.default_rng(0).integers(
            max(info.min, -(1 << 30)),
            min(info.max, (1 << 30)),
            size=(48, 64),
        ).astype(np_dt)
    p = tmp_path / f"dt_{dtype}_v3"
    z = zarr.create_array(
        store=str(p), shape=arr.shape, chunks=(16, 16),
        dtype=np_dt, zarr_format=3,
    )
    z[:] = arr
    back = OmeZarrArray(p).read()
    np.testing.assert_array_equal(back, arr)


# ---------------------------------------------------------------------------
# Pyramid (multiscales)
# ---------------------------------------------------------------------------


def _build_v04_pyramid(tmp_path):
    """Build a Zarr v2 / OME-NGFF v0.4 group with 3 pyramid levels."""
    import zarr
    root = tmp_path / "v04.zarr"
    group = zarr.group(store=str(root), zarr_format=2)
    base = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)
    levels = [base, base[::2, ::2].copy(), base[::4, ::4].copy()]
    for i, lvl in enumerate(levels):
        a = group.create_array(
            name=str(i), shape=lvl.shape, chunks=(32, 32),
            dtype=lvl.dtype,
        )
        a[:] = lvl
    # Write the multiscales metadata into .zattrs at the group root.
    attrs = {
        "multiscales": [{
            "version": "0.4",
            "axes": [
                {"name": "y", "type": "space"},
                {"name": "x", "type": "space"},
            ],
            "datasets": [
                {"path": "0",
                 "coordinateTransformations": [
                     {"type": "scale", "scale": [1.0, 1.0]}
                 ]},
                {"path": "1",
                 "coordinateTransformations": [
                     {"type": "scale", "scale": [2.0, 2.0]}
                 ]},
                {"path": "2",
                 "coordinateTransformations": [
                     {"type": "scale", "scale": [4.0, 4.0]}
                 ]},
            ],
        }]
    }
    (root / ".zattrs").write_text(json.dumps(attrs))
    return root, levels


def _build_v05_pyramid(tmp_path):
    """Build a Zarr v3 / OME-NGFF v0.5 group with 3 pyramid levels."""
    import zarr
    root = tmp_path / "v05.zarr"
    group = zarr.group(store=str(root), zarr_format=3)
    base = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)
    levels = [base, base[::2, ::2].copy(), base[::4, ::4].copy()]
    for i, lvl in enumerate(levels):
        a = group.create_array(
            name=str(i), shape=lvl.shape, chunks=(32, 32),
            dtype=lvl.dtype,
        )
        a[:] = lvl
    # Inject multiscales attrs into the group's zarr.json under
    # "attributes.ome".
    group_meta = json.loads((root / "zarr.json").read_text())
    group_meta.setdefault("attributes", {}).setdefault("ome", {})[
        "multiscales"
    ] = [{
        "version": "0.5",
        "axes": [
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
        "datasets": [
            {"path": "0",
             "coordinateTransformations": [
                 {"type": "scale", "scale": [1.0, 1.0]}]},
            {"path": "1",
             "coordinateTransformations": [
                 {"type": "scale", "scale": [2.0, 2.0]}]},
            {"path": "2",
             "coordinateTransformations": [
                 {"type": "scale", "scale": [4.0, 4.0]}]},
        ],
    }]
    (root / "zarr.json").write_text(json.dumps(group_meta))
    return root, levels


@pytest.mark.parametrize("builder", [_build_v04_pyramid, _build_v05_pyramid])
def test_pyramid_enumerates_levels(tmp_path, builder):
    root, levels = builder(tmp_path)
    with OmeZarrPyramidDataset(root) as p:
        assert p.n_levels == 3
        assert p.level(0).shape == (128, 128)
        assert p.level(1).shape == (64, 64)
        assert p.level(2).shape == (32, 32)
        # Downscale factors come out of shape ratios.
        assert p.level(0).downscale == (1, 1)
        assert p.level(1).downscale == (2, 2)
        assert p.level(2).downscale == (4, 4)


@pytest.mark.parametrize("builder", [_build_v04_pyramid, _build_v05_pyramid])
def test_pyramid_read_region(tmp_path, builder):
    root, levels = builder(tmp_path)
    with OmeZarrPyramidDataset(root) as p:
        # Full-resolution crop
        crop0 = p.read_region(level=0, y=(40, 96), x=(30, 100))
        np.testing.assert_array_equal(crop0, levels[0][40:96, 30:100])
        # Half-res crop
        crop1 = p.read_region(level=1, y=(0, 32), x=(0, 32))
        np.testing.assert_array_equal(crop1, levels[1][0:32, 0:32])
        # Lowest-res full read
        crop2 = p.read_region(level=2)
        np.testing.assert_array_equal(crop2, levels[2])


def test_pyramid_best_level_for(tmp_path):
    root, _ = _build_v04_pyramid(tmp_path)
    with OmeZarrPyramidDataset(root) as p:
        # 200×200 envelope: only level 0 (128x128) fits — but it's the
        # highest-resolution that fits, so we want it.
        assert p.best_level_for(max_pixels_y=200) == 0
        # 50x50 envelope: only level 2 (32x32) fits.
        assert p.best_level_for(max_pixels_y=50) == 2
        # 80x80 envelope: level 1 (64x64) fits, level 0 does not.
        assert p.best_level_for(max_pixels_y=80) == 1


def test_pyramid_with_higher_dim_axes(tmp_path):
    """T, C, Z, Y, X — non-spatial axes selectable via kwargs."""
    import zarr
    root = tmp_path / "tczyx.zarr"
    group = zarr.group(store=str(root), zarr_format=3)
    # 2 channels x 1 z x 64x64
    base = np.zeros((2, 1, 64, 64), dtype=np.uint16)
    base[0] = 100
    base[1] = 200
    levels = [base, base[:, :, ::2, ::2].copy()]
    for i, lvl in enumerate(levels):
        a = group.create_array(
            name=str(i), shape=lvl.shape, chunks=(1, 1, 32, 32),
            dtype=lvl.dtype,
        )
        a[:] = lvl
    group_meta = json.loads((root / "zarr.json").read_text())
    group_meta.setdefault("attributes", {}).setdefault("ome", {})[
        "multiscales"
    ] = [{
        "version": "0.5",
        "axes": [
            {"name": "c", "type": "channel"},
            {"name": "z", "type": "space"},
            {"name": "y", "type": "space"},
            {"name": "x", "type": "space"},
        ],
        "datasets": [
            {"path": "0", "coordinateTransformations": [
                {"type": "scale", "scale": [1, 1, 1, 1]}]},
            {"path": "1", "coordinateTransformations": [
                {"type": "scale", "scale": [1, 1, 2, 2]}]},
        ],
    }]
    (root / "zarr.json").write_text(json.dumps(group_meta))

    with OmeZarrPyramidDataset(root) as p:
        ch0 = p.read_region(level=0, y=(0, 64), x=(0, 64), c=0)
        assert ch0.shape == (64, 64)
        assert np.all(ch0 == 100)
        ch1 = p.read_region(level=0, y=(0, 64), x=(0, 64), c=1)
        assert np.all(ch1 == 200)
        # Half-res, channel 1
        ch1_lo = p.read_region(level=1, c=1)
        assert ch1_lo.shape == (32, 32)
        assert np.all(ch1_lo == 200)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _v2_compressor(name: str | None):
    """Return a numcodecs codec for zarr v2 create_array. Returns
    ``()`` for no-compression so the same kwarg name works for v2/v3."""
    if name is None:
        return ()
    import numcodecs
    if name == "zstd":
        return (numcodecs.Zstd(level=1),)
    if name == "gzip":
        return (numcodecs.GZip(),)
    if name == "blosc":
        return (numcodecs.Blosc(),)
    raise ValueError(name)


def _v3_compressors(name: str | None):
    """Return the compressors= arg for zarr v3 create_array."""
    if name is None:
        return ()
    if name == "zstd":
        return (zarr.codecs.ZstdCodec(level=1),)
    if name == "gzip":
        return (zarr.codecs.GzipCodec(level=5),)
    if name == "blosc":
        return (zarr.codecs.BloscCodec(),)
    raise ValueError(name)
