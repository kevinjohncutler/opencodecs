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


# ---------------------------------------------------------------------------
# Local HTTP server fixture — serve a Zarr directory tree
# ---------------------------------------------------------------------------


import functools
import http.server
import socketserver
import threading


@pytest.fixture
def http_zarr_server(tmp_path):
    """Spin up a localhost HTTP server rooted at a directory the
    caller fills in. Yields ``(base_url, directory)``. Used to test
    :meth:`OmeZarrArray.from_http` and
    :meth:`OmeZarrPyramidDataset.from_http` without needing network."""
    directory = tmp_path / "zarr_server_root"
    directory.mkdir()

    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=str(directory),
    )

    class _Quiet(http.server.ThreadingHTTPServer):
        allow_reuse_address = True

    httpd = _Quiet(("127.0.0.1", 0), handler)
    # Suppress per-request log spam.
    handler.log_message = lambda *_, **__: None
    th = threading.Thread(target=httpd.serve_forever, daemon=True)
    th.start()
    base_url = f"http://127.0.0.1:{httpd.server_address[1]}"
    try:
        yield base_url, directory
    finally:
        httpd.shutdown()
        th.join(timeout=2)


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


# ---------------------------------------------------------------------------
# HTTP (OmeZarrArray.from_http / OmeZarrPyramidDataset.from_http)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_array_from_http_matches_local(tmp_path, http_zarr_server, zarr_format):
    """from_http() reads the same array as the local-path constructor."""
    base_url, root = http_zarr_server
    rng = np.random.default_rng(0)
    arr_src = rng.integers(0, 4000, size=(64, 96), dtype=np.uint16)
    arr_dir = root / f"a_v{zarr_format}"
    z = zarr.create_array(
        store=str(arr_dir), shape=arr_src.shape, chunks=(16, 24),
        dtype=arr_src.dtype, zarr_format=zarr_format,
    )
    z[:] = arr_src

    local = OmeZarrArray(arr_dir)
    remote = OmeZarrArray.from_http(f"{base_url}/{arr_dir.name}")
    np.testing.assert_array_equal(local.read(), remote.read())


def test_array_from_http_partial_read_only_touches_relevant_chunks(
    tmp_path, http_zarr_server,
):
    """Partial reads must touch only the chunks intersecting the bbox.
    The HTTP store's stats() reports request count and bytes fetched —
    if the partial read were materializing the whole array we'd see
    100+ chunk requests."""
    base_url, root = http_zarr_server
    rng = np.random.default_rng(1)
    arr_src = rng.integers(0, 4000, size=(128, 192), dtype=np.uint16)
    arr_dir = root / "partial"
    z = zarr.create_array(
        store=str(arr_dir), shape=arr_src.shape, chunks=(32, 32),
        dtype=arr_src.dtype, zarr_format=2,
    )
    z[:] = arr_src
    arr = OmeZarrArray.from_http(f"{base_url}/partial")
    # Read a (10, 10) crop entirely inside one 32x32 chunk.
    one_chunk_crop = arr.read_region(
        (slice(0, 10), slice(0, 10))
    )
    np.testing.assert_array_equal(one_chunk_crop, arr_src[0:10, 0:10])
    s = arr._store.stats()
    # 1 metadata fetch (the .zarray) + 1 chunk fetch + at most one
    # __contains__ probe for the metadata file = ~2-3 requests total.
    # Anything close to 6×8=48 (total chunks) would indicate a bug.
    assert s["requests"] <= 5, (
        f"partial read fetched {s['requests']} HTTP requests for a "
        f"single-chunk crop; expected ≤5 (metadata + 1 chunk)"
    )


@pytest.mark.parametrize("ngff_version", ["v04", "v05"])
def test_pyramid_from_http_matches_local(
    tmp_path, http_zarr_server, ngff_version,
):
    """Building an OmeZarrPyramidDataset over HTTP returns the same
    pixels per level as the local-path version."""
    base_url, root = http_zarr_server
    if ngff_version == "v04":
        local_root, levels = _build_v04_pyramid(root)
    else:
        local_root, levels = _build_v05_pyramid(root)
    # local_root is inside `root` so served under {base_url}/{name}
    rel = local_root.name
    with OmeZarrPyramidDataset(local_root) as local:
        with OmeZarrPyramidDataset.from_http(f"{base_url}/{rel}") as remote:
            assert local.n_levels == remote.n_levels
            for i in range(local.n_levels):
                assert local.level(i).shape == remote.level(i).shape
                local_lvl = local.read_region(
                    i, y=(0, levels[i].shape[0]),
                    x=(0, levels[i].shape[1]),
                )
                remote_lvl = remote.read_region(
                    i, y=(0, levels[i].shape[0]),
                    x=(0, levels[i].shape[1]),
                )
                np.testing.assert_array_equal(local_lvl, remote_lvl)


def test_http_store_lru_cache_hits(tmp_path, http_zarr_server):
    """The LRU cache satisfies repeated reads of the same chunk
    without re-fetching."""
    base_url, root = http_zarr_server
    arr_src = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    arr_dir = root / "cached"
    z = zarr.create_array(
        store=str(arr_dir), shape=arr_src.shape, chunks=(32, 32),
        dtype=arr_src.dtype, zarr_format=2,
    )
    z[:] = arr_src
    arr = OmeZarrArray.from_http(f"{base_url}/cached")
    # First read: cache miss; second: cache hit.
    arr.read_region((slice(0, 32), slice(0, 32)))
    s1 = arr._store.stats()
    arr.read_region((slice(0, 32), slice(0, 32)))
    s2 = arr._store.stats()
    # Misses didn't increase, hits did
    assert s2["misses"] == s1["misses"]
    assert s2["hits"] >= s1["hits"] + 1


# ---------------------------------------------------------------------------
# Bonus: live test against IDR (only runs when network + IDR are reachable)
# ---------------------------------------------------------------------------


_IDR_URL = "https://uk1s3.embassy.ebi.ac.uk/idr/zarr/v0.4/idr0062A/6001240.zarr"


def _idr_reachable() -> bool:
    """Quick HEAD probe — used as a skip gate so CI without network
    just skips this test rather than failing."""
    import urllib.request, urllib.error
    try:
        urllib.request.urlopen(_IDR_URL + "/.zattrs", timeout=3).close()
        return True
    except Exception:
        return False


@pytest.mark.skipif(
    not _idr_reachable(),
    reason="IDR HTTPS endpoint unreachable (offline or blocked)",
)
def test_pyramid_from_http_real_idr_endpoint():
    """Live network test against a real IDR-hosted OME-NGFF dataset.
    Validates the full streaming pipeline end-to-end on actual S3
    object storage."""
    p = OmeZarrPyramidDataset.from_http(_IDR_URL)
    try:
        assert p.n_levels == 3
        # Lowest-res level is small enough to fully fetch
        lo = p.read_region(2, c=0)
        assert lo.shape == (68, 67)
        # bytes_fetched should be a small multiple of the level-2 size,
        # not the full pyramid
        s = p._store.stats()
        assert s["bytes_fetched"] > 0
    finally:
        p.close()


# ---------------------------------------------------------------------------
# Zarr v3 sharded storage
# ---------------------------------------------------------------------------


def test_sharded_v3_full_array_round_trip(tmp_path):
    """A Zarr v3 array written with shards= round-trips pixel-equal
    through OmeZarrArray. zarr-python is the reference writer; we
    must match its decoded output exactly."""
    p = tmp_path / "sharded.zarr"
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 4000, size=(64, 64), dtype=np.uint16)
    z = zarr.create_array(
        store=str(p), shape=arr.shape,
        chunks=(16, 16), shards=(32, 32),
        dtype=arr.dtype,
    )
    z[:] = arr
    back = OmeZarrArray(p).read()
    np.testing.assert_array_equal(back, arr)


def test_sharded_v3_partial_region(tmp_path):
    """Partial reads on a sharded array — only the shards that contain
    intersecting chunks should be touched. Verify pixels match."""
    p = tmp_path / "sharded_partial.zarr"
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 4000, size=(128, 128), dtype=np.uint16)
    z = zarr.create_array(
        store=str(p), shape=arr.shape,
        chunks=(16, 16), shards=(64, 64),
        dtype=arr.dtype,
    )
    z[:] = arr
    reader = OmeZarrArray(p)
    # The user-facing chunk shape is the INNER chunk (16, 16),
    # matching zarr-python's arr.chunks.
    assert reader.chunks == (16, 16)
    # Crop spanning two shards (left half goes through shard (0,0),
    # right half through shard (0,1)).
    crop = reader[30:60, 40:120]
    np.testing.assert_array_equal(crop, arr[30:60, 40:120])


def test_sharded_v3_fill_value_for_absent_chunk(tmp_path):
    """A sharded array with no data written returns fill_value for
    every chunk. Tests the EMPTY-marker path inside the shard index."""
    p = tmp_path / "sharded_empty.zarr"
    z = zarr.create_array(
        store=str(p), shape=(32, 32), chunks=(16, 16), shards=(32, 32),
        dtype="uint8", fill_value=77,
    )
    # No write — all chunks absent. zarr-python doesn't create the
    # shard file at all in this case, so OmeZarrArray's missing-chunk
    # path returns fill_value.
    out = OmeZarrArray(p).read()
    assert np.all(out == 77)


@pytest.mark.parametrize("compressor", ["zstd", "gzip", None])
def test_sharded_v3_with_inner_compression(tmp_path, compressor):
    """Sharded arrays support an arbitrary inner codec chain — the
    bytes inside each shard go through the inner codecs before
    storage. Confirm we decode that correctly with each compressor."""
    p = tmp_path / f"sharded_{compressor or 'raw'}.zarr"
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 4000, size=(48, 48), dtype=np.uint16)
    if compressor is None:
        comps = ()
    elif compressor == "zstd":
        comps = (zarr.codecs.ZstdCodec(level=1),)
    else:
        comps = (zarr.codecs.GzipCodec(level=5),)
    z = zarr.create_array(
        store=str(p), shape=arr.shape,
        chunks=(16, 16), shards=(48, 48),
        dtype=arr.dtype,
        compressors=comps,
    )
    z[:] = arr
    back = OmeZarrArray(p).read()
    np.testing.assert_array_equal(back, arr)


# ---------------------------------------------------------------------------
# Writer (write_zarr_array + write_omezarr_pyramid)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("compressor", ["zstd", "gzip", "none"])
def test_writer_array_round_trips_through_zarr_python(
    tmp_path, zarr_format, compressor,
):
    """write_zarr_array output must read back pixel-equal via
    zarr-python — the reference reader."""
    from opencodecs import write_zarr_array
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 4000, size=(48, 64), dtype=np.uint16)
    p = tmp_path / f"out_v{zarr_format}_{compressor}"
    write_zarr_array(
        p, arr, chunks=(16, 32),
        compressor=compressor, zarr_format=zarr_format,
    )
    z = zarr.open(str(p), mode="r")
    back = np.asarray(z[:])
    np.testing.assert_array_equal(back, arr)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_writer_array_round_trips_through_opencodecs(tmp_path, zarr_format):
    """write_zarr_array → OmeZarrArray round-trip (our writer +
    our reader). Confirms both halves agree on the on-disk format."""
    from opencodecs import write_zarr_array
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 4000, size=(64, 96), dtype=np.uint16)
    p = tmp_path / f"selfrt_v{zarr_format}"
    write_zarr_array(p, arr, chunks=(16, 16), compressor="zstd",
                     zarr_format=zarr_format)
    back = OmeZarrArray(p).read()
    np.testing.assert_array_equal(back, arr)


def test_writer_array_dtype_matrix(tmp_path):
    """Every numeric dtype we map round-trips at zarr_format=3."""
    from opencodecs import write_zarr_array
    rng = np.random.default_rng(2)
    for dt in [np.uint8, np.uint16, np.int32, np.float32, np.float64]:
        if np.issubdtype(dt, np.floating):
            arr = rng.standard_normal((16, 24)).astype(dt)
        else:
            info = np.iinfo(dt)
            arr = rng.integers(
                max(info.min, -(1 << 30)), min(info.max, (1 << 30)),
                size=(16, 24),
            ).astype(dt)
        p = tmp_path / f"dt_{np.dtype(dt).name}"
        write_zarr_array(p, arr, chunks=(8, 12), zarr_format=3)
        back = OmeZarrArray(p).read()
        np.testing.assert_array_equal(back, arr)


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_writer_pyramid_round_trips_through_pyramid_reader(
    tmp_path, zarr_format,
):
    """OME-NGFF pyramid writer → OmeZarrPyramidDataset round-trip."""
    from opencodecs import write_omezarr_pyramid
    base = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)
    levels = [base, base[::2, ::2].copy(), base[::4, ::4].copy()]
    p = tmp_path / f"pyr_v{zarr_format}.zarr"
    write_omezarr_pyramid(
        p, levels, chunks=(32, 32),
        compressor="zstd", zarr_format=zarr_format,
    )
    with OmeZarrPyramidDataset(p) as ds:
        assert ds.n_levels == 3
        for i, lvl in enumerate(levels):
            assert ds.level(i).shape == lvl.shape
            out = ds.read_region(
                i, y=(0, lvl.shape[0]), x=(0, lvl.shape[1]),
            )
            np.testing.assert_array_equal(out, lvl)


def test_writer_partial_chunks_at_edges_pad_with_fill_value(tmp_path):
    """When array shape isn't a multiple of chunks, edge chunks are
    padded with fill_value. Read-back must crop back to the original
    shape correctly."""
    from opencodecs import write_zarr_array
    arr = np.arange(48 * 56, dtype=np.uint8).reshape(48, 56)
    # chunks 16x16; edge chunks at right (col 48..56) and bottom
    # (row 48..64 within full chunks)
    p = tmp_path / "edge"
    write_zarr_array(p, arr, chunks=(16, 16), zarr_format=2)
    back = OmeZarrArray(p).read()
    np.testing.assert_array_equal(back, arr)
