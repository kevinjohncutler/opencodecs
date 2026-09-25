"""Regional reads that decode each tile straight into its output window.

Zstd tiles no longer become owned arrays that the calling thread pastes:
one native call decompresses, unshuffles and writes the tile's part of the
request into the output. With enough work, workers write in any order, each
tile writing only the pieces no later tile on its plane covers, so overlap
still resolves to the later tile. Ground truth here is a canvas painted in
directory order from the source tiles themselves, never the reader.

Tiles are large enough (128x128 uint16, 32 KiB) that a full read spans
several 2 MiB batches, so the parallel path and its conflict handling run.
"""
from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(
    not oc.has_codec("czi"),
    reason="czi codec requires native zstd + bytetools extensions",
)

from _czi_fixture import mosaic_czi_bytes  # noqa: E402

from opencodecs._czi_reader import CziError, CziPyramidReader  # noqa: E402


def _tiles(rows, cols, th, tw, step_y, step_x, *, seed=0, dtype=np.uint16,
           samples=1, jitter=0):
    rs = np.random.RandomState(seed)
    out = []
    for r in range(rows):
        for c in range(cols):
            shape = (th, tw) + ((samples,) if samples > 1 else ())
            hi = 256 if dtype == np.uint8 else 65535
            t = rs.randint(0, hi, shape).astype(dtype)
            dy = int(rs.randint(0, jitter + 1)) if jitter else 0
            dx = int(rs.randint(0, jitter + 1)) if jitter else 0
            out.append((t, (r * step_y + dy, c * step_x + dx)))
    order = rs.permutation(len(out))  # directory order unrelated to position
    return [out[i] for i in order]


def _canvas(tiles):
    """Paint tiles in directory order; the image starts at the first tile."""
    oy = min(sy for _, (sy, sx) in tiles)
    ox = min(sx for _, (sy, sx) in tiles)
    h = max(sy + t.shape[0] for t, (sy, sx) in tiles) - oy
    w = max(sx + t.shape[1] for t, (sy, sx) in tiles) - ox
    canvas = np.zeros((h, w) + tiles[0][0].shape[2:], dtype=tiles[0][0].dtype)
    for t, (sy, sx) in tiles:
        canvas[sy - oy:sy - oy + t.shape[0], sx - ox:sx - ox + t.shape[1]] = t
    return canvas


def _reader(data, **kw):
    return CziPyramidReader(oc.get_codec("czi").open(data), **kw)


@pytest.mark.parametrize("workers", [None, 1, 4])
def test_overlapping_mosaic_full_read_keeps_directory_order(workers):
    # 16 px overlap on both axes, shuffled directory order: every seam is
    # decided by which of two tiles comes later in the directory.
    tiles = _tiles(10, 12, 128, 128, 112, 112, seed=1)
    canvas = _canvas(tiles)
    with _reader(mosaic_czi_bytes(tiles, compression=6, hilo=True),
                 decode_workers=workers) as p:
        got = p.read_region(0)
        stats = p.region_stats
    np.testing.assert_array_equal(got, canvas)
    if workers == 4:
        # Every tile overlaps a neighbor, and they still decode on workers.
        assert stats["workers"] > 1 and stats["batches"] >= 2


def test_parallel_read_with_gaps_and_uneven_overlaps():
    # Jittered grid: gaps stay zero, overlaps of every shape resolve to the
    # later tile, and the image starts at the first tile, not at zero.
    tiles = _tiles(9, 9, 128, 128, 136, 136, seed=2, jitter=24)
    canvas = _canvas(tiles)
    with _reader(mosaic_czi_bytes(tiles, compression=6, hilo=True),
                 decode_workers=4) as p:
        got = p.read_region(0)
        stats = p.region_stats
    np.testing.assert_array_equal(got, canvas)
    assert stats["workers"] > 1


def test_random_crops_match_the_canvas_on_every_path():
    tiles = _tiles(8, 8, 128, 128, 120, 104, seed=3)
    canvas = _canvas(tiles)
    data = mosaic_czi_bytes(tiles, compression=6, hilo=True)
    rs = np.random.RandomState(4)
    with _reader(data, decode_workers=4) as par, _reader(data, decode_workers=1) as ser:
        for _ in range(40):
            h, w = canvas.shape
            y0 = int(rs.randint(0, h - 1))
            x0 = int(rs.randint(0, w - 1))
            y1 = int(rs.randint(y0 + 1, min(h, y0 + 700) + 1))
            x1 = int(rs.randint(x0 + 1, min(w, x0 + 700) + 1))
            expected = canvas[y0:y1, x0:x1]
            np.testing.assert_array_equal(par.read_region(0, y=(y0, y1), x=(x0, x1)), expected)
            np.testing.assert_array_equal(ser.read_region(0, y=(y0, y1), x=(x0, x1)), expected)


@pytest.mark.parametrize("dtype,samples,hilo,compression", [
    (np.uint8, 3, True, 6),     # BGR24: itemsize 1, nothing to unshuffle
    (np.uint16, 3, True, 6),    # BGR48: three shuffled samples per pixel
    (np.uint16, 1, False, 6),   # ZSTDHDR without the hi/lo shuffle
    (np.uint16, 1, False, 5),   # plain zstd (ZSTD0)
])
def test_window_decode_covers_samples_and_unshuffled_payloads(dtype, samples, hilo, compression):
    tiles = _tiles(4, 5, 96, 80, 90, 70, seed=5, dtype=dtype, samples=samples)
    canvas = _canvas(tiles)
    with _reader(mosaic_czi_bytes(tiles, compression=compression, hilo=hilo),
                 decode_workers=4) as p:
        np.testing.assert_array_equal(p.read_region(0), canvas)
        np.testing.assert_array_equal(p.read_region(0, y=(33, 301), x=(7, 290)),
                                      canvas[33:301, 7:290])


def test_tile_cache_keeps_the_owned_tile_path_and_agrees():
    tiles = _tiles(6, 6, 128, 128, 112, 112, seed=6)
    canvas = _canvas(tiles)
    with _reader(mosaic_czi_bytes(tiles, compression=6, hilo=True),
                 decode_workers=4, decoded_cache_bytes=64 << 20) as p:
        np.testing.assert_array_equal(p.read_region(0), canvas)
        np.testing.assert_array_equal(p.read_region(0), canvas)
        assert p.region_stats["cache_hits"] == len(tiles)


def test_uncompressed_level_uses_the_owned_tile_path():
    tiles = _tiles(3, 3, 64, 64, 60, 60, seed=7)
    with _reader(mosaic_czi_bytes(tiles, compression=0)) as p:
        np.testing.assert_array_equal(p.read_region(0), _canvas(tiles))
        assert "tiles_hidden" not in p.region_stats


def test_fully_hidden_tiles_are_not_decoded():
    # A second copy of every tile, later in the directory, covers the first
    # exactly: the first copies must never be decoded.
    base = _tiles(6, 6, 128, 128, 128, 128, seed=12)
    cover = _tiles(6, 6, 128, 128, 128, 128, seed=13)
    tiles = base + cover
    with _reader(mosaic_czi_bytes(tiles, compression=6, hilo=True),
                 decode_workers=4) as p:
        np.testing.assert_array_equal(p.read_region(0), _canvas(tiles))
        assert p.region_stats["tiles_hidden"] == len(base)


def test_corrupt_payload_raises_czi_error_from_a_worker():
    tiles = _tiles(8, 8, 128, 128, 128, 128, seed=8)
    data = mosaic_czi_bytes(tiles, compression=6, hilo=True)
    with _reader(data) as p:
        view, _ = p._czi._pixel_data_view(p.levels[0].reader[37])
        # ZSTDHDR: 3 header bytes, then the zstd frame and its magic number.
        needle = bytes(view[3:64])
    pos = data.find(needle)
    assert pos > 0 and data.find(needle, pos + 1) == -1
    bad = bytearray(data)
    bad[pos:pos + 4] = bytes(4)
    with _reader(bytes(bad), decode_workers=4) as p:
        with pytest.raises(CziError, match="does not decode"):
            p.read_region(0)


def test_visible_pieces_repaint_the_level_exactly():
    """Painting every tile's visible pieces in any order gives the canvas."""
    tiles = _tiles(7, 7, 128, 128, 120, 136, seed=9, jitter=20)
    canvas = _canvas(tiles)
    with _reader(mosaic_czi_bytes(tiles, compression=6, hilo=True)) as p:
        level = p.levels[0]
        index = p._level_index(level)
        owner = np.full(level.shape, -1, np.int64)
        for i in np.random.RandomState(14).permutation(index.n):
            for y0, y1, x0, x1 in index.visible(int(i)):
                assert (owner[y0:y1, x0:x1] == -1).all(), "pieces overlap"
                owner[y0:y1, x0:x1] = i
        # Directory-order painting decides the owner of every pixel.
        expected = np.full(level.shape, -1, np.int64)
        for i in range(index.n):
            y0, y1, x0, x1 = index.bounds[i]
            expected[y0:y1, x0:x1] = i
        np.testing.assert_array_equal(owner, expected)
        np.testing.assert_array_equal(p.read_region(0), canvas)


# ---- the native call on its own ----

def test_native_window_matches_numpy():
    z = pytest.importorskip("opencodecs.codecs._zstd")
    rs = np.random.RandomState(11)
    for itemsize, samples in ((1, 1), (2, 1), (2, 3), (4, 1), (8, 2)):
        h, w = 37, 53
        tile = rs.randint(0, 256, (h, w, samples * itemsize)).astype(np.uint8)
        elements = tile.reshape(h * w * samples, itemsize)
        shuffled = np.ascontiguousarray(elements.T).tobytes() if itemsize > 1 else tile.tobytes()
        frame = z.encode(shuffled, level=3)
        out = np.full((60, 70 * samples * itemsize), 7, np.uint8)
        scratch = bytearray(h * w * samples * itemsize)
        windows = [(5, 30, 11, 50, 3, 20), (0, 2, 0, 53, 40, 0), (30, 37, 50, 53, 50, 66)]
        n = z.decode_unshuffle_windows(frame, scratch, out, itemsize, samples, h, w,
                                       windows, z.DecodeContext())
        assert n == len(scratch)
        expected = np.full_like(out, 7)
        px = samples * itemsize
        for sy0, sy1, sx0, sx1, dy, dx in windows:
            expected[dy:dy + sy1 - sy0, dx * px:(dx + sx1 - sx0) * px] = \
                tile[sy0:sy1, sx0:sx1].reshape(sy1 - sy0, -1)
        np.testing.assert_array_equal(out, expected)


def test_native_window_rejects_out_of_bounds_and_wrong_sizes():
    z = pytest.importorskip("opencodecs.codecs._zstd")
    frame = z.encode(bytes(range(256)) * 8, level=3)  # 2048 bytes
    scratch = bytearray(2048)
    out = np.zeros((32, 64), np.uint8)
    full = [(0, 32, 0, 32, 0, 0)]
    with pytest.raises(ValueError):  # window outside the tile
        z.decode_unshuffle_windows(frame, scratch, out, 2, 1, 32, 32, [(0, 33, 0, 32, 0, 0)])
    with pytest.raises(ValueError):  # window outside the destination
        z.decode_unshuffle_windows(frame, scratch, out, 2, 1, 32, 32, [(0, 32, 0, 32, 1, 0)])
    # A bad window anywhere in the list refuses the call before decoding.
    with pytest.raises(ValueError):
        z.decode_unshuffle_windows(frame, scratch, out, 2, 1, 32, 32, full + [(0, 1, 0, 40, 0, 0)])
    assert not out.any()
    # A frame of the wrong size reports its size and leaves out alone.
    small = z.encode(bytes(100), level=3)
    assert z.decode_unshuffle_windows(small, scratch, out, 2, 1, 32, 32, full) == 100
    assert not out.any()
    with pytest.raises(z.ZstdError):
        z.decode_unshuffle_windows(b"not zstd", scratch, out, 2, 1, 32, 32, full)
    # The right frame with no windows decodes and writes nothing.
    assert z.decode_unshuffle_windows(frame, scratch, out, 2, 1, 32, 32, []) == 2048
    assert not out.any()


def test_overlap_follows_mosaic_index_not_directory_order():
    """Where tiles overlap, the higher mosaic index (M) is on top.

    That is the libCZI and czifile rule; Zen writes tiles out of M order,
    so directory order put the wrong tile on top. Directory order is the
    reverse of M order here, and the parallel path must agree.
    """
    rs = np.random.RandomState(15)
    tiles = []
    for r in range(5):
        for c in range(5):
            t = rs.randint(0, 65535, (128, 128)).astype(np.uint16)
            tiles.append((t, (r * 100, c * 100)))
    m_of = list(range(len(tiles)))
    directory = [(t, pos, [(b"M", m)]) for (t, pos), m in reversed(list(zip(tiles, m_of)))]
    canvas = _canvas(tiles)  # painted in ascending M order
    for workers in (1, 4):
        with _reader(mosaic_czi_bytes(directory, compression=6, hilo=True),
                     decode_workers=workers) as p:
            np.testing.assert_array_equal(p.read_region(0), canvas)
            np.testing.assert_array_equal(p.read_region(0, y=(90, 330), x=(40, 410)),
                                          canvas[90:330, 40:410])
