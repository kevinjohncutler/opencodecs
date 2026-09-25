"""Batched overlapping region requests (priority 6).

``PyramidReader.iter_regions``/``read_regions`` plan a bounded batch of
boxes, decode the union of their tiles once, and paste each tile into every
output it intersects in composition order. What must hold: every per-box
result equals ``read_region`` for the same box; adjacent boxes decode fewer
tiles than they request while disjoint boxes decode exactly what they
request; request order survives mixed sizes and batch boundaries; closing
the iterator early leaves the reader usable; a backend without the hooks is
served by the per-box fallback in order; overlap and plane semantics carry
over from the single-region path.
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
from opencodecs.core.pyramid import PyramidLevel, PyramidReader  # noqa: E402


def _tile(shape, seed):
    return np.random.RandomState(seed).randint(0, 65535, shape).astype(np.uint16)


def _grid(rows, cols, th=32, tw=32, seed=0):
    return [(_tile((th, tw), seed + r * cols + c), (r * th, c * tw))
            for r in range(rows) for c in range(cols)]


def _canvas(tiles, dtype=np.uint16):
    h = max(sy + t.shape[0] for t, (sy, sx), *_ in tiles)
    w = max(sx + t.shape[1] for t, (sy, sx), *_ in tiles)
    canvas = np.zeros((h, w) + tiles[0][0].shape[2:], dtype=dtype)
    for t, (sy, sx), *_ in tiles:
        canvas[sy:sy + t.shape[0], sx:sx + t.shape[1]] = t
    return canvas


def _open(tiles, **kw):
    r = oc.get_codec("czi").open(mosaic_czi_bytes(tiles, compression=6, hilo=True))
    return CziPyramidReader(r, **kw)


def _boxes_as_pairs(boxes):
    return [((y0, y1), (x0, x1)) for y0, y1, x0, x1 in boxes]


def test_adjacent_crops_share_tiles_and_match_single_reads():
    tiles = _grid(8, 8)
    canvas = _canvas(tiles)
    p = _open(tiles)
    try:
        # 3x3 crops of 48x48 stepping 32: neighbors share a tile column/row.
        boxes = [(r * 32, r * 32 + 48, c * 32, c * 32 + 48)
                 for r in range(3) for c in range(3)]
        got = p.read_regions(0, _boxes_as_pairs(boxes))
        assert len(got) == len(boxes)
        for out, (y0, y1, x0, x1) in zip(got, boxes):
            np.testing.assert_array_equal(out, canvas[y0:y1, x0:x1])
            np.testing.assert_array_equal(out, p.read_region(0, y=(y0, y1), x=(x0, x1)))
        st = p.batch_stats
        assert st["boxes"] == 9 and st["batches"] == 1 and st["fallback_boxes"] == 0
        assert st["requested_tiles"] == 9 * 4          # each 48x48 box spans 2x2 tiles
        assert st["unique_tiles"] == 16                # the union is a 4x4 block
    finally:
        p.close()


def test_disjoint_crops_decode_exactly_what_they_request():
    tiles = _grid(6, 6)
    canvas = _canvas(tiles)
    p = _open(tiles)
    try:
        boxes = [(0, 32, 0, 32), (64, 96, 64, 96), (128, 160, 0, 32), (32, 64, 160, 192)]
        got = p.read_regions(0, _boxes_as_pairs(boxes))
        for out, (y0, y1, x0, x1) in zip(got, boxes):
            np.testing.assert_array_equal(out, canvas[y0:y1, x0:x1])
        st = p.batch_stats
        assert st["requested_tiles"] == st["unique_tiles"] == 4
    finally:
        p.close()


def test_one_large_box_matches_read_region():
    tiles = _grid(6, 6)
    p = _open(tiles)
    try:
        (out,) = p.read_regions(0, [((10, 180), (5, 190))])
        np.testing.assert_array_equal(out, p.read_region(0, y=(10, 180), x=(5, 190)))
        assert p.batch_stats["unique_tiles"] == 36
    finally:
        p.close()


def test_request_order_survives_mixed_sizes_and_batch_boundaries():
    tiles = _grid(6, 6)
    canvas = _canvas(tiles)
    p = _open(tiles)
    try:
        rs = np.random.RandomState(4)
        boxes = []
        for _ in range(40):
            h, w = int(rs.randint(1, 90)), int(rs.randint(1, 90))
            y0, x0 = int(rs.randint(0, 192 - h)), int(rs.randint(0, 192 - w))
            boxes.append((y0, y0 + h, x0, x0 + w))
        # A budget of one tile forces many batches; order must still hold.
        got = p.read_regions(0, _boxes_as_pairs(boxes), max_batch_bytes=32 * 32 * 2)
        assert p.batch_stats["batches"] > 1
        for out, (y0, y1, x0, x1) in zip(got, boxes):
            np.testing.assert_array_equal(out, canvas[y0:y1, x0:x1])
    finally:
        p.close()


def test_iter_regions_can_be_abandoned_and_reader_stays_usable():
    tiles = _grid(4, 4)
    canvas = _canvas(tiles)
    p = _open(tiles, decode_workers=2)
    try:
        boxes = [(0, 64, 0, 64), (64, 128, 64, 128), (0, 128, 0, 128)]
        it = p.iter_regions(0, _boxes_as_pairs(boxes), max_batch_bytes=1)
        first = next(it)
        it.close()
        np.testing.assert_array_equal(first, canvas[0:64, 0:64])
        np.testing.assert_array_equal(p.read_region(0, y=(64, 128), x=(64, 128)),
                                      canvas[64:128, 64:128])
    finally:
        p.close()


def test_overlapping_tiles_keep_last_wins_in_every_output():
    a, b = _tile((32, 32), 1), _tile((32, 32), 2)
    tiles = [(a, (0, 0)), (b, (16, 16))]
    canvas = _canvas(tiles)
    p = _open(tiles)
    try:
        boxes = [(0, 48, 0, 48), (16, 32, 16, 32), (8, 40, 8, 40)]
        got = p.read_regions(0, _boxes_as_pairs(boxes))
        for out, (y0, y1, x0, x1) in zip(got, boxes):
            np.testing.assert_array_equal(out, canvas[y0:y1, x0:x1])
        np.testing.assert_array_equal(got[1], b[:16, :16])
        assert p.batch_stats["unique_tiles"] == 2 and p.batch_stats["requested_tiles"] == 6
    finally:
        p.close()


def test_plane_refusal_propagates_from_the_planner():
    tiles = [(_tile((32, 32), 7), (0, 0), [(b"C", 0)]),
             (_tile((32, 32), 8), (0, 0), [(b"C", 1)])]
    p = _open(tiles)
    try:
        with pytest.raises(CziError, match="distinct planes"):
            p.read_regions(0, [((0, 32), (0, 32))])
    finally:
        p.close()


def test_serial_and_parallel_batched_reads_agree():
    tiles = [(_tile((512, 512), 500 + i), ((i // 4) * 512, (i % 4) * 512))
             for i in range(16)]
    canvas = _canvas(tiles)
    serial = _open(tiles, decode_workers=1)
    parallel = _open(tiles, decode_workers=4)
    try:
        boxes = [(0, 1500, 0, 1500), (600, 1100, 600, 1100), (1500, 2048, 0, 2048)]
        a = serial.read_regions(0, _boxes_as_pairs(boxes))
        b = parallel.read_regions(0, _boxes_as_pairs(boxes))
        for x, y, (y0, y1, x0, x1) in zip(a, b, boxes):
            np.testing.assert_array_equal(x, y)
            np.testing.assert_array_equal(x, canvas[y0:y1, x0:x1])
        # Box 1 spans tile rows/cols 0..2 (9), box 2 lies inside it, box 3
        # spans rows 2..3 across all columns and adds row 2 col 3 plus row 3
        # (5 new): 14 distinct tiles, decoded once each.
        assert serial.batch_stats["unique_tiles"] == parallel.batch_stats["unique_tiles"] == 14
        assert serial.batch_stats["requested_tiles"] == 9 + 4 + 8
    finally:
        serial.close()
        parallel.close()


class _NoHooks(PyramidReader):
    """A backend with only _read_region: the planner must fall back per box."""

    def __init__(self, canvas):
        self._canvas = canvas
        self._levels = [PyramidLevel(reader=None, downscale=(1, 1),
                                     shape=canvas.shape, dtype=canvas.dtype)]
        self.calls = []

    @property
    def levels(self):
        return self._levels

    def _read_region(self, level, y0, y1, x0, x1):
        self.calls.append((y0, y1, x0, x1))
        return self._canvas[y0:y1, x0:x1].copy()


def test_backend_without_hooks_falls_back_in_request_order():
    canvas = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    p = _NoHooks(canvas)
    boxes = [(0, 10, 0, 10), (50, 64, 20, 40), (5, 6, 5, 6)]
    got = p.read_regions(0, _boxes_as_pairs(boxes))
    for out, (y0, y1, x0, x1) in zip(got, boxes):
        np.testing.assert_array_equal(out, canvas[y0:y1, x0:x1])
    assert p.calls == boxes
    assert p.batch_stats["fallback_boxes"] == 3 and p.batch_stats["batches"] == 0


def test_empty_request_list_yields_nothing():
    p = _open(_grid(2, 2))
    try:
        assert p.read_regions(0, []) == []
        assert p.batch_stats["boxes"] == 0
    finally:
        p.close()
