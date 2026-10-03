"""Rewriting a CZI with different compression, keeping the container.

``czi_recompress`` is the operation ZEISS's ``czicompress`` performs. Doing it
here is only worth anything if the result is the same file with a different
payload encoding, so these check the container as hard as the pixels: a
rewrite that silently drops a channel index, a scene, a mosaic position or a
pyramid level still decodes to plausible-looking images, and nothing about
comparing arrays would catch it.

Three properties of ``CziSubBlockEntry`` have to be undone to rebuild what the
file stores, and each of them was a real bug found against real files:

- ``dims`` is in REVERSE file order, so writing it straight back produced a
  file whose axes read back reversed;
- its last element is a synthetic ``"S"`` for samples-per-pixel, not a
  dimension, and emitting it made ``scene_index`` read back as 0 rather than
  absent;
- ``pyramid_type`` is not in ``dims`` at all, so every level of a slide scan
  came back marked as full resolution.
"""
from __future__ import annotations

import contextlib
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from _czi_fixture import mosaic_czi_bytes, pyramid_czi_bytes  # noqa: E402

from opencodecs._czi_reader import CziReader  # noqa: E402
from opencodecs._czi_writer import (  # noqa: E402
    CziCancelled, CziWriter, CziWriterError, czi_recompress, subblock_dims,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
CORPUS = ROOT / ".test_data" / "czi"


def _key(e):
    """Everything about a sub-block except where its bytes happen to sit."""
    return (e.dims, e.start, e.shape, e.stored_shape,
            e.mosaic_index, e.scene_index, e.pixel_type, e.pyramid_type)


def _tile(h, w, seed):
    return np.random.RandomState(seed).randint(0, 4096, (h, w)).astype(np.uint16)


def _assert_faithful(src, dst):
    """The rewrite kept every sub-block's identity, place and pixels."""
    with CziReader(str(src)) as a, CziReader(str(dst)) as b:
        assert len(a.entries) == len(b.entries), "sub-block count changed"
        assert [_key(e) for e in a.entries] == [_key(e) for e in b.entries], \
            "a sub-block's dimensions, position, scene, mosaic index or " \
            "pyramid level changed"
        assert a.metadata_bytes == b.metadata_bytes, "metadata XML not carried"
        for i in range(len(a.entries)):
            assert np.array_equal(np.squeeze(a[i]), np.squeeze(b[i])), \
                f"sub-block {i} pixels differ"


# ------------------------------------------------------------ subblock_dims

def test_subblock_dims_restores_file_order():
    """The reader reports axes slowest-first; the file stores them the other way."""
    tiles = [(_tile(32, 48, 1), (0, 0), [(b"C", 0)]),
             (_tile(32, 48, 2), (0, 48), [(b"C", 1)])]
    with CziReader(buffer=mosaic_czi_bytes(tiles)) as r:
        entry = r.entries[0]
        names = [d[0].rstrip(b"\x00").decode() for d in subblock_dims(entry)]
        # reader order is reversed, and its trailing samples axis is not a
        # dimension of the file at all
        assert names == list(reversed(entry.dims[:-1]))
        assert entry.dims[-1] == "S", "reader still appends a samples axis"


def test_subblock_dims_drops_the_samples_axis():
    tiles = [(_tile(16, 16, 3), (0, 0))]
    with CziReader(buffer=mosaic_czi_bytes(tiles)) as r:
        entry = r.entries[0]
        assert len(subblock_dims(entry)) == len(entry.dims) - 1


# --------------------------------------------------------------- round trip

@pytest.mark.parametrize("compression", ["zstdhdr", "zstd", "none"])
def test_roundtrip_keeps_container_and_pixels(tmp_path, compression):
    tiles = [(_tile(64, 64, i), (0, i * 64), [(b"C", i)]) for i in range(4)]
    src = tmp_path / "src.czi"
    src.write_bytes(mosaic_czi_bytes(tiles, compression=0,
                                     metadata_xml=b"<Metadata><X/></Metadata>"))
    dst = tmp_path / "dst.czi"
    info = czi_recompress(src, dst, compression=compression)
    assert info["subblocks"] == 4
    assert info["compression"] == compression
    _assert_faithful(src, dst)


def test_roundtrip_keeps_pyramid_levels(tmp_path):
    """A level marked as down-scaled must not come back as full resolution."""
    base = _tile(128, 128, 7)
    src = tmp_path / "src.czi"
    src.write_bytes(pyramid_czi_bytes([base, base[::2, ::2], base[::4, ::4]]))
    with CziReader(str(src)) as r:
        assert any(e.is_pyramid for e in r.entries), "fixture is not pyramidal"
    dst = tmp_path / "dst.czi"
    czi_recompress(src, dst)
    _assert_faithful(src, dst)


def test_roundtrip_keeps_negative_mosaic_starts(tmp_path):
    """Stage coordinates go negative on a real slide; that is not an error."""
    tiles = [(_tile(32, 32, 11), (-64, -128)), (_tile(32, 32, 12), (-64, -96))]
    src = tmp_path / "src.czi"
    src.write_bytes(mosaic_czi_bytes(tiles))
    dst = tmp_path / "dst.czi"
    czi_recompress(src, dst)
    _assert_faithful(src, dst)
    with CziReader(str(dst)) as r:
        assert min(e.start[-2] for e in r.entries) < 0


def test_recompress_reports_what_it_did(tmp_path):
    tiles = [(_tile(64, 64, 21), (0, 0))]
    src = tmp_path / "src.czi"
    src.write_bytes(mosaic_czi_bytes(tiles, compression=0))
    dst = tmp_path / "dst.czi"
    info = czi_recompress(src, dst, compression="zstdhdr")
    assert info["subblocks"] == 1
    assert info["src_bytes"] == src.stat().st_size
    assert info["dst_bytes"] == dst.stat().st_size


def test_recompress_verifies_when_asked(tmp_path):
    tiles = [(_tile(64, 64, 31), (0, 0), [(b"C", 0)]),
             (_tile(64, 64, 32), (0, 64), [(b"C", 1)])]
    src = tmp_path / "src.czi"
    src.write_bytes(mosaic_czi_bytes(tiles, compression=0))
    dst = tmp_path / "dst.czi"
    czi_recompress(src, dst, compression="zstdhdr", verify=True)
    _assert_faithful(src, dst)


def test_recompress_refuses_an_empty_file(tmp_path):
    src = tmp_path / "src.czi"
    src.write_bytes(mosaic_czi_bytes([(_tile(8, 8, 41), (0, 0))]))
    with CziReader(str(src)) as r:
        r.entries.clear()
    # a real empty directory, built by hand
    with pytest.raises(Exception):
        czi_recompress(tmp_path / "missing.czi", tmp_path / "out.czi")


# ------------------------------------------------------------- writer input

def test_write_frame_places_a_sub_block(tmp_path):
    out = tmp_path / "out.czi"
    a, b = _tile(32, 32, 51), _tile(32, 32, 52)
    with CziWriter(out, compression="zstdhdr") as w:
        w.write_frame(a, dims=[("X", 0, 32, 32), ("Y", 0, 32, 32), ("C", 0, 1, 1)])
        w.write_frame(b, dims=[("X", 0, 32, 32), ("Y", 0, 32, 32), ("C", 1, 1, 1)])
    with CziReader(str(out)) as r:
        assert len(r.entries) == 2
        cs = sorted(e.start[r.entries[0].dims.index("C")] for e in r.entries)
        assert cs == [0, 1], "the channel index did not survive"
        assert np.array_equal(np.squeeze(r[0]), a)
        assert np.array_equal(np.squeeze(r[1]), b)


def test_write_many_accepts_dims_and_pyramid_type(tmp_path):
    out = tmp_path / "out.czi"
    tiles = [_tile(32, 32, 60 + i) for i in range(3)]
    frames = [
        (tiles[0], [("X", 0, 32, 32), ("Y", 0, 32, 32)], 0),
        (tiles[1], [("X", 0, 64, 32), ("Y", 0, 64, 32)], 2),
        (tiles[2], [("X", 0, 32, 32), ("Y", 0, 32, 32)]),
    ]
    with CziWriter(out, compression="zstdhdr") as w:
        w.write_many(frames, workers=2)
    with CziReader(str(out)) as r:
        assert [e.pyramid_type for e in r.entries] == [0, 2, 0]
        for i, t in enumerate(tiles):
            assert np.array_equal(np.squeeze(r[i]), t)


@pytest.mark.parametrize("dims,message", [
    ([("X", 0, 32)], "dimension is"),
    ([("XYZAB", 0, 32, 32)], "1-4 characters"),
    ([("X", 0, -1, 32)], "negative size"),
    ([("X", 0, 32, -5)], "negative stored_size"),
    ([], "at least one dimension"),
])
def test_bad_dims_are_refused(tmp_path, dims, message):
    # Not a `with` block: the rejected write leaves the file empty, and
    # close() then raises its own "no frames written", masking the error
    # under test.
    w = CziWriter(tmp_path / "out.czi", compression="none")
    try:
        with pytest.raises(CziWriterError, match=message):
            w.write_frame(_tile(32, 32, 71), dims=dims)
    finally:
        with contextlib.suppress(CziWriterError):
            w.close()


def test_write_many_refuses_a_malformed_frame(tmp_path):
    w = CziWriter(tmp_path / "out.czi", compression="none")
    try:
        with pytest.raises(CziWriterError, match="array, "):
            w.write_many([(_tile(8, 8, 81), None, 0, "extra")])
    finally:
        with contextlib.suppress(CziWriterError):
            w.close()


# ------------------------------------------------------- against real files

@pytest.mark.skipif(not (CORPUS / "idr0011_plate1_scene1.czi").is_file(),
                    reason="corpus CZI not present")
def test_roundtrip_a_real_multichannel_file(tmp_path):
    src = CORPUS / "idr0011_plate1_scene1.czi"
    dst = tmp_path / "out.czi"
    info = czi_recompress(src, dst, compression="zstdhdr", workers=4)
    assert info["subblocks"] == 63
    _assert_faithful(src, dst)


@pytest.mark.slow
@pytest.mark.skipif(not (CORPUS / "ome_axioscan_pyramid.czi").is_file(),
                    reason="corpus slide scan not present")
def test_roundtrip_a_real_slide_scan(tmp_path):
    """481 sub-blocks, JPEG XR in, pyramidal, negative stage coordinates."""
    src = CORPUS / "ome_axioscan_pyramid.czi"
    dst = tmp_path / "out.czi"
    czi_recompress(src, dst, compression="zstdhdr", workers=4)
    _assert_faithful(src, dst)


# ------------------------------------------------------- cancellation
# A cancelled rewrite has one job beyond stopping: leave nothing behind. A CZI
# missing sub-blocks still has a valid header and directory, so a reader opens
# it without complaint and the loss is silent.
#
# THE FIXTURE SIZE IS LOAD-BEARING, and an earlier version of these tests was
# vacuous because of it. write_many groups frames into ~2 MiB tasks, a failing
# task emits nothing, and the writer opens the output file lazily on the FIRST
# emitted sub-block. With few small tiles the generator raises while tasks are
# still being submitted, so nothing is ever emitted, the file is never created,
# and "no partial file survives" passes whether or not the cleanup exists.
# Measured on 2026-10-02:
#     tiles=4  side=1200 stop=3  workers=None -> no file  (cleanup unreachable)
#     tiles=8  side=1200 stop=5  workers=1    -> 8.7 MB partial file
#     tiles=16 side=1200 stop=10 workers=1    -> 19.7 MB partial file
# Enough tiles that the producer must drain results to make room is what forces
# real emission before the cancel. That is also the real case: a slide scan.

PARTIAL_TILES, PARTIAL_SIDE, PARTIAL_STOP, PARTIAL_WORKERS = 8, 1200, 5, 1


def _src(tmp_path, ntiles=4, side=64):
    tiles = [(_tile(side, side, 90 + i), (0, i * side), [(b"C", i)])
             for i in range(ntiles)]
    src = tmp_path / "src.czi"
    src.write_bytes(mosaic_czi_bytes(tiles, compression=0))
    return src


def test_cancel_removes_a_partial_file_that_really_exists(tmp_path):
    """The case that actually exercises the cleanup.

    Sized so sub-blocks are emitted before the cancel; without the cleanup this
    leaves a multi-megabyte file that still parses as a CZI.
    """
    src = _src(tmp_path, PARTIAL_TILES, PARTIAL_SIDE)
    dst = tmp_path / "dst.czi"
    seen = {"n": 0}

    def should_continue():
        seen["n"] += 1
        return seen["n"] <= PARTIAL_STOP

    with pytest.raises(CziCancelled):
        czi_recompress(src, dst, workers=PARTIAL_WORKERS,
                       should_continue=should_continue)
    assert not dst.exists(), (
        "a cancelled rewrite left a partial CZI behind; without the cleanup "
        "this file is several MB and readers accept it")
    assert src.exists(), "the source must never be touched"


@pytest.mark.parametrize("stop_after", [0, 1, 3])
def test_cancel_raises_at_any_point(tmp_path, stop_after):
    src = _src(tmp_path)
    dst = tmp_path / "dst.czi"
    seen = {"n": 0}

    def should_continue():
        seen["n"] += 1
        return seen["n"] <= stop_after

    with pytest.raises(CziCancelled) as exc:
        czi_recompress(src, dst, should_continue=should_continue)
    assert "cancelled after" in str(exc.value)
    assert not dst.exists()


def test_cancel_is_a_writer_error():
    """Existing callers that catch CziWriterError keep working."""
    assert issubclass(CziCancelled, CziWriterError)


def test_should_continue_true_completes_normally(tmp_path):
    src = _src(tmp_path)
    dst = tmp_path / "dst.czi"
    calls = {"n": 0}

    def always():
        calls["n"] += 1
        return True

    info = czi_recompress(src, dst, should_continue=always)
    assert info["subblocks"] == 4
    assert calls["n"] == 4, "should_continue is polled once per sub-block"
    _assert_faithful(src, dst)


def test_default_is_unaffected(tmp_path):
    """Omitting should_continue must not change anything."""
    src = _src(tmp_path)
    dst = tmp_path / "dst.czi"
    czi_recompress(src, dst)
    _assert_faithful(src, dst)


def test_a_failing_predicate_also_leaves_no_output(tmp_path):
    """A buggy predicate is a failure like any other, and is sized to emit."""
    src = _src(tmp_path, PARTIAL_TILES, PARTIAL_SIDE)
    dst = tmp_path / "dst.czi"
    calls = {"n": 0}

    def boom():
        calls["n"] += 1
        if calls["n"] > PARTIAL_STOP:
            raise ValueError("predicate blew up")
        return True

    with pytest.raises(ValueError):
        czi_recompress(src, dst, workers=PARTIAL_WORKERS, should_continue=boom)
    assert not dst.exists()
