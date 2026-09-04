"""``Writer`` is one of the three core ABCs. Nothing implemented it.

``core/codec.py`` opens by naming three abstractions -- Codec, Reader and
"Writer -- uniform write_frame / close" -- and exports all three. The
reader half was made true after a review; this file is the writer half.

What the audit found: six streaming writers, none of them a ``Writer``,
and only three with a ``write_frame`` at all. TIFF called it
``write_page``, CZI called it ``write``, the CZI pyramid writer called
it ``write_level``. NDTiff was the interesting one -- it had a method
named ``write_frame`` whose first positional argument was the axes dict
rather than the array, so it satisfied the name and not the contract,
and generic code calling ``w.write_frame(arr)`` broke on it alone.

The test that matters here is not ``isinstance``. It is that a function
which knows nothing about the format can drive any of them.
"""

from __future__ import annotations

import numpy as np
import pytest

from opencodecs.core.codec import Writer

FRAME = np.arange(32 * 48, dtype="u1").reshape(32, 48)


def _tiff(tmp_path):
    from opencodecs._tiff_writer import TiffWriter
    return TiffWriter(tmp_path / "w.tif"), FRAME


def _czi(tmp_path):
    from opencodecs._czi_writer import CziWriter
    return CziWriter(tmp_path / "w.czi"), FRAME


def _czi_pyramid(tmp_path):
    from opencodecs._czi_writer import CziPyramidWriter
    return CziPyramidWriter(tmp_path / "wp.czi"), FRAME


def _ndtiff(tmp_path):
    from opencodecs._ndtiff_writer import NDTiffWriter
    d = tmp_path / "nd"
    d.mkdir()
    return NDTiffWriter(d), FRAME.astype("u2")


def _gif(tmp_path):
    gif = pytest.importorskip("opencodecs.codecs._gif")
    return gif.GifWriter(width=48, height=32, loop=0), FRAME


def _jxl(tmp_path):
    pytest.importorskip("opencodecs.codecs._jxl")
    from opencodecs.jxl import JxlWriter
    # dest=None means "return the bytes from close()"
    return JxlWriter(None, lossless=True, animation=True), FRAME


WRITERS = [
    ("tiff", _tiff),
    ("czi", _czi),
    ("czi-pyramid", _czi_pyramid),
    ("ndtiff", _ndtiff),
    ("gif", _gif),
    ("jxl", _jxl),
]


@pytest.fixture(params=WRITERS, ids=[w[0] for w in WRITERS])
def writer(request, tmp_path):
    name, make = request.param
    try:
        w, frame = make(tmp_path)
    except (ImportError, RuntimeError) as exc:            # backend absent
        pytest.skip(f"{name} writer unavailable: {exc}")
    try:
        yield name, w, frame
    finally:
        try:
            w.close()
        except Exception:                                 # noqa: BLE001
            pass


def test_every_writer_implements_the_abc(writer):
    name, w, _ = writer
    assert isinstance(w, Writer), (
        f"{name}: {type(w).__name__} is not a Writer")


def test_write_frame_takes_the_frame_first(writer):
    """The NDTiff bug in one assertion.

    A caller holding a writer it did not choose passes the array. If a
    format wants more than that, the extra goes in keyword or later
    positional arguments -- not in front of the pixels.
    """
    name, w, frame = writer
    w.write_frame(frame)


def test_a_format_agnostic_function_can_drive_any_of_them(writer):
    """The whole point of the ABC, stated as code.

    This helper knows nothing about the format. Before this change it
    worked for three of the six: two had no write_frame at all, and
    NDTiff's took the axes dict where the array belongs.
    """
    name, w, frame = writer

    def write_stack(dest: Writer, frames) -> int:
        n = 0
        for f in frames:
            dest.write_frame(f)
            n += 1
        return n

    if name == "czi-pyramid":
        stack = [frame, frame[::2, ::2].copy()]
    elif name == "gif":
        stack = [frame, frame + 1]
    else:
        stack = [frame, frame + 1, frame + 2]
    assert write_stack(w, stack) == len(stack)


def test_close_is_idempotent(writer):
    """Closing twice is what a context manager plus an explicit close
    does, and it should not raise the second time."""
    name, w, frame = writer
    w.write_frame(frame)
    w.close()
    w.close()


def test_ndtiff_still_accepts_the_old_argument_order(tmp_path):
    """The order changed, so the previous spelling has to keep working.

    A dict in the first position is unambiguous, because a frame is
    never a dict.
    """
    from opencodecs._ndtiff_writer import NDTiffWriter
    d = tmp_path / "legacy"
    d.mkdir()
    a = FRAME.astype("u2")
    with NDTiffWriter(d) as w:
        rec_new = w.write_frame(a, {"z": 0})
        rec_old = w.write_frame({"z": 1}, a)          # legacy order
    assert rec_new["axes"] == {"z": 0}
    assert rec_old["axes"] == {"z": 1}


def test_ndtiff_defaults_axes_so_the_bare_call_works(tmp_path):
    """Without a default, the contract-conforming call is impossible
    for this format and the conformance would be cosmetic."""
    from opencodecs._ndtiff_writer import NDTiffWriter
    d = tmp_path / "defaulted"
    d.mkdir()
    a = FRAME.astype("u2")
    with NDTiffWriter(d) as w:
        recs = [w.write_frame(a) for _ in range(3)]
    assert [r["axes"]["z"] for r in recs] == [0, 1, 2]


def test_the_format_specific_names_still_work(tmp_path):
    """write_frame is added, not substituted: write_page and write are
    what each format calls the operation and what callers already use."""
    from opencodecs._tiff_writer import TiffWriter
    from opencodecs._czi_writer import CziWriter

    with TiffWriter(tmp_path / "p.tif") as w:
        w.write_page(FRAME)
    with CziWriter(tmp_path / "p.czi") as w:
        w.write(FRAME)


# --------------------------------------------------------------------
# oc.writer(): obtaining a writer the way oc.open() obtains a reader
# --------------------------------------------------------------------
#
# The ABC being honest is half of it. Until now there was no uniform way
# to *get* a writer: oc.open() hands back a Reader for any format, and
# the write side made you import TiffWriter or CziWriter by name. So
# Codec.writer() mirrors Codec.open(), including the default -- where
# open() decodes eagerly and wraps the result, writer() buffers frames
# and encodes on close, so a caller can drive any encoder through one
# interface and only pays the buffering where the format cannot stream.

import pathlib                                            # noqa: E402

import opencodecs as oc                                   # noqa: E402

VOLUME = [np.full((32, 48), i * 20, dtype="u1") for i in range(3)]


@pytest.mark.parametrize("name,suffix", [
    ("tiff", ".tif"), ("czi", ".czi"), ("gif", ".gif"),
])
def test_oc_writer_dispatches_by_extension(tmp_path, name, suffix):
    """One loop, four formats, no format-specific code in the caller."""
    if not oc.has_codec(name):
        pytest.skip(f"{name} codec not built")
    dest = tmp_path / f"stack{suffix}"
    with oc.writer(str(dest)) as w:
        assert isinstance(w, Writer)
        for plane in VOLUME:
            w.write_frame(plane)
    assert dest.is_file() and dest.stat().st_size > 0


def test_oc_writer_round_trips_through_the_reader(tmp_path):
    """Written by the Writer contract, read back by the Reader one."""
    pytest.importorskip("tifffile")
    dest = tmp_path / "rt.tif"
    with oc.writer(str(dest)) as w:
        for plane in VOLUME:
            w.write_frame(plane)
    with oc.open(str(dest)) as r:
        frames = list(r.iter_frames())
    assert len(frames) == len(VOLUME)
    for got, want in zip(frames, VOLUME):
        assert np.array_equal(got, want)


def test_the_buffering_default_works_for_a_non_streaming_codec():
    """png has no streaming writer, and the interface is the same."""
    if not oc.has_codec("png"):
        pytest.skip("png codec not built")
    w = oc.writer(format="png")
    assert isinstance(w, Writer)
    w.write_frame(np.zeros((16, 16, 3), dtype="u1"))
    blob = w.close()
    assert isinstance(blob, bytes) and blob[:8] == b"\x89PNG\r\n\x1a\n"
    assert w.close() is blob            # idempotent, same result


def test_a_codec_that_cannot_encode_refuses_clearly():
    if not oc.has_codec("dicom"):
        pytest.skip("dicom codec not registered")
    with pytest.raises(NotImplementedError, match="cannot encode"):
        oc.writer(format="dicom")


def test_writer_needs_a_format_when_dest_is_not_a_path():
    with pytest.raises(ValueError, match="format="):
        oc.writer(None)


def test_closing_with_no_frames_is_an_error():
    if not oc.has_codec("png"):
        pytest.skip("png codec not built")
    w = oc.writer(format="png")
    with pytest.raises(ValueError, match="without writing a frame"):
        w.close()


def test_gif_infers_the_canvas_from_the_first_frame(tmp_path):
    """GifWriter needs width/height up front; nothing else here does.

    A caller driving writers generically has no geometry to pass, so the
    canvas comes from the first frame -- which is what it had to be
    anyway, since GIF requires every frame to match it.
    """
    if not oc.has_codec("gif"):
        pytest.skip("gif codec not built")
    dest = tmp_path / "inferred.gif"
    with oc.writer(str(dest)) as w:
        for plane in VOLUME:
            w.write_frame(plane)
    with oc.open(str(dest), format="gif") as r:
        assert r.n_frames == len(VOLUME)
        assert r.shape[1:3] == VOLUME[0].shape


def test_jxl_animation_written_through_the_contract_decodes(tmp_path):
    """The quiet one: same byte count either way, only one decodes.

    libjxl marks the final frame in its header when it is submitted, and
    the Writer contract's write_frame(arr) cannot say which frame that
    is. Relying on close() produces a stream of exactly the same length
    that fails to decode, so a size check would not catch it. The
    codec's writer holds one frame back to mark it.
    """
    if not oc.has_codec("jxl"):
        pytest.skip("jxl codec not built")
    dest = tmp_path / "anim.jxl"
    with oc.writer(str(dest), animation=True, lossless=True) as w:
        for plane in VOLUME:
            w.write_frame(plane)
    with oc.open(str(dest), format="jxl") as r:
        frames = list(r.iter_frames())
    assert len(frames) == len(VOLUME)
    for got, want in zip(frames, VOLUME):
        assert np.array_equal(got, want)


def test_jxl_single_frame_still_needs_no_flag(tmp_path):
    if not oc.has_codec("jxl"):
        pytest.skip("jxl codec not built")
    dest = tmp_path / "still.jxl"
    with oc.writer(str(dest), lossless=True) as w:
        w.write_frame(VOLUME[0])
    assert np.array_equal(oc.read(str(dest)), VOLUME[0])
