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
