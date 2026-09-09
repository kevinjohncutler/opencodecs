"""Multi-frame DICOM decodes across threads.

Every encapsulated frame is its own codestream and every native frame
is its own run of bytes at a known offset, so frames are independent
and threading them needs no format work. What that is worth depends
entirely on whether the per-frame decode releases the GIL, which is a
property of the codec rather than of DICOM, so these tests measure it
instead of assuming it.

The corpus has no multi-frame encapsulated file -- the largest is 10
native 64x64 frames -- so the timing fixture is built here.
"""

from __future__ import annotations

import struct
import time

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._dicom import DicomFile

from test_dicom_encodings import _meta, explicit_le

J2K_LOSSLESS = "1.2.840.10008.1.2.4.90"


def build_encapsulated(frames: np.ndarray, ts: str = J2K_LOSSLESS) -> bytes:
    """A multi-frame file with one codestream fragment per frame.

    Includes a real Basic Offset Table, because the reader's fragment
    mapping has a fallback for files that omit it and the point here is
    to exercise the direct path.
    """
    enc = explicit_le()
    n, rows, cols = frames.shape
    codec = oc.get_codec("jpeg2k")
    fragments = [codec.encode(frames[i], level=0) for i in range(n)]
    fragments = [f + b"\x00" * (len(f) % 2) for f in fragments]

    bot, off = b"", 0
    for f in fragments:
        bot += struct.pack("<I", off)
        off += 8 + len(f)

    items = struct.pack("<HHI", 0xFFFE, 0xE000, len(bot)) + bot
    for f in fragments:
        items += struct.pack("<HHI", 0xFFFE, 0xE000, len(f)) + f
    items += struct.pack("<HHI", 0xFFFE, 0xE0DD, 0)

    body = (enc.us((0x0028, 0x0002), 1)
            + enc.element((0x0028, 0x0008), b"IS",
                          str(n).encode().ljust((len(str(n)) + 1) // 2 * 2))
            + enc.us((0x0028, 0x0010), rows)
            + enc.us((0x0028, 0x0011), cols)
            + enc.us((0x0028, 0x0100), frames.dtype.itemsize * 8)
            + enc.us((0x0028, 0x0101), frames.dtype.itemsize * 8)
            + enc.us((0x0028, 0x0102), frames.dtype.itemsize * 8 - 1)
            + enc.us((0x0028, 0x0103), 0))
    body += (struct.pack("<HH", 0x7FE0, 0x0010) + b"OB" + b"\x00\x00"
             + struct.pack("<I", 0xFFFFFFFF) + items)
    return _meta(ts) + body


@pytest.fixture(scope="module")
def stack():
    rng = np.random.default_rng(0)
    return (rng.integers(0, 4000, (24, 256, 256))).astype("u2")


@pytest.fixture(scope="module")
def encapsulated(stack, tmp_path_factory):
    p = tmp_path_factory.mktemp("dcm") / "multi.dcm"
    p.write_bytes(build_encapsulated(stack))
    return p


@pytest.mark.parametrize("numthreads", [None, 1, 2, 4, 8])
def test_threaded_matches_serial(encapsulated, stack, numthreads):
    with DicomFile(str(encapsulated)) as d:
        got = d.asarray(numthreads=numthreads)
    assert got.shape == stack.shape
    assert np.array_equal(got, stack)


def test_native_multiframe_matches_serial(tmp_path):
    """Native Pixel Data takes the has_decode_work=False branch, so an
    explicit thread count is deliberately ignored. It must still be
    correct, which is the half that matters."""
    from test_dicom_encodings import build_dicom
    a = (np.arange(12 * 16 * 20) % 4000).astype("u2").reshape(12, 16, 20)
    p = tmp_path / "native.dcm"
    p.write_bytes(build_dicom(explicit_le(), a))
    with DicomFile(str(p)) as d:
        assert np.array_equal(d.asarray(numthreads=8), a)
        assert np.array_equal(d.asarray(numthreads=1), a)


def test_single_frame_is_unchanged_by_threading(tmp_path):
    from test_dicom_encodings import build_dicom
    a = (np.arange(16 * 20) % 4000).astype("u2").reshape(16, 20)
    p = tmp_path / "one.dcm"
    p.write_bytes(build_dicom(explicit_le(), a))
    with DicomFile(str(p)) as d:
        out = d.asarray(numthreads=8)
    assert out.shape == a.shape          # not (1, 16, 20)
    assert np.array_equal(out, a)


@pytest.mark.perf
def test_threading_actually_helps(encapsulated):
    """The claim in the manifest is a speedup, so measure one.

    A codec that holds the GIL through its decode would pass every
    correctness test above and make parallel_decode a lie.
    """
    def run(nt):
        with DicomFile(str(encapsulated)) as d:
            d.asarray(numthreads=nt)          # warm
        best = min(_time(lambda: _read(encapsulated, nt)) for _ in range(3))
        return best

    def _read(p, nt):
        with DicomFile(str(p)) as d:
            d.asarray(numthreads=nt)

    def _time(fn):
        t = time.perf_counter()
        fn()
        return time.perf_counter() - t

    serial, threaded = run(1), run(8)
    got = serial / threaded
    # 1.2x, not the 1.86x measured on an 8-core workstation, and the
    # gap is not slack -- it is what this particular speedup is worth
    # on a small machine. openjpeg runs its own T1 threads even at
    # numthreads=1, so the "serial" baseline here is already parallel
    # and the frame pool only adds what is left. A 4-core CI runner
    # measured exactly 1.50 against a `> 1.5` bar and failed.
    #
    # The strong, machine-independent form of this claim is in
    # test_decode_releases_gil.py, where openjpeg is pinned to one
    # thread and the fan-out shows 7.4x. That is the test protecting
    # the GIL fix; this one only has to catch the frame pool being
    # removed outright.
    assert got > 1.2, (
        f"{got:.2f}x on 8 threads; the per-frame pool looks gone")
