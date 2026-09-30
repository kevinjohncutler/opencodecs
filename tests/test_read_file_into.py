"""core.io.read_file_into: parallel positioned reads straight into an array.

Every layout of the split (one part, several, a last short part, chunks
smaller than a part) must give the bytes of the file, with and without
os.preadv. Removing it here runs the path Windows takes on every
platform, so CI checks both wherever it runs.
"""
from __future__ import annotations

import io
import os

import numpy as np
import pytest

from opencodecs.core import io as core_io
from opencodecs.core.io import read_file_into


@pytest.fixture
def blob(tmp_path):
    data = np.random.default_rng(3).integers(0, 256, (40 << 20) + 12345, dtype=np.uint8)
    path = tmp_path / "blob.bin"
    path.write_bytes(data.tobytes())
    return path, data


@pytest.mark.parametrize("positioned", [True, False])
@pytest.mark.parametrize("numthreads", [None, 1, 3, 8])
def test_reads_exactly_the_requested_range(blob, monkeypatch, positioned, numthreads):
    path, data = blob
    if not positioned:
        monkeypatch.delattr(os, "preadv", raising=False)
    elif not hasattr(os, "preadv"):
        pytest.skip("this platform has no os.preadv")
    # The kernel may return short; make it, so every part takes several calls.
    if positioned:
        real_preadv = os.preadv
        monkeypatch.setattr(os, "preadv", lambda h, bufs, off: real_preadv(
            h, [memoryview(bufs[0])[:3 << 20]], off))
    else:
        class Stingy(io.RawIOBase):
            def __init__(self, *a, **k):
                self.raw = open(*a, **k)
            def seek(self, *a):
                return self.raw.seek(*a)
            def readinto(self, b):
                return self.raw.readinto(memoryview(b)[:3 << 20])
            def close(self):
                self.raw.close()
        monkeypatch.setattr(core_io, "open", Stingy, raising=False)
    for offset, n in ((0, data.size), (1001, 25 << 20), (7, 1), (data.size - 5, 5)):
        out = np.zeros(n, np.uint8)
        read_file_into(str(path), offset, out, numthreads=numthreads)
        np.testing.assert_array_equal(out, data[offset:offset + n])


def test_a_short_file_raises(blob):
    path, data = blob
    out = np.zeros(1024, np.uint8)
    with pytest.raises(OSError, match="ends"):
        read_file_into(str(path), data.size - 100, out)


def test_empty_view_reads_nothing(blob):
    path, _ = blob
    read_file_into(str(path), 0, np.zeros(0, np.uint8))


def _which_path(monkeypatch):
    """Record the strategy each call takes: "reads", "mapping" (one copy)
    or "parts" (a copy split across threads, which then logs its pieces)."""
    used = []
    real_copy, real_parts = core_io._copy_from_mapping, core_io._copy_parts_from_mapping
    real_reads = core_io._positioned_reads
    monkeypatch.setattr(core_io, "_copy_from_mapping",
                        lambda *a, **k: (used.append("mapping"), real_copy(*a, **k))[1])
    monkeypatch.setattr(core_io, "_copy_parts_from_mapping",
                        lambda *a: (used.append("parts"), real_parts(*a))[1])
    monkeypatch.setattr(core_io, "_positioned_reads",
                        lambda *a: (used.append("reads"), real_reads(*a))[1])
    return used


def _set_kernel(monkeypatch, alone, among_others):
    monkeypatch.setattr(core_io, "_COPY_BEATS_READ_ALONE", alone)
    monkeypatch.setattr(core_io, "_COPY_BEATS_READ_AMONG_OTHERS", among_others)


@pytest.mark.parametrize("alone", [True, False])
def test_a_lone_read_copies_one_threads_worth_where_copying_wins(blob, monkeypatch, alone):
    """Every setting of the kernel properties runs here whatever the platform,
    and a lone read ignores the one for reads among others."""
    import mmap
    path, data = blob
    _set_kernel(monkeypatch, alone, not alone)
    used = _which_path(monkeypatch)
    with open(path, "rb") as fh:
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            for numthreads in (1, None):
                out = np.zeros(20 << 20, np.uint8)
                read_file_into(str(path), 12345, out, numthreads=numthreads, mapping=mm)
                np.testing.assert_array_equal(out, data[12345:12345 + out.size])
        finally:
            mm.close()
    # One thread's worth copies where copying wins alone; a split read never does.
    assert used == ["mapping" if alone else "reads", "reads"]


@pytest.mark.parametrize("among_others", [True, False])
@pytest.mark.parametrize("numthreads", [1, 3])
def test_a_read_among_others_follows_the_kernel(blob, monkeypatch, among_others, numthreads):
    """Both settings run here whatever the platform, so both stay tested; a
    copy is split into the parts the budget gives, as a read would be."""
    import mmap
    from opencodecs.core import parallel
    path, data = blob
    _set_kernel(monkeypatch, not among_others, among_others)
    used = _which_path(monkeypatch)
    with open(path, "rb") as fh, parallel._in_flight():  # another call running
        mm = mmap.mmap(fh.fileno(), 0, access=mmap.ACCESS_READ)
        try:
            out = np.zeros(20 << 20, np.uint8)
            read_file_into(str(path), 7, out, numthreads=numthreads, mapping=mm)
        finally:
            mm.close()
    np.testing.assert_array_equal(out, data[7:7 + out.size])
    if not among_others:
        assert used == ["reads"]
    elif numthreads == 1:
        assert used == ["mapping"]
    else:
        assert used == ["parts"] + ["mapping"] * numthreads
