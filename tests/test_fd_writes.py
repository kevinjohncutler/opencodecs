"""Raw-descriptor writes that the kernel may accept only partly.

os.write, os.writev and os.pwrite may all write less than asked. The TIFF
and NDTiff writers used to carry their own copies of the retry logic, and
the NDTiff copy advanced its position twice for the buffers after a
partial writev. Here the kernel is made to stop short on every call, and
the file, the returned sizes and the descriptor position must all still
be exact. Removing writev and pwrite also runs the paths Windows takes.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from opencodecs.core import _write_helpers as wh


@pytest.fixture
def stingy(monkeypatch):
    """os.write/writev/pwrite that write at most 5 bytes per call."""
    real_write, real_pwrite = os.write, getattr(os, "pwrite", None)

    def write(handle, data):
        return real_write(handle, memoryview(data)[:5])

    def writev(handle, buffers):
        first = next((b for b in buffers if len(memoryview(b))), b"")
        return real_write(handle, memoryview(first).cast("B")[:5])

    monkeypatch.setattr(os, "write", write)
    if hasattr(os, "writev"):
        monkeypatch.setattr(os, "writev", writev)
    if real_pwrite is not None:
        monkeypatch.setattr(os, "pwrite",
                            lambda handle, data, off: real_pwrite(handle, memoryview(data)[:5], off))


def _open(tmp_path):
    from opencodecs.core.io import O_BINARY
    return os.open(tmp_path / "f.bin", os.O_RDWR | os.O_CREAT | os.O_TRUNC | O_BINARY)


@pytest.mark.parametrize("have_vector", [True, False])
def test_partial_writes_still_write_everything_once(tmp_path, stingy, monkeypatch, have_vector):
    if not have_vector:
        monkeypatch.delattr(os, "writev", raising=False)
        monkeypatch.delattr(os, "pwrite", raising=False)
    handle = _open(tmp_path)
    try:
        parts = [b"header--", np.arange(37, dtype="<u2"), memoryview(b""), bytearray(b"tail")]
        expected = b"".join(bytes(memoryview(p).cast("B")) for p in parts)
        assert wh.fd_write_all(handle, b"0123456789ab") == 12
        assert wh.fd_writev_all(handle, parts) == len(expected)
        assert os.lseek(handle, 0, os.SEEK_CUR) == 12 + len(expected)
        wh.fd_pwrite_all(handle, b"XYZWVU", 2)
        assert os.lseek(handle, 0, os.SEEK_CUR) == 12 + len(expected)  # unmoved
    finally:
        os.close(handle)
    data = (tmp_path / "f.bin").read_bytes()
    assert data == b"01XYZWVU89ab" + expected


def test_writev_splits_more_buffers_than_the_kernel_takes(tmp_path, monkeypatch):
    monkeypatch.setattr(wh, "_IOV_MAX", 7)
    handle = _open(tmp_path)
    try:
        parts = [bytes([i]) * (i % 5 + 1) for i in range(40)]
        assert wh.fd_writev_all(handle, parts) == sum(map(len, parts))
    finally:
        os.close(handle)
    assert (tmp_path / "f.bin").read_bytes() == b"".join(parts)


def test_writers_survive_partial_writes(tmp_path, stingy):
    """Both writers, end to end, with every write cut short."""
    tifffile = pytest.importorskip("tifffile")
    import opencodecs as oc
    from opencodecs._ndtiff import NDTiffDataset
    from opencodecs._ndtiff_writer import NDTiffWriter
    frames = [np.random.default_rng(i).integers(0, 4000, (24, 31), dtype=np.uint16)
              for i in range(4)]
    with NDTiffWriter(tmp_path / "nd", summary={"PixelType": "uint16"}) as w:
        for i, arr in enumerate(frames):
            w.write_frame({"z": i}, arr, metadata={"z_um": i})
    with NDTiffDataset(tmp_path / "nd") as ds:
        for i, arr in enumerate(frames):
            np.testing.assert_array_equal(ds.read_frame(z=i), arr)
    path = tmp_path / "t.tif"
    with oc.TiffWriter(str(path)) as w:
        for arr in frames:
            w.write_page(arr)
    np.testing.assert_array_equal(tifffile.imread(path), np.stack(frames))
