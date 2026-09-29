"""Owned destinations and bounded array serialization for container writers."""

from contextlib import contextmanager
import os

import numpy as np


@contextmanager
def binary_destination(dest):
    """Borrow a caller's file object or open and close a destination path."""
    if hasattr(dest, "write"):
        yield dest
    else:
        with open(os.fspath(dest), "wb") as stream:
            yield stream


def write_all(dest, data):
    """Write all bytes, including to destinations accepting short writes."""
    view = memoryview(data).cast("B")
    while view:
        written = dest.write(view)
        if written is None or written <= 0 or written > len(view):
            raise OSError("destination did not accept a valid byte count")
        view = view[written:]


def iter_array_buffers(array, *, dtype=None, order="C", buffer_bytes=1 << 20):
    """Yield bounded byte views in storage order, converting only one block.

    Views may borrow a reusable iterator buffer. Consume each view before
    advancing the iterator. Contiguous arrays with matching dtype are borrowed.
    """
    arr = np.asarray(array)
    target = np.dtype(dtype if dtype is not None else arr.dtype)
    if arr.dtype.hasobject or target.hasobject:
        raise ValueError("object arrays cannot be serialized as raw bytes")
    if order not in ("C", "F"):
        raise ValueError("order must be C or F")
    if target.itemsize == 0:
        return
    if buffer_bytes < target.itemsize:
        raise ValueError("buffer_bytes must hold at least one element")
    count = max(1, buffer_bytes // target.itemsize)
    contiguous = arr.flags.c_contiguous if order == "C" else arr.flags.f_contiguous
    if contiguous and arr.dtype == target:
        flat = arr.reshape(-1, order=order)
        for offset in range(0, flat.size, count):
            yield memoryview(flat[offset:offset + count]).cast("B")
        return
    with np.nditer(arr, flags=["external_loop", "buffered", "zerosize_ok"],
                   op_flags=["readonly"], op_dtypes=[target], order=order,
                   casting="unsafe", buffersize=count) as iterator:
        for block in iterator:
            for offset in range(0, block.size, count):
                # A strided external loop can bypass the conversion buffer.
                chunk = np.ascontiguousarray(block[offset:offset + count])
                yield memoryview(chunk).cast("B")


# Raw file descriptors ------------------------------------------------------
#
# The TIFF and NDTiff writers each carried a copy of this, and the copies
# had drifted: one chunked writev by IOV_MAX and one did not, and one
# counted a partially written writev twice. The platform differences are
# capabilities, probed rather than named: writev and pwrite are missing on
# Windows.

try:
    # Half the platform's iovec limit, so a resubmitted tail has headroom.
    _IOV_MAX = max(64, min(512, os.sysconf("SC_IOV_MAX") // 2))
except (ValueError, OSError, AttributeError):  # no sysconf (Windows)
    _IOV_MAX = 512


def _byte_view(data):
    return memoryview(data).cast("B")


def fd_write_all(handle: int, data) -> int:
    """Write all of ``data`` at the descriptor's position; returns its size."""
    view = _byte_view(data)
    done = 0
    while done < view.nbytes:
        n = os.write(handle, view[done:])
        if n <= 0:
            raise OSError("short write")
        done += n
    return view.nbytes


def fd_writev_all(handle: int, buffers) -> int:
    """Write every buffer in order, as few writev calls as the kernel allows.

    Returns the total size. A writev that stops partway is finished with
    plain writes before the next call, so the file and the returned size
    always match the buffers exactly.
    """
    views = [_byte_view(b) for b in buffers]
    total = sum(v.nbytes for v in views)
    if not hasattr(os, "writev"):
        for v in views:
            fd_write_all(handle, v)
        return total
    for i in range(0, len(views), _IOV_MAX):
        chunk = views[i:i + _IOV_MAX]
        n = os.writev(handle, chunk)
        for v in chunk:
            if n >= v.nbytes:
                n -= v.nbytes
                continue
            fd_write_all(handle, v[n:])
            n = 0
    return total


def fd_pwrite_all(handle: int, data, offset: int) -> None:
    """Write all of ``data`` at ``offset`` without moving the position."""
    view = _byte_view(data)
    if hasattr(os, "pwrite"):
        done = 0
        while done < view.nbytes:
            n = os.pwrite(handle, view[done:], offset + done)
            if n <= 0:
                raise OSError("short positional write")
            done += n
        return
    saved = os.lseek(handle, 0, os.SEEK_CUR)
    try:
        os.lseek(handle, offset, os.SEEK_SET)
        fd_write_all(handle, view)
    finally:
        os.lseek(handle, saved, os.SEEK_SET)


class CompleteWriter:
    """Adapt a borrowed destination for libraries that assume complete writes."""

    def __init__(self, dest):
        self.dest = dest

    def write(self, data):
        write_all(self.dest, data)
        return memoryview(data).nbytes

    def flush(self):
        flush = getattr(self.dest, "flush", None)
        if flush is not None:
            flush()
