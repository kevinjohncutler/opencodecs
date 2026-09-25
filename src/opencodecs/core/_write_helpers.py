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
