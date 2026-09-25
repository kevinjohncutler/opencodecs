"""Validation and ownership for caller-supplied decode destinations.

Byte decoders return a memoryview of the written prefix when given storage.
This keeps the returned result attached to the caller's buffer, including
bytearray destinations, whose ordinary slices would copy. Integer size hints
retain the native codec's allocated-bytes behavior.
"""

from __future__ import annotations

import numpy as np


def byte_output(out):
    """Validate byte storage and return a writable contiguous byte view.

    The returned view owns its own buffer export. Releasing it does not release
    a caller's memoryview. Native codecs check the required output capacity.
    """
    if out is None:
        return None
    if isinstance(out, int):
        if out < 0:
            raise ValueError("decode out size must be nonnegative")
        return out
    try:
        view = memoryview(out)
    except TypeError as exc:
        raise TypeError("decode out must be an integer or writable buffer") from exc
    if view.readonly:
        raise ValueError("decode out must be writable")
    if not view.c_contiguous:
        raise ValueError("decode out must be C-contiguous")
    return view.cast("B")


def array_output(out):
    """Validate array storage before a native decoder writes through a pointer.

    The native decoder remains responsible for its exact shape and dtype.
    """
    if out is None:
        return None
    if not isinstance(out, np.ndarray):
        raise TypeError("decode out must be an ndarray")
    if not out.flags.writeable:
        raise ValueError("decode out must be writable")
    if not out.flags.c_contiguous:
        raise ValueError("decode out must be C-contiguous")
    return out


class CallbackDestination:
    """Borrow a sink and retain callback exceptions for native error propagation."""

    def __init__(self, dest):
        self.dest = dest
        self.error = None

    def write(self, data):
        from ._write_helpers import write_all
        write_all(self.dest, data)


class ImageDecoderContext:
    """Public source/output normalization around an owned native decoder."""

    def __init__(self, native):
        self._native = native

    def decode(self, src, *, out=None, **options):
        from ._io_helpers import read_src
        return self._native.decode(read_src(src), out=array_output(out), **options)

    def close(self):
        self._native.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


class SeekableDestination(CallbackDestination):
    """Relative native offsets within a borrowed seekable destination."""

    def __init__(self, dest):
        super().__init__(dest)
        self.base = dest.tell()

    def write_at(self, offset, data):
        self.dest.seek(self.base + offset)
        self.write(data)

    def finish(self, size):
        self.dest.seek(self.base + size)
