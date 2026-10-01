"""Validation and ownership for caller-supplied decode destinations.

Byte decoders return a memoryview of the written prefix when given storage.
This keeps the returned result attached to the caller's buffer, including
bytearray destinations, whose ordinary slices would copy. Decoders that
pass their result through :func:`byte_result` return the destination
itself when the result fills it, and an ndarray view of the prefix for
an ndarray destination, as imagecodecs does. Integer size hints
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


def _deliver(data, out, what):
    """Copy finished ``data`` into the caller's ``out``, as imagecodecs does.

    ``out`` follows imagecodecs' contract: None returns ``data``; the
    ``bytearray`` type returns a new bytearray; an int is a capacity
    (``data`` must fit in it); a writable buffer receives ``data`` at
    its start, and the result is that buffer when ``data`` fills it
    exactly, otherwise the written prefix: a uint8 ndarray view when
    ``out`` is an ndarray, a memoryview for any other buffer. Data that
    does not fit raises ``ValueError``: it is never truncated.
    """
    if out is None:
        return data
    if out is bytearray:
        return bytearray(data)
    if isinstance(out, int):
        if out < 0:
            raise ValueError(f"{what} out size must be nonnegative")
        if len(data) > out:
            raise ValueError(
                f"{what}: the result is {len(data)} bytes, more than "
                f"out={out}")
        return data
    try:
        view = memoryview(out)
    except TypeError as exc:
        raise TypeError(
            f"{what} out must be an integer, the bytearray type or a "
            f"writable buffer, got {type(out).__name__}") from exc
    if view.readonly:
        raise ValueError(f"{what} out must be writable")
    if not view.c_contiguous:
        raise ValueError(f"{what} out must be C-contiguous")
    view = view.cast("B")
    if len(view) < len(data):
        raise ValueError(
            f"{what}: the result is {len(data)} bytes, the out= buffer "
            f"holds {len(view)}")
    view[:len(data)] = data
    if len(view) == len(data):
        return out
    if isinstance(out, np.ndarray):
        return _ndarray_prefix(out, len(data))
    return view[:len(data)]


def _ndarray_prefix(out, nbytes):
    """Return the first ``nbytes`` of C-contiguous ``out`` as imagecodecs does.

    imagecodecs returns ``out`` itself when the result fills it and an
    ndarray slice of it otherwise. A view, never a copy.
    """
    if nbytes == out.nbytes:
        return out
    return out.reshape(-1).view(np.uint8)[:nbytes]


def byte_result(result, out):
    """Give a native decoder's result the type imagecodecs returns.

    A native decoder handed :func:`byte_output`'s view of ``out`` returns
    a memoryview of the written prefix. imagecodecs returns ``out``
    itself when the result fills it exactly, and an ndarray view of the
    prefix when ``out`` is an ndarray. Any other result, including one
    that does not start at ``out``'s first byte, is returned unchanged.
    """
    if out is None or isinstance(out, int) or not isinstance(result, memoryview):
        return result
    nbytes = result.nbytes
    target = (out if isinstance(out, np.ndarray)
              else np.frombuffer(memoryview(out).cast("B"), np.uint8))
    if nbytes and (np.frombuffer(result, np.uint8).ctypes.data
                   != target.ctypes.data):
        return result
    if nbytes == target.nbytes:
        return out
    if isinstance(out, np.ndarray):
        return _ndarray_prefix(out, nbytes)
    return result


def native_decoded(decode, data, out, **kwargs):
    """Call a native byte decoder under imagecodecs' decode ``out=`` contract.

    ``decode`` is a native decoder that takes ``out=`` as None, an int
    capacity or a writable byte view. This adds the rest of the contract:
    the ``bytearray`` type returns a new bytearray, and a buffer the
    result fills exactly is returned as is (see :func:`byte_result`).
    """
    if out is None:
        return decode(data, **kwargs)
    if out is bytearray:
        return bytearray(decode(data, **kwargs))
    return byte_result(decode(data, out=byte_output(out), **kwargs), out)


def encoded_output(encoded, out, codec, dest=None):
    """Return ``encoded`` through imagecodecs' encode ``out=`` contract.

    Byte-stream encoders that build their result as bytes use this so
    ``out=`` is honored rather than ignored; see :func:`_deliver`.
    Without ``out`` the bytes go to ``dest`` as ``write_dest`` sends
    them; the two are different destinations, so passing both raises.
    """
    if out is None:
        from ._io_helpers import write_dest
        return write_dest(encoded, dest)
    if dest is not None:
        raise ValueError(f"{codec} encode: pass dest= or out=, not both")
    return _deliver(encoded, out, f"{codec} encode")


def decoded_output(decoded, out, codec):
    """Return already-decoded bytes through the decode ``out=`` contract.

    For decoders that cannot write into the caller's storage directly;
    the copy is the price of honoring ``out=``. See :func:`_deliver`.
    """
    return _deliver(decoded, out, f"{codec} decode")


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
