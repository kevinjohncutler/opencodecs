# opencodecs/codecs/_bytetools.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Tight nogil helpers for byte-level data shuffling.

Used by the CZI reader (and potentially other parsers) to undo the
"byte-plane" shuffling that compressors apply before zstd. Doing this
in Cython with nogil is ~50× faster than numpy's transpose+copy AND
runs in parallel across threads (vs the GIL-serialised numpy path).
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t


def byteshuffle_encode(data, int itemsize, Py_ssize_t n_elements, *, out=None):
    """Byte-plane shuffle (inverse of :func:`byteshuffle_decode`).

    Rearranges memory from natural ``[e0_byte0, e0_byte1, ..., e0_byte{k-1},
    e1_byte0, ...]`` layout into ``[byte0_of_e0, byte0_of_e1, ..., byte0_of_e_{n-1},
    byte1_of_e0, byte1_of_e1, ...]`` — the format that compresses better
    under zstd / lz4 / deflate because byte-plane streams are more
    redundant than interleaved-byte streams.

    Parameters
    ----------
    data : buffer-protocol object
        Source bytes; length must equal ``itemsize * n_elements``.
    itemsize, n_elements : int
        Same as :func:`byteshuffle_decode`.
    out : int | bytearray | memoryview | None, optional
        See ``_zstd.decode`` for the full ``out=`` contract.
    """
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] out_view
        const uint8_t* sp
        uint8_t* dp
        Py_ssize_t i, b
        Py_ssize_t n = n_elements
        Py_ssize_t k = itemsize
        Py_ssize_t total = k * n
        bytes out_bytes

    if itemsize < 1:
        raise ValueError(f"itemsize must be >= 1, got {itemsize}")
    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    if <Py_ssize_t> src.shape[0] != total:
        raise ValueError(
            f"data length {src.shape[0]} != itemsize ({itemsize}) "
            f"* n_elements ({n_elements}) = {total}"
        )
    if total == 0:
        if out is None or isinstance(out, int):
            return b''
        return out[:0]

    if out is not None and not isinstance(out, int):
        try:
            out_view = out
        except (TypeError, ValueError, BufferError) as e:
            raise TypeError(
                f"byteshuffle_encode: out= must be int or writable buffer, "
                f"got {type(out).__name__}"
            ) from e
        if out_view.shape[0] < total:
            raise ValueError(
                f"byteshuffle_encode: out= buffer is {out_view.shape[0]} "
                f"bytes but output is {total} bytes")
        dp = &out_view[0]
    else:
        if isinstance(out, int) and out < total:
            raise ValueError(
                f"byteshuffle_encode: out=int({out}) is less than the "
                f"required {total} bytes")
        out_bytes = PyBytes_FromStringAndSize(NULL, total)
        dp = <uint8_t*> PyBytes_AsString(out_bytes)
    sp = &src[0]

    if k == 1:
        with nogil:
            for i in range(n):
                dp[i] = sp[i]
    elif k == 2:
        with nogil:
            for i in range(n):
                dp[i] = sp[2 * i]
                dp[n + i] = sp[2 * i + 1]
    else:
        # General case: split into k byte-planes.
        with nogil:
            for b in range(k):
                for i in range(n):
                    dp[b * n + i] = sp[i * k + b]

    if out is not None and not isinstance(out, int):
        del out_view
        return out[:total]
    return out_bytes


def byteshuffle_decode(data, int itemsize, Py_ssize_t n_elements, *, out=None):
    """Inverse byte-plane shuffle.

    Input bytes layout (the format zstd-with-byteshuffle leaves behind):

        [byte0_of_e0, byte0_of_e1, ..., byte0_of_e_{n-1},
         byte1_of_e0, byte1_of_e1, ..., byte1_of_e_{n-1},
         ...
         byte{itemsize-1}_of_e0, ..., byte{itemsize-1}_of_e_{n-1}]

    Output: bytes interleaved as [e0_byte0, e0_byte1, ..., e0_byte{itemsize-1},
                                   e1_byte0, ...] which is the natural memory
    layout for an array of ``n_elements`` ``itemsize``-byte values.

    Parameters
    ----------
    data : buffer-protocol object
        Source bytes; length must equal ``itemsize * n_elements``.
    itemsize : int
        Bytes per element (1 = no-op, 2 = uint16/int16, 4 = uint32/float32, ...).
    n_elements : int
        Number of elements.
    out : int | bytearray | memoryview | None, optional
        See ``_zstd.decode`` for the full ``out=`` contract. Output
        size is always ``itemsize * n_elements`` bytes.
    """
    cdef:
        const uint8_t[::1] src
        uint8_t[::1] out_view             # writable view of caller buffer
        const uint8_t* sp
        uint8_t* dp
        Py_ssize_t i, b
        Py_ssize_t n = n_elements
        Py_ssize_t k = itemsize
        Py_ssize_t total = k * n
        bytes out_bytes

    if itemsize < 1:
        raise ValueError(f"itemsize must be >= 1, got {itemsize}")

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)

    if <Py_ssize_t> src.shape[0] != total:
        raise ValueError(
            f"data length {src.shape[0]} != itemsize ({itemsize}) "
            f"* n_elements ({n_elements}) = {total}"
        )

    if total == 0:
        if out is None or isinstance(out, int):
            return b''
        return out[:0]

    # Resolve output destination.
    if out is not None and not isinstance(out, int):
        try:
            out_view = out
        except (TypeError, ValueError, BufferError) as e:
            raise TypeError(
                f"byteshuffle_decode: out= must be int or writable buffer, "
                f"got {type(out).__name__}"
            ) from e
        if out_view.shape[0] < total:
            raise ValueError(
                f"byteshuffle_decode: out= buffer is {out_view.shape[0]} "
                f"bytes but output is {total} bytes")
        dp = &out_view[0]
    else:
        if isinstance(out, int) and out < total:
            raise ValueError(
                f"byteshuffle_decode: out=int({out}) is less than the "
                f"required {total} bytes")
        out_bytes = PyBytes_FromStringAndSize(NULL, total)
        dp = <uint8_t*> PyBytes_AsString(out_bytes)
    sp = &src[0]

    if k == 1:
        # No-op shuffle.
        with nogil:
            for i in range(n):
                dp[i] = sp[i]
    elif k == 2:
        # Hot path for uint16 / int16. Two tight passes over memory in
        # the natural read order; the writes touch alternating positions.
        with nogil:
            for i in range(n):
                dp[2 * i] = sp[i]
                dp[2 * i + 1] = sp[n + i]
    else:
        # General case: k byte-planes.
        with nogil:
            for b in range(k):
                for i in range(n):
                    dp[i * k + b] = sp[b * n + i]

    if out is not None and not isinstance(out, int):
        del out_view
        return out[:total]
    return out_bytes
