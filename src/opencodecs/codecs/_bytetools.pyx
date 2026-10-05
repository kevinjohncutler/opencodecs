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
runs in parallel across threads (vs the GIL-serialized numpy path).
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t, uint32_t, uint64_t
cimport cython

cdef extern from "Python.h":
    const Py_ssize_t PY_SSIZE_T_MAX

cdef extern from "byteplanes.h" nogil:
    void oc_shuffle(uint8_t* dst, const uint8_t* src, size_t plane,
                    size_t count, size_t k)
    void oc_unshuffle(uint8_t* dst, const uint8_t* src, size_t plane,
                      size_t count, size_t k)


cdef Py_ssize_t _plane_bytes(Py_ssize_t itemsize, Py_ssize_t n_elements) except -1:
    """itemsize * n_elements, refusing arguments whose product would not
    fit. A product that wrapped could equal len(data) and let the length
    check pass while the kernels walked n_elements far past the buffers."""
    if itemsize < 1:
        raise ValueError(f"itemsize must be >= 1, got {itemsize}")
    if n_elements < 0 or n_elements > PY_SSIZE_T_MAX // itemsize:
        raise ValueError(f"n_elements must be in [0, {PY_SSIZE_T_MAX // itemsize}] "
                         f"for itemsize {itemsize}, got {n_elements}")
    return itemsize * n_elements


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
        Py_ssize_t n = n_elements
        Py_ssize_t k = itemsize
        Py_ssize_t total
        bytes out_bytes

    total = _plane_bytes(itemsize, n_elements)
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
    with nogil:
        oc_shuffle(dp, sp, <size_t> n, <size_t> n, <size_t> k)

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
        Py_ssize_t n = n_elements
        Py_ssize_t k = itemsize
        Py_ssize_t total
        bytes out_bytes

    total = _plane_bytes(itemsize, n_elements)

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
    with nogil:
        oc_unshuffle(dp, sp, <size_t> n, <size_t> n, <size_t> k)

    if out is not None and not isinstance(out, int):
        del out_view
        return out[:total]
    return out_bytes


# ---------------------------------------------------------------------------
# Delta predictor: prefix sum along the last (contiguous) axis
# ---------------------------------------------------------------------------
#
# numpy has cumsum, and for this shape it is 5.5x slower than a plain C
# loop: 34.6 ms against 6.3 ms on 17 MB of uint8. A prefix sum is serial,
# so neither version vectorizes; the difference is that np.cumsum carries
# per-element dispatch that a specialized loop does not. This is the same
# specialization argument as libspng's filter_scanline, applied to a
# numpy call rather than to a switch.
#
# Encode does not need a kernel: np.roll allocates a whole shifted copy,
# and replacing it with a slice-subtract already matches the reference
# exactly. Only decode was worth compiling.

ctypedef fused delta_t:
    cython.uchar
    cython.schar
    cython.ushort
    cython.short
    cython.uint
    cython.int
    cython.ulonglong
    cython.longlong


cdef void _chains(delta_t* p0, Py_ssize_t rows, Py_ssize_t n, Py_ssize_t dist,
                  bint use_xor) noexcept nogil:
    """Running sum (or XOR) along each of ``rows`` rows of ``n`` over
    ``dist`` > 1 interleaved chains, each walked with its running value in
    a register.

    Only the value's low bits are stored, which agrees with the dtype's
    wrapping arithmetic for every width and signedness. Rereading
    ``p[i - dist]`` instead makes every step wait on the store before it:
    4096 rows of 12288 uint8 with dist=3 measured 83.4 against 14.5 ms on
    GCC, 80.8 against 25.7 ms on MSVC, and level on Apple Clang.
    """
    cdef Py_ssize_t r, i, start
    cdef uint64_t acc
    cdef delta_t* p
    for r in range(rows):
        p = p0 + r * n
        for start in range(dist if dist < n else n):
            acc = <uint64_t> p[start]
            i = start + dist
            if use_xor:
                while i < n:
                    acc = acc ^ <uint64_t> p[i]
                    p[i] = <delta_t> acc
                    i += dist
            else:
                while i < n:
                    acc = acc + <uint64_t> p[i]
                    p[i] = <delta_t> acc
                    i += dist


cdef void _triples(delta_t* p0, Py_ssize_t rows, Py_ssize_t n,
                   bint use_xor) noexcept nogil:
    """``_chains`` for ``dist`` == 3 (three interleaved channels): all three
    chains in one pass over the row, a register each.

    Three independent sums per step instead of one, and one loop test per
    three elements. On 4096 rows of 12288 uint8 this measured 12.6 against
    15.2 ms with GCC (Zen 2), 9.9 against 22.9 with Apple Clang, and 17.8
    against 24.9 with MSVC 14.51 (Kaby Lake). Two and four channels have
    their own kernels below.
    """
    cdef Py_ssize_t r, i
    cdef uint64_t a0, a1, a2
    cdef delta_t* p
    if n <= 3:
        return   # every chain is a single element
    for r in range(rows):
        p = p0 + r * n
        a0 = <uint64_t> p[0]
        a1 = <uint64_t> p[1]
        a2 = <uint64_t> p[2]
        i = 3
        if use_xor:
            while i + 3 <= n:
                a0 = a0 ^ <uint64_t> p[i]; p[i] = <delta_t> a0
                a1 = a1 ^ <uint64_t> p[i + 1]; p[i + 1] = <delta_t> a1
                a2 = a2 ^ <uint64_t> p[i + 2]; p[i + 2] = <delta_t> a2
                i += 3
            if i < n:
                p[i] = <delta_t> (a0 ^ <uint64_t> p[i])
            if i + 1 < n:
                p[i + 1] = <delta_t> (a1 ^ <uint64_t> p[i + 1])
        else:
            while i + 3 <= n:
                a0 = a0 + <uint64_t> p[i]; p[i] = <delta_t> a0
                a1 = a1 + <uint64_t> p[i + 1]; p[i + 1] = <delta_t> a1
                a2 = a2 + <uint64_t> p[i + 2]; p[i + 2] = <delta_t> a2
                i += 3
            if i < n:
                p[i] = <delta_t> (a0 + <uint64_t> p[i])
            if i + 1 < n:
                p[i + 1] = <delta_t> (a1 + <uint64_t> p[i + 1])


cdef void _twos(delta_t* p0, Py_ssize_t rows, Py_ssize_t n,
                bint use_xor) noexcept nogil:
    """``_chains`` for ``dist`` == 2: both chains in one pass, a register
    each, over a counted loop.

    ``_chains``' loop is one element per step, 20 bytes of code under
    MSVC, and whether its branch straddles a 32-byte line moved with
    unrelated edits to this file: MSVC 14.51 builds of the same kernel ran
    16.6 or 19.2 ms on Kaby Lake. Two elements per step leave the loop
    bound by its stores instead. On 4096 rows of 8192 uint8 this measured
    9.2 against 10.5 ms with GCC (Zen 2), 6.6 against 15.3 with Apple Clang
    and 10.4 against 19.6 with MSVC 14.51 (Kaby Lake). A test of ``i + 2
    <= n`` on each step instead of a count cost MSVC two instructions.
    """
    cdef Py_ssize_t r, k, steps = (n - 2) // 2
    cdef uint64_t a0, a1
    cdef delta_t* q
    for r in range(rows):
        q = p0 + r * n
        a0 = <uint64_t> q[0]
        a1 = <uint64_t> q[1]
        q += 2
        if use_xor:
            for k in range(steps):
                a0 = a0 ^ <uint64_t> q[0]; q[0] = <delta_t> a0
                a1 = a1 ^ <uint64_t> q[1]; q[1] = <delta_t> a1
                q += 2
            if n % 2:
                q[0] = <delta_t> (a0 ^ <uint64_t> q[0])
        else:
            for k in range(steps):
                a0 = a0 + <uint64_t> q[0]; q[0] = <delta_t> a0
                a1 = a1 + <uint64_t> q[1]; q[1] = <delta_t> a1
                q += 2
            if n % 2:
                q[0] = <delta_t> (a0 + <uint64_t> q[0])


cdef void _quads(delta_t* p0, Py_ssize_t rows, Py_ssize_t n,
                 bint use_xor) noexcept nogil:
    """``_chains`` for ``dist`` == 4 (RGBA): all four chains in one pass,
    each with its own register and its own pointer.

    Through one pointer, GCC merges the four narrow stores of a step into
    one word built from shifts and ORs, which ran 10% slower than
    ``_chains``; with a pointer each it cannot see that they touch. On 4096
    rows of 16384 uint8 this measured 16.7 against 19.8 ms with GCC (Zen
    2), 12.3 against 30.8 with Apple Clang and 21.3 against 39.2 with MSVC
    14.51 (Kaby Lake).
    """
    cdef Py_ssize_t r, k, steps = (n - 4) // 4, rest = (n - 4) % 4
    cdef uint64_t a0, a1, a2, a3
    cdef delta_t* q0
    cdef delta_t* q1
    cdef delta_t* q2
    cdef delta_t* q3
    for r in range(rows):
        q0 = p0 + r * n
        a0 = <uint64_t> q0[0]
        a1 = <uint64_t> q0[1]
        a2 = <uint64_t> q0[2]
        a3 = <uint64_t> q0[3]
        q0 += 4
        q1 = q0 + 1
        q2 = q0 + 2
        q3 = q0 + 3
        if use_xor:
            for k in range(steps):
                a0 = a0 ^ <uint64_t> q0[0]; q0[0] = <delta_t> a0; q0 += 4
                a1 = a1 ^ <uint64_t> q1[0]; q1[0] = <delta_t> a1; q1 += 4
                a2 = a2 ^ <uint64_t> q2[0]; q2[0] = <delta_t> a2; q2 += 4
                a3 = a3 ^ <uint64_t> q3[0]; q3[0] = <delta_t> a3; q3 += 4
            if rest > 0:
                q0[0] = <delta_t> (a0 ^ <uint64_t> q0[0])
            if rest > 1:
                q1[0] = <delta_t> (a1 ^ <uint64_t> q1[0])
            if rest > 2:
                q2[0] = <delta_t> (a2 ^ <uint64_t> q2[0])
        else:
            for k in range(steps):
                a0 = a0 + <uint64_t> q0[0]; q0[0] = <delta_t> a0; q0 += 4
                a1 = a1 + <uint64_t> q1[0]; q1[0] = <delta_t> a1; q1 += 4
                a2 = a2 + <uint64_t> q2[0]; q2[0] = <delta_t> a2; q2 += 4
                a3 = a3 + <uint64_t> q3[0]; q3[0] = <delta_t> a3; q3 += 4
            if rest > 0:
                q0[0] = <delta_t> (a0 + <uint64_t> q0[0])
            if rest > 1:
                q1[0] = <delta_t> (a1 + <uint64_t> q1[0])
            if rest > 2:
                q2[0] = <delta_t> (a2 + <uint64_t> q2[0])


cdef void _pairs(delta_t* p0, Py_ssize_t rows, Py_ssize_t n,
                 bint use_xor) noexcept nogil:
    """Running sum (or XOR) along each of ``rows`` rows of ``n``, the
    running value in a register and two elements per counted step.

    The obvious ``p[i] += p[i - 1]`` leaves the register to the compiler,
    and they disagree: MSVC rereads the element it just stored, and GCC's
    code for it moved between 5.5 and 7.1 ms with unrelated edits
    elsewhere in this file. On 4096 x 4096 uint16 this form measured 5.9
    against 26.5 ms on MSVC and 5.1 against 7.1 on GCC. Stepping by four
    lets GCC merge the four stores into one built from shifts, which
    loses on x86. Apple Clang unrolls the obvious loop itself and was up
    to 3% slower here on 2-D input, level on 1-D.
    """
    cdef Py_ssize_t r, k
    cdef uint64_t acc
    cdef delta_t* q
    for r in range(rows):
        q = p0 + r * n
        acc = <uint64_t> q[0]
        q += 1
        if use_xor:
            for k in range((n - 1) // 2):
                acc = acc ^ <uint64_t> q[0]; q[0] = <delta_t> acc
                acc = acc ^ <uint64_t> q[1]; q[1] = <delta_t> acc
                q += 2
            if (n - 1) % 2:
                q[0] = <delta_t>(acc ^ <uint64_t> q[0])
        else:
            for k in range((n - 1) // 2):
                acc = acc + <uint64_t> q[0]; q[0] = <delta_t> acc
                acc = acc + <uint64_t> q[1]; q[1] = <delta_t> acc
                q += 2
            if (n - 1) % 2:
                q[0] = <delta_t>(acc + <uint64_t> q[0])


def delta_decode_inplace(delta_t[:, ::1] arr, Py_ssize_t dist=1):
    """In-place prefix sum along the last axis, wrapping like the dtype.

    ``arr`` is (rows, n) and C-contiguous, which is what the codec
    reshapes any axis into before calling. Wraparound is what the format
    means by delta on unsigned types, and C's unsigned arithmetic is
    already modular, so nothing special is needed to get it.
    """
    cdef Py_ssize_t rows = arr.shape[0]
    cdef Py_ssize_t n = arr.shape[1]
    if n < 2 or dist < 1 or dist >= n:
        # dist >= n: no element has one dist before it, and start + dist
        # would overflow for a dist near PY_SSIZE_T_MAX.
        return
    with nogil:
        if dist == 1:
            _pairs(&arr[0, 0], rows, n, False)
        elif dist == 2:
            _twos(&arr[0, 0], rows, n, False)
        elif dist == 3:
            _triples(&arr[0, 0], rows, n, False)
        elif dist == 4:
            _quads(&arr[0, 0], rows, n, False)
        else:
            _chains(&arr[0, 0], rows, n, dist, False)


def xor_decode_inplace(delta_t[:, ::1] arr, Py_ssize_t dist=1):
    """Running XOR along the last axis, the sibling of the prefix sum.

    np.bitwise_xor.accumulate has the same per-element dispatch cost as
    np.cumsum, and the same 5x gap against a specialized loop. XOR needs
    no wraparound reasoning -- it cannot carry -- so this is the simpler
    of the two.
    """
    cdef Py_ssize_t rows = arr.shape[0]
    cdef Py_ssize_t n = arr.shape[1]
    if n < 2 or dist < 1 or dist >= n:
        # dist >= n: no element has one dist before it, and start + dist
        # would overflow for a dist near PY_SSIZE_T_MAX.
        return
    with nogil:
        if dist == 1:
            _pairs(&arr[0, 0], rows, n, True)
        elif dist == 2:
            _twos(&arr[0, 0], rows, n, True)
        elif dist == 3:
            _triples(&arr[0, 0], rows, n, True)
        elif dist == 4:
            _quads(&arr[0, 0], rows, n, True)
        else:
            _chains(&arr[0, 0], rows, n, dist, True)


cdef uint32_t _crc32c_table[256]


cdef void _initialize_crc32c() noexcept nogil:
    cdef unsigned int index, bit
    cdef uint32_t value
    for index in range(256):
        value = index
        for bit in range(8):
            if value & 1:
                value = (value >> 1) ^ <uint32_t> 0x82f63b78
            else:
                value >>= 1
        _crc32c_table[index] = value


_initialize_crc32c()


def crc32c(data):
    """Return the Castagnoli checksum while releasing the interpreter lock."""
    cdef const uint8_t[::1] source
    cdef Py_ssize_t index, size
    cdef uint32_t value = <uint32_t> 0xffffffff
    try:
        source = data
    except (TypeError, ValueError, BufferError):
        source = bytes(data)
    size = source.shape[0]
    with nogil:
        for index in range(size):
            value = _crc32c_table[(value ^ source[index]) & 255] ^ (value >> 8)
    return value ^ <uint32_t> 0xffffffff



def unpackints_into(data, out, int bits, Py_ssize_t count,
                    int itemsize, bint little_endian=True):
    """Unpack most-significant-bit-first samples into caller byte storage.

    Up to 56 bits, whole input bytes shift into a 64-bit accumulator and
    each sample is one shift and mask off its top, with the store loop
    specialized for the common item sizes. The general form below, which
    assembles each sample from the bits left in the current byte, measured
    2 to 5x slower: 16 M 12-bit samples into big-endian uint16 took 130
    against 31 ms with MSVC 14.51, 80 against 29 with GCC and 77 against
    21 with Apple Clang, and its time moved 15% between MSVC builds of the
    same source.
    """
    cdef const uint8_t[::1] source = data
    cdef uint8_t[::1] target = out
    cdef Py_ssize_t i, byte_index = 0, dest_index
    cdef int bit_index = 0, left, take, j, nacc = 0
    cdef uint64_t value, acc = 0, mask
    cdef const uint8_t* s
    cdef uint8_t* d
    if bits < 1 or bits > 64 or itemsize not in (1, 2, 4, 8):
        raise ValueError("invalid packed integer width or output itemsize")
    if count < 0 or count > (source.shape[0] * 8) // bits:
        raise ValueError(
            f"packed integer input is truncated: {count} samples of {bits} "
            f"bits need {(count * bits + 7) // 8} bytes, got {source.shape[0]}")
    if count > target.shape[0] // itemsize:
        raise ValueError("packed integer output is too small")
    if count == 0:
        return out
    if bits <= 56:
        # The accumulator holds at most bits + 7 <= 63 unread bits, and it
        # reads exactly the (count * bits + 7) // 8 bytes checked above.
        s = &source[0]
        d = &target[0]
        mask = ((<uint64_t> 1) << bits) - 1
        with nogil:
            if itemsize == 2 and not little_endian:
                for i in range(count):
                    while nacc < bits:
                        acc = (acc << 8) | s[byte_index]
                        byte_index += 1
                        nacc += 8
                    nacc -= bits
                    value = (acc >> nacc) & mask
                    d[2 * i] = <uint8_t> (value >> 8)
                    d[2 * i + 1] = <uint8_t> value
            elif itemsize == 2:
                for i in range(count):
                    while nacc < bits:
                        acc = (acc << 8) | s[byte_index]
                        byte_index += 1
                        nacc += 8
                    nacc -= bits
                    value = (acc >> nacc) & mask
                    d[2 * i] = <uint8_t> value
                    d[2 * i + 1] = <uint8_t> (value >> 8)
            elif itemsize == 1:
                for i in range(count):
                    while nacc < bits:
                        acc = (acc << 8) | s[byte_index]
                        byte_index += 1
                        nacc += 8
                    nacc -= bits
                    d[i] = <uint8_t> ((acc >> nacc) & mask)
            else:
                for i in range(count):
                    while nacc < bits:
                        acc = (acc << 8) | s[byte_index]
                        byte_index += 1
                        nacc += 8
                    nacc -= bits
                    value = (acc >> nacc) & mask
                    for j in range(itemsize):
                        d[i * itemsize + (j if little_endian else itemsize - 1 - j)] = <uint8_t> (value >> (8 * j))
        return out
    with nogil:
        for i in range(count):
            value = 0
            left = bits
            while left:
                take = 8 - bit_index
                if take > left:
                    take = left
                value = (value << take) | ((source[byte_index] >> (8 - bit_index - take)) & ((1 << take) - 1))
                left -= take
                bit_index += take
                if bit_index == 8:
                    byte_index += 1
                    bit_index = 0
            for j in range(itemsize):
                dest_index = i * itemsize + (j if little_endian else itemsize - 1 - j)
                target[dest_index] = <uint8_t> (value >> (8 * j))
    return out


# ---------------------------------------------------------------------------
# Reading many files, and placing decoded chunks, without the GIL
# ---------------------------------------------------------------------------

from libc.stdint cimport int32_t, int64_t
from libc.stdlib cimport calloc, free
from libc.string cimport memcpy
from libc.errno cimport errno

cdef extern from "oc_readfiles.h":
    const void* oc_path_acquire(object obj, void** owner) except NULL
    void oc_path_release(void* owner)
    int oc_file_open(const void* path) nogil
    void oc_file_close(int fd) nogil
    int64_t oc_file_size(int fd) nogil
    int64_t oc_file_pread(int fd, uint8_t* dst, int64_t n, int64_t offset) nogil


def read_ranges(paths, const int64_t[::1] offsets, const int64_t[::1] lengths,
                uint8_t[::1] buffer, int64_t[::1] starts, int64_t[::1] sizes,
                int32_t[::1] errnos):
    """Read a byte range of each file in ``paths`` into ``buffer``, in order.

    Item ``i`` reads ``lengths[i]`` bytes at ``offsets[i]`` of ``paths[i]``
    (fewer at the end of the file); a negative length reads to the end of
    the file, and a negative offset counts back from the end, clamped at the
    start, as ``_FsStore.read_range`` does. The bytes go to ``buffer`` back
    to back: item ``i`` at ``starts[i]``, ``sizes[i]`` long. An item that
    does not fit in what is left of ``buffer`` gets size -2 and is skipped,
    so the caller can grow the buffer and read it again; an item whose
    file could not be opened or read gets size -1 and its ``errno``.
    Consecutive items naming the same path object share one open.

    Opening, sizing and reading all run without the GIL. Returns the
    number of buffer bytes used.
    """
    cdef:
        Py_ssize_t n = len(paths), i
        const void** native = NULL
        void** owners = NULL
        int fd = -1
        int64_t pos = 0, cap = buffer.shape[0], size, offset, want, got
        uint8_t* base = &buffer[0] if buffer.shape[0] else NULL
    if (offsets.shape[0] < n or lengths.shape[0] < n or starts.shape[0] < n
            or sizes.shape[0] < n or errnos.shape[0] < n):
        raise ValueError("every per-item array needs one entry per path")
    native = <const void**> calloc(n + 1, sizeof(void*))
    owners = <void**> calloc(n + 1, sizeof(void*))
    if native == NULL or owners == NULL:
        free(native)
        free(owners)
        raise MemoryError()
    try:
        for i in range(n):
            if i and paths[i] is paths[i - 1]:
                native[i] = native[i - 1]
            else:
                native[i] = oc_path_acquire(paths[i], &owners[i])
        with nogil:
            for i in range(n):
                starts[i] = pos
                sizes[i] = -1
                errnos[i] = 0
                if i == 0 or native[i] != native[i - 1] or fd < 0:
                    if fd >= 0:
                        oc_file_close(fd)
                    fd = oc_file_open(native[i])
                    if fd < 0:
                        errnos[i] = errno
                        continue
                offset = offsets[i]
                want = lengths[i]
                if offset < 0 or want < 0:
                    size = oc_file_size(fd)
                    if size < 0:
                        errnos[i] = errno
                        continue
                    if offset < 0:
                        offset = size + offset if size + offset > 0 else 0
                    if want < 0:
                        want = size - offset if size > offset else 0
                if want > cap - pos:
                    sizes[i] = -2
                    continue
                got = oc_file_pread(fd, base + pos if base != NULL else NULL,
                                    want, offset)
                if got < 0:
                    errnos[i] = errno
                    continue
                sizes[i] = got
                pos += got
            if fd >= 0:
                oc_file_close(fd)
    finally:
        for i in range(n):
            oc_path_release(owners[i])
        free(native)
        free(owners)
    return pos


def place_windows(src, out, const int64_t[:, ::1] windows):
    """Copy rectangles of rows from ``src`` into the 2D byte array ``out``.

    Each row of ``windows`` is ``(src_offset, src_stride, rows, nbytes,
    dst_row, dst_col)``: ``rows`` runs of ``nbytes`` bytes, the r-th
    starting at ``src[src_offset + r * src_stride]``, land at
    ``out[dst_row + r, dst_col:dst_col + nbytes]``. Every window is checked
    against both buffers first; the copies then run without the GIL.
    """
    cdef:
        const uint8_t[::1] s = src
        uint8_t[:, ::1] d = out
        Py_ssize_t m = windows.shape[0], i, r
        int64_t so, ss, rows, nb, dr, dc
        int64_t slen = s.shape[0]
    if windows.shape[1] != 6:
        raise ValueError(f"windows must have 6 columns, got {windows.shape[1]}")
    for i in range(m):
        so = windows[i, 0]; ss = windows[i, 1]; rows = windows[i, 2]
        nb = windows[i, 3]; dr = windows[i, 4]; dc = windows[i, 5]
        if rows <= 0 or nb <= 0:
            continue
        if (so < 0 or ss < 0 or dr < 0 or dc < 0
                or dr + rows > d.shape[0] or dc + nb > d.shape[1]
                or so + (rows - 1) * ss + nb > slen):
            raise ValueError(f"window {i} lies outside the source or destination")
    with nogil:
        for i in range(m):
            so = windows[i, 0]; ss = windows[i, 1]; rows = windows[i, 2]
            nb = windows[i, 3]; dr = windows[i, 4]; dc = windows[i, 5]
            if rows <= 0 or nb <= 0:
                continue
            for r in range(rows):
                memcpy(&d[dr + r, dc], &s[so + r * ss], <size_t> nb)
