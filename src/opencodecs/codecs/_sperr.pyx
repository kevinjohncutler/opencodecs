# opencodecs/codecs/_sperr.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native SPERR codec — wavelet-based error-bounded lossy compression.

SPERR (2022+) is a SPECK/wavelet-based compressor for scientific
floating-point arrays. It often hits much smaller bitstreams than ZFP
or SZ3 at the same PSNR target on smooth fields (climate, CFD,
seismic, scientific simulation).

C API (``SPERR_C_API.h``) accepts 2D slices and 3D volumes in either
float32 (``is_float=1``) or float64 (``is_float=0``).

``encode`` writes SPERR's own format, what the sperr2d / sperr3d tools
and ``imagecodecs.sperr_encode`` write: a 2-D slice with the 10-byte
header ``sperr_comp_2d`` adds when asked (version, flags, two uint32
dimensions), and a 3-D volume as ``sperr_comp_3d`` returns it, which
always starts with its own header. ``decode_native`` reads the
dimensions and precision from that header. A 2-D stream written
without the header (``header=False``) needs ``shape`` and ``dtype``.

Releases up to 0.4.0 dropped SPERR's 2-D header and wrote a private
48-byte preamble in front of the stream instead::

    bytes  0..3   ASCII magic 'SPRR'
    byte   4      dtype: 1 = float32, 0 = float64
    byte   5      ndim:  2 or 3
    bytes  6..7   reserved
    bytes  8..47  shape: 5 × uint64 little-endian
                  (dimx fastest, dimy, dimz, _, _)

``decode_framed`` still reads those blobs, recognized by the magic.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t
from libc.string cimport memcpy

import struct as _struct

import numpy as np
cimport numpy as cnp

from sperr cimport (
    sperr_comp_2d, sperr_decomp_2d,
    sperr_comp_3d, sperr_decomp_3d,
    sperr_parse_header,
    free,
    SPERR_VERSION_MAJOR,
)


cnp.import_array()


_HEADER_LEN = 48
_HEADER_MAGIC = b'SPRR'
_HEADER_FMT = '<4sBB2x5Q'


class SperrError(RuntimeError):
    """Raised on SPERR encode/decode failures."""


_MODE_ALIASES = {
    "bpp": 1,
    "psnr": 2,
    "pwe": 3,
}
_MODE_NAMES = {v: k for k, v in _MODE_ALIASES.items()}

# SPERR's stream header: version byte, then eight flags packed most
# significant bit first (sperr::pack_8_booleans).
_FLAG_3D = 0x40
_FLAG_FLOAT = 0x20
_FLAG_MULTI_CHUNK = 0x10
_HEADER_2D_LEN = 10


def mode_name(mode):
    """'bpp' / 'psnr' / 'pwe' for a name in any case or for imagecodecs'
    SPERR.MODE integer value."""
    if isinstance(mode, str):
        key = mode.lower()
        if key in _MODE_ALIASES:
            return key
    else:
        try:
            value = int(mode)
        except (TypeError, ValueError):
            value = None
        if value in _MODE_NAMES:
            return _MODE_NAMES[value]
    raise ValueError(
        f"sperr encode: unknown mode {mode!r}; expected one of "
        f"{sorted(_MODE_ALIASES.keys())}"
    )


# Shared with sz3 and pcodec. Sperr's callers pass the three volume
# dimensions positionally rather than as a sequence, so the thin
# wrappers below keep that signature over the shared implementation.
from opencodecs.core._sidecar_header import make_sidecar_header as \
    _make_sidecar_header
_HEADER_LEN, _sidecar_pack, _sidecar_unpack = _make_sidecar_header(
    _HEADER_MAGIC, 5, SperrError, "sperr")


def _pack_header(int is_float, int ndim, dimx, dimy, dimz):
    return _sidecar_pack(is_float, ndim, (dimx, dimy, dimz))


def _unpack_header(buf):
    is_float, ndim, dims = _sidecar_unpack(buf)
    return is_float, ndim, dims[0], dims[1], dims[2]


def encode(arr, *,
           mode="psnr",
           psnr: float = 80.0,
           bpp: float = 4.0,
           pwe: float = 1e-3,
           chunk=None,
           nthreads: int = 0,
           bint header=True) -> bytes:
    """Compress ndarray with SPERR into SPERR's own stream format.

    Parameters
    ----------
    arr : np.ndarray
        2D or 3D float32 / float64 array.
    mode : {"psnr", "bpp", "pwe"}
        Quality control mode.
        - "psnr": target peak signal-to-noise ratio in dB (default 80).
        - "bpp" : target bits per pixel (default 4.0).
        - "pwe" : point-wise absolute error bound (default 1e-3).
    psnr, bpp, pwe : float
        Target value for the chosen mode.
    chunk : tuple of 3 ints or None
        Chunk dims for 3D mode, in numpy order (ignored for 2D). None
        (the default, as in imagecodecs) compresses the volume as one
        chunk.
    nthreads : int
        OpenMP threads for 3D mode (0 = auto). 2D is single-threaded.
    header : bool
        Write SPERR's 10-byte header on a 2-D slice (default). A 3-D
        stream always carries its header, so False is refused there.
    """
    cdef:
        cnp.ndarray contig
        int is_float
        int err_mode
        size_t dimx = 0, dimy = 0, dimz = 0
        size_t cx, cy, cz
        size_t out_size = 0
        void* dst = NULL
        int rc
        bytes payload
        double quality
        int inc_header

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    contig = np.ascontiguousarray(arr)
    if contig.dtype == np.dtype(np.float32):
        is_float = 1
    elif contig.dtype == np.dtype(np.float64):
        is_float = 0
    else:
        raise ValueError(
            f"sperr encode: only float32/float64 supported "
            f"(got {contig.dtype!r}); use 'zfp' or 'sz3' for integer arrays"
        )

    mode = mode_name(mode)
    err_mode = _MODE_ALIASES[mode]
    if mode == "psnr":
        quality = float(psnr)
    elif mode == "bpp":
        quality = float(bpp)
    else:  # pwe
        quality = float(pwe)

    if contig.ndim not in (2, 3):
        raise ValueError(f"sperr encode: ndim must be 2 or 3, got {contig.ndim}")
    if contig.ndim == 3 and not header:
        raise ValueError(
            "sperr encode: a 3-D SPERR stream always carries its header; "
            "header=False applies to 2-D slices only")
    inc_header = 1 if header else 0

    # numpy is row-major: outermost dim is shape[0] = slowest. SPERR takes
    # dimx as the fastest-varying.
    if contig.ndim == 2:
        dimx = <size_t> contig.shape[1]
        dimy = <size_t> contig.shape[0]
        dimz = 1
    else:
        dimx = <size_t> contig.shape[2]
        dimy = <size_t> contig.shape[1]
        dimz = <size_t> contig.shape[0]

    cdef void* data_ptr = <void*> contig.data
    cdef size_t nthreads_c = <size_t> int(nthreads)

    if contig.ndim == 2:
        with nogil:
            rc = sperr_comp_2d(
                data_ptr, is_float, dimx, dimy,
                err_mode, quality, inc_header,
                &dst, &out_size,
            )
    else:
        # The caller passes chunk in numpy order (z, y, x); SPERR takes
        # (x, y, z). The values go to SPERR as given, as sperr3d and
        # imagecodecs pass them; SPERR fits a chunk larger than the
        # volume to the volume itself.
        if chunk is None:
            chunk = tuple(arr.shape)
        chunk = tuple(int(c) for c in chunk)
        if len(chunk) != 3 or not all(c >= 1 for c in chunk):
            raise ValueError(
                f"sperr encode: chunk must be three positive sizes, got {chunk}")
        # A multi-chunk header stores the chunk size in 16 bits per
        # axis (sperr::chunk_volume decides how many chunks there are).
        if (_chunk_count(tuple(arr.shape), chunk) > 1
                and not all(c <= 0xffff for c in chunk)):
            raise ValueError(
                f"sperr encode: chunk sizes must be at most 65535 when the "
                f"volume is split, got {chunk}")
        cx = <size_t> int(chunk[0])
        cy = <size_t> int(chunk[1])
        cz = <size_t> int(chunk[2])
        with nogil:
            rc = sperr_comp_3d(
                data_ptr, is_float,
                dimx, dimy, dimz,
                cz, cy, cx,
                err_mode, quality, nthreads_c,
                &dst, &out_size,
            )

    if rc != 0 or dst == NULL or out_size == 0:
        if dst != NULL:
            free(dst)
        raise SperrError(f"SPERR compression failed (rc={rc})")

    try:
        payload = PyBytes_FromStringAndSize(<const char*> dst, <Py_ssize_t> out_size)
    finally:
        free(dst)

    if _has_infinite_step(payload, contig.ndim, header):
        raise ValueError(
            f"sperr encode: SPERR's quantization step overflowed to infinity "
            f"for this data in mode {mode!r} (its values are too large), and "
            f"the stream would decode to NaN; scale the data, or use "
            f"mode='pwe'")
    return payload


def _coded_spans(data, int ndim, bint header):
    """(start, end) of each SPECK_FLT stream in a stream SPERR wrote."""
    if ndim == 2:
        return [(_HEADER_2D_LEN if header else 0, len(data))]
    vol = _struct.unpack_from('<III', data, 2)
    if data[1] & _FLAG_MULTI_CHUNK:
        chunk = _struct.unpack_from('<HHH', data, 14)
        fixed = 20
    else:
        chunk = vol
        fixed = 14
    nchunks = _chunk_count(vol, chunk)
    spans = []
    pos = fixed + 4 * nchunks
    for length in _struct.unpack_from(f'<{nchunks}I', data, fixed):
        spans.append((pos, pos + length))
        pos += length
    return spans


def _has_infinite_step(data, int ndim, bint header):
    for start, end in _coded_spans(data, ndim, header):
        if (end - start >= _CONDI_LEN and not data[start] & _CONDI_CONSTANT
                and _struct.unpack_from('<d', data, start + 9)[0]
                == float('inf')):
            return True
    return False


def _chunk_count(vol, chunk):
    """How many chunks sperr::chunk_volume cuts ``vol`` into: whole
    chunks per axis, plus one for a remainder longer than half a chunk,
    and at least one."""
    count = 1
    for v, c in zip(vol, chunk):
        segs = v // c
        if v % c > c // 2:
            segs += 1
        count *= max(segs, 1)
    return count


def decode_framed(data, *, int nthreads=0) -> 'np.ndarray':
    """Decode a blob in the framed layout releases up to 0.4.0 wrote
    ('SPRR' preamble + bitstream) to an ndarray."""
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        Py_ssize_t header_off = <Py_ssize_t> _HEADER_LEN
        Py_ssize_t payload_len
        void* dst = NULL
        size_t out_dimx = 0, out_dimy = 0, out_dimz = 0
        int rc
        cnp.ndarray out_arr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    if srcsize < _HEADER_LEN:
        raise SperrError("sperr blob too short to contain header")

    is_float, ndim, dimx, dimy, dimz = _unpack_header(bytes(src[:_HEADER_LEN]))
    payload_len = srcsize - _HEADER_LEN
    if payload_len <= 0:
        raise SperrError("sperr blob payload missing")

    dtype = np.float32 if is_float else np.float64
    if ndim == 2:
        shape = (int(dimy), int(dimx))
        _check_speck(src, header_off, srcsize, int(dimx) * int(dimy))
    elif ndim == 3:
        shape = (int(dimz), int(dimy), int(dimx))
        # The payload is a whole SPERR 3-D stream: check its header,
        # chunk table and chunks, and that it matches the preamble.
        p_ndim, p_float, p_zyx = stream_info(src[header_off:])
        if p_ndim != 3 or p_zyx != shape:
            raise SperrError(
                f"sperr blob: preamble shape {shape} does not match the "
                f"stream's {p_zyx}")
    else:
        raise SperrError(f"sperr header has invalid ndim={ndim}")

    out = np.empty(shape, dtype=dtype)
    out_arr = out

    cdef const void* src_ptr = <const void*> &src[header_off]
    cdef size_t pl = <size_t> payload_len
    cdef size_t dx = <size_t> dimx, dy = <size_t> dimy
    cdef int isf = int(is_float)
    cdef size_t workers = max(0, nthreads)

    if ndim == 2:
        with nogil:
            rc = sperr_decomp_2d(src_ptr, pl, isf, dx, dy, &dst)
    else:
        with nogil:
            rc = sperr_decomp_3d(
                src_ptr, pl, isf, workers,
                &out_dimx, &out_dimy, &out_dimz,
                &dst,
            )

    if rc != 0 or dst == NULL:
        if dst != NULL:
            free(dst)
        raise SperrError(f"SPERR decompression failed (rc={rc})")

    try:
        memcpy(<void*> out_arr.data, dst, <size_t>(out.size * out.dtype.itemsize))
    finally:
        free(dst)
    return out


# The coded data of a 2-D slice, and of each chunk of a volume, is a
# SPECK_FLT stream (SPECK_FLT.cpp, use_bitstream): a 17-byte
# conditioner header (Conditioner.h condi_type: a flags byte, then for
# a constant field the value count and the value, otherwise the mean
# and from byte 9 the quantization step q), then for a non-constant
# field the SPECK stream (SPECK_INT.h: a bit-plane count byte and a
# uint64 bit count, 9 bytes, then the bits), then optionally the
# outlier stream in the same layout. SPERR's decoder checks none of
# this in a release build: its asserts are compiled out and the C API
# ignores what use_bitstream returns.
_CONDI_LEN = 17
_SPECK_HEADER_LEN = 9
_CONDI_CONSTANT = 0x01       # flag 7 of Conditioner's meta
_CONDI_UNUSED = 0x7E         # flags 1..6, which SPERR always clears


def _speck_bits_limit(nvals, bitplanes):
    """The most bits a SPECK stream of ``nvals`` values can declare.

    SPECK spends, per bit plane, a significance or a refinement bit on
    each value, a sign bit once per value and one test per set of its
    partition, so a few bits per value per plane; SPERR's own streams
    stay near one (at most 1.03 in our measurements over float32 and
    float64, 2-D and 3-D, smooth, random and heavy-tailed data at every
    mode). Eight per value per plane, plus a margin for tiny arrays, is
    far above what its encoder writes. SPERR reserves memory for the
    declared count before decoding, so a damaged count would otherwise
    ask for up to 2**64 bits."""
    return 8 * nvals * (bitplanes + 1) + 4096


def _check_coded_bits(head, Py_ssize_t at, nvals, what):
    planes = head[at]
    if planes > 64:
        raise SperrError(f"sperr: {what} has {planes} bit planes, more than 64")
    bits = _struct.unpack_from('<Q', head, at + 1)[0]
    if bits > _speck_bits_limit(nvals, planes):
        raise SperrError(
            f"sperr: {what} declares {bits} bits, more than {nvals} values "
            f"in {planes} bit planes can take")
    return _SPECK_HEADER_LEN + (bits + 7) // 8


def _check_speck(data, Py_ssize_t start, Py_ssize_t end, nvals,
                 bint whole=True):
    """Raise SperrError unless ``data[start:end]`` has the layout of a
    SPECK_FLT stream, so SPERR's decoder is not handed a stream that
    ends inside its fixed fields or holds values its writer never
    produces. ``nvals`` is the number of values the stream codes, or
    with ``whole=False`` (one chunk of several) an upper bound on it.

    A stream cut short inside its coded bits passes: SPERR reads such
    a prefix by design (progressive decoding)."""
    n = end - start
    if n < _CONDI_LEN:
        raise SperrError("sperr: coded data too short for its header")
    head = bytes(data[start:start + min(n, _CONDI_LEN + _SPECK_HEADER_LEN)])
    meta = head[0]
    if meta & _CONDI_UNUSED:
        raise SperrError(
            f"sperr: coded data header has flags {meta:#04x}, which SPERR "
            "does not write")
    if meta & _CONDI_CONSTANT:
        if n != _CONDI_LEN:
            raise SperrError(
                "sperr: a constant field's coded data must be its 17-byte "
                f"header, got {n} bytes")
        count = _struct.unpack_from('<Q', head, 1)[0]
        if count != nvals if whole else count > nvals:
            raise SperrError(
                f"sperr: constant field of {count} values, the header "
                f"dimensions hold {nvals}")
        return
    if n < _CONDI_LEN + _SPECK_HEADER_LEN:
        raise SperrError("sperr: coded data too short for its SPECK header")
    q = _struct.unpack_from('<d', head, 9)[0]
    # SPERR writes an infinite step when the field's range overflows
    # its arithmetic (psnr mode on values near the float64 limit); such
    # a stream decodes to NaN, as in imagecodecs and 0.4.0, so it is
    # let through. encode refuses to write one.
    if not q > 0.0:
        raise SperrError(f"sperr: invalid quantization step {q!r}")
    speck_len = _check_coded_bits(head, _CONDI_LEN, nvals, "SPECK stream")
    rest = n - _CONDI_LEN - speck_len
    if rest >= _SPECK_HEADER_LEN:
        o = start + _CONDI_LEN + speck_len
        ohead = bytes(data[o:o + _SPECK_HEADER_LEN])
        if _check_coded_bits(ohead, 0, nvals, "outlier stream") < rest:
            raise SperrError(
                "sperr: coded data is longer than the streams it declares")


def stream_info(data):
    """Read a SPERR stream's header without decoding.

    Returns ``(ndim, is_float, (dimz, dimy, dimx))`` for a stream that
    carries SPERR's header, after checking that the header is whole and
    that a 3-D stream's chunk table accounts for every byte, so the C
    decoder is never handed a header that points past the buffer.
    """
    head = bytes(data[:20])
    if len(head) < _HEADER_2D_LEN:
        raise SperrError("sperr: stream too short for a SPERR header")
    if head[0] != SPERR_VERSION_MAJOR:
        raise SperrError(
            f"sperr: stream major version {head[0]} is not this SPERR "
            f"build's {SPERR_VERSION_MAJOR}")
    flags = head[1]
    is_float = 1 if flags & _FLAG_FLOAT else 0
    if not flags & _FLAG_3D:
        dimx, dimy = _struct.unpack_from('<II', head, 2)
        if dimx == 0 or dimy == 0:
            raise SperrError("sperr: header has a zero dimension")
        if len(data) <= _HEADER_2D_LEN:
            raise SperrError("sperr: stream payload missing")
        _check_speck(data, _HEADER_2D_LEN, len(data), dimx * dimy)
        return 2, is_float, (1, int(dimy), int(dimx))
    if len(head) < 14:
        raise SperrError("sperr: stream too short for a 3-D header")
    vol = _struct.unpack_from('<III', head, 2)
    if 0 in vol:
        raise SperrError("sperr: header has a zero dimension")
    if flags & _FLAG_MULTI_CHUNK:
        if len(head) < 20:
            raise SperrError("sperr: stream too short for a 3-D header")
        chunk = _struct.unpack_from('<HHH', head, 14)
        if 0 in chunk:
            raise SperrError("sperr: header has a zero chunk dimension")
        fixed = 20
    else:
        chunk = vol
        fixed = 14
    nchunks = _chunk_count(vol, chunk)
    header_len = fixed + 4 * nchunks
    if len(data) < header_len:
        raise SperrError("sperr: stream too short for its chunk table")
    lens = _struct.unpack_from(f'<{nchunks}I', bytes(data[fixed:header_len]))
    if header_len + sum(lens) != len(data):
        raise SperrError(
            f"sperr: header accounts for {header_len + sum(lens)} bytes, "
            f"the stream has {len(data)}")
    nvals = vol[0] * vol[1] * vol[2]
    pos = header_len
    for length in lens:
        _check_speck(data, pos, pos + length, nvals, nchunks == 1)
        pos += length
    return 3, is_float, (int(vol[2]), int(vol[1]), int(vol[0]))


def decode_native(data, *, shape=None, dtype=None, bint header=True,
                  int nthreads=0) -> 'np.ndarray':
    """Decode a stream in SPERR's own format.

    With ``header`` (the default) the dimensions and precision come
    from the stream. ``dtype`` may still ask for float32 or float64
    output, which SPERR converts to; ``shape`` must match the stream.
    A 2-D stream written without its header needs ``shape`` (2-D) and
    ``dtype``.
    """
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        Py_ssize_t offset
        void* dst = NULL
        size_t out_dimx = 0, out_dimy = 0, out_dimz = 0
        size_t dx, dy
        int rc
        int isf
        int ndim
        size_t workers
        cnp.ndarray out_arr

    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    src = data
    srcsize = src.shape[0]
    want = None if dtype is None else np.dtype(dtype)
    if want is not None and want not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"sperr decode: dtype must be float32 or float64, got {want}")
    if shape is not None:
        shape = (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape)

    if header:
        ndim, stream_float, zyx = stream_info(data)
        stream_shape = zyx[1:] if ndim == 2 else zyx
        if shape is not None and shape != stream_shape:
            raise ValueError(
                f"sperr decode: shape {shape} does not match the stream's "
                f"{stream_shape}")
        shape = stream_shape
        if want is None:
            want = np.dtype(np.float32 if stream_float else np.float64)
        offset = _HEADER_2D_LEN if ndim == 2 else 0
    else:
        if shape is None or want is None:
            raise ValueError(
                "sperr decode: a stream without its header needs shape= and dtype=")
        if len(shape) != 2:
            raise ValueError(
                "sperr decode: only a 2-D stream can lack its header")
        if 0 in shape:
            raise ValueError("sperr decode: shape has a zero dimension")
        ndim = 2
        offset = 0
        _check_speck(data, 0, srcsize, shape[0] * shape[1])
    if srcsize <= offset:
        raise SperrError("sperr blob payload missing")

    isf = 1 if want == np.dtype(np.float32) else 0
    workers = <size_t> max(0, nthreads)
    if ndim == 2:
        dy = <size_t> shape[0]
        dx = <size_t> shape[1]
        with nogil:
            rc = sperr_decomp_2d(<const void*> &src[offset],
                                 <size_t> (srcsize - offset), isf, dx, dy, &dst)
    else:
        with nogil:
            rc = sperr_decomp_3d(<const void*> &src[0], <size_t> srcsize, isf,
                                 workers, &out_dimx, &out_dimy, &out_dimz, &dst)
        if rc == 0 and (out_dimz, out_dimy, out_dimx) != shape:
            if dst != NULL:
                free(dst)
            raise SperrError(
                f"sperr: decoded dimensions {(out_dimz, out_dimy, out_dimx)} "
                f"differ from the header's {shape}")
    if rc != 0 or dst == NULL:
        if dst != NULL:
            free(dst)
        raise SperrError(f"SPERR decompression failed (rc={rc})")
    out_arr = np.empty(shape, dtype=want)
    try:
        memcpy(<void*> out_arr.data, dst, <size_t> out_arr.nbytes)
    finally:
        free(dst)
    return out_arr


def _looks_like_speck(head, Py_ssize_t start, Py_ssize_t length, nvals):
    """The fixed fields of one SPECK_FLT stream of ``length`` bytes at
    ``head[start:]``, as far as ``head`` holds them, have values SPERR
    writes. The coded bits may be fewer than the count declares: a
    ``bpp`` stream is cut short by design."""
    if length < _CONDI_LEN:
        return False
    if len(head) <= start:
        return True
    meta = head[start]
    if meta & _CONDI_UNUSED:
        return False
    if meta & _CONDI_CONSTANT:
        if length != _CONDI_LEN:
            return False
        return (len(head) < start + 9
                or 0 < _struct.unpack_from('<Q', head, start + 1)[0] <= nvals)
    if length < _CONDI_LEN + _SPECK_HEADER_LEN:
        return False
    if len(head) < start + _CONDI_LEN:
        return True
    if not _struct.unpack_from('<d', head, start + 9)[0] > 0.0:
        return False
    at = start + _CONDI_LEN
    if len(head) < at + _SPECK_HEADER_LEN:
        return True
    planes = head[at]
    return (planes <= 64 and _struct.unpack_from('<Q', head, at + 1)[0]
            <= _speck_bits_limit(nvals, planes))


_SNIFF_MAX_VALUES = 1 << 40


def _looks_like_stream(head):
    """True if ``head``, the first bytes of a buffer (all of it when
    shorter than 512), starts the way a SPERR stream with its header
    does: this build's version byte, flags SPERR sets, nonzero
    dimensions holding at most 2**40 values (8 TiB of float64, more
    than any array decoded in memory, which bounds the bit count a
    coded stream may declare), chunk lengths SPERR writes for every
    entry of the chunk table that ``head`` holds, and values SPERR
    writes in the fixed fields of the first coded stream that ``head``
    holds. A chunk table of 117 chunks or more can push some or all of
    those fields past the 512th byte; the table entries in ``head``
    (at least 117 of them) are then checked without them. A buffer
    that fits in ``head`` must also pass the whole check
    ``stream_info`` makes."""
    if len(head) < _HEADER_2D_LEN or head[0] != SPERR_VERSION_MAJOR:
        return False
    flags = head[1]
    if flags & ~(_FLAG_3D | _FLAG_FLOAT | _FLAG_MULTI_CHUNK) or (
            flags & _FLAG_MULTI_CHUNK and not flags & _FLAG_3D):
        return False
    whole = len(head) < 512
    if not flags & _FLAG_3D:
        dimx, dimy = _struct.unpack_from('<II', head, 2)
        nvals = dimx * dimy
        start, length = _HEADER_2D_LEN, len(head) - _HEADER_2D_LEN
    else:
        if len(head) < 14:
            return False
        vol = _struct.unpack_from('<III', head, 2)
        nvals = vol[0] * vol[1] * vol[2]
        if flags & _FLAG_MULTI_CHUNK:
            if len(head) < 20:
                return False
            chunk = _struct.unpack_from('<HHH', head, 14)
            if 0 in chunk:
                return False
            fixed = 20
        else:
            chunk = vol
            fixed = 14
        if not 0 < nvals <= _SNIFF_MAX_VALUES:
            return False
        nchunks = _chunk_count(vol, chunk)
        start = fixed + 4 * nchunks
        if len(head) < fixed + 4:
            return False
        # Every chunk length the head holds must be one SPERR writes: a
        # constant chunk's 17 bytes, or a conditioner header, a SPECK
        # header and at most two streams of bits for the values of one
        # chunk. A chunk spans at most twice the chunk size along each
        # axis (sperr::chunk_volume adds a short remainder to the last
        # chunk), and never more than the volume.
        chunk_vals = 1
        for v, c in zip(vol, chunk):
            chunk_vals *= min(v, 2 * c)
        longest = _CONDI_LEN + 2 * (_SPECK_HEADER_LEN
                                    + _speck_bits_limit(chunk_vals, 64) // 8 + 1)
        seen = min(nchunks, (len(head) - fixed) // 4)
        for n in _struct.unpack_from(f'<{seen}I', head, fixed):
            if not (n == _CONDI_LEN
                    or _CONDI_LEN + _SPECK_HEADER_LEN <= n <= longest):
                return False
        length = _struct.unpack_from('<I', head, fixed)[0]
    if not 0 < nvals <= _SNIFF_MAX_VALUES:
        return False
    if not _looks_like_speck(head, start, length, nvals):
        return False
    if whole:
        try:
            stream_info(head)
        except Exception:
            return False
    return True


def check_signature(data) -> bool:
    """Match a SPERR stream with its header (see ``_looks_like_stream``;
    SPERR's header has no magic, it starts with the version number), or
    the 'SPRR' preamble of releases up to 0.4.0. A 2-D stream written
    without its header, or a stream of more than 2**40 values, is not
    recognized."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:512])
    else:
        try:
            head = bytes(data)[:512]
        except Exception:
            return False
    if head[:4] == _HEADER_MAGIC:
        return True
    return _looks_like_stream(head)
