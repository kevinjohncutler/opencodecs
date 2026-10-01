# opencodecs/codecs/_sz3.pyx
# distutils: language = c
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""Native SZ3 codec — error-bounded lossy compression for scientific arrays.

SZ3 is a modern (2022+) prediction-based compressor that often
beats ZFP at the same error budget for scientifically-correlated data
(time series, simulation snapshots). The C API is ``SZ_compress_args``.

``encode`` writes SZ3's own stream as the library produces it
(SZ3/api/sz.hpp): a 16-byte header (magic 0xF342F310, data version,
payload size), the payload, then the serialized configuration, which
records the dimensions with the ones of size 1 dropped. That is the
stream ``imagecodecs.sz3_encode`` writes. The configuration's data
type byte is not reliable (the C API leaves it at SZ_FLOAT for
float64), so the reader passes ``dtype``, as SZ3's own tools and
``imagecodecs.sz3_decode`` require.

Releases up to 0.4.0 wrote a private 48-byte preamble in front of the
stream instead::

    bytes  0..3   ASCII magic 'SZ3O'
    byte   4      dtype enum (SZ_FLOAT/SZ_DOUBLE/...)
    byte   5      ndim (1..5)
    bytes  6..7   reserved
    bytes  8..47  shape: 5 × uint64 little-endian (r1, r2, r3, r4, r5)

``decode_framed`` still reads those blobs, recognized by the magic.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize, PyBytes_AsString
from libc.stdint cimport uint8_t
from libc.string cimport memcpy

import numpy as np
cimport numpy as cnp

from sz3c cimport (
    ABS, REL, ABS_AND_REL, ABS_OR_REL,
    SZ_FLOAT, SZ_DOUBLE,
    SZ_UINT8, SZ_INT8, SZ_UINT16, SZ_INT16,
    SZ_UINT32, SZ_INT32, SZ_UINT64, SZ_INT64,
    SZ_compress_args, SZ_decompress, free_buf,
)


cnp.import_array()


_HEADER_LEN = 48
_HEADER_MAGIC = b'SZ3O'
_SZ3_MAGIC = b'\x10\xf3\x42\xf3'   # 0xF342F310, little-endian
import struct as _struct
_HEADER_FMT = '<4sBB2x5Q'


class Sz3Error(RuntimeError):
    """Raised on SZ3 encode/decode failures."""


_DTYPE_TO_SZ = {
    np.dtype(np.float32): SZ_FLOAT,
    np.dtype(np.float64): SZ_DOUBLE,
    np.dtype(np.uint8):   SZ_UINT8,
    np.dtype(np.int8):    SZ_INT8,
    np.dtype(np.uint16):  SZ_UINT16,
    np.dtype(np.int16):   SZ_INT16,
    np.dtype(np.uint32):  SZ_UINT32,
    np.dtype(np.int32):   SZ_INT32,
    np.dtype(np.uint64):  SZ_UINT64,
    np.dtype(np.int64):   SZ_INT64,
}
_SZ_TO_DTYPE = {v: k for k, v in _DTYPE_TO_SZ.items()}


# The error-bound modes SZ_compress_args implements. sz3c.h also
# declares PSNR (4), NORM (5) and the point-wise relative modes, but the
# C API answers those by printing "not support" and exiting the
# process, so they are refused here with an error instead.
_MODE_ALIASES = {
    "abs": ABS, "rel": REL, "abs_and_rel": ABS_AND_REL,
    "abs_or_rel": ABS_OR_REL,
}
_MODE_VALUES = frozenset(_MODE_ALIASES.values())


def mode_value(mode):
    """The sz3c error-bound mode for a name ('abs', 'ABS_OR_REL', ...)
    or for its integer value (imagecodecs' SZ3.MODE)."""
    if isinstance(mode, str):
        key = mode.lower()
        if key not in _MODE_ALIASES:
            raise ValueError(
                f"sz3 encode: unknown mode {mode!r}; expected one of "
                f"{sorted(_MODE_ALIASES.keys())}"
            )
        return _MODE_ALIASES[key]
    value = int(mode)
    if value not in _MODE_VALUES:
        raise ValueError(f"sz3 encode: unknown mode value {value}")
    return value


# Shared with sperr and pcodec, which prefix the same shape of header.
from opencodecs.core._sidecar_header import make_sidecar_header as \
    _make_sidecar_header
_HEADER_LEN, _pack_header, _unpack_header = _make_sidecar_header(
    _HEADER_MAGIC, 5, Sz3Error, "sz3")


def encode(arr, *,
           mode="abs",
           abs_err: float = 0.0,
           rel_err: float = 0.0) -> bytes:
    """Compress an ndarray into a bare SZ3 stream.

    Parameters
    ----------
    arr : np.ndarray
        float32 or float64, at most four dimensions longer than 1.
    mode : str or int
        "abs" (absolute err bound), "rel" (value-range relative),
        "abs_and_rel", "abs_or_rel", or the matching sz3c.h value.
    abs_err : float
        Used in "abs" / mixed modes. Absolute error per pixel
        (default 0.0, as in imagecodecs).
    rel_err : float
        Used in "rel" / mixed modes. Fraction of value range
        (default 0.0).
    """
    cdef:
        cnp.ndarray contig
        int dtype_enum
        int err_mode
        size_t r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0
        size_t out_size = 0
        unsigned char* sz_buf
        bytes payload

    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    contig = np.ascontiguousarray(arr)
    if contig.dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        # The C API declares the integer types but exits the process
        # on them, like the unsupported modes.
        raise ValueError(
            f"sz3 encode: only float32/float64 supported (got {contig.dtype!r})")
    dtype_enum = _DTYPE_TO_SZ[contig.dtype]

    err_mode = mode_value(mode)

    if contig.ndim < 1 or contig.ndim > 5:
        raise ValueError(f"sz3 encode: ndim must be 1..5, got {contig.ndim}")
    # SZ3 drops dimensions of size 1 and handles at most four of the
    # rest; past that the library throws, which would end the process.
    if sum(1 for d in arr.shape if d > 1) > 4:
        raise ValueError(
            f"sz3 encode: shape {arr.shape} has more than four dimensions "
            "longer than 1, which SZ3 cannot compress")
    if contig.size == 0:
        raise ValueError("sz3 encode: cannot compress an empty array")

    # SZ3 takes (r5, r4, r3, r2, r1) — innermost dim is r1 (fastest-varying).
    # numpy is row-major: outermost dim is shape[0]. Map: r1 = shape[ndim-1],
    # r2 = shape[ndim-2], ..., r5 = shape[0]. Unused dims pass 0.
    if contig.ndim >= 1: r1 = <size_t> contig.shape[contig.ndim - 1]
    if contig.ndim >= 2: r2 = <size_t> contig.shape[contig.ndim - 2]
    if contig.ndim >= 3: r3 = <size_t> contig.shape[contig.ndim - 3]
    if contig.ndim >= 4: r4 = <size_t> contig.shape[contig.ndim - 4]
    if contig.ndim >= 5: r5 = <size_t> contig.shape[contig.ndim - 5]

    cdef void* data_ptr = <void*> contig.data
    cdef double abs_e = float(abs_err)
    cdef double rel_e = float(rel_err)
    # The point-wise relative bound, used only by modes refused above.
    cdef double pwr_e = 0.0

    with nogil:
        sz_buf = SZ_compress_args(
            dtype_enum, data_ptr, &out_size,
            err_mode,
            abs_e, rel_e, pwr_e,
            r5, r4, r3, r2, r1,
        )
    if sz_buf == NULL or out_size == 0:
        raise Sz3Error("SZ_compress_args returned NULL")

    try:
        payload = PyBytes_FromStringAndSize(<const char*> sz_buf, <Py_ssize_t> out_size)
    finally:
        free_buf(<void*> sz_buf)

    return payload


def decode_framed(data, *, out=None) -> 'np.ndarray':
    """Decode a blob in the framed layout releases up to 0.4.0 wrote
    ('SZ3O' preamble + SZ3 stream) to an ndarray.

    ``out=`` is a preallocated ndarray; see ``_png.decode`` for the full
    contract. SZ3's library decompresses into its own malloc'd buffer
    which we memcpy into the destination, so out= saves the second
    alloc (not the SZ3-internal one).
    """
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        int dtype_enum
        int ndim
        size_t r1 = 0, r2 = 0, r3 = 0, r4 = 0, r5 = 0
        Py_ssize_t payload_len
        void* sz_out
        size_t total_elems = 1
        cnp.ndarray out_arr

    try:
        src = data
    except (TypeError, ValueError, BufferError):
        src = bytes(data)
    srcsize = src.shape[0]
    if srcsize < _HEADER_LEN:
        raise Sz3Error("sz3 blob too short to contain header")

    dtype_enum, ndim, shape5 = _unpack_header(bytes(src[:_HEADER_LEN]))
    r1, r2, r3, r4, r5 = shape5

    if dtype_enum not in _SZ_TO_DTYPE:
        raise Sz3Error(f"sz3 blob has unsupported dtype enum {dtype_enum}")
    dtype = _SZ_TO_DTYPE[dtype_enum]

    # Reconstruct numpy shape from r1..r5 and ndim.
    if ndim == 1:
        shape = (int(r1),)
    elif ndim == 2:
        shape = (int(r2), int(r1))
    elif ndim == 3:
        shape = (int(r3), int(r2), int(r1))
    elif ndim == 4:
        shape = (int(r4), int(r3), int(r2), int(r1))
    elif ndim == 5:
        shape = (int(r5), int(r4), int(r3), int(r2), int(r1))
    else:
        raise Sz3Error(f"sz3 header has invalid ndim={ndim}")

    payload_len = srcsize - _HEADER_LEN
    if payload_len <= 0:
        raise Sz3Error("sz3 blob payload missing")
    if dtype_enum != SZ_FLOAT and dtype_enum != SZ_DOUBLE:
        raise Sz3Error(f"sz3 blob has unsupported dtype enum {dtype_enum}")
    # Check the SZ3 stream before handing it to the library, which ends
    # the process on a stream it cannot read, and make sure it holds as
    # many values as the preamble's shape.
    version, _, num = stream_info(memoryview(data)[_HEADER_LEN:]
                                  if isinstance(data, (bytes, bytearray, memoryview))
                                  else bytes(data)[_HEADER_LEN:])
    if (version >> 8) != (library_data_version() >> 8):
        raise Sz3Error("sz3: stream data version differs from this SZ3 build's")
    if num != int(np.prod(shape, dtype=np.uint64)):
        raise Sz3Error(
            f"sz3 blob: preamble shape {shape} does not match the stream's "
            f"{num} values")
    try:
        check_payload(memoryview(data)[_HEADER_LEN:]
                      if isinstance(data, (bytes, bytearray, memoryview))
                      else bytes(data)[_HEADER_LEN:], dtype)
    except ValueError as exc:
        raise Sz3Error(f"sz3 blob: {exc}") from None

    if out is not None:
        if not isinstance(out, np.ndarray):
            raise TypeError(
                f"sz3 decode: out= must be an ndarray, "
                f"got {type(out).__name__}")
        if out.shape != shape:
            raise ValueError(
                f"sz3 decode: out= shape {out.shape} does not match "
                f"expected {shape}")
        if out.dtype != dtype:
            raise ValueError(
                f"sz3 decode: out= dtype {out.dtype} does not match "
                f"expected {dtype}")
        if not out.flags['C_CONTIGUOUS']:
            raise ValueError("sz3 decode: out= must be C-contiguous")
        out_arr = out
    else:
        out_arr = np.empty(shape, dtype=dtype)
    total_elems = <size_t> out_arr.size

    cdef Py_ssize_t header_off = <Py_ssize_t> _HEADER_LEN
    with nogil:
        sz_out = SZ_decompress(
            dtype_enum,
            <unsigned char*> &src[header_off], <size_t> payload_len,
            r5, r4, r3, r2, r1,
        )
    if sz_out == NULL:
        raise Sz3Error("SZ_decompress returned NULL")
    try:
        memcpy(<void*> out_arr.data, sz_out, total_elems * out_arr.dtype.itemsize)
    finally:
        free_buf(sz_out)
    return out_arr


_library_version = None


def library_data_version():
    """The SZ3 data version the linked library writes and accepts.

    SZ3 throws, ending the process, on a stream of another data
    version, so ``decode_raw`` checks first. The library reports the
    version only inside a stream, so compress one value once to read it.
    """
    global _library_version
    if _library_version is None:
        probe = encode(np.zeros(1, dtype=np.float32), mode="abs", abs_err=1.0)
        _library_version = int.from_bytes(probe[4:8], 'little')
    return _library_version


def stream_info(data):
    """Parse a bare SZ3 stream's header and configuration.

    Returns ``(version, dims, num)``, ``dims`` slowest first with the
    size-1 dimensions SZ3 drops already gone. Raises Sz3Error for
    anything that is not a whole, well-formed SZ3 stream.
    """
    buf = bytes(data[:16])
    if len(buf) < 16 or buf[:4] != _SZ3_MAGIC:
        raise Sz3Error("sz3: not an SZ3 stream (no 0xF342F310 magic)")
    version = int.from_bytes(buf[4:8], 'little')
    payload = int.from_bytes(buf[8:16], 'little')
    total = len(data)
    conf_at = 16 + payload
    if payload > total or conf_at + 3 > total:
        raise Sz3Error("sz3: stream is truncated")
    conf = bytes(data[conf_at:conf_at + 256])
    conf_size = conf[0]
    ndim = conf[1]
    width = conf[2]
    if not (1 <= ndim <= 4) or not (1 <= width <= 64):
        raise Sz3Error(f"sz3: corrupt configuration (N={ndim}, width={width})")
    nbytes = (ndim * width + 7) // 8
    if conf_size > len(conf) or 3 + nbytes + 8 > conf_size:
        raise Sz3Error("sz3: corrupt or truncated configuration")
    packed = int.from_bytes(conf[3:3 + nbytes], 'little')
    mask = (1 << width) - 1
    dims = tuple((packed >> (i * width)) & mask for i in range(ndim))
    num = int.from_bytes(conf[3 + nbytes:3 + nbytes + 8], 'little')
    prod = 1
    for d in dims:
        prod *= d
    if 0 in dims or prod != num:
        raise Sz3Error(f"sz3: corrupt configuration (dims {dims}, num {num})")
    return version, dims, num


# SZ3 data versions whose payload layout ``check_payload`` knows. The
# layout below is SZ3 3.3.2's (api/impl/SZAlgo*.hpp, the generic
# compressor, LinearQuantizer, HuffmanEncoder, Lossless_zstd); every
# opencodecs build links that release (bench/build_codec_libs.sh).
_LAYOUT_VERSIONS = frozenset({0x03030200})
_ALGO_LORENZO_REG, _ALGO_INTERP, _ALGO_NOPRED, _ALGO_LOSSLESS = 0, 2, 3, 4
_ALGO_NAMES = {0: "LORENZO_REG", 1: "INTERP_LORENZO", 2: "INTERP",
               3: "NOPRED", 4: "LOSSLESS", 5: "BIOMD", 6: "BIOMDXTC"}
# Bytes of error bound the configuration stores for each EB_* mode.
_EB_BOUND_BYTES = {0: 8, 1: 8, 2: 8, 3: 8, 4: 16, 5: 16}


def stream_config(data):
    """The configuration fields ``check_payload`` needs, as a dict:
    version, dims, num, payload (its byte count), algo, lorenzo,
    lorenzo2, regression and openmp."""
    version, dims, num = stream_info(data)
    payload = int.from_bytes(bytes(data[8:16]), 'little')
    conf = bytes(data[16 + payload:16 + payload + 256])
    conf_size = conf[0]
    width = conf[2]
    p = 3 + (len(dims) * width + 7) // 8 + 8
    if p + 2 > conf_size:
        raise Sz3Error("sz3: truncated configuration")
    algo = conf[p]
    eb_mode = conf[p + 1]
    if eb_mode not in _EB_BOUND_BYTES:
        raise Sz3Error(f"sz3: corrupt configuration (error bound mode {eb_mode})")
    p += 2 + _EB_BOUND_BYTES[eb_mode]
    # Config::load keeps its defaults for fields an older writer left out.
    lorenzo, lorenzo2, regression, openmp = 1, 0, 1, 0
    if p < conf_size:
        flags = conf[p]
        lorenzo = (flags >> 7) & 1
        lorenzo2 = (flags >> 6) & 1
        regression = (flags >> 5) & 1
        openmp = (flags >> 3) & 1
    return dict(version=version, dims=dims, num=num, payload=payload,
                algo=algo, lorenzo=lorenzo, lorenzo2=lorenzo2,
                regression=regression, openmp=openmp)


class _Short(Exception):
    pass


cdef class _Cursor:
    """Reads the fields of an SZ3 payload, raising _Short past its end."""
    cdef const uint8_t[::1] buf
    cdef public Py_ssize_t pos

    def __init__(self, buf):
        self.buf = buf
        self.pos = 0

    def size(self):
        return self.buf.shape[0]

    def skip(self, n):
        if n < 0 or n > self.buf.shape[0] - self.pos:
            raise _Short()
        self.pos += n

    def u8(self):
        if self.pos + 1 > self.buf.shape[0]:
            raise _Short()
        self.pos += 1
        return self.buf[self.pos - 1]

    def le(self, n):
        cdef Py_ssize_t i
        cdef unsigned long long v = 0
        if self.pos + n > self.buf.shape[0]:
            raise _Short()
        for i in range(n - 1, -1, -1):
            v = (v << 8) | self.buf[self.pos + i]
        self.pos += n
        return v

    def be32(self):
        cdef Py_ssize_t i
        cdef long long v = 0
        if self.pos + 4 > self.buf.shape[0]:
            raise _Short()
        for i in range(4):
            v = (v << 8) | self.buf[self.pos + i]
        self.pos += 4
        if v >= 0x80000000:
            v -= 0x100000000
        return v

    def at_end(self):
        return self.pos == self.buf.shape[0]


def _quantizer(cur, int itemsize):
    # LinearQuantizer<T>::save: uid (0b10), error_bound (double),
    # radius (int), the count of unpredictable values (size_t), then
    # those values as T.
    if cur.u8() != 2:
        raise _Short()
    cur.skip(8 + 4)
    n = cur.le(8)
    if n > (cur.size() - cur.pos) // itemsize:
        raise _Short()
    cur.skip(n * itemsize)


def _huffman(cur):
    # HuffmanEncoder<int>::save, then encode's output: offset (int),
    # node count and half the state count (big-endian ints), the tree
    # (sized as HuffmanEncoder::load sizes it), then the encoded
    # length (size_t) and that many bytes. A one-node tree is a
    # constant, for which decode() reads the length and nothing more.
    cur.skip(4)
    nodes = cur.be32()
    cur.be32()
    if nodes < 1:
        raise _Short()
    if nodes <= 256:
        tree = 1 + 3 * nodes + 4 * nodes
    elif nodes <= 65536:
        tree = 1 + 2 * nodes * 2 + nodes + 4 * nodes
    else:
        tree = 1 + 2 * nodes * 4 + nodes + 4 * nodes
    cur.skip(tree)
    return nodes


def _coded(cur, nodes):
    length = cur.le(8)
    if nodes > 1:
        cur.skip(length)


def _regression(cur, int itemsize):
    # RegressionPredictor::save
    if cur.le(8):
        _quantizer(cur, itemsize)
        _quantizer(cur, itemsize)
        _coded(cur, _huffman(cur))


def _layout_fits(inner, cfg, int itemsize):
    """True if the decompressed payload has exactly the layout SZ3
    writes for this configuration with ``itemsize``-byte values."""
    cur = _Cursor(inner)
    try:
        algo = cfg['algo']
        if algo == _ALGO_INTERP:
            # InterpolationDecomposition::save: the dimensions, block
            # size, interpolator and direction ids, anchor stride and
            # two double tuning factors.
            for d in cfg['dims']:
                if cur.le(8) != d:
                    return False
            cur.skip(4 + 4 + 4 + 8 + 8 + 8)
        elif algo == _ALGO_LORENZO_REG:
            # BlockwiseDecomposition::save: the Lorenzo fallback and
            # Lorenzo predictors save nothing; a regression predictor
            # and a composed predictor's selection do.
            count = cfg['lorenzo'] + cfg['lorenzo2'] + cfg['regression']
            if count == 0:
                return False
            if cfg['regression']:
                _regression(cur, itemsize)
            if count > 1:
                if cur.le(8):
                    _coded(cur, _huffman(cur))
        elif algo != _ALGO_NOPRED:
            return False
        _quantizer(cur, itemsize)
        nodes = _huffman(cur)
        cur.le(8)  # how many quantization indices follow
        _coded(cur, nodes)
        return cur.at_end()
    except _Short:
        return False


def check_payload(data, dtype):
    """Check that an SZ3 stream's payload holds ``dtype`` values and has
    the layout SZ3 writes for its configuration, before SZ3 sees it.

    The stream does not record the value type reliably, and SZ3 reads
    a payload with no bounds checks: decoding with the wrong type (or a
    damaged payload) reads and writes past its buffers, or throws an
    exception the C API does not catch, ending the process. The type
    does show in the payload's layout, in the size of each block of
    unpredictable values, so this walks that layout for float32 and
    float64 and raises ValueError if only the other type fits, Sz3Error
    if neither does. When the stream holds no unpredictable values both
    fit and the type cannot be told; SZ3 then reads the same bytes
    either way. Streams of SZ3 data versions whose layout this does not
    know pass unchecked.
    """
    dt = np.dtype(dtype)
    cfg = stream_config(data)
    if cfg['version'] not in _LAYOUT_VERSIONS:
        return
    other = np.dtype(np.float64 if dt.itemsize == 4 else np.float32)
    if cfg['openmp']:
        raise Sz3Error(
            "sz3: the stream was written with SZ3's OpenMP layout, which "
            "this build cannot check or read")
    algo = cfg['algo']
    if algo not in (_ALGO_LORENZO_REG, _ALGO_INTERP, _ALGO_NOPRED,
                    _ALGO_LOSSLESS):
        raise Sz3Error(
            f"sz3: cannot check a stream of SZ3 algorithm "
            f"{_ALGO_NAMES.get(algo, algo)} before decoding it, and SZ3 "
            f"ends the process on one it cannot read")
    payload = memoryview(data)[16:16 + cfg['payload']]
    if len(payload) < 8:
        raise Sz3Error("sz3: payload is truncated")
    inner_len = int.from_bytes(bytes(payload[:8]), 'little')
    num = cfg['num']
    if algo == _ALGO_LOSSLESS:
        # Lossless_zstd: the raw byte count, then a zstd frame of the
        # values; SZ3 inflates that many bytes into a buffer of num T.
        if inner_len == num * dt.itemsize:
            return
        if inner_len == num * other.itemsize:
            raise ValueError(
                f"sz3 decode: the stream holds {other} values, not {dt}")
        raise Sz3Error(
            f"sz3: the payload holds {inner_len} bytes of values, not "
            f"{num} values")
    # Lossy: the payload is a zstd frame of the compressor's buffer.
    # That buffer is at most twice the values plus their indices.
    if inner_len > 64 * num + (1 << 24):
        raise Sz3Error("sz3: corrupt payload (implausible size)")
    from opencodecs.codecs import _zstd
    try:
        inner = _zstd.decode(payload[8:], out=inner_len)
    except Exception as exc:
        raise Sz3Error(f"sz3: corrupt payload ({exc})") from None
    if len(inner) != inner_len:
        raise Sz3Error(
            f"sz3: payload inflates to {len(inner)} bytes, its header "
            f"says {inner_len}")
    if _layout_fits(inner, cfg, dt.itemsize):
        return
    if _layout_fits(inner, cfg, other.itemsize):
        raise ValueError(
            f"sz3 decode: the stream holds {other} values, not {dt}")
    raise Sz3Error(
        f"sz3: the payload does not have the layout SZ3 writes for its "
        f"{_ALGO_NAMES.get(algo, algo)} configuration; it is corrupt")


def decode_raw(data, *, dtype, shape=None, out=None) -> 'np.ndarray':
    """Decode a bare SZ3 stream into ``dtype`` (float32 or float64).

    ``shape`` defaults to the stream's dimensions (which omit the
    dimensions of size 1) and must hold the same number of values.
    ``out``, if given, is a C-contiguous array of that shape and dtype.
    """
    cdef:
        const uint8_t[::1] src
        Py_ssize_t srcsize
        int dtype_enum
        void* sz_out
        size_t total_elems
        size_t r1, r2, r3, r4, r5
        cnp.ndarray out_arr

    if not isinstance(data, (bytes, bytearray, memoryview)):
        data = bytes(data)
    src = data
    srcsize = src.shape[0]
    dt = np.dtype(dtype)
    if dt not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"sz3 decode: dtype must be float32 or float64, got {dt}")
    dtype_enum = _DTYPE_TO_SZ[dt]
    version, dims, num = stream_info(data)
    if (version >> 8) != (library_data_version() >> 8):
        raise Sz3Error(
            f"sz3: stream data version {version >> 24}.{(version >> 16) & 255}."
            f"{(version >> 8) & 255} differs from this SZ3 build's")
    check_payload(data, dt)
    if shape is None:
        shape = dims
    else:
        shape = (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape)
        if tuple(d for d in shape if d > 1) != tuple(d for d in dims if d > 1):
            raise ValueError(
                f"sz3 decode: shape {shape} does not match the stream's "
                f"dimensions {dims}")

    if out is not None:
        if not isinstance(out, np.ndarray):
            raise TypeError(
                f"sz3 decode: out= must be an ndarray, "
                f"got {type(out).__name__}")
        if out.shape != shape:
            raise ValueError(
                f"sz3 decode: out= shape {out.shape} does not match "
                f"expected {shape}")
        if out.dtype != dt:
            raise ValueError(
                f"sz3 decode: out= dtype {out.dtype} does not match "
                f"expected {dt}")
        if not out.flags['C_CONTIGUOUS']:
            raise ValueError("sz3 decode: out= must be C-contiguous")
        out_arr = out
    else:
        out_arr = np.empty(shape, dtype=dt)
    total_elems = <size_t> num
    # SZ_decompress sizes its result from the stream's configuration,
    # which was checked above to hold exactly ``num`` values. Pass the
    # same dimensions, fastest last, right-aligned onto r5..r1.
    rdims = (0,) * (5 - len(dims)) + tuple(dims)
    r5, r4, r3, r2, r1 = rdims
    with nogil:
        sz_out = SZ_decompress(
            dtype_enum, <unsigned char*> &src[0], <size_t> srcsize,
            r5, r4, r3, r2, r1,
        )
    if sz_out == NULL:
        raise Sz3Error("SZ_decompress returned NULL")
    try:
        memcpy(<void*> out_arr.data, sz_out, total_elems * out_arr.dtype.itemsize)
    finally:
        free_buf(sz_out)
    return out_arr


def check_signature(data) -> bool:
    """Match SZ3's own magic, or the 'SZ3O' preamble of releases up
    to 0.4.0."""
    cdef bytes head
    if isinstance(data, (bytes, bytearray)):
        head = bytes(data[:4])
    else:
        try:
            head = bytes(data)[:4]
        except Exception:
            return False
    return head == _SZ3_MAGIC or head == _HEADER_MAGIC
