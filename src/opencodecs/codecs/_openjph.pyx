# opencodecs/codecs/_openjph.pyx
# distutils: language = c++
# cython: boundscheck = False
# cython: wraparound = False
# cython: cdivision = True
# cython: nonecheck = False
# cython: language_level = 3

"""HTJ2K codec via OpenJPH (libopenjph).

High-Throughput JPEG-2000 (ISO/IEC 15444-15) — a block-coder
replacement for the Part-1 EBCOT entropy coder. Same wavelet front
end as classic JPEG-2000 but ~10-20x faster encode/decode.

Built on top of OpenJPH's C++ ``ojph::codestream`` API via a thin
C shim (``openjph_shim.cpp``). All I/O goes through OpenJPH's
``mem_infile`` / ``mem_outfile`` so there are no temp files.

Supported pixels
================

  * 1 to 16384 components (the SIZ limit), all with the same precision
  * 1-32 bit precision: (u)int8, (u)int16 or (u)int32 samples
  * float32 (and float16 on decode), carried as the samples' bit
    patterns under an NLT type 3 marker (ISO/IEC 15444-2), the way
    imagecodecs and OpenJPH write floating point data
  * 2-D ``(H, W)``, 3-D ``(H, W, C)``, or planar ``(C, H, W)`` arrays

The keyword names and their meanings follow ``imagecodecs.htj2k_encode``
and ``imagecodecs.htj2k_decode``.

Output codestream is always raw HTJ2K (.j2c-style) — no JP2 box wrapping.
"""

from cpython.bytes cimport PyBytes_FromStringAndSize
from libc.stdint cimport uint8_t, uint16_t
from libc.string cimport memcpy

import numpy as np
cimport numpy as cnp

from openjph cimport (
    opencodecs_htj2k_encode_params,
    opencodecs_htj2k_info,
    opencodecs_htj2k_encode,
    opencodecs_htj2k_decode,
    opencodecs_htj2k_decode_info,
    opencodecs_htj2k_free,
    opencodecs_htj2k_last_error,
    opencodecs_htj2k_last_warnings,
    opencodecs_htj2k_clear_warnings,
)

cnp.import_array()


class OpenJphError(RuntimeError):
    """Raised on OpenJPH (HTJ2K) encode/decode failures."""


class OpenJphUnsupportedFeature(OpenJphError):
    """The codestream uses a marker segment OpenJPH does not implement.

    OpenJPH warns about these and keeps decoding, which means the pixels
    it returns come from a codestream it did not fully read. Measured
    against the JPEG committee's reference images, such a decode can be
    far off: one conformance file with an unread QCD-in-tile marker
    differed from the reference by 954 levels. Returning that quietly
    would be worse than failing, so it is raised instead.

    Pass ``ignore_unsupported=True`` to get the image anyway.
    """


def last_warnings() -> str:
    """Warnings OpenJPH raised during the most recent call."""
    return opencodecs_htj2k_last_warnings().decode("utf-8", errors="replace")


cdef _check_warnings(bint ignore_unsupported):
    """Turn 'not supported' warnings into an exception.

    Collecting the warnings at all also stops OpenJPH printing them to
    the process's stdout, where they corrupt whatever the caller was
    writing.
    """
    warned = opencodecs_htj2k_last_warnings().decode("utf-8", errors="replace")
    if not warned:
        return
    unsupported = [ln for ln in warned.split("\n") if "not supported" in ln]
    if unsupported and not ignore_unsupported:
        seen = sorted(set(unsupported))
        raise OpenJphUnsupportedFeature(
            "OpenJPH could not read part of this codestream, so the decoded "
            "image would not match the encoder's: "
            + "; ".join(seen)
            + ". Pass ignore_unsupported=True to decode anyway.")
    import warnings as _w
    for line in sorted(set(warned.split("\n"))):
        if line and line not in unsupported:
            _w.warn(f"OpenJPH: {line}", RuntimeWarning, stacklevel=3)


cdef _raise(int rc, str where):
    msg = opencodecs_htj2k_last_error().decode("utf-8", errors="replace")
    # The shim already prefixes with the failing call; don't say it twice.
    if msg.startswith(where + ":"):
        msg = msg[len(where) + 1:].strip()
    raise OpenJphError(f"{where}: {msg} (rc={rc})")


_PROG_ORDERS = ("LRCP", "RLCP", "RPCL", "PCRL", "CPRL")
_PROFILES = ("IMF", "BROADCAST")


def _quantization(level):
    """imagecodecs' reading of ``level``: (qstep, qfactor).

    Below 1 it is the irreversible quantization step, clamped to
    [0, 1] and held as a float32, and a step under 1e-5 means none at
    all. From 1 up it is a
    JPEG-style quality factor, clamped to [1, 100].
    """
    if level is None:
        return 0.0, 0
    level = float(level)
    if level < 1.0:
        # imagecodecs holds the step in a C float before the 1e-5 test,
        # so a level of exactly 1e-5 (9.99999975e-6 as a float) is
        # lossless there; rounding it the same way keeps the bytes equal.
        qstep = float(np.float32(min(max(level, 0.0), 1.0)))
        if qstep < 0.00001:
            qstep = 0.0
        return qstep, 0
    return 0.0, int(min(level, 100.0))


def _max_decompositions(int width, int height, int tile_w, int tile_h):
    cdef int m = width if width < height else height
    if tile_w and tile_h:
        m = min(m, tile_w, tile_h)
    cdef int n = 0
    while m > 1:
        m >>= 1
        n += 1
    return n


def encode(
    data,
    level=None,
    *,
    rgb=None,
    planar=None,
    tile=None,
    resolutions=None,
    reversible=None,
    tlm=None,
    tilepart=None,
    block_size=None,
    prog_order=None,
    profile=None,
    num_decomp=None,
) -> bytes:
    """Encode an ndarray as an HTJ2K codestream.

    Parameters
    ----------
    data
        2-D ``(H, W)``, 3-D ``(H, W, C)``, or with ``planar=True``
        ``(C, H, W)``. uint8, int8, uint16, int16, uint32, int32 or
        float32. A float32 image is written as the bit patterns of its
        samples with an NLT type 3 marker; it round-trips exactly only
        on the reversible path.
    level : float, optional
        ``None`` (default) is the reversible, mathematically lossless
        path. Below 1, the irreversible quantization step (smaller is
        closer to lossless; under 1e-5 means lossless). From 1 to 100,
        a JPEG-style quality factor on the irreversible path. This is
        ``imagecodecs.htj2k_encode``'s ``level``.
    rgb : bool, optional
        Apply the component transform (RCT when reversible, ICT when
        not) to components 0..2. ``None`` turns it on for interleaved
        3- and 4-component integer input, as imagecodecs and OpenJPH's
        ``ojph_compress`` do; ``False`` leaves the components alone.
        ``True`` applies it to any integer input with 3 or more
        components, planar included (imagecodecs ignores ``rgb=True``
        for planar input), and raises ValueError for float32 input or
        fewer than 3 components.
    planar : bool, optional
        ``True`` reads ``data`` as ``(C, H, W)``. ``None`` does so only
        when the last axis is longer than 4 and the first is 4 or
        shorter, imagecodecs' rule.
    tile : (int, int), optional
        Tile ``(width, height)``; ``None`` writes one tile.
    resolutions : int, optional
        DWT decomposition levels, clamped to what the image (or tile)
        size allows. 0 or ``None`` keeps OpenJPH's default of 5.
    reversible : bool, optional
        Force the reversible (5/3) or irreversible (9/7) path. ``None``
        picks reversible exactly when ``level`` asks for no loss.
        ``reversible=True`` with a lossy ``level`` raises ValueError.
    tlm : bool, optional
        Write a TLM (tile-part length) marker.
    tilepart : int, optional
        Tile-part divisions: 1 at resolutions, 2 at components, 3 both.
    block_size : (int, int), optional
        Code-block ``(width, height)``; default 64 x 64.
    prog_order : str, optional
        Progression order, one of LRCP, RLCP, RPCL, PCRL, CPRL.
    profile : str, optional
        ``"IMF"`` or ``"BROADCAST"``.
    num_decomp : int, optional
        DWT decomposition levels, used exactly as given (no clamping).
        Kept from earlier opencodecs releases; ``resolutions`` is the
        imagecodecs name.

    Returns
    -------
    bytes
        Raw HTJ2K codestream (.j2c).
    """
    cdef:
        opencodecs_htj2k_encode_params p
        void* out_buf = NULL
        size_t out_size = 0
        int rc
        bytes prog_b = None
        bytes profile_b = None

    arr = np.ascontiguousarray(data)
    kind = arr.dtype.kind
    itemsize = arr.dtype.itemsize
    is_float = False
    if kind in "ui" and itemsize in (1, 2, 4):
        is_signed = kind == "i"
    elif kind == "f" and itemsize == 4:
        is_float = True
        is_signed = True
    else:
        raise OpenJphError(
            f"HTJ2K encode: unsupported dtype {arr.dtype}; expected "
            f"uint8/int8/uint16/int16/uint32/int32/float32")

    if arr.ndim == 2:
        planar = False
        height, width = arr.shape
        components = 1
    elif arr.ndim == 3:
        if planar is None:
            planar = arr.shape[2] > 4 and arr.shape[0] <= 4
        if planar:
            components, height, width = arr.shape
        else:
            height, width, components = arr.shape
    else:
        raise OpenJphError(f"HTJ2K encode: unsupported ndim {arr.ndim}")
    if height < 1 or width < 1 or components < 1:
        raise OpenJphError(f"HTJ2K encode: empty image {arr.shape}")
    if components > 16384:
        # SIZ Csiz is 1 to 16384 (ISO/IEC 15444-1 Table A.9); OpenJPH
        # would write a codestream outside that range without an error.
        raise OpenJphError(
            f"HTJ2K encode: {components} components; a codestream holds "
            f"at most 16384")

    qstep, qfactor = _quantization(level)
    lossy_level = qstep > 0.0 or qfactor > 0
    if reversible is None:
        is_reversible = not lossy_level
    else:
        is_reversible = bool(reversible)
        if is_reversible and lossy_level:
            raise ValueError(
                f"HTJ2K encode: level={level!r} asks for lossy "
                f"quantization, which the reversible path does not do; "
                f"drop reversible=True or the level")
    if is_float and not is_reversible and qstep <= 0.0:
        # imagecodecs' minimum step for irreversible float32
        qstep = 1.0 / 16384.0

    # rgb=None follows imagecodecs: the transform for interleaved 3- and
    # 4-component integer input only. An explicit rgb=True is honored
    # for planar input too, and refused where it cannot apply, rather
    # than dropped as imagecodecs drops it.
    if rgb is None:
        color_transform = (components in (3, 4) and not planar
                           and not is_float)
    elif rgb:
        if components < 3:
            raise ValueError(
                f"HTJ2K encode: rgb=True needs 3 or more components, "
                f"got {components}")
        if is_float:
            raise ValueError(
                "HTJ2K encode: rgb=True is not supported for float32 "
                "input; the component transform would mix the NLT bit "
                "patterns of different components. Pass rgb=False or "
                "rgb=None")
        color_transform = True
    else:
        color_transform = False

    tile_w = tile_h = 0
    if tile is not None:
        tile_w, tile_h = (int(v) for v in tile)

    if num_decomp is not None and resolutions is not None:
        raise ValueError(
            "HTJ2K encode: pass resolutions= or num_decomp=, not both")
    decomp = -1
    if num_decomp is not None:
        decomp = int(num_decomp)
        if decomp < 0:
            raise ValueError(f"num_decomp must be >= 0, got {num_decomp}")
    elif resolutions:
        decomp = min(int(resolutions),
                     _max_decompositions(width, height, tile_w, tile_h))

    tp = 0 if tilepart is None else int(tilepart)
    block_w = block_h = 0
    if block_size is not None:
        block_w, block_h = (int(v) for v in block_size)
    if prog_order is not None:
        if str(prog_order).upper() not in _PROG_ORDERS:
            raise ValueError(
                f"prog_order must be one of {_PROG_ORDERS}, got {prog_order!r}")
        prog_b = str(prog_order).upper().encode()
    if profile is not None:
        if str(profile).upper() not in _PROFILES:
            raise ValueError(
                f"profile must be one of {_PROFILES}, got {profile!r}")
        profile_b = str(profile).upper().encode()

    p.width = <int> width
    p.height = <int> height
    p.components = <int> components
    p.bit_depth = 8 * itemsize
    p.is_signed = 1 if is_signed else 0
    p.bytes_per_sample = <int> itemsize
    p.src_planar = 1 if (planar or components == 1) else 0
    p.reversible = 1 if is_reversible else 0
    p.irrev_delta = <float> qstep
    p.qfactor = <int> qfactor
    p.num_decomp = <int> decomp
    p.color_transform = 1 if color_transform else 0
    p.nlt_binary_complement = 1 if is_float else 0
    p.tile_w = <int> tile_w
    p.tile_h = <int> tile_h
    p.tlm = 1 if tlm else 0
    p.tilepart_resolutions = 1 if tp & 1 else 0
    p.tilepart_components = 1 if tp & 2 else 0
    p.block_w = <int> block_w
    p.block_h = <int> block_h
    p.prog_order = NULL
    p.profile = NULL
    if prog_b is not None:
        p.prog_order = <const char*> prog_b
    if profile_b is not None:
        p.profile = <const char*> profile_b

    opencodecs_htj2k_clear_warnings()
    rc = opencodecs_htj2k_encode(
        <const void*> cnp.PyArray_DATA(<cnp.ndarray> arr), &p, &out_buf,
        &out_size)
    if rc != 0:
        _raise(rc, "encode")
    try:
        return PyBytes_FromStringAndSize(<const char*> out_buf,
                                         <Py_ssize_t> out_size)
    finally:
        opencodecs_htj2k_free(out_buf)


def _reductions(reduce, skipres):
    """Resolve ``reduce`` and imagecodecs' ``skipres`` to (data, recon)."""
    if skipres is None:
        r = int(reduce)
        if r < 0:
            raise ValueError(f"reduce must be >= 0, got {reduce}")
        return r, r
    try:
        rd, rr = skipres
    except TypeError:
        rd = rr = skipres
    rd, rr = int(rd), int(rr)
    if reduce and (rd, rr) != (int(reduce), int(reduce)):
        raise ValueError(
            f"reduce={reduce} and skipres={skipres!r} disagree; pass one")
    if rd < 0 or rr < 0:
        raise ValueError(f"skipres must be >= 0, got {skipres!r}")
    if rr > rd:
        raise ValueError(
            f"skipres: the reconstruction cannot skip more resolutions "
            f"({rr}) than are left unread ({rd})")
    return rd, rr


def _output_dtype(int bit_depth, bint is_signed, int nlt_type):
    """dtype for a uniform codestream, as imagecodecs.htj2k_decode picks it."""
    if bit_depth < 1 or bit_depth > 32:
        raise OpenJphError(f"HTJ2K decode: unsupported bit depth {bit_depth}")
    itemsize = 1 if bit_depth <= 8 else (2 if bit_depth <= 16 else 4)
    if nlt_type == 3:
        # NLT type 3 (binary complement, ISO/IEC 15444-2) is how
        # floating point samples travel: the decoded integers are the
        # bit patterns of IEEE floats of the same width.
        if itemsize == 1:
            raise OpenJphUnsupportedFeature(
                "HTJ2K decode: an NLT type 3 component of 8 bits or "
                "fewer has no floating point type to return")
        return np.dtype(f"f{itemsize}")
    if nlt_type != 0:
        raise OpenJphUnsupportedFeature(
            f"HTJ2K decode: nonlinearity (NLT) type {nlt_type} is not "
            f"supported; the samples would come back without it applied")
    return np.dtype(f"{'i' if is_signed else 'u'}{itemsize}")


def decode_info(data, *, int reduce=0, skipres=None,
                bint resilient=False) -> dict:
    """Read the HTJ2K headers without decoding any samples.

    ``reduce`` reports the geometry a decode at that reduction would
    produce, so a pyramid's level shapes cost only a header parse.
    ``num_decompositions`` in the result is the largest ``reduce`` the
    codestream supports. ``dtype`` is what :func:`decode` returns, or
    ``None`` when the codestream is one decode refuses.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        opencodecs_htj2k_info info
        int rd, rr, rc

    rd, rr = _reductions(reduce, skipres)
    opencodecs_htj2k_clear_warnings()

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize == 0:
        raise OpenJphError("decode_info: empty input")

    rc = opencodecs_htj2k_decode_info(
        <const void*> &src[0], srcsize, rd, rr, 1 if resilient else 0, &info)
    if rc != 0:
        _raise(rc, "decode_info")
    try:
        dtype = (_output_dtype(info.bit_depth, info.is_signed, info.nlt_type)
                 if info.uniform else None)
    except OpenJphError:
        dtype = None
    return {
        "width": info.width,
        "height": info.height,
        "components": info.components,
        "bit_depth": info.bit_depth,
        "signed": bool(info.is_signed),
        "num_decompositions": info.num_decompositions,
        "color_transform": bool(info.color_transform),
        "nlt_type": info.nlt_type,
        "uniform": bool(info.uniform),
        "dtype": dtype,
    }


def _decode_target(out, shape, dtype):
    """Validate a caller's ``out=`` for :func:`decode`; return the array."""
    nbytes = int(np.prod(shape)) * dtype.itemsize
    if isinstance(out, np.ndarray):
        if out.shape != tuple(shape):
            raise ValueError(
                f"HTJ2K decode: out.shape={out.shape} does not match the "
                f"output shape {tuple(shape)}")
        if out.dtype != dtype:
            raise ValueError(
                f"HTJ2K decode: out.dtype={out.dtype} does not match the "
                f"output dtype {dtype}")
        if not out.flags.c_contiguous or not out.flags.writeable:
            raise ValueError(
                "HTJ2K decode: out must be C-contiguous and writable")
        return out
    try:
        view = memoryview(out)
    except TypeError:
        raise TypeError(
            f"HTJ2K decode: out must be an ndarray or a writable buffer, "
            f"got {type(out).__name__}") from None
    if view.readonly or not view.c_contiguous:
        raise ValueError(
            "HTJ2K decode: out must be C-contiguous and writable")
    if view.nbytes != nbytes:
        raise ValueError(
            f"HTJ2K decode: out holds {view.nbytes} bytes; the output "
            f"needs {nbytes}")
    return np.frombuffer(view.cast("B"), dtype=dtype).reshape(shape)


def decode(data, *, bint ignore_unsupported=False, int reduce=0,
           planar=None, skipres=None, bint resilient=False,
           out=None) -> np.ndarray:
    """Decode an HTJ2K codestream to an ndarray.

    Raises :class:`OpenJphUnsupportedFeature` when OpenJPH reports that
    it skipped a marker segment it does not implement; pass
    ``ignore_unsupported=True`` to accept the image regardless.

    The dtype follows the codestream: (u)int8, (u)int16 or (u)int32 by
    precision and sign, and float16/float32 when the components carry
    an NLT type 3 marker. A codestream whose components differ in
    precision, sign or sampling raises :class:`OpenJphError`.

    Parameters
    ----------
    reduce : int, optional
        Skip this many of the finest wavelet resolutions, returning an
        image about ``2**reduce`` times smaller on each axis. OpenJPH
        never reads those subbands, so this costs proportionally less
        than a full decode rather than being a downscale of one. Raises
        when it exceeds the codestream's decomposition count, which
        :func:`decode_info` reports as ``num_decompositions``.
    planar : bool, optional
        ``True`` returns a multi-component image as ``(C, H, W)`` and
        ``False`` as ``(H, W, C)``. ``None``, as in imagecodecs,
        returns ``(H, W, C)`` when the codestream uses the component
        transform (an RGB or RGBA image) and ``(C, H, W)`` when it does
        not. A single component is always ``(H, W)``.
    skipres : int or (int, int), optional
        imagecodecs' name for the reduction: an int is the same as
        ``reduce``; a pair is (resolutions left unread, resolutions
        left out of the reconstruction).
    resilient : bool, optional
        Ask OpenJPH to tolerate damaged codestreams.
    out : numpy.ndarray or writable buffer, optional
        Decode into this array instead of a new one, as in
        imagecodecs. An ndarray must have exactly the output shape and
        dtype and be C-contiguous and writable; any other writable
        buffer must hold exactly the output's bytes and is returned
        viewed as the output array. A mismatch raises ValueError.
    """
    cdef:
        const uint8_t[::1] src
        size_t srcsize
        opencodecs_htj2k_info info
        int rd, rr, rc
        int bytes_per_sample
        int res_flag = 1 if resilient else 0
        int planar_flag
        void* out_ptr
        size_t out_nbytes
        cnp.ndarray result

    rd, rr = _reductions(reduce, skipres)
    opencodecs_htj2k_clear_warnings()

    if isinstance(data, (bytes, bytearray)):
        src = data
    else:
        src = bytes(data)
    srcsize = <size_t> src.shape[0]
    if srcsize == 0:
        raise OpenJphError("decode: empty input")

    # Sized at the SAME reduction the decode will use, so the buffer
    # matches the reconstructed extent rather than the full image.
    with nogil:
        rc = opencodecs_htj2k_decode_info(
            <const void*> &src[0], srcsize, rd, rr, res_flag, &info)
    if rc != 0:
        _raise(rc, "decode_info")
    if not info.uniform:
        raise OpenJphError(
            "HTJ2K decode: the components differ in precision, sign, "
            "nonlinearity or subsampling; not supported")

    if info.nlt_type != 0 and info.nlt_type != 3 and ignore_unsupported:
        dtype = _output_dtype(info.bit_depth, info.is_signed, 0)
    else:
        dtype = _output_dtype(info.bit_depth, info.is_signed, info.nlt_type)
    bytes_per_sample = dtype.itemsize

    if planar is None:
        # imagecodecs.htj2k_decode: interleave only what the encoder
        # marked as color (COD SGcod), keep other component sets planar.
        planar = not info.color_transform
    if info.components == 1:
        shape = (info.height, info.width)
        planar_flag = 1
    elif planar:
        shape = (info.components, info.height, info.width)
        planar_flag = 1
    else:
        shape = (info.height, info.width, info.components)
        planar_flag = 0

    # The shim writes integers; a float result is the same bits.
    int_dtype = np.dtype(f"{'i' if (info.is_signed or dtype.kind == 'f') else 'u'}"
                         f"{bytes_per_sample}")
    if out is None:
        result = np.empty(shape, dtype=int_dtype)
    else:
        result = _decode_target(out, shape, dtype)

    # OpenJPH works over raw pointers here and its warning buffer in
    # the shim is already thread_local, so nothing in the decode
    # touches Python or shared state.
    out_ptr = <void*> cnp.PyArray_DATA(result)
    out_nbytes = <size_t> result.nbytes
    with nogil:
        rc = opencodecs_htj2k_decode(
            <const void*> &src[0], srcsize,
            out_ptr, out_nbytes,
            bytes_per_sample, rd, rr, res_flag, planar_flag)
    if rc != 0:
        _raise(rc, "decode")

    _check_warnings(ignore_unsupported)

    if result.dtype != dtype:
        return result.view(dtype)
    return result
