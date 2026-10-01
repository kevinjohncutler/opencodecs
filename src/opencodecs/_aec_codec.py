"""AecCodec — Codec adapter wrapping the native _aec extension.

AEC = Adaptive Entropy Coding (CCSDS 121.0-B-2). Used by NetCDF-4 SZIP,
HDF5 SZIP filter, and most satellite/Earth-observation pipelines.
Optimized for *integer* arrays with locally predictable runs; typical
ratios are 2–4× on satellite radiance data.

Wire format
-----------
``encode`` writes the bare CCSDS 121.0-B-2 coded stream libaec
produces, the stream GRIB2 template 5.42 stores and
``imagecodecs.aec_encode`` writes. The standard defines no container:
the coding parameters travel out of band, so ``decode`` needs the ones
the stream was written with. The defaults are imagecodecs' (and the
libaec ``aec`` tool's): ``block_size=8``, ``rsi=2``, preprocessing on.
NetCDF-4 and HDF5 szip data commonly use ``block_size=32, rsi=128``,
which also compresses better; pass those to both calls to use them.

Parameters take opencodecs' names or imagecodecs' (``bitspersample``,
``blocksize``, ``flags``). ``bits_per_sample`` defaults to the array's
item size in bits, and to 8 for bytes. A signed 16 or 32-bit array
sets the signed-sample flag, on encode from the input and on decode
from ``dtype=`` or ``out=``, as imagecodecs does; an int8 array is
coded as unsigned bytes, also as imagecodecs writes it. Pass
``signed=`` to override either.

``flags`` sets the AEC_DATA_* bits wholesale (default preprocessing
only), and the boolean options (``signed``, ``msb``, ``preprocess``,
``three_byte``, ``restricted``, ``pad_rsi``) then set or clear one bit
each. When neither ``flags`` nor ``msb`` is given, the byte order bit
(``AEC_DATA_MSB``) follows the array's byte order, so a big-endian
array is coded by its values, and decoding with ``dtype=`` returns
values; imagecodecs codes such an array's bytes as little-endian
samples. An explicit ``flags`` or ``msb`` is kept as given: libaec
then reads the array's bytes in that order, and decoding with
``dtype=`` or ``out=`` returns the decoded bytes as they are, both as
in imagecodecs. Encoding raises ``ValueError`` when a sample read that
way does not fit ``bits_per_sample``, for bytes input too.

libaec takes a signed sample narrower than its item as a
``bits_per_sample``-bit two's complement number and sign-extends it
on decode. A signed array holds its values sign-extended to the full
item instead: they must lie in the signed range of ``bits_per_sample``
bits, and the negative ones are masked to that many bits before
coding (imagecodecs passes them as they are, and its stream then
decodes to other values). The stream is the one imagecodecs writes
for the masked array. Bytes input is taken as libaec takes it, the
low ``bits_per_sample`` bits of each sample, so masked samples are
accepted there.

``pad_rsi`` (``AEC_PAD_RSI``) pads each reference sample interval to
a byte boundary only in a libaec built with ``ENABLE_RSI_PADDING``; a
libaec built without it accepts the flag and writes the stream
without the padding, which no decoder then reads with the flag set.
Encoding with ``pad_rsi`` therefore checks once whether the linked
libaec pads and raises ``ValueError`` if it does not. Decoding a
padded stream does not depend on that build option.

Blobs from releases up to 0.4.0 carry a private 16-byte preamble, and
``decode`` still reads them. The preamble has no magic, so a blob is
read as an old one when every field holds a value the old encoder
could write and agrees with each coding parameter the caller passes
(and with the byte count ``dtype`` and ``shape`` give), and its payload
decodes. If the same bytes also
decode as a bare stream with the caller's parameters, the reading
whose values encode back to exactly the input wins, the bare one
first; when neither does, ``decode`` raises rather than guess.

Example::

    import numpy as np
    import opencodecs as oc

    arr = np.random.randint(0, 4096, size=10000, dtype=np.uint16)
    blob = oc.write(None, arr, format="aec", bits_per_sample=12)
    back = oc.read(blob, format="aec", bits_per_sample=12,
                   dtype=np.uint16, shape=arr.shape)
    assert np.array_equal(back, arr)
"""

from __future__ import annotations

import math
import sys
from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import byte_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _aec_encode, _aec_decode_raw, _aec_decode_framed,
    _aec_read_framed_header, _aec_sample_bytes, _aec_check_signature,
    _AecError, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._aec",
    "encode", "decode_raw", "decode_framed", "read_framed_header",
    "sample_bytes", "check_signature", "AecError",
)

# libaec.h AEC_DATA_* / AEC_* flag values.
_SIGNED, _3BYTE, _MSB, _PREPROCESS = 1, 2, 4, 8
_RESTRICTED, _PAD_RSI, _NOT_ENFORCE = 16, 32, 64

# imagecodecs' defaults, which are also the libaec ``aec`` tool's.
_DEFAULT_BLOCK_SIZE = 8
_DEFAULT_RSI = 2


# ndarray -> bits_per_sample inference. AEC's "bits_per_sample" is the
# *meaningful* bit-width, not the storage word size; the user can
# override if their data fits in fewer bits than the dtype allows.
_DTYPE_BITS = {
    np.dtype(np.uint8):  (8,  False),
    # imagecodecs codes int8 without the signed flag; do the same, so
    # int8 streams are the same bytes and read the same both ways.
    np.dtype(np.int8):   (8,  False),
    np.dtype(np.uint16): (16, False),
    np.dtype(np.int16):  (16, True),
    np.dtype(np.uint32): (32, False),
    np.dtype(np.int32):  (32, True),
}


def _alias(label, ours, theirs):
    if ours is not None and theirs is not None and int(ours) != int(theirs):
        raise ValueError(
            f"aec: {label} given twice with different values "
            f"({ours} and {theirs})")
    value = ours if ours is not None else theirs
    return None if value is None else int(value)


def _combine_flags(flags, *, signed, msb, preprocess, three_byte,
                   restricted, pad_rsi):
    f = _PREPROCESS if flags is None else int(flags)
    for bit, value in ((_SIGNED, signed), (_MSB, msb),
                       (_PREPROCESS, preprocess), (_3BYTE, three_byte),
                       (_RESTRICTED, restricted), (_PAD_RSI, pad_rsi)):
        if value is not None:
            f = (f | bit) if value else (f & ~bit)
    return f


def _reject_unknown(opts):
    if opts:
        raise TypeError(f"aec: unexpected keyword argument(s) {sorted(opts)}")


_pad_rsi_encodes = None


def _check_pad_rsi_encodes():
    """Raise unless the linked libaec pads the RSI on encode.

    libaec's encoder pads only when built with ENABLE_RSI_PADDING; a
    build without it accepts AEC_PAD_RSI and ignores it when encoding,
    and the unpadded stream it writes does not decode with the flag
    set. Compare one input coded with and without the flag: with
    padding, 100 one-byte samples in RSIs of 16 come out 2 bytes
    longer.
    """
    global _pad_rsi_encodes
    if _pad_rsi_encodes is None:
        probe = bytes(i * 37 % 251 for i in range(100))
        plain = _aec_encode(probe, bits_per_sample=8, block_size=8, rsi=2,
                            flags=_PREPROCESS)
        padded = _aec_encode(probe, bits_per_sample=8, block_size=8, rsi=2,
                             flags=_PREPROCESS | _PAD_RSI)
        _pad_rsi_encodes = len(padded) > len(plain)
    if not _pad_rsi_encodes:
        raise ValueError(
            "aec encode: this libaec was built without ENABLE_RSI_PADDING "
            "and ignores AEC_PAD_RSI when encoding, so the stream would not "
            "decode with pad_rsi set; encode without pad_rsi")


def _fit_samples(buf, bps, f, hint, *, values):
    """Return the samples of ``buf`` in the form libaec codes.

    libaec reads each sample as ``sample_bytes`` bytes in the flags'
    byte order and codes its low ``bits_per_sample`` bits; a signed
    sample must be a ``bits_per_sample``-bit two's complement number,
    which libaec sign-extends (encode.c, preprocess_signed). Unsigned
    samples above ``2**bits_per_sample - 1`` would lose their high
    bits, so they raise.

    With ``values`` (an array), signed samples are the array's values,
    sign-extended to the full item: they must lie in the signed range
    of ``bits_per_sample`` bits, and the negative ones are masked to
    that many bits, in a copy, so they code the values they hold. A
    value above that range raises rather than being read as a negative
    number. Without ``values`` (bytes), samples are libaec's own: the
    low ``bits_per_sample`` bits are taken as two's complement, and
    sign-extended negatives are masked the same way.
    """
    sb = _aec_sample_bytes(bps, f)
    raw = np.frombuffer(buf, np.uint8)
    if bps >= 8 * sb or not raw.size or raw.size % sb:
        return buf          # every sample fits, or the encoder raises
    signed = bool(f & _SIGNED)
    if sb == 3:
        b = raw.reshape(-1, 3).astype(np.int64)
        if f & _MSB:
            b = b[:, ::-1]
        samples = b[:, 0] | b[:, 1] << 8 | b[:, 2] << 16
        if signed:
            samples = samples - ((samples >> 23) << 24)
    else:
        samples = raw.view(np.dtype(
            f"{'>' if f & _MSB else '<'}{'i' if signed else 'u'}{sb}"))
    mask = (1 << bps) - 1
    lo = -(1 << (bps - 1)) if signed else 0
    hi = (1 << (bps - 1)) - 1 if signed and values else mask
    smin, smax = int(samples.min()), int(samples.max())
    if smin < lo or smax > hi:
        if signed and values and smin >= 0 and smax <= mask:
            hint += "; pass signed=False to code unsigned samples"
        raise ValueError(
            f"aec encode: {'signed ' if signed else ''}samples outside "
            f"[{lo}, {hi}] do not fit in bits_per_sample={bps}{hint}")
    if not signed or smin >= 0:
        return buf
    masked = samples & mask
    if sb != 3:
        return masked.astype(samples.dtype).view(np.uint8)
    out = np.empty((masked.size, 3), np.uint8)
    for i in range(3):
        out[:, 2 - i if f & _MSB else i] = (masked >> (8 * i)) & 0xFF
    return out.reshape(-1)


def _sample_dtype(dtype):
    dt = np.dtype(dtype)
    if dt.kind not in "iu" or dt.itemsize not in (1, 2, 4):
        raise ValueError(
            f"aec: dtype must be an 8, 16 or 32-bit integer, got {dt}")
    return dt


class AecCodec(Codec):
    """Native libaec — CCSDS 121.0-B-2 adaptive entropy coding."""

    name = "aec"
    aliases = ("ccsds", "szip")
    file_extensions = (".aec",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (
        np.uint8, np.int8,
        np.uint16, np.int16,
        np.uint32, np.int32,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return _aec_check_signature(head)

    def encode(self, data: Any, *, dest=None,
               bits_per_sample: int | None = None,
               block_size: int | None = None,
               rsi: int | None = None,
               flags: int | None = None,
               signed: bool | None = None,
               msb: bool | None = None,
               preprocess: bool | None = None,
               three_byte: bool | None = None,
               restricted: bool | None = None,
               pad_rsi: bool | None = None,
               bitspersample: int | None = None,
               blocksize: int | None = None,
               **opts) -> bytes | None:
        """Return the bare libaec stream of ``data``.

        ``flags`` sets the AEC_DATA_* bits wholesale (default
        preprocessing only); the boolean options then set or clear one
        bit each. Without ``flags`` or ``msb`` an ndarray is coded by
        value in its own byte order; with either, libaec reads its
        bytes in the order the flags give (see the module docstring).
        """
        _reject_unknown(opts)
        bps = _alias("bits_per_sample", bits_per_sample, bitspersample)
        bsz = _alias("block_size", block_size, blocksize)
        if isinstance(data, np.ndarray):
            dtype = data.dtype.newbyteorder("=")
            if dtype not in _DTYPE_BITS:
                raise ValueError(
                    f"aec encode: unsupported ndarray dtype {data.dtype!r}; "
                    f"expected uint8/int8/uint16/int16/uint32/int32"
                )
            inferred_bps, inferred_signed = _DTYPE_BITS[dtype]
            if bps is None:
                bps = inferred_bps
            if signed is None and inferred_signed:
                signed = True
            if msb is None and flags is None:
                # libaec reads samples least significant byte first
                # unless told otherwise; tell it the array's order, so
                # the stream codes the array's values. An explicit
                # flags or msb is the caller's byte order and is kept:
                # libaec then reads the array's bytes in that order,
                # as it does in imagecodecs.
                msb = _is_big_endian(data.dtype)
            f = _combine_flags(flags, signed=signed, msb=msb,
                               preprocess=preprocess, three_byte=three_byte,
                               restricted=restricted, pad_rsi=pad_rsi)
            if _aec_sample_bytes(bps, f) != data.dtype.itemsize:
                raise ValueError(
                    f"aec encode: bits_per_sample={bps} stores each sample "
                    f"in {_aec_sample_bytes(bps, f)} bytes, but the array "
                    f"has {data.dtype.itemsize}-byte items")
            # Pass the ndarray's bytes straight to the Cython encoder
            # via the buffer protocol, with no .tobytes() copy; a 0-d
            # array is one sample, as in imagecodecs.
            buf = (data if data.flags["C_CONTIGUOUS"]
                   else np.ascontiguousarray(data)).reshape(-1).view(np.uint8)
            hint = ""
            if data.dtype.itemsize > 1 and (
                    bool(f & _MSB) != _is_big_endian(data.dtype)):
                hint = ("; the flags read the array's bytes in the other "
                        "byte order (AEC_DATA_MSB)")
            elif not f & _SIGNED and data.dtype.kind == "i":
                hint = "; pass signed=True to code signed samples"
            buf = _fit_samples(buf, bps, f, hint, values=True)
        else:
            if bps is None:
                bps = 8
            f = _combine_flags(flags, signed=signed, msb=msb,
                               preprocess=preprocess, three_byte=three_byte,
                               restricted=restricted, pad_rsi=pad_rsi)
            buf = data if isinstance(data, (bytes, bytearray, memoryview)) else bytes(data)
            if isinstance(buf, memoryview) and not buf.c_contiguous:
                buf = buf.tobytes()
            buf = _fit_samples(buf, bps, f, "", values=False)

        if f & _PAD_RSI:
            _check_pad_rsi_encodes()
        out = _aec_encode(
            buf,
            bits_per_sample=bps,
            block_size=_DEFAULT_BLOCK_SIZE if bsz is None else bsz,
            rsi=_DEFAULT_RSI if rsi is None else int(rsi),
            flags=f,
        )
        return _write_dest(out, dest)

    def decode(self, src: Any, *, out=None,
               bits_per_sample: int | None = None,
               block_size: int | None = None,
               rsi: int | None = None,
               flags: int | None = None,
               signed: bool | None = None,
               msb: bool | None = None,
               preprocess: bool | None = None,
               three_byte: bool | None = None,
               restricted: bool | None = None,
               pad_rsi: bool | None = None,
               bitspersample: int | None = None,
               blocksize: int | None = None,
               dtype=None, shape=None,
               **opts):
        """Decode a libaec stream.

        Returns bytes; with ``dtype=`` an ndarray (of ``shape`` if
        given, which needs ``dtype``); with a buffer ``out=`` a
        memoryview of the bytes written into it; with an int ``out=``
        bytes decoded into a buffer of that capacity.

        Without ``out`` or ``shape`` the output size is not known, and
        the result can carry libaec's padding past the data (part of a
        block, or zero blocks to the end of a 64-block segment), as
        ``imagecodecs.aec_decode`` does.
        """
        _reject_unknown(opts)
        buf = _read_src(src)
        bps = _alias("bits_per_sample", bits_per_sample, bitspersample)
        bsz = _alias("block_size", block_size, blocksize)
        if shape is not None and dtype is None:
            raise ValueError("aec decode: shape= needs dtype=")
        sample_dt = None
        if dtype is not None:
            sample_dt = _sample_dtype(dtype)
        elif isinstance(out, np.ndarray) and out.dtype.kind in "iu":
            sample_dt = out.dtype
        target_shape = None if shape is None else (
            (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape))
        size = -1
        if dtype is not None and target_shape is not None:
            size = math.prod(target_shape) * sample_dt.itemsize
        as_bytes = isinstance(out, int) and not isinstance(out, bool)
        if as_bytes:
            dest = memoryview(bytearray(byte_output(out)))
        else:
            dest = byte_output(out)

        # A release-0.4.0 blob is a candidate when its preamble agrees
        # with everything the caller says about the stream.
        legacy = None
        hdr = _aec_read_framed_header(buf)
        if hdr is not None and _legacy_agrees(
                hdr, size, bits_per_sample=bps, block_size=bsz, rsi=rsi,
                flags=flags, signed=signed, msb=msb, preprocess=preprocess,
                three_byte=three_byte, restricted=restricted,
                pad_rsi=pad_rsi):
            if dest is not None and hdr[0] > len(dest):
                raise ValueError(
                    f"aec decode: out= holds {len(dest)} bytes but this "
                    f"blob's header declares {hdr[0]}")
            try:
                legacy = bytes(_aec_decode_framed(buf))
            except Exception:
                legacy = None  # not a framed blob: read it bare

        # With an explicit flags or msb the caller sets the byte order
        # libaec writes samples in, and the decoded bytes are returned
        # as they are (viewed as ``dtype``), as imagecodecs writes them
        # into its ``out``. Otherwise the order follows the output, so
        # the result holds the stream's values.
        by_value = msb is None and flags is None
        if sample_dt is not None:
            if bps is None:
                bps = 8 * sample_dt.itemsize
            if signed is None and sample_dt.kind == "i" and sample_dt.itemsize > 1:
                signed = True
            if by_value:
                # Samples land in the caller's array (``out``) or in a
                # native-order buffer (``dtype``): write them in that
                # byte order.
                msb = _is_big_endian(
                    sample_dt if dtype is None else sample_dt.newbyteorder("="))
        if bps is None:
            bps = 8
        f = _combine_flags(flags, signed=signed, msb=msb,
                           preprocess=preprocess, three_byte=three_byte,
                           restricted=restricted, pad_rsi=pad_rsi)
        params_fit = (sample_dt is None
                      or _aec_sample_bytes(bps, f) == sample_dt.itemsize)
        coding = dict(
            bits_per_sample=bps,
            block_size=_DEFAULT_BLOCK_SIZE if bsz is None else bsz,
            rsi=_DEFAULT_RSI if rsi is None else int(rsi),
            flags=f)

        if legacy is None:
            if not params_fit:
                raise ValueError(
                    f"aec decode: bits_per_sample={bps} stores each sample "
                    f"in {_aec_sample_bytes(bps, f)} bytes, but {sample_dt} "
                    f"has {sample_dt.itemsize}")
            result = _aec_decode_raw(
                buf, out=dest, size=size if dest is None else -1, **coding)
            return self._finish(result, dest, as_bytes, dtype, sample_dt,
                                target_shape, by_value)

        # The bytes read as an old blob. See whether they also read as
        # a bare stream with these parameters, and if so keep the
        # reading that encodes back to exactly the input.
        bare = None
        if params_fit:
            try:
                bare = bytes(_aec_decode_raw(
                    buf, out=None if dest is None else bytearray(len(dest)),
                    size=size if dest is None else -1, **coding))
            except Exception:
                bare = None
        chosen = legacy
        if bare is not None:
            if _encodes_to(bare, coding, buf):
                chosen = bare
            elif not _encodes_to(
                    legacy, dict(bits_per_sample=hdr[1], block_size=hdr[2],
                                 rsi=hdr[3], flags=hdr[4]),
                    memoryview(buf)[16:]):
                raise _AecError(
                    "aec decode: these bytes read both as a blob from "
                    "opencodecs 0.4.0 and as a bare stream with the given "
                    "parameters, and neither reading encodes back to them")
        if dest is not None:
            dest[:len(chosen)] = chosen
            result = dest[:len(chosen)]
        else:
            result = chosen
        return self._finish(result, dest, as_bytes, dtype, sample_dt,
                            target_shape, by_value)

    @staticmethod
    def _finish(result, dest, as_bytes, dtype, sample_dt, target_shape,
                by_value):
        if as_bytes:
            return bytes(result)
        if dest is not None or dtype is None:
            return result
        if len(result) % sample_dt.itemsize:
            raise ValueError(
                f"aec decode: {len(result)} decoded bytes are not a whole "
                f"number of {sample_dt} items")
        if by_value:
            # Decoded in native order: convert values to the order asked.
            arr = np.frombuffer(result, dtype=sample_dt.newbyteorder("="))
            arr = arr.astype(sample_dt) if not sample_dt.isnative else arr.copy()
        else:
            # Decoded in the order the caller's flags give: keep the bytes.
            arr = np.frombuffer(result, dtype=sample_dt).copy()
        if target_shape is not None:
            arr = arr.reshape(target_shape)
        return arr


def _legacy_agrees(hdr, size, *, bits_per_sample, block_size, rsi, flags,
                   **bits):
    """True if a 0.4.0 preamble agrees with every parameter the caller
    passed (None means not passed) and with the expected byte count."""
    hsize, hbps, hblock, hrsi, hflags = hdr
    if size >= 0 and hsize != size:
        return False
    for given, have in ((bits_per_sample, hbps), (block_size, hblock),
                        (rsi, hrsi)):
        if given is not None and int(given) != have:
            return False
    if flags is not None and _combine_flags(flags, **bits) != hflags:
        return False
    for name, bit in (("signed", _SIGNED), ("msb", _MSB),
                      ("preprocess", _PREPROCESS), ("three_byte", _3BYTE),
                      ("restricted", _RESTRICTED), ("pad_rsi", _PAD_RSI)):
        value = bits[name]
        if value is not None and bool(value) != bool(hflags & bit):
            return False
    return True


def _encodes_to(samples, coding, stream) -> bool:
    try:
        return bytes(_aec_encode(samples, **coding)) == bytes(stream)
    except Exception:
        return False


def _is_big_endian(dt) -> bool:
    if dt.itemsize == 1:
        return False
    order = dt.byteorder
    if order == "=":
        order = "<" if sys.byteorder == "little" else ">"
    return order == ">"


__all__ = ["AecCodec"]
