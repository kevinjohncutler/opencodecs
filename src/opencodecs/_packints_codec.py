"""PackintsCodec — bit-pack arbitrary-width unsigned integers.

Camera raws (10/12/14-bit ADCs), scientific imaging cameras, and
embedded data formats commonly emit integers that don't fit cleanly
into 8/16/32-bit slots — e.g. 12-bit pixels packed 8 to a 12-byte
block. ``packints`` is the filter that converts between the packed
on-wire form and a standard numpy dtype.

The parameters are imagecodecs' ``packints_encode`` / ``packints_decode``
ones, and ``bitorder`` selects one of three published layouts:

* ``None`` (default): the TIFF bitstream, most significant bit first
  (TIFF 6.0 FillOrder 1: the first sample in the high-order bits of the
  first byte). ``runlen=N`` starts every run of ``N`` samples on a byte
  boundary, which is how TIFF stores rows of sub-byte samples ("leaving
  no unused bits except at the end of a row"); ``runlen=0`` packs one
  continuous stream.
* ``"<"``: GenICam PFNC least-significant-bit-first packing (``Mono12p``
  and its siblings): sample bits fill each byte from bit 0 upward.
* ``">"``: GigE Vision paired-pixel packing, 10 or 12 bits
  (``Mono10Packed``, ``Mono12Packed``): two samples in three bytes, the
  high bits of each in its own byte and both low parts in the middle one.

Whole-byte widths (8, 16, 32, 64) in the default layout are the same
MSB-first stream, so they are big-endian bytes on every machine. This
differs from imagecodecs, which copies them in memory order (little
endian on x86 and arm64 unless the dtype says otherwise), and agrees
with imagecodecs' own definition of the default as an "MSB-first
continuous bitstream" and with what both libraries write for 24-bit and
other widths. TIFF itself never routes whole-byte samples through this
codec: they are stored in the file's byte order.

Arbitrary widths use a native direct-output bit reader when available,
with a NumPy fallback. One-bit input uses NumPy unpackbits.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from ._filter_args import reject_unknown


# Samples converted to bits per step when packing: bounds the temporary
# bit matrix (one byte per bit) at a few megabytes. A multiple of 8, so
# every block of a continuous stream ends on a byte boundary.
_PACK_BLOCK = 1 << 18


def _check_layout(bitspersample: int, runlen: int, bitorder) -> None:
    if not 1 <= bitspersample <= 64:
        raise ValueError(
            f"packints: bitspersample must be in 1..64, got {bitspersample}")
    if runlen < 0:
        raise ValueError(f"packints: runlen must be >= 0, got {runlen}")
    if bitorder is None:
        return
    if bitorder not in ("<", ">"):
        raise ValueError(
            f"packints: bitorder must be None, '<' or '>', got {bitorder!r}")
    if runlen:
        raise ValueError(
            "packints: runlen pads TIFF rows and is not defined for bitorder "
            f"{bitorder!r}")
    if bitorder == ">" and bitspersample not in (10, 12):
        raise ValueError(
            "packints: bitorder '>' (GigE Vision paired pixels) needs "
            f"bitspersample 10 or 12, got {bitspersample}")


def _pack_rows(values: np.ndarray, bitspersample: int, lsb_first: bool) -> bytes:
    """Pack each row of ``values`` (2-D) as one bitstream padded to a byte."""
    bps = bitspersample
    shifts = (np.arange(bps, dtype=np.uint64) if lsb_first
              else np.arange(bps - 1, -1, -1, dtype=np.uint64))
    rows, n = values.shape
    parts = []
    step = max(1, _PACK_BLOCK // max(n, 1))
    for r in range(0, rows, step):
        block = values[r:r + step].astype(np.uint64)
        bits = ((block[..., None] >> shifts) & np.uint64(1)).astype(np.uint8)
        parts.append(np.packbits(bits.reshape(len(block), n * bps), axis=-1,
                                 bitorder="little" if lsb_first else "big"))
    return b"".join(part.tobytes() for part in parts)


def _bitpack(values: np.ndarray, bitspersample: int, runlen: int = 0,
             bitorder=None) -> bytes:
    """Pack unsigned integers in the layout ``bitorder`` and ``runlen`` name."""
    _check_layout(bitspersample, runlen, bitorder)
    flat = values.reshape(-1)
    if bitspersample < flat.dtype.itemsize * 8 and flat.size:
        top = int(flat.max())
        if top >> bitspersample:
            # imagecodecs masks such samples to their low bits, which
            # loses data without a word.
            raise ValueError(
                f"packints: sample value {top} does not fit in "
                f"{bitspersample} bits")
    if bitorder == ">":
        return _pack_paired(flat, bitspersample)
    if runlen and flat.size % runlen:
        # imagecodecs drops a trailing partial run without a word.
        raise ValueError(
            f"packints: {flat.size} samples are not a whole number of runs "
            f"of {runlen}")
    if bitorder is None and bitspersample in (8, 16, 32, 64):
        # Whole bytes, most significant first: the big-endian integers.
        return flat.astype(f">u{bitspersample // 8}", copy=False).tobytes()
    lsb_first = bitorder == "<"
    if runlen and (runlen * bitspersample) % 8:
        return _pack_rows(flat.reshape(-1, runlen), bitspersample, lsb_first)
    # One continuous stream (a row padding that adds nothing is the same).
    # Blocks of _PACK_BLOCK samples end on byte boundaries.
    parts = [_pack_rows(flat[i:i + _PACK_BLOCK].reshape(1, -1),
                        bitspersample, lsb_first)
             for i in range(0, flat.size, _PACK_BLOCK)]
    return b"".join(parts)


def _pack_paired(flat: np.ndarray, bitspersample: int) -> bytes:
    """GigE Vision Mono10Packed / Mono12Packed: two samples in three bytes.

    12 bits: byte 0 = p0[11:4], byte 1 = p1[3:0] << 4 | p0[3:0], byte 2 =
    p1[11:4]. 10 bits: byte 0 = p0[9:2], byte 1 = p1[1:0] << 4 | p0[1:0],
    byte 2 = p1[9:2].
    """
    if flat.size % 2:
        raise ValueError(
            "packints: bitorder '>' packs samples in pairs and needs an even "
            f"count, got {flat.size}")
    low = bitspersample - 8
    mask = (1 << bitspersample) - 1
    pairs = (flat.astype(np.uint32) & np.uint32(mask)).reshape(-1, 2)
    out = np.empty((len(pairs), 3), np.uint8)
    lowmask = np.uint32((1 << low) - 1)
    out[:, 0] = pairs[:, 0] >> np.uint32(low)
    out[:, 1] = ((pairs[:, 1] & lowmask) << np.uint32(4)) | (pairs[:, 0] & lowmask)
    out[:, 2] = pairs[:, 1] >> np.uint32(low)
    return out.tobytes()


def _unpack_paired(buf: bytes, dtype: np.dtype, bitspersample: int,
                   n_elements: int) -> np.ndarray:
    """Inverse of :func:`_pack_paired`."""
    if n_elements % 2:
        raise ValueError(
            "packints: bitorder '>' holds samples in pairs and needs an even "
            f"count, got {n_elements}")
    need = n_elements // 2 * 3
    if len(buf) < need:
        raise ValueError(
            f"packints decode: input holds {len(buf)} bytes, need {need} "
            f"for {n_elements} paired samples")
    raw = np.frombuffer(buf, np.uint8, count=need).reshape(-1, 3).astype(np.uint32)
    low = bitspersample - 8
    lowmask = np.uint32((1 << low) - 1)
    out = np.empty((len(raw), 2), np.uint32)
    out[:, 0] = (raw[:, 0] << np.uint32(low)) | (raw[:, 1] & lowmask)
    out[:, 1] = (raw[:, 2] << np.uint32(low)) | ((raw[:, 1] >> np.uint32(4)) & lowmask)
    return out.reshape(-1).astype(dtype)


def _unpack_stream(buf: bytes, dtype: np.dtype, bitspersample: int,
                   n_elements: int, lsb_first: bool) -> np.ndarray:
    """A continuous stream to samples, ``_PACK_BLOCK`` samples at a time.

    Each block starts on a byte boundary (``_PACK_BLOCK`` is a multiple
    of 8). Up to 57 bits a sample lies inside the 8 bytes starting at its
    first byte, so it is one unaligned 64-bit load, read little endian
    for LSB-first and big endian for MSB-first, then a shift and a mask:
    the temporaries are a few words per sample of one block. Wider
    samples fold a block's bits with a dot product. Expanding the whole
    stream to one byte per bit at once, as this used to, took 1.9 GB of
    temporaries for one 4096 x 4096 12-bit frame.
    """
    bps = bitspersample
    need_bits = n_elements * bps
    if len(buf) * 8 < need_bits:
        raise ValueError(
            f"packints decode: input holds {len(buf) * 8} bits, need "
            f"{need_bits} for {n_elements} samples of {bps} bits")
    raw = np.frombuffer(buf, dtype=np.uint8, count=(need_bits + 7) // 8)
    result = np.empty(n_elements, dtype)
    mask = np.uint64((1 << bps) - 1)
    word = np.dtype("<u8" if lsb_first else ">u8")
    weights = (np.uint64(1) << (np.arange(bps, dtype=np.uint64) if lsb_first
                                else np.arange(bps - 1, -1, -1, dtype=np.uint64)))
    for start in range(0, n_elements, _PACK_BLOCK):
        count = min(_PACK_BLOCK, n_elements - start)
        first = start * bps // 8                     # exact: start % 8 == 0
        nbytes = (count * bps + 7) // 8
        if bps <= 57:
            chunk = np.zeros(nbytes + 8, np.uint8)  # 8 zero bytes past the end
            chunk[:nbytes] = raw[first:first + nbytes]
            words = np.ndarray((nbytes + 1,), word, chunk, strides=(1,))
            pos = np.arange(count, dtype=np.int64) * bps
            shift = (pos & 7).astype(np.uint64)
            if not lsb_first:
                shift = np.uint64(64 - bps) - shift
            vals = (words[pos >> 3].astype(np.uint64) >> shift) & mask
        else:
            bits = np.unpackbits(raw[first:first + nbytes],
                                 bitorder="little" if lsb_first else "big")
            vals = bits[:count * bps].reshape(count, bps).astype(np.uint64) @ weights
        result[start:start + count] = vals.astype(dtype, copy=False)
    return result


def _unpack_lsb(buf: bytes, dtype: np.dtype, bitspersample: int,
                n_elements: int) -> np.ndarray:
    """GenICam LSB-first continuous stream to samples."""
    return _unpack_stream(buf, dtype, bitspersample, n_elements, True)


def _bitunpack(buf: bytes, dtype: np.dtype, bitspersample: int,
               n_elements: int) -> np.ndarray:
    """Inverse of :func:`_bitpack`."""
    if bitspersample <= 0 or bitspersample > 64:
        raise ValueError(
            f"packints: bitspersample must be in 1..64, got {bitspersample}")
    if bitspersample == 1:
        if n_elements > len(buf) * 8:
            raise ValueError("packints input is truncated")
        return np.unpackbits(np.frombuffer(buf, dtype=np.uint8), count=n_elements).astype(dtype, copy=False)
    if bitspersample % 8 == 0:
        target_dt = {
            8: ">u1", 16: ">u2", 32: ">u4", 64: ">u8",
        }.get(bitspersample)
        if target_dt is not None:
            arr = np.frombuffer(buf, dtype=target_dt, count=n_elements)
            return arr.astype(dtype, copy=False)
    # General path, for widths that do not land on a byte boundary when
    # the native bit reader is absent or cannot write the output. This
    # used to be a Python loop doing one numpy scalar store per element,
    # which cost 81 seconds for a million 4-bit samples against
    # imagecodecs' 0.02 ms.
    return _unpack_stream(buf, dtype, bitspersample, n_elements, False)


def _sample_count(nbytes: int, bitspersample: int, runlen: int, bitorder) -> int:
    """Samples a stream of ``nbytes`` holds when the caller gives no count.

    As imagecodecs: whole runs only with ``runlen``, pairs for ``'>'``,
    otherwise every whole sample the bits hold (trailing pad bits of a
    final byte cannot tell a sample from padding, so pass a shape when the
    count matters).
    """
    if bitorder == ">":
        return nbytes // 3 * 2
    if runlen:
        return nbytes // _run_bytes(runlen, bitspersample) * runlen
    return nbytes * 8 // bitspersample


def _run_bytes(runlen: int, bitspersample: int) -> int:
    return (runlen * bitspersample + 7) // 8


class PackintsCodec(Codec):
    """Bit-pack / unpack arbitrary-width unsigned integers.

    ``bitspersample`` (1 to 64), ``runlen`` and ``bitorder`` mean what
    they mean in imagecodecs; see the module docstring for the layouts.
    """

    name = "packints"
    aliases = ()
    file_extensions = ()

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (
        np.uint8, np.uint16, np.uint32, np.uint64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return False  # opaque filter

    def encode(
        self,
        data: Any,
        *,
        dest=None,
        bitspersample: int,
        runlen: int = 0,
        bitorder=None,
        **opts,
    ) -> bytes | None:
        reject_unknown("packints encode", opts)
        arr = np.ascontiguousarray(data)
        if arr.dtype.kind != "u":
            raise ValueError(
                f"packints encode: requires unsigned int dtype, got {arr.dtype}")
        out = _bitpack(arr, int(bitspersample), int(runlen or 0), bitorder)
        return _write_dest(out, dest)

    def decode(
        self,
        src: Any,
        *,
        dtype,
        bitspersample: int,
        runlen: int = 0,
        bitorder=None,
        n_elements: int | None = None,
        shape=None,
        out=None,
        **opts,
    ) -> np.ndarray:
        reject_unknown("packints decode", opts)
        if dtype is None:
            raise ValueError("packints decode: dtype= is required")
        bitspersample = int(bitspersample)
        runlen = int(runlen or 0)
        _check_layout(bitspersample, runlen, bitorder)
        kind = np.dtype(dtype).kind
        if not (kind in "iu" and np.dtype(dtype).itemsize * 8 >= bitspersample
                or kind == "b" and bitspersample == 1):
            # A float dtype took the integer values and a narrow integer one
            # kept their low bits, without a word. imagecodecs raises for
            # both; it takes bool for one-bit samples, as tifffile asks.
            raise ValueError(
                f"packints decode: dtype {np.dtype(dtype)} is not an integer "
                f"type of at least {bitspersample} bits")
        buf = _read_src(src)
        if n_elements is None:
            if shape is None:
                n_elements = _sample_count(len(buf), bitspersample, runlen, bitorder)
            else:
                n_elements = math.prod(shape)
        n_elements = int(n_elements)
        target_dtype = np.dtype(dtype)
        target_shape = tuple(shape) if shape is not None else (n_elements,)
        if n_elements < 0 or math.prod(target_shape) != n_elements:
            raise ValueError("packints shape does not match sample count")
        if runlen and bitorder is None and n_elements % runlen:
            # Encode raises for this too; whole-byte rows used to be read
            # as one stream without a word.
            raise ValueError(
                f"packints decode: {n_elements} samples are not a whole "
                f"number of runs of {runlen}")
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError("packints out must be an ndarray")
            if out.dtype != target_dtype or out.shape != target_shape:
                raise ValueError("packints out shape/dtype mismatch")
            if not out.flags.writeable:
                raise ValueError("packints out must be writable")
        if bitorder is not None:
            unpack = _unpack_paired if bitorder == ">" else _unpack_lsb
            result = unpack(buf, target_dtype, bitspersample, n_elements)
        elif runlen and (runlen * bitspersample) % 8:
            return self._decode_runs(buf, target_dtype, target_shape,
                                     bitspersample, runlen, n_elements, out)
        else:
            return self._decode_stream(buf, target_dtype, target_shape,
                                       bitspersample, n_elements, out)
        result = result.reshape(target_shape)
        if out is not None:
            np.copyto(out, result)
            return out
        return result

    def _decode_runs(self, buf, target_dtype, target_shape, bitspersample,
                     runlen, n_elements, out):
        """TIFF rows: each run of ``runlen`` samples starts on a byte."""
        if n_elements % runlen:
            raise ValueError(
                f"packints decode: {n_elements} samples are not a whole "
                f"number of runs of {runlen}")
        rows = n_elements // runlen
        row_bytes = _run_bytes(runlen, bitspersample)
        if len(buf) < rows * row_bytes:
            raise ValueError(
                f"packints decode: input holds {len(buf)} bytes, need "
                f"{rows * row_bytes} for {rows} runs of {runlen} samples of "
                f"{bitspersample} bits")
        result = (np.empty(target_shape, dtype=target_dtype)
                  if out is None or not out.flags.c_contiguous else out)
        flat = result.reshape(rows, runlen)
        for r in range(rows):
            chunk = buf[r * row_bytes:(r + 1) * row_bytes]
            self._decode_stream(chunk, target_dtype, (runlen,),
                                bitspersample, runlen, flat[r])
        if out is not None and result is not out:
            np.copyto(out, result)
            return out
        return result

    @staticmethod
    def _decode_stream(buf, target_dtype, target_shape, bitspersample,
                       n_elements, out):
        """The default continuous MSB-first stream."""
        # Whole-byte widths already use a fast NumPy view/conversion. The
        # native bit reader avoids a samples-by-bits matrix for other widths.
        if (bitspersample not in (1, 8, 16, 32, 64) and target_dtype.kind in "iu" and
                target_dtype.itemsize in (1, 2, 4, 8) and
                (out is None or out.flags.c_contiguous)):
            try:
                from .codecs._bytetools import unpackints_into
            except ImportError:
                pass
            else:
                import sys
                result = np.empty(target_shape, dtype=target_dtype) if out is None else out
                little = (target_dtype.byteorder == "<" or
                          target_dtype.byteorder in ("=", "|") and sys.byteorder == "little")
                unpackints_into(buf, memoryview(result).cast("B"), bitspersample,
                                n_elements, target_dtype.itemsize, little)
                return result
        result = _bitunpack(buf, target_dtype, bitspersample, n_elements)
        result = result.reshape(target_shape)
        if out is not None:
            np.copyto(out, result)
            return out
        return result


__all__ = ["PackintsCodec"]
