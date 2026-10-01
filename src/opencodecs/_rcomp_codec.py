"""RcompCodec — Rice compression for FITS astronomy data.

Rice coding (a special case of Golomb-Rice) is a lightweight entropy
coder optimized for streams of signed integers concentrated near
zero. It's the canonical compression for FITS BINTABLE columns and
the ``RICE_1`` tile-compression algorithm in compressed FITS images.

Implementation: thin wrapper around the Cython ``_rcomp`` extension,
which binds cfitsio's vendored ``ricecomp.c``.

Wire format
-----------
``encode`` writes the bare cfitsio Rice stream: the bytes FITS stores
for a ``RICE_1`` tile (FITS Standard 4.0, section 10.4.1) and the
bytes ``imagecodecs.rcomp_encode`` writes. The stream does not record
its element count, pixel size or block size, so ``decode`` takes them
the way ``imagecodecs.rcomp_decode`` does::

    blob = codec.encode(arr)                      # nblock defaults to 32
    back = codec.decode(blob, shape=arr.shape, dtype=arr.dtype)

A big-endian array is coded by its values, as FITS defines the tile.
``imagecodecs.rcomp_encode`` codes such an array's bytes as if they
were native integers, so for it the bytes differ, and opencodecs
decodes an imagecodecs stream of one to the byte-swapped integers it
holds.

Releases up to 0.4.0 wrote a private 12-byte header in front of the
stream. ``decode`` still reads those blobs. The header has no magic,
so a blob is read as an old one only when each header field holds a
value the old encoder could have written, agrees with ``shape``,
``dtype`` and the block size when the caller passes them, and the
rest of the blob is exactly the Rice stream of the values it decodes
to. A buffer that is also a whole bare stream of the requested shape
and type, again exactly the Rice stream of what it decodes to, is read
as the bare stream.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _rcomp_encode, _rcomp_decode_raw, _rcomp_decode_framed,
    _rcomp_read_framed_header, _rcomp_check_signature, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._rcomp", "encode", "decode_raw", "decode_framed",
    "read_framed_header", "check_signature",
)

# FITS RICE_1 default (ZVALn for BLOCKSIZE) and imagecodecs' default.
_DEFAULT_BLOCKSIZE = 32


def _resolve_blocksize(blocksize, nblock):
    if blocksize is not None and nblock is not None and int(blocksize) != int(nblock):
        raise ValueError(
            f"rcomp: blocksize={blocksize} and nblock={nblock} disagree; "
            "they name the same parameter")
    value = blocksize if blocksize is not None else nblock
    return None if value is None else int(value)


def _reject_unknown(opts):
    if opts:
        raise TypeError(
            f"rcomp: unexpected keyword argument(s) {sorted(opts)}")


class RcompCodec(Codec):
    """Rice compression (Golomb-Rice) for FITS-style integer streams."""

    name = "rcomp"
    aliases = ("rice", "rice1", "ricecomp")
    file_extensions = ()

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    # cfitsio's ricecomp.c supports int8 / int16 / int32 (i.e. 1/2/4
    # byte signed integers); unsigned inputs of the same itemsize are
    # passed through bit-for-bit. int64 is not supported by the cfitsio
    # backend — callers with 8-byte ints should down-convert.
    supported_dtypes = (
        np.int8, np.uint8, np.int16, np.uint16,
        np.int32, np.uint32,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return False  # a Rice stream has no magic

    def encode(self, data: Any, *, dest=None, blocksize: int | None = None,
               nblock: int | None = None, **opts) -> bytes | None:
        """Return the bare cfitsio Rice stream of an integer array.

        ``blocksize`` (alias ``nblock``, imagecodecs' name) is the
        number of pixels per coding block; 32 by default, as in FITS.
        """
        _reject_unknown(opts)
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        bs = _resolve_blocksize(blocksize, nblock)
        out = _rcomp_encode(data, blocksize=_DEFAULT_BLOCKSIZE if bs is None else bs)
        return _write_dest(out, dest)

    def decode(self, src: Any, *, shape=None, dtype=None,
               blocksize: int | None = None, nblock: int | None = None,
               out=None, **opts) -> np.ndarray:
        """Decode a Rice stream into an array of ``shape`` and ``dtype``.

        A bare stream needs ``shape`` and ``dtype`` (or an ``out``
        array to take them from), and the ``blocksize`` it was written
        with (alias ``nblock``; 32 by default). A blob in the framed
        layout of releases up to 0.4.0 carries its own byte count and
        pixel size, so for it ``shape`` and ``dtype`` are optional.
        """
        _reject_unknown(opts)
        buf = _read_src(src)
        bs = _resolve_blocksize(blocksize, nblock)
        if out is not None:
            if not isinstance(out, np.ndarray):
                raise TypeError(
                    f"rcomp decode: out= must be an ndarray, "
                    f"got {type(out).__name__}")
            if not out.flags.writeable:
                raise ValueError("rcomp decode: out= must be writable")
            if dtype is None:
                dtype = out.dtype
            if shape is None:
                shape = out.shape
        target = None if dtype is None else np.dtype(dtype)
        if target is not None and (target.kind not in "iu"
                                   or target.itemsize not in (1, 2, 4)):
            raise ValueError(
                f"rcomp decode: dtype must be an 8, 16 or 32-bit integer, "
                f"got {target}")
        target_shape = None if shape is None else (
            (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape))

        framed = self._framed_header(buf, target, target_shape, bs)
        if framed is not None:
            raw = self._decode_legacy(buf, framed[1])
            if raw is not None and not (
                    target is not None and target_shape is not None
                    and self._bare_round_trips(buf, target, target_shape, bs)):
                return self._finish(raw, target, target_shape, out)

        if target is None or target_shape is None:
            raise ValueError(
                "rcomp decode: a bare Rice stream does not record its size "
                "or pixel type; pass shape= and dtype= (or out=)")
        n = math.prod(target_shape)
        if (out is not None and out.flags.c_contiguous and target.isnative
                and out.dtype == target and out.shape == target_shape):
            # Decode straight into the caller's storage.
            _rcomp_decode_raw(
                buf, nelements=n,
                blocksize=_DEFAULT_BLOCKSIZE if bs is None else bs,
                bytes_per_pixel=target.itemsize,
                out=out.reshape(-1).view(f"u{target.itemsize}"),
            )
            return out
        raw = _rcomp_decode_raw(
            buf, nelements=n,
            blocksize=_DEFAULT_BLOCKSIZE if bs is None else bs,
            bytes_per_pixel=target.itemsize,
        )
        return self._finish(raw, target, target_shape, out)

    @staticmethod
    def _framed_header(buf, target, target_shape, bs):
        """Header fields if ``buf`` reads as a release-0.4.0 framed blob
        consistent with what the caller asked for, else None."""
        hdr = _rcomp_read_framed_header(buf)
        if hdr is None:
            return None
        nbytes, blk, bpp = hdr
        if bs is not None and blk != bs:
            return None
        if target is not None and target.itemsize != bpp:
            return None
        if target_shape is not None:
            itemsize = bpp if target is None else target.itemsize
            if math.prod(target_shape) * itemsize != nbytes:
                return None
        return hdr

    @staticmethod
    def _decode_legacy(buf, blocksize):
        """Decode ``buf`` as a release-0.4.0 framed blob, or return None
        if it is not one.

        The old header has no magic, so a bare stream can start with
        bytes that pass for one. Rice coding is deterministic and the
        old encoder was the same cfitsio coder, so a real old blob's
        payload is exactly what encoding its decoded values gives
        back; anything else is not taken for an old blob.
        """
        try:
            raw = _rcomp_decode_framed(buf)
        except Exception:
            return None
        try:
            again = _rcomp_encode(raw, blocksize=blocksize)
        except Exception:
            return None
        if bytes(again) != bytes(memoryview(buf)[12:]):
            return None
        return raw

    @staticmethod
    def _bare_round_trips(buf, target, target_shape, bs):
        """True if ``buf`` is a whole bare Rice stream of this size and
        pixel type that re-encodes to itself. A bare stream that also
        passes for an old framed blob is read as the standard stream."""
        blocksize = _DEFAULT_BLOCKSIZE if bs is None else bs
        try:
            raw = _rcomp_decode_raw(
                buf, nelements=math.prod(target_shape), blocksize=blocksize,
                bytes_per_pixel=target.itemsize)
            again = _rcomp_encode(raw, blocksize=blocksize)
        except Exception:
            return False
        return bytes(again) == bytes(buf)

    @staticmethod
    def _finish(raw, target, target_shape, out):
        # cfitsio writes unsigned words; reinterpret the bit pattern as
        # the requested type (``view``, so negative values survive),
        # then fix the byte order if a non-native one was asked for.
        arr = raw
        if target is not None:
            arr = arr.view(target.newbyteorder("="))
            if not target.isnative:
                arr = arr.astype(target)
        if target_shape is not None:
            if math.prod(target_shape) != arr.size:
                raise ValueError(
                    f"rcomp decode: shape {target_shape} does not match the "
                    f"{arr.size} decoded elements")
            arr = arr.reshape(target_shape)
        if out is not None:
            if out.shape != arr.shape or out.dtype != arr.dtype:
                raise ValueError("rcomp decode: out= shape/dtype mismatch")
            np.copyto(out, arr)
            return out
        return arr


__all__ = ["RcompCodec"]
