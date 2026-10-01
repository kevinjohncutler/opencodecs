"""PcodecCodec — Codec adapter wrapping the native _pcodec extension.

pcodec (https://github.com/mwlon/pcodec) is a 2024+ lossless numerical
compressor that beats zstd on dense numerical arrays by ~1.5-3× without
filtering. It's a drop-in replacement for zstd-on-floats in scientific
pipelines.

Wire format
-----------
``encode`` writes pcodec's standalone format (magic ``pco!``), the
stream the pcodec package, ``numcodecs.PCodec`` and
``imagecodecs.pcodec_encode`` write. It records the number type and
a hint of the element count (pcodec writes the count; the format
allows 0 for unknown), but not an N-D shape. ``shape=`` (or an
``out=`` array) sets the count, as ``imagecodecs.pcodec_decode`` takes
it; without one, ``decode`` returns a 1-D array of the elements the
stream holds, up to the hinted count, and a stream that holds more
than its hint (any data at all for hint 0) needs ``shape``. Blobs from
releases up to 0.4.0 start with a private 'PCOO' preamble that
carried the shape; ``decode`` still reads them.

A big-endian array is coded by its values. ``imagecodecs.pcodec_encode``
codes such an array's bytes as if they were native numbers, so for it
the bytes differ, and opencodecs decodes an imagecodecs stream of one
to the byte-swapped numbers it holds.

Example::

    arr = np.random.rand(100000).astype(np.float32)
    blob = oc.write(None, arr, format="pcodec", level=8)
    back = oc.read(blob, format="pcodec")
    np.testing.assert_array_equal(arr, back)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import array_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _pco_encode, _pco_decode_framed, _pco_decode_standalone,
    _pco_standalone_info, _pco_check_signature, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._pcodec",
    "encode", "decode_framed", "decode_standalone", "standalone_info",
    "check_signature",
)


def _reject_unknown(opts):
    if opts:
        raise TypeError(
            f"pcodec: unexpected keyword argument(s) {sorted(opts)}")


class PcodecCodec(Codec):
    """Native pcodec — modern lossless numerical compressor."""

    name = "pcodec"
    aliases = ("pco",)
    file_extensions = (".pco",)

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (
        np.uint8, np.int8,
        np.uint16, np.int16, np.float16,
        np.uint32, np.int32, np.float32,
        np.uint64, np.int64, np.float64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return _pco_check_signature(head)

    def encode(self, data: Any, *, dest=None,
               level: int | None = None,
               max_page_n: int | None = None,
               pagesize: int | None = None,
               **opts) -> bytes | None:
        """Return the pcodec standalone stream of ``data`` (C order).

        ``level`` is 0..12 (default 8, pcodec's own). ``max_page_n``
        (alias ``pagesize``, imagecodecs' name) caps the elements per
        page; 0 or None is the library default.
        """
        _reject_unknown(opts)
        if (max_page_n is not None and pagesize is not None
                and int(max_page_n) != int(pagesize)):
            raise ValueError(
                f"pcodec: max_page_n={max_page_n} and pagesize={pagesize} "
                "disagree; they name the same parameter")
        page = max_page_n if max_page_n is not None else pagesize
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        if not data.dtype.isnative:
            data = data.astype(data.dtype.newbyteorder("="))
        out = _pco_encode(data, level=8 if level is None else int(level),
                          max_page_n=0 if page is None else int(page))
        return _write_dest(out, dest)

    def decode(self, src: Any, *, shape=None, dtype=None, out=None,
               **opts) -> np.ndarray:
        """Decode a pcodec standalone stream.

        The number type comes from the stream. ``shape`` (or ``out``)
        sets the element count, and the decode must produce exactly
        that many; without either, the header's count hint is used and
        the result is 1-D. ``dtype`` is needed only for a stream that
        does not record its type, and must otherwise match it.
        """
        _reject_unknown(opts)
        buf = _read_src(src)
        out = array_output(out)
        target_shape = None if shape is None else (
            (int(shape),) if np.isscalar(shape) else tuple(int(s) for s in shape))
        target = None if dtype is None else np.dtype(dtype)
        head = bytes(buf[:4])

        if head == b"PCOO":
            if out is not None:
                if target_shape is None or target_shape == out.shape:
                    return _pco_decode_framed(buf, out=out)
            arr = _pco_decode_framed(buf)
            return self._finish(arr, target, target_shape, out)

        if head != b"pco!":
            raise ValueError(
                f"pcodec decode: not a pcodec stream (magic {head!r}, "
                "expected b'pco!')")
        version, stream_dt, hint = _pco_standalone_info(buf)
        if out is not None:
            if target is None:
                target = out.dtype
            if target_shape is None:
                target_shape = out.shape
        if target is not None and stream_dt is not None and target != stream_dt:
            raise ValueError(
                f"pcodec decode: the stream holds {stream_dt} numbers, "
                f"not {target}")
        dt = stream_dt if stream_dt is not None else target
        if dt is None:
            raise ValueError(
                "pcodec decode: this stream does not record its number "
                "type; pass dtype=")
        # The header's count is a hint (pcodec docs/format.md: the
        # count "if known; 0 otherwise"); the stream itself ends with a
        # termination byte. A shape or out decides the count; the hint
        # is only the default.
        if target_shape is not None:
            n = math.prod(target_shape)
            exact = True
        else:
            n = hint or 0
            exact = False
        if out is not None:
            if out.shape != target_shape or out.dtype != dt:
                raise ValueError("pcodec decode: out= shape/dtype mismatch")
            _pco_decode_standalone(buf, dtype=dt, n=n, out=out)
            return out
        try:
            arr = _pco_decode_standalone(buf, dtype=dt, n=n, exact=exact)
        except RuntimeError as exc:
            if exact:
                raise
            if not n:
                # Hint 0: an empty stream, or one of unknown count.
                raise ValueError(
                    f"pcodec decode: this version {version} standalone "
                    "stream does not record its element count; pass "
                    "shape= or out=") from exc
            raise type(exc)(
                f"{exc}; the header's count hint is {n}: pass shape= if "
                "the stream holds more") from exc
        return arr if target_shape is None else arr.reshape(target_shape)

    @staticmethod
    def _finish(arr, target, target_shape, out):
        if target is not None and arr.dtype != target:
            raise ValueError(
                f"pcodec decode: the stream holds {arr.dtype} numbers, "
                f"not {target}")
        if target_shape is not None and target_shape != arr.shape:
            if math.prod(target_shape) != arr.size:
                raise ValueError(
                    f"pcodec decode: shape {target_shape} does not match "
                    f"the {arr.size} decoded elements")
            arr = arr.reshape(target_shape)
        if out is not None:
            if out.shape != arr.shape or out.dtype != arr.dtype:
                raise ValueError("pcodec decode: out= shape/dtype mismatch")
            np.copyto(out, arr)
            return out
        return arr


__all__ = ["PcodecCodec"]
