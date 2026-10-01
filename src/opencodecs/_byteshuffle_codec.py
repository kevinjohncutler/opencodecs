"""ByteshuffleCodec — element-byte-plane shuffle for compression preprocessors.

For multi-byte element arrays (uint16 / uint32 / float32 / ...) the
high-byte and low-byte streams typically have very different
entropy: high bytes are often constant or slowly-varying, low bytes
look random. A bytes-in/bytes-out compressor (zstd, lz4, deflate)
sees one interleaved stream of all bytes and matches both streams
together — losing the redundancy in the high-byte plane.

Byteshuffle rearranges memory so all the high bytes come first, then
all the low bytes (etc. for >2 byte types). The result is a byte
stream the compressor can squeeze ~1.5-3× harder for typical
scientific arrays. Bitshuffle (a finer-grained sibling — see
``BitshuffleCodec``) often beats it on noisy data; byteshuffle is
cheaper to encode/decode and frequently wins on smooth data.

This is the whole-buffer shuffle of the HDF5 shuffle filter
(H5Z_FILTER_SHUFFLE), Blosc's ``BLOSC_SHUFFLE`` and
``numcodecs.Shuffle``, byte for byte, and the layout this package's
HDF5, FITS ``GZIP_2`` and CZI readers undo. It is not imagecodecs'
``byteshuffle_encode``, which shuffles each row along an ``axis``
separately (the byte-plane step of TIFF predictor 3; see the
``floatpred`` codec here). The two agree only for 1-D input. That
function's ``axis``, ``dist``, ``delta`` and ``reorder`` keywords have
no meaning for a whole-buffer shuffle and raise ``TypeError`` here
rather than being ignored.

Composes with any byte-level compressor:

    byteshuffled = oc.get_codec("byteshuffle").encode(arr.tobytes(), itemsize=2)
    compressed = oc.get_codec("zstd").encode(byteshuffled)

The underlying nogil loops live in
``opencodecs.codecs._bytetools.byteshuffle_{encode,decode}``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import byte_output
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from ._filter_args import reject_unknown
from .codecs._bytetools import (
    byteshuffle_encode as _bs_encode,
    byteshuffle_decode as _bs_decode,
)


class ByteshuffleCodec(Codec):
    """Element-byte-plane shuffle (compression preprocessor)."""

    name = "byteshuffle"
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
        np.uint8, np.int8, np.uint16, np.int16,
        np.uint32, np.int32, np.uint64, np.int64,
        np.float16, np.float32, np.float64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        # Byteshuffle is a filter, not a container — no magic bytes.
        return False

    def encode(
        self,
        data: Any,
        *,
        dest=None,
        itemsize: int | None = None,
        **opts,
    ) -> bytes | None:
        """Shuffle bytes.

        ``data`` may be a numpy array (itemsize inferred from dtype) or
        a bytes-like; for bytes-like input, ``itemsize`` is required.
        """
        _reject(opts, "encode")
        if isinstance(data, np.ndarray):
            if itemsize is None:
                itemsize = data.dtype.itemsize
            n_elements = data.size
            buf = np.ascontiguousarray(data).tobytes()
        else:
            if itemsize is None:
                raise ValueError(
                    "byteshuffle encode: itemsize= is required for "
                    "non-ndarray input")
            buf = bytes(data) if not isinstance(data, (bytes, bytearray)) else data
            if len(buf) % itemsize != 0:
                raise ValueError(
                    f"byteshuffle encode: data length {len(buf)} is not "
                    f"a multiple of itemsize {itemsize}")
            n_elements = len(buf) // itemsize
        out = _bs_encode(buf, int(itemsize), int(n_elements))
        return _write_dest(out, dest)

    def decode(
        self,
        src: Any,
        *,
        itemsize: int | None = None,
        n_elements: int | None = None,
        out=None,
        **opts,
    ) -> bytes | np.ndarray:
        """Reverse a byteshuffle.

        The shuffled bytes do not record the element size: it comes from
        ``itemsize``, or, by default, from an ndarray ``src``'s dtype, which
        then also gives the result's dtype and shape. Bytes-like input
        needs ``itemsize``. ``n_elements`` defaults to
        ``len(src) // itemsize``.
        """
        _reject(opts, "decode")
        array_src = src if isinstance(src, np.ndarray) else None
        if itemsize is None:
            if array_src is None:
                raise ValueError(
                    "byteshuffle decode: itemsize= is required for "
                    "non-ndarray input")
            itemsize = array_src.dtype.itemsize
        buf = _read_src(src)
        if n_elements is None:
            if len(buf) % itemsize != 0:
                raise ValueError(
                    f"byteshuffle decode: data length {len(buf)} is not "
                    f"a multiple of itemsize {itemsize}")
            n_elements = len(buf) // itemsize
        if out is not None:
            return _bs_decode(buf, int(itemsize), int(n_elements),
                              out=byte_output(out))
        result = _bs_decode(buf, int(itemsize), int(n_elements))
        if array_src is None or len(result) != array_src.nbytes:
            return result
        return np.frombuffer(result, dtype=array_src.dtype).reshape(array_src.shape).copy()


_PER_ROW_KEYWORDS = ("axis", "dist", "delta", "reorder")


def _reject(opts: dict, which: str) -> None:
    hint = ""
    if any(name in opts for name in _PER_ROW_KEYWORDS):
        hint = ("this codec is the whole-buffer HDF5/Blosc shuffle; "
                "imagecodecs' per-row byteshuffle with axis, dist, delta "
                "and reorder is TIFF predictor 3's byte-plane step, the "
                "floatpred codec here")
    reject_unknown(f"byteshuffle {which}", opts, hint)


__all__ = ["ByteshuffleCodec"]
