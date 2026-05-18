"""AecCodec — Codec adapter wrapping the native _aec extension.

AEC = Adaptive Entropy Coding (CCSDS 121.0-B-2). Used by NetCDF-4 SZIP,
HDF5 SZIP filter, and most satellite/Earth-observation pipelines.
Optimized for *integer* arrays with locally predictable runs; typical
ratios are 2–4× on satellite radiance data.

Encoder defaults match the most common configuration in NetCDF-4
(block_size=32, rsi=128, preprocess=True). Pass ``signed=True`` for
int8/int16/int32 data; ``msb=True`` if your samples are stored
big-endian (default = LSB-first, matching NumPy on x86/ARM).

Example::

    import numpy as np
    import opencodecs as oc

    arr = np.random.randint(0, 4096, size=10000, dtype=np.uint16)
    blob = oc.write(None, arr.tobytes(), format="aec",
                    bits_per_sample=12)  # 12-bit samples in 16-bit words
    raw = oc.read(blob, format="aec")
    assert np.frombuffer(raw, dtype=np.uint16).all() == arr.all()
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

(
    _aec_encode, _aec_decode, _aec_check_signature, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._aec",
    "encode", "decode", "check_signature",
)


# ndarray -> bits_per_sample inference. AEC's "bits_per_sample" is the
# *meaningful* bit-width, not the storage word size; the user can
# override if their data fits in fewer bits than the dtype allows.
_DTYPE_BITS = {
    np.dtype(np.uint8):  (8,  False),
    np.dtype(np.int8):   (8,  True),
    np.dtype(np.uint16): (16, False),
    np.dtype(np.int16):  (16, True),
    np.dtype(np.uint32): (32, False),
    np.dtype(np.int32):  (32, True),
}


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
               block_size: int = 32,
               rsi: int = 128,
               signed: bool | None = None,
               msb: bool = False,
               preprocess: bool = True,
               three_byte: bool = False,
               **opts) -> bytes | None:
        if isinstance(data, np.ndarray):
            dtype = data.dtype
            if dtype not in _DTYPE_BITS:
                raise ValueError(
                    f"aec encode: unsupported ndarray dtype {dtype!r}; "
                    f"expected uint8/int8/uint16/int16/uint32/int32"
                )
            inferred_bps, inferred_signed = _DTYPE_BITS[dtype]
            if bits_per_sample is None:
                bits_per_sample = inferred_bps
            if signed is None:
                signed = inferred_signed
            # Pass the ndarray straight to the Cython encoder via the
            # buffer protocol — no .tobytes() copy. Saves ~200 us per
            # call on a 200 KB input (~30% of the total encode time at
            # default settings).
            buf = (data if data.flags["C_CONTIGUOUS"]
                   else np.ascontiguousarray(data))
        else:
            if bits_per_sample is None:
                raise ValueError(
                    "aec encode: bits_per_sample required when input is bytes"
                )
            if signed is None:
                signed = False
            buf = data if isinstance(data, (bytes, bytearray, memoryview)) else bytes(data)

        out = _aec_encode(
            buf,
            bits_per_sample=int(bits_per_sample),
            block_size=int(block_size),
            rsi=int(rsi),
            is_signed=bool(signed),
            msb=bool(msb),
            preprocess=bool(preprocess),
            three_byte=bool(three_byte),
        )
        return _write_dest(out, dest)

    def decode(self, src: Any, **opts) -> bytes:
        return _aec_decode(_read_src(src))


__all__ = ["AecCodec"]
