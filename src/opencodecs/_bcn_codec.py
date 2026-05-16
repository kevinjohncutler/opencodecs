"""BcnCodec — top-level BC1-7 (DXT / S3TC / RGTC / BPTC) GPU texture codec.

The actual decompression lives in ``opencodecs.codecs._bcdec`` (one
function per BC variant, sharing a Cython inner loop). This module
wraps them in the standard Codec API so callers can dispatch by
codec name + a ``format=`` parameter, matching imagecodecs's
``bcn_decode`` interface.

BC1-7 are the GPU texture-compression formats DirectX / Vulkan / WebGPU
use; they're also embedded in DDS files, KTX/KTX2 containers, and a
handful of game-asset formats. Single image (no multi-frame), fixed
4×4 pixel blocks at codec-specific bytes-per-block.

BC1-3, BC7 → ``uint8`` RGBA.
BC4 → ``uint8`` / ``int8`` (single channel).
BC5 → ``uint8`` / ``int8`` (two channels).
BC6H → ``float32`` (or ``float16`` if ``format='half'``) RGB (HDR).

Encode is NOT yet implemented (BCn encoders are far more complex
than decoders and we don't yet have a Cython BC encoder; ``imagecodecs``
ships ``bcn_encode`` via the upstream NVTT-derived encoder which we
haven't ported).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src
from .core._optional_backend import import_or_stubs

(
    _decode_bc1, _decode_bc2, _decode_bc3, _decode_bc4,
    _decode_bc5, _decode_bc6h, _decode_bc7, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._bcdec",
    "decode_bc1", "decode_bc2", "decode_bc3", "decode_bc4",
    "decode_bc5", "decode_bc6h", "decode_bc7",
)


# Normalized aliases for the BC format strings (matches imagecodecs's
# tj3-style enum vs string lookups).
_FORMAT_MAP = {
    "bc1": "bc1", "dxt1": "bc1",
    "bc2": "bc2", "dxt3": "bc2",
    "bc3": "bc3", "dxt5": "bc3",
    "bc4": "bc4", "ati1n": "bc4", "ati1": "bc4",
    "bc5": "bc5", "ati2n": "bc5", "ati2": "bc5", "3dc": "bc5",
    "bc6h": "bc6h", "bc6": "bc6h",
    "bc7": "bc7",
}


class BcnCodec(Codec):
    """Top-level BC1-7 texture codec dispatcher."""

    name = "bcn"
    aliases = ("bc1", "bc2", "bc3", "bc4", "bc5", "bc6h", "bc7", "dxt1", "dxt3", "dxt5")
    file_extensions = ()

    has_native = True
    has_delegate = False
    can_encode = False        # No native BCn encoder yet
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8, np.int8, np.float16, np.float32)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        # Raw BCn blocks have no magic. DDS containers have a magic but
        # those are handled by a dedicated DDS reader (not implemented
        # yet); return False so codec_for_bytes doesn't false-positive.
        return False

    def encode(self, data: Any, *, dest=None, **opts) -> bytes | None:
        raise NotImplementedError(
            "bcn encode: native BCn encoder is not implemented yet. "
            "For encoding to DDS / KTX use the corresponding GPU tool."
        )

    def decode(
        self,
        src: Any,
        *,
        format: str,
        width: int,
        height: int,
        out=None,
        is_signed: bool = False,
        fp16: bool = False,
        **opts,
    ) -> np.ndarray:
        """Decode BCn-compressed bytes to an ndarray.

        Parameters
        ----------
        src
            Compressed BCn bytes (4-byte-aligned to a 4×4-block grid).
        format : str
            Which BC variant. Common values: ``"bc1"``/``"dxt1"``,
            ``"bc2"``/``"dxt3"``, ``"bc3"``/``"dxt5"``, ``"bc4"``,
            ``"bc5"``, ``"bc6h"``, ``"bc7"`` (case-insensitive).
        width, height : int
            Output raster dimensions in pixels. Both must be
            multiples of 4 (BCn's block size).
        is_signed : bool, optional
            Only meaningful for BC4 / BC5 / BC6H. Selects the
            signed variant.
        fp16 : bool, optional
            Only meaningful for BC6H — returns float16 instead of
            float32.
        out : np.ndarray, optional
            Preallocated destination. See ``_png.decode`` for the full
            contract; the ``out=`` value is forwarded to the underlying
            ``decode_bcN``.
        """
        buf = _read_src(src)
        fmt = _FORMAT_MAP.get(str(format).lower().strip())
        if fmt is None:
            raise ValueError(
                f"bcn decode: unknown format {format!r}; expected one of "
                f"{sorted(set(_FORMAT_MAP))}")
        if fmt == "bc1":
            return _decode_bc1(buf, width=width, height=height, out=out)
        if fmt == "bc2":
            return _decode_bc2(buf, width=width, height=height, out=out)
        if fmt == "bc3":
            return _decode_bc3(buf, width=width, height=height, out=out)
        if fmt == "bc4":
            return _decode_bc4(buf, width=width, height=height,
                               is_signed=is_signed, out=out)
        if fmt == "bc5":
            return _decode_bc5(buf, width=width, height=height,
                               is_signed=is_signed, out=out)
        if fmt == "bc6h":
            return _decode_bc6h(
                buf, width=width, height=height,
                is_signed=is_signed,
                format="half" if fp16 else "float",
                out=out,
            )
        if fmt == "bc7":
            return _decode_bc7(buf, width=width, height=height, out=out)
        raise AssertionError(f"unhandled BC format {fmt}")


__all__ = ["BcnCodec"]
