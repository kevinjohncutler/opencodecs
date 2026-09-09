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
    # BC1/2/3/7 decode in bands of block rows across threads. Every
    # 4x4 block is at a computed offset with no dependencies, which is
    # the whole point of block compression. Measured on 4096x4096:
    # BC7 69.8 ms to 11.3 ms.
    # decode_rows() decodes a band of block rows and nothing else.
    # Every 4x4 block is at a computed offset with no dependencies, so
    # the band's blocks are a contiguous slice of the input and decode
    # as a surface in their own right. 64 rows of a 4096-row BC7
    # texture: 1.11 ms against 69.8 ms for the whole thing.
    chunked = True
    parallel_decode = True

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
        numthreads : int, optional
            Decode bands of block rows across threads. Every 4x4 block
            sits at a computed offset and depends on nothing else, so a
            band reads a contiguous run of the input and writes a
            contiguous band of the output. BC1/2/3/7 only; the
            single-channel and float variants go through their own
            decoders.
        """
        buf = _read_src(src)
        numthreads = opts.pop("numthreads", None)
        fmt = _FORMAT_MAP.get(str(format).lower().strip())
        if fmt is None:
            raise ValueError(
                f"bcn decode: unknown format {format!r}; expected one of "
                f"{sorted(set(_FORMAT_MAP))}")
        rgba = {"bc1": 1, "bc2": 2, "bc3": 3, "bc7": 7}.get(fmt)
        if rgba is not None:
            return self._decode_rgba(buf, rgba, width, height, out,
                                     numthreads)
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
        raise AssertionError(f"unhandled BC format {fmt}")

    #: Block byte size per RGBA BC format. BC1 packs two endpoints and
    #: 2-bit indices into 8 bytes; the rest carry an alpha block too.
    _RGBA_BLOCK_BYTES = {1: 8, 2: 16, 3: 16, 7: 16}

    def decode_rows(self, src: Any, *, format: str, width: int, height: int,
                    y0: int, y1: int, numthreads: int | None = None):
        """Decode pixel rows ``[y0, y1)`` without touching the rest.

        A BCn surface is a grid of independent 4x4 blocks at computed
        offsets, so a horizontal band costs its own blocks and nothing
        else. That makes the band a whole surface in its own right:
        slice the input to the band's block rows and it decodes as an
        ordinary image of that height, with no offset arithmetic.

        This is random access the format has always had and this codec
        did not offer. Decoding a 4096-row BC7 texture to reach 64 rows
        of it read every block to produce 1.5% of the output.

        Bounds are pixel rows and must fall on the 4-pixel block grid:
        a partial block row cannot be decoded without decoding the
        block. BC1/2/3/7 only, since the single-channel and float
        variants go through their own decoders.
        """
        buf = _read_src(src)
        fmt = _FORMAT_MAP.get(str(format).lower().strip())
        fmt_id = {"bc1": 1, "bc2": 2, "bc3": 3, "bc7": 7}.get(fmt)
        if fmt_id is None:
            raise ValueError(
                f"bcn decode_rows: {format!r} is not an RGBA BC format; "
                f"row access is bc1, bc2, bc3 or bc7")
        if y0 % 4 or y1 % 4:
            raise ValueError(
                f"bcn decode_rows: rows [{y0}:{y1}) must be multiples of 4; "
                f"a partial block row cannot be decoded without its block")
        if not 0 <= y0 < y1 <= height:
            raise ValueError(
                f"bcn decode_rows: [{y0}:{y1}) is out of range for "
                f"{height} rows")
        row_bytes = (width // 4) * self._RGBA_BLOCK_BYTES[fmt_id]
        band = buf[(y0 // 4) * row_bytes:(y1 // 4) * row_bytes]
        return self._decode_rgba(band, fmt_id, width, y1 - y0, None,
                                 numthreads)

    @staticmethod
    def _decode_rgba(buf, fmt_id: int, width: int, height: int, out,
                     numthreads):
        """BC1/2/3/7 into (H, W, 4) uint8, in bands across threads."""
        from .codecs._bcdec import decode_block_rows
        from .core.parallel import resolve_workers, run_batched

        n_block_rows = height // 4
        if out is None:
            out = np.empty((height, width, 4), dtype=np.uint8)
        workers = resolve_workers(numthreads, n_block_rows,
                                  output_bytes=out.nbytes,
                                  min_items=8)
        if workers <= 1:
            return decode_block_rows(buf, width=width, height=height,
                                     fmt_id=fmt_id, out=out,
                                     by0=0, by1=n_block_rows)
        step = (n_block_rows + workers - 1) // workers
        bands = [(i, min(i + step, n_block_rows))
                 for i in range(0, n_block_rows, step)]
        run_batched(
            lambda b: decode_block_rows(buf, width=width, height=height,
                                        fmt_id=fmt_id, out=out,
                                        by0=b[0], by1=b[1]),
            bands, workers, name="bcn")
        return out


__all__ = ["BcnCodec"]
