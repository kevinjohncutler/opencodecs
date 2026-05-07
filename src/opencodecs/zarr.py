"""numcodecs / zarr adapter for the JPEG XL codec.

Registers ``opencodecs.zarr.JxlCodec`` as a numcodecs codec so it can be
used as a zarr compressor::

    import zarr
    from opencodecs.zarr import JxlCodec

    z = zarr.open(
        "stack.zarr", mode="w",
        shape=(100, 1024, 1024, 3), chunks=(1, 1024, 1024, 3),
        dtype="uint8",
        compressor=JxlCodec(lossless=True, color="display-p3"),
    )
    z[:] = arr

Each chunk is encoded as a single JXL still image (no animation container) —
that's the natural mapping when chunks are already (Y, X[, C]) shaped, and
it keeps each chunk independently decodable.

Multi-frame chunks (T, Y, X[, C]) are supported by setting
``animation=True``; the codec writes them as a JXL animation and decodes
back to the same shape.
"""

from __future__ import annotations

from typing import Any

import numpy as np

try:
    from numcodecs.abc import Codec
    from numcodecs.compat import ensure_contiguous_ndarray
    from numcodecs.registry import register_codec
except ImportError as e:  # pragma: no cover - numcodecs-missing branch
    raise ImportError(
        "opencodecs.zarr requires numcodecs (`pip install numcodecs`)"
    ) from e

from .codecs._jxl import encode as _jxl_encode, decode as _jxl_decode


class JxlCodec(Codec):
    """JPEG XL codec for numcodecs/zarr.

    All kwargs map directly to ``opencodecs.jxl.write`` / encode().
    See ``opencodecs.core.color.parse_color`` for accepted color strings.
    """

    codec_id = "opencodecs_jxl"

    def __init__(
        self,
        *,
        lossless: bool = True,
        quality: float | None = None,
        distance: float | None = None,
        effort: int = 5,
        decoding_speed: int = 0,
        color: str | None = None,
        animation: bool = False,
        numthreads: int | None = None,
    ):
        self.lossless = bool(lossless)
        self.quality = quality
        self.distance = distance
        self.effort = int(effort)
        self.decoding_speed = int(decoding_speed)
        self.color = color
        self.animation = bool(animation)
        self.numthreads = numthreads

    def encode(self, buf: Any) -> bytes:
        # numcodecs.ensure_contiguous_ndarray flattens by default — we need
        # the original shape for image codecs. Pass shape-preserved.
        if isinstance(buf, np.ndarray):
            arr = np.ascontiguousarray(buf)
        else:
            arr = ensure_contiguous_ndarray(buf, flatten=False)

        # zarr chunks are typically (T, Y, X[, C]) where T (or any leading
        # 1-size dim) makes ndim > 3. JXL accepts 2D / 3D arrays. Strip
        # leading length-1 axes; the decode side restores them.
        squeezed = arr
        while squeezed.ndim > 3 and squeezed.shape[0] == 1:
            squeezed = squeezed[0]

        if squeezed.ndim > 3 and self.animation:
            # Multi-frame chunk: encode as JXL animation
            return _jxl_encode(
                squeezed, color=self.color, lossless=self.lossless,
                quality=self.quality, distance=self.distance,
                effort=self.effort, decoding_speed=self.decoding_speed,
                numthreads=self.numthreads, animation=True,
                container=False, dest=None,
            )
        if squeezed.ndim > 3:
            raise ValueError(
                f"JxlCodec: chunk shape {arr.shape} not encodable as a "
                f"single image. Use chunks with at most one non-leading "
                f"axis, or set animation=True."
            )
        return _jxl_encode(
            squeezed, color=self.color, lossless=self.lossless,
            quality=self.quality, distance=self.distance,
            effort=self.effort, decoding_speed=self.decoding_speed,
            numthreads=self.numthreads, animation=self.animation,
            container=False, dest=None,
        )

    def decode(self, buf: bytes, out: Any | None = None) -> Any:
        arr = _jxl_decode(buf, numthreads=self.numthreads, parse_color=False)
        if out is not None:
            # zarr passes an `out` ndarray view of its target chunk buffer.
            # Reshape decoded pixels to fit (zarr's `out` may have leading
            # length-1 axes that we squeezed away on encode).
            target = np.asarray(out)
            try:
                view = target.reshape(arr.shape)
            except ValueError:  # pragma: no cover - shape-mismatch defense; reshape works when sizes equal
                # try squeezing target's leading singletons
                view = target
                while view.shape != arr.shape and view.ndim > arr.ndim and view.shape[0] == 1:
                    view = view[0]
            np.copyto(view.view(arr.dtype), arr)
            return out
        return arr

    def __repr__(self) -> str:
        flags = []
        if self.lossless:
            flags.append("lossless")
        else:
            flags.append(f"distance={self.distance}")
        if self.color:
            flags.append(f"color={self.color!r}")
        if self.animation:
            flags.append("animation")
        return f"JxlCodec({', '.join(flags)})"


# Register on import so numcodecs.get_codec({"id": "opencodecs_jxl", ...})
# resolves correctly when reading existing zarr stores.
register_codec(JxlCodec)


__all__ = ["JxlCodec"]
