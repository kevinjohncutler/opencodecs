"""NrrdCodec — Codec adapter for the native NRRD reader."""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec


class NrrdCodec(Codec):
    """NRRD reader (3D Slicer, ITK, medical image computing)."""

    name = "nrrd"
    aliases = ("nhdr",)
    file_extensions = (".nrrd", ".nhdr")

    has_native = True
    has_delegate = False
    can_encode = False
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (
        np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
        np.int64, np.uint64, np.float32, np.float64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return head[:4] == b"NRRD"

    def open(self, src: Any):
        from ._nrrd import NrrdFile
        return NrrdFile(src)

    def decode(self, src: Any, **opts) -> np.ndarray:
        with self.open(src) as r:
            return r.asarray()

    def encode(self, data: Any, *, dest=None, **opts):
        raise NotImplementedError(
            "nrrd: encoding is not implemented; opencodecs reads NRRD")


__all__ = ["NrrdCodec"]
