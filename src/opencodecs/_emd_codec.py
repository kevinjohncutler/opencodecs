"""EmdCodec — Codec adapter for the EMD reader."""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec


class EmdCodec(Codec):
    """EMD reader (Berkeley/NCEM and Thermo Velox schemas)."""

    name = "emd"
    aliases = ("velox", "ncem")
    file_extensions = (".emd",)

    has_native = True
    has_delegate = False
    can_encode = False
    can_decode = True
    multi_frame = True
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (
        np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
        np.float32, np.float64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        """EMD is HDF5, and so are Imaris and MINC.

        The magic alone cannot tell them apart, so this deliberately
        does not claim HDF5 files: .emd dispatches by extension and the
        reader then checks the structure. Claiming the magic here would
        route every HDF5 file to this codec.
        """
        return False

    def open(self, src: Any):
        from ._emd import EmdFile
        return EmdFile(src)

    def decode(self, src: Any, **opts) -> np.ndarray:
        index = opts.pop("index", 0)
        with self.open(src) as f:
            return f.asarray(index)

    def encode(self, data: Any, *, dest=None, **opts):
        raise NotImplementedError(
            "emd: encoding is not implemented; opencodecs reads EMD")


__all__ = ["EmdCodec"]
