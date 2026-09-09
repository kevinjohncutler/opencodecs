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
    # A 3-D NRRD is a stack of slices and ArrayReader already reports
    # them through n_frames, exactly as for MRC and NIfTI, both of
    # which say so. This said False while iterating really did yield
    # frames.
    multi_frame = True
    # _frame() reads one slice at its own offset for the raw encoding,
    # so iterating does not materialize the volume first, and slice N
    # does not cost slices 0..N-1.
    streaming_decode = True
    chunked = True
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
