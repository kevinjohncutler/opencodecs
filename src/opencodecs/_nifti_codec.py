"""NiftiCodec — Codec adapter for the native NIfTI-1 / NIfTI-2 reader.

NIfTI is a container, so this decodes and does not encode, matching
``FitsCodec`` and ``MrcCodec``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec


class NiftiCodec(Codec):
    """NIfTI-1 / NIfTI-2 neuroimaging volume reader."""

    name = "nifti"
    aliases = ("nii", "nifti1", "nifti2")
    file_extensions = (".nii", ".nii.gz")

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = True
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (
        np.uint8, np.int8, np.int16, np.uint16, np.int32, np.uint32,
        np.int64, np.uint64, np.float32, np.float64,
        np.complex64, np.complex128,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        """Match the magic, which moved between the two versions.

        NIfTI-2 puts it at byte 4, right after sizeof_hdr; NIfTI-1 puts
        it at 344, at the end of its header. Gzipped files cannot be
        sniffed this way at all, so ``.nii.gz`` dispatches by extension.
        """
        if len(head) >= 8 and head[4:7] in (b"n+2", b"ni2"):
            return True
        return len(head) >= 348 and head[344:347] in (b"n+1", b"ni1")

    def open(self, src: Any):
        from ._nifti import NiftiStream
        return NiftiStream(src)

    def decode(self, src: Any, **opts) -> np.ndarray:
        """Read the volume. ``scaled=False`` returns raw stored values."""
        scaled = opts.pop("scaled", True)
        out = opts.pop("out", None)
        with self.open(src) as r:
            arr = r.asarray(scaled=scaled)
            if out is not None:
                out[...] = arr
                return out
            return arr

    def encode(self, data: Any, *, dest=None, **opts) -> bytes | None:
        """Serialize an array as a single-file NIfTI-1 (.nii).

        ``compress=True`` gzips it. NIfTI-1 rather than NIfTI-2 because
        every tool reads NIfTI-1, and its int16 dimension limit is
        checked rather than silently truncated.
        """
        from ._nifti_writer import encode_nifti
        blob = encode_nifti(data, **opts)
        if dest is None:
            return blob
        from .core._io_helpers import write_dest as _write_dest
        return _write_dest(dest, blob)


__all__ = ["NiftiCodec"]
