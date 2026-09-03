"""DmCodec — Codec adapter for the Gatan Digital Micrograph reader."""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec


class DmCodec(Codec):
    """Gatan Digital Micrograph reader (.dm3 / .dm4)."""

    name = "dm"
    aliases = ("dm3", "dm4", "gatan")
    file_extensions = (".dm3", ".dm4")

    has_native = True
    has_delegate = False
    can_encode = False
    can_decode = True
    multi_frame = True
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (
        np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32,
        np.int64, np.uint64, np.float32, np.float64,
        np.complex64, np.complex128,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        """Version 3 or 4 as a big-endian int, then a plausible root length.

        DM has no magic string, so the check has to lean on structure.
        Requiring the root length to be positive and the byte-order flag
        to be 0 or 1 keeps it from claiming arbitrary files whose first
        four bytes happen to be 3.
        """
        if len(head) < 12:
            return False
        version = int.from_bytes(head[:4], "big")
        if version == 3:
            root, order = (int.from_bytes(head[4:8], "big"),
                           int.from_bytes(head[8:12], "big"))
        elif version == 4 and len(head) >= 24:
            root, order = (int.from_bytes(head[4:12], "big"),
                           int.from_bytes(head[16:20], "big"))
        else:
            return False
        return root > 0 and order in (0, 1)

    def open(self, src: Any):
        from ._dm import DmFile
        return DmFile(src)

    def decode(self, src: Any, **opts) -> np.ndarray:
        """Decode one image; the embedded thumbnail is skipped."""
        index = opts.pop("index", 0)
        include_thumbnails = opts.pop("include_thumbnails", False)
        with self.open(src) as f:
            return f.asarray(index, include_thumbnails=include_thumbnails)

    def encode(self, data: Any, *, dest=None, **opts):
        raise NotImplementedError(
            "dm: encoding is not implemented; opencodecs reads Digital "
            "Micrograph files")


__all__ = ["DmCodec"]
