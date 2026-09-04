"""DicomCodec — Codec adapter for the native DICOM file reader.

DICOM is a container, not a compression codec: the compression, when
there is any, is named by the transfer syntax and handled by whichever
codec that syntax points at. So this exposes decode and the ``open(src)``
reader contract and does not encode, the same shape as ``MrcCodec``.

Without this the reader was reachable only as ``DicomFile``, which meant
``oc.read("study.dcm")`` did not work while every other format in the
package answered to it.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec


class DicomCodec(Codec):
    """DICOM Part 10 file reader (.dcm)."""

    name = "dicom"
    aliases = ("dcm",)
    file_extensions = (".dcm", ".dicom")

    has_native = True
    has_delegate = False
    can_encode = False
    can_decode = True
    multi_frame = True
    streaming_decode = True
    parallel_decode = False

    supported_dtypes = (np.uint8, np.int8, np.uint16, np.int16,
                        np.uint32, np.int32)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        """The DICM magic, at 128 or at 0.

        Files written without the preamble are legal and common off a
        network stream, but sniffing those means guessing from a group
        number, which would claim other files. They can still be opened
        explicitly with ``format="dicom"``.
        """
        if len(head) >= 132 and head[128:132] == b"DICM":
            return True
        return head[:4] == b"DICM"

    def open(self, src: Any):
        from ._dicom import DicomFile
        return DicomFile(src)

    def decode(self, src: Any, **opts) -> np.ndarray:
        """Decode the image.

        ``frame=i`` decodes one frame instead of the series, which for
        native pixel data is a single read at a known offset.
        ``rescale=True`` applies Rescale Slope and Intercept, turning
        stored values into the real units a CT is measured in.
        """
        frame = opts.pop("frame", None)
        out = opts.pop("out", None)
        rescale = bool(opts.pop("rescale", False))
        with self.open(src) as r:
            arr = r.frame(int(frame)) if frame is not None \
                else r.asarray(rescale=rescale)
        if out is not None:
            out[...] = arr
            return out
        return arr

    def encode(self, data: Any, *, dest=None, **opts) -> bytes | None:
        raise NotImplementedError(
            "dicom: writing DICOM files is not supported; this codec reads "
            "them and routes the pixel data to the transfer syntax's codec")


__all__ = ["DicomCodec"]
