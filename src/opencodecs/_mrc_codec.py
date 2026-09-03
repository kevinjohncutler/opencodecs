"""MrcCodec — Codec adapter for the native MRC / CCP4 map reader.

MRC is a container, not a compression codec, so this exposes decode and
the ``open(src)`` reader contract and does not encode. Same shape as
``FitsCodec``: the format's own byte layout is the "compression".
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec


class MrcCodec(Codec):
    """MRC2014 / CCP4 map reader (cryo-EM volumes, EMDB density maps)."""

    name = "mrc"
    aliases = ("ccp4", "map", "mrcs")
    file_extensions = (".mrc", ".mrcs", ".map", ".ccp4", ".rec")

    has_native = True
    has_delegate = False
    can_encode = False
    can_decode = True
    multi_frame = True
    streaming_decode = True
    parallel_decode = False

    supported_dtypes = (
        np.int8, np.int16, np.uint16, np.float16, np.float32, np.complex64,
    )
    supports_color = False

    def signature(self, head: bytes) -> bool:
        """Match on the MAP identifier at byte 208.

        Deliberately strict. Sniffing MRC by plausible dimensions would
        claim any file whose first four bytes happen to be a small
        integer, which is most binary formats. Pre-2014 files without the
        identifier can still be opened explicitly with ``format="mrc"``.
        """
        return len(head) >= 212 and head[208:212] == b"MAP "

    def open(self, src: Any):
        from ._mrc import MrcStream
        return MrcStream(src)

    def decode(self, src: Any, **opts) -> np.ndarray:
        """Read the whole map.

        ``plane=i`` reads one z-section instead. ``canonical=True``
        reorients a map whose MAPC/MAPR/MAPS is not (1, 2, 3) so the
        result really is (z, y, x).
        """
        plane = opts.pop("plane", None)
        out = opts.pop("out", None)
        canonical = opts.pop("canonical", False)
        with self.open(src) as r:
            if plane is not None:
                arr = r.plane(int(plane))
                if out is not None:
                    out[...] = arr
                    return out
                return arr
            return r.asarray(out=out, canonical=canonical)

    def encode(self, data: Any, *, dest=None, **opts):
        raise NotImplementedError(
            "mrc: encoding is not implemented; MRC is a container format "
            "and opencodecs reads it rather than writing it")


__all__ = ["MrcCodec"]
