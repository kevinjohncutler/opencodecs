"""OIR (Olympus FluoView Newer) codec stub.

OIR is the successor to OIB in Olympus FluoView / Evident software.
Each file starts with the 16-byte ASCII signature
``OLYMPUSRAWFORMAT`` followed by an undocumented proprietary
binary container with embedded compressed frames.

We expose this codec ONLY for format detection — ``oc.read``,
``oc.list_codecs``, and signature dispatch all work. Calling
``decode()`` or ``open()`` raises ``NotImplementedError`` with a
clear message pointing at bioformats (the only public reader)
until a native parser lands.

Adding a native OIR parser is a separate project: the format isn't
documented publicly, so it requires reverse-engineering from real
files (and ideally cross-checking against bioformats' OirReader).
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec, Reader


_OIR_MAGIC = b"OLYMPUSRAWFORMAT"


class OirCodec(Codec):
    """Olympus OIR — format detection only (no decoder yet).

    Files start with the 16-byte ASCII signature ``OLYMPUSRAWFORMAT``;
    the rest of the container is undocumented. Signature detection
    works; decode raises NotImplementedError pointing at bioformats.
    """

    name = "oir"
    file_extensions = (".oir",)
    aliases = ()

    has_native = False
    has_delegate = False
    can_encode = False
    can_decode = False
    multi_frame = True
    chunked = True
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8, np.uint16)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        return len(head) >= 16 and head[:16] == _OIR_MAGIC

    def decode(self, src: Any, **opts) -> np.ndarray:
        raise NotImplementedError(
            "OIR: Olympus FluoView Newer format. The format is "
            "OLYMPUSRAWFORMAT-prefixed but the binary container is "
            "undocumented. No native parser yet — use bioformats "
            "(via python-bioformats / scyjava) for the time being. "
            "Tracking issue: opencodecs#future-oir-native.")

    def open(self, src: Any, **opts) -> Reader:
        raise NotImplementedError(
            "OIR: Olympus FluoView Newer — see OirCodec.decode().")


__all__ = ["OirCodec"]
