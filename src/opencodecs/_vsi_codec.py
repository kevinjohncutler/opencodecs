"""VSI (Olympus CellSens) codec — TIFF-backed index reader.

VSI is a multi-file format produced by Olympus CellSens / Evident
microscope software. The top-level ``foo.vsi`` is a **TIFF**
(``II*\\0`` magic) containing:

  * A thumbnail / overview image (typically a 256x256 RGB jpg-in-TIFF)
  * Olympus-specific metadata in private IFD tags
  * Pointers into the sibling ``_foo_/stack<N>/frame_t.ets`` directory
    that holds the full-resolution pyramid data

Our native TIFF reader already handles the top-level ``.vsi`` file
end-to-end — it returns the thumbnail and exposes the IFD tags.
What we DON'T yet have is the ``.ets`` parser for full-resolution
tile data. That's a future native upgrade.

This codec wires VSI into the registry so:

  * ``oc.read("foo.vsi")`` returns the thumbnail (uses TIFF reader)
  * ``oc.get_codec("vsi").open(...)`` returns a TiffStream
  * ``codec.signature(head)`` detects VSI by extension hint + TIFF magic
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec, Reader


class VsiCodec(Codec):
    """Olympus VSI (CellSens virtual slide) — delegates to TIFF.

    The top-level VSI is a TIFF; full-resolution data lives in a
    sibling ``_NAME_/stackN/frame_t.ets`` companion tree we don't
    yet read natively. For typical "what's in this slide?" use cases
    (thumbnail + metadata) this still works.
    """

    name = "vsi"
    file_extensions = (".vsi",)
    aliases = ()

    has_native = True   # TIFF reader handles the top-level container
    has_delegate = False
    can_encode = False
    can_decode = True
    multi_frame = True
    chunked = True
    streaming_decode = True
    parallel_decode = False

    supported_dtypes = (
        np.uint8, np.uint16, np.float32,
    )
    supports_color = True

    def signature(self, head: bytes) -> bool:
        """VSI files start with the standard TIFF magic. There's no
        VSI-specific magic in the header — the format is detected by
        the .vsi extension; we accept any TIFF-magic bytes here so
        codec_for_bytes() can still route a .vsi blob correctly."""
        return len(head) >= 4 and head[:4] in (b"II*\x00", b"MM\x00*")

    def decode(self, src: Any, **opts) -> np.ndarray:
        with self.open(src, **opts) as reader:
            return reader.read()

    def open(self, src: Any, **opts) -> Reader:
        # The .vsi top-level container is a TIFF. Delegate to our
        # native TIFF reader, which handles JPEG-in-TIFF compression
        # (the typical thumbnail encoding) and exposes the pages
        # iterator for any extra IFDs.
        from ._tiff_codec import TiffStream
        return TiffStream(src, **opts)


__all__ = ["VsiCodec"]
