"""DeflateCodec — Codec adapter wrapping the native _deflate extension."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .core.codec import Codec
from .core.buffers import encoded_output, native_decoded
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs

_zlib_encode, _zlib_decode, _zlib_check_signature, _HAVE_BACKEND = import_or_stubs(
    "opencodecs.codecs._deflate",
    "encode", "decode", "check_signature",
)

# Optional ISA-L backend — x86_64-only. Measured against libdeflate on
# 1 MB natural-image bytes (Kodak photo, 64-core x86_64):
#
#   encode: 4.10x FASTER  (ISA-L 3.55 ms vs libdeflate 14.56 ms)
#   decode: 0.83x slower  (ISA-L 2.98 ms vs libdeflate 2.48 ms)
#   output size: 19% bigger (ISA-L 926 KB vs libdeflate 779 KB)
#
# Not strictly Pareto-better than libdeflate (bigger output + slower
# decode), so it's *opt-in* via ``backend="isal"`` on encode/decode
# rather than the new default. Pick it when you have an
# encode-bound batch job and storage cost matters less than throughput.
_isal_encode, _isal_decode, _isal_check_signature, _HAVE_ISAL = import_or_stubs(
    "opencodecs.codecs._isal",
    "encode", "decode", "check_signature",
)


class DeflateCodec(Codec):
    """Native zlib / deflate codec (matches imagecodecs.deflate_encode/decode).

    The default stream is zlib (RFC 1950). ``raw=True`` on encode and
    decode selects a bare DEFLATE stream (RFC 1951), exactly as
    imagecodecs's ``raw=`` does.
    """

    name = "deflate"
    file_extensions = (".zlib",)
    # ``zlibng`` is exposed for ic-compatible callers (tifffile / zarr
    # filter-chains that name backends explicitly). The codec is the
    # same wire format regardless — we already link zlib-ng-compat
    # when it's available (see _deflate.pyx's backend probe).
    aliases = ("zlib", "zlibng")

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8,)
    supports_color = False

    def signature(self, head: bytes) -> bool:
        return _zlib_check_signature(head)

    def encode(self, data: Any, *, dest=None, level: int | None = None,
               raw: bool = False,
               backend: str | None = None, out=None) -> bytes | None:
        if isinstance(data, np.ndarray):
            data = data.tobytes()
        use_isal = backend is not None and backend.lower() in ("isal", "igzip")
        if use_isal and raw:
            raise ValueError(
                "deflate encode: raw=True is not available with "
                "backend='isal', which writes zlib streams only")
        if use_isal:
            if not _HAVE_ISAL:
                raise RuntimeError(
                    "deflate encode: backend='isal' requested but the "
                    "ISA-L extension was not built (Linux x86_64 only — "
                    "install libisal-dev)"
                )
            # ISA-L compression levels top out at 3; clamp so a caller
            # asking for zlib level 6 gets ISA-L's best (3) without
            # raising.
            isal_level = None if level is None else min(3, max(0, int(level)))
            compressed = _isal_encode(data, level=isal_level)
        else:
            compressed = _zlib_encode(data, level=level, raw=bool(raw))
        return encoded_output(compressed, out, self.name, dest)

    def decode(self, src: Any, *, raw: bool = False,
               backend: str | None = None,
               out=None) -> bytes | memoryview:
        use_isal = backend is not None and backend.lower() in ("isal", "igzip")
        if use_isal and raw:
            raise ValueError(
                "deflate decode: raw=True is not available with "
                "backend='isal', which reads zlib streams only")
        if use_isal and not _HAVE_ISAL:
            raise RuntimeError(
                "deflate decode: backend='isal' requested but the "
                "ISA-L extension was not built (Linux x86_64 only)"
            )
        data = _read_src(src)
        if use_isal:
            return native_decoded(_isal_decode, data, out)
        return native_decoded(_zlib_decode, data, out, raw=bool(raw))



__all__ = ["DeflateCodec"]
