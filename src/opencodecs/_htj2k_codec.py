"""Htj2kCodec — HTJ2K (JPEG-2000 Part 15) via OpenJPH.

HTJ2K is the high-throughput JPEG-2000 codestream defined in ISO/IEC
15444-15. It targets 10-30× faster encode/decode than classic JPEG-2000
at near-identical compression ratios, while staying within the
JPEG-2000 ecosystem (same wavelet basis, same image model).

Used in DICOM medical imaging (transfer syntax 1.2.840.10008.1.2.4.201)
and increasingly in the broadcast / cinema pipeline.

Modes (the meaning of ``level`` is imagecodecs.htj2k_encode's)::

    level=None     # reversible, mathematically lossless (default)
    level=0.01     # irreversible, quantization step below 1
    level=75       # irreversible, quality factor from 1 to 100

RGB and RGBA input use the component transform (RCT/ICT) by default,
as imagecodecs and OpenJPH's own ojph_compress do; ``rgb=False`` turns
it off.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .core.codec import Codec
from .core._io_helpers import read_src as _read_src, write_dest as _write_dest
from .core._optional_backend import import_or_stubs
from .backends import select as _select_backend

(
    _htj2k_encode, _htj2k_decode, _HAVE_BACKEND,
) = import_or_stubs(
    "opencodecs.codecs._openjph",
    "encode", "decode",
)


class Htj2kCodec(Codec):
    """HTJ2K (JPEG-2000 Part-15) via OpenJPH."""

    name = "htj2k"
    aliases = ("openjph", "jph", "j2c")
    file_extensions = (".j2c", ".jph")

    has_native = True
    has_delegate = False
    can_encode = True
    can_decode = True
    multi_frame = False
    streaming_decode = False
    parallel_decode = False

    supported_dtypes = (np.uint8, np.uint16, np.int8, np.int16,
                        np.uint32, np.int32, np.float32)
    supports_color = True

    def signature(self, head: bytes) -> bool:
        # HTJ2K raw codestream starts with SOC marker 0xFF4F + SIZ
        # marker 0xFF51. JP2-wrapped HTJ2K starts with the JP2
        # signature box (\x00\x00\x00\x0Cjp2 \r\n\x87\n) — same as
        # classic JPEG-2000. Match either.
        if len(head) < 4:
            return False
        if head[0] == 0xFF and head[1] == 0x4F and \
           head[2] == 0xFF and head[3] == 0x51:
            return True
        return len(head) >= 12 and bytes(head[:12]) == (
            b"\x00\x00\x00\x0Cjp2 \r\n\x87\n"
        )

    # imagecodecs.htj2k_encode's keywords, all implemented by the native
    # encoder. Anything else raises rather than being dropped.
    _ENCODE_OPTIONS = ("rgb", "planar", "tile", "resolutions", "reversible",
                       "tlm", "tilepart", "block_size", "prog_order",
                       "profile", "num_decomp")

    def encode(self, data: Any, *, dest=None,
               level: float | None = None, backend: str | None = None,
               **opts) -> bytes | None:
        """Encode as HTJ2K; keywords follow ``imagecodecs.htj2k_encode``.

        ``level=None`` is lossless. See
        :func:`opencodecs.codecs._openjph.encode` for every option.

        ``backend="nvimgcodec"`` encodes on an NVIDIA GPU (opt-in; see
        :mod:`opencodecs.backends`): uint8, uint16 or int16, lossless or
        a quality ``level`` from 1 to 100, from a numpy or CuPy array.
        """
        unknown = sorted(set(opts) - set(self._ENCODE_OPTIONS))
        if unknown:
            raise TypeError(f"htj2k encode: unsupported options {unknown}")
        hardware = _select_backend(backend, self.name, "encode")
        if hardware is not None:
            return _write_dest(hardware.encode_htj2k(data, level, **opts), dest)
        if not isinstance(data, np.ndarray):
            data = np.asarray(data)
        out = _htj2k_encode(data, level, **opts)
        return _write_dest(out, dest)

    def decode(self, src: Any, *, reduce: int = 0,
               ignore_unsupported: bool = False,
               planar: bool | None = None,
               skipres: Any = None,
               resilient: bool = False, out=None,
               numthreads: int | None = None,
               backend: str | None = None,
               **opts) -> np.ndarray:
        """Decode HTJ2K; keywords follow ``imagecodecs.htj2k_decode``.

        ``out=`` receives the image in place, and ``numthreads`` bounds
        the threads a codestream of several tiles decodes on; see
        :func:`opencodecs.codecs._openjph.decode`.

        ``backend="nvimgcodec"`` decodes on an NVIDIA GPU (opt-in; see
        :mod:`opencodecs.backends`): same dtype, shape and pixels.
        ``out=`` may then also be a CuPy array or one from
        :func:`opencodecs.backends.pinned_empty`; ``reduce``, ``skipres``
        and ``resilient`` are not available there.
        """
        if opts:
            raise TypeError(
                f"htj2k decode: unsupported options {sorted(opts)}")
        hardware = _select_backend(backend, self.name, "decode")
        if hardware is not None:
            if reduce or skipres is not None or resilient:
                raise ValueError(
                    "htj2k decode: reduce=, skipres= and resilient= are not "
                    "available with backend='nvimgcodec'")
            return hardware.decode(self.name, _read_src(src), out=out,
                                   planar=planar)
        return _htj2k_decode(_read_src(src), reduce=reduce,
                             ignore_unsupported=ignore_unsupported,
                             planar=planar, skipres=skipres,
                             resilient=resilient, out=out,
                             numthreads=numthreads)


__all__ = ["Htj2kCodec"]
