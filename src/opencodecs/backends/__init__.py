"""Opt-in hardware backends: NVIDIA nvImageCodec and Apple ImageIO.

A codec never picks one of these by itself. They are chosen per call with
the ``backend=`` keyword, the same keyword deflate uses for ISA-L::

    oc.get_codec("htj2k").decode(blob, backend="nvimgcodec")
    oc.read("photo.heic", backend="imageio")

=================  ==========================  ===========================
``backend=``       codecs                      needs
=================  ==========================  ===========================
``"nvimgcodec"``   jpeg, jpeg2k, htj2k:        an NVIDIA GPU, CuPy and
                   encode and decode           ``nvidia-nvimgcodec``
``"imageio"``      heif: decode, and a lossy   macOS (ImageIO through
                   hardware HEVC encode        ctypes; no extra package)
=================  ==========================  ===========================

``None`` or ``"native"`` is the CPU path, as before.

Why opt-in only: the first nvImageCodec call in a process costs 1 to 1.6 s
(importing CuPy and nvImageCodec, creating the CUDA context, the codec's
own first-call setup), so a script that decodes one image is much slower
with it. Each backend is created once per process, lazily, and reused.

Importing ``opencodecs`` or this package imports none of CuPy,
nvImageCodec or any Apple framework; that happens on the first call that
asks for the backend. Asking for a backend that cannot run raises
:class:`BackendUnavailable`, never a silent fallback to the CPU. See
docs/hardware_backends.md for measurements, and for what each backend
does with input it cannot handle.
"""

from __future__ import annotations

from typing import Any

from ..core.errors import OpenCodecsError

__all__ = ["BackendUnavailable", "available", "pinned_empty", "BACKENDS"]


class BackendUnavailable(OpenCodecsError, ImportError):
    """The requested hardware backend cannot run here.

    Raised when its packages are missing, there is no usable device, or
    the platform is wrong. A subclass of ImportError as well as
    :class:`~opencodecs.core.errors.OpenCodecsError`, so either catches it.
    """


# backend name -> {codec name: operations it serves}
BACKENDS = {
    "nvimgcodec": {"jpeg": ("decode", "encode"),
                   "jpeg2k": ("decode", "encode"),
                   "htj2k": ("decode", "encode")},
    "imageio": {"heif": ("decode", "encode")},
}


def select(backend: Any, codec: str, op: str):
    """The backend module ``backend`` names, or None for the CPU path.

    Raises ValueError for a name this codec does not offer for ``op``.
    Importing the returned module is cheap; the heavy imports happen on
    its first real call.
    """
    if backend is None:
        return None
    name = str(backend).lower()
    if name == "native":
        return None
    if op not in BACKENDS.get(name, {}).get(codec, ()):
        offered = sorted(b for b, codecs in BACKENDS.items()
                         if op in codecs.get(codec, ()))
        choices = ", ".join(repr(c) for c in ["native", *offered])
        raise ValueError(
            f"{codec} {op}: unknown backend={backend!r}; choose None, {choices}")
    if name == "nvimgcodec":
        from . import _nvimgcodec as module
    else:
        from . import _imageio as module
    return module


def available(name: str) -> bool:
    """True if backend ``name`` can run in this process.

    This loads the backend (and so pays its startup cost) the first
    time; afterwards it is free.
    """
    name = str(name).lower()
    if name not in BACKENDS:
        raise ValueError(f"unknown backend {name!r}; one of {sorted(BACKENDS)}")
    module = select(name, next(iter(BACKENDS[name])), "decode")
    try:
        module.load()
    except BackendUnavailable:
        return False
    return True


def pinned_empty(shape, dtype) -> "Any":
    """A numpy array in page-locked (pinned) host memory, for ``out=``.

    The GPU copies decoded pixels into pinned memory faster than into
    ordinary pageable memory (measured 5.7 ms against 9.8 ms for a
    4096 x 4096 uint16 HTJ2K decode), so a loop that decodes many images
    of one shape should allocate its output once with this and pass it
    as ``out=`` to ``decode(..., backend="nvimgcodec")``. Needs CuPy.
    """
    from ._nvimgcodec import pinned_empty as _pinned_empty
    return _pinned_empty(shape, dtype)
