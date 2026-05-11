"""Public Python API for the streaming JPEG XL codec.

This module re-exports the cdef classes from opencodecs.codecs._jxl plus a
small set of convenience helpers (`open`, `read`, `write`, `iter_frames`).

Examples
--------
Decode all frames at once::

    import opencodecs.jxl as jxl
    arr = jxl.read("frame.jxl")            # ndarray

Stream frames one at a time (no full-stack materialization)::

    with jxl.open("stack.jxl") as r:
        print(r.shape, r.dtype, r.color)
        for frame in r.iter_frames():
            ...

Encode with HDR color (BT.2100 PQ)::

    jxl.write("out.jxl", arr, color="rec2020-pq", lossless=False, distance=1.0)

Encode an animation/multi-frame stack incrementally::

    with jxl.JxlWriter("stack.jxl", animation=True, lossless=True) as w:
        for plane in stack:
            w.write_frame(plane)
"""

from __future__ import annotations

from typing import Any, Iterator

import numpy as np

from .core.color import ColorSpec, parse_color

# Backend (libjxl Cython extension) is optional. If it didn't build —
# typically Windows / fresh Linux without libjxl-dev — the module still
# imports cleanly so `import opencodecs` works; calling any JXL function
# raises a clear error rather than ImportError-on-import.
try:
    from .codecs._jxl import (
        JxlReader,
        JxlWriter,
        encode,
        decode,
        check_signature,
        libjxl_version,
    )
    _HAVE_BACKEND = True
except ImportError as _exc:  # pragma: no cover - libjxl-missing stub; tested via import_or_stubs
    _HAVE_BACKEND = False
    _IMPORT_ERROR = _exc

    def _missing(*_a, **_kw):
        raise ImportError(
            "opencodecs.jxl backend (libjxl Cython extension) is not "
            f"available: {_IMPORT_ERROR}. Build with libjxl present "
            "(see INSTALL.md)."
        )

    JxlReader = JxlWriter = None  # type: ignore[assignment]
    encode = decode = check_signature = libjxl_version = _missing  # type: ignore[assignment]


__all__ = [
    "JxlReader",
    "JxlWriter",
    "encode",
    "decode",
    "check_signature",
    "libjxl_version",
    "open",
    "read",
    "write",
    "iter_frames",
    "ColorSpec",
    "parse_color",
]


# Provide an `open(path)` -> JxlReader for the streaming reader pattern.
def open(  # noqa: A001  (shadows builtin intentionally for the public API)
    src: Any,
    *,
    numthreads: int | None = None,
    keep_orientation: bool = False,
    coalesce: bool = True,
    parse_color: bool = True,
    streaming: bool = False,
) -> JxlReader:
    """Open a JPEG XL stream and parse the header.

    Parameters
    ----------
    src : str | os.PathLike | bytes | bytearray | memoryview | file-like
        The input stream. Path / file-like is read entirely into memory by
        default; pass ``streaming=True`` for the bg-thread chunked path.
    numthreads : int, optional
        Number of decoder threads. Default: libjxl's default (CPU count).
    keep_orientation : bool, default False
        If True, ignore the EXIF/JXL orientation tag.
    coalesce : bool, default True
        If True (default), animations are returned as composited frames.
    parse_color : bool, default True
        Subscribe to JXL_DEC_COLOR_ENCODING so .color (and lazy .icc_profile)
        are populated. Set False to skip — saves ~2x decode time on Linux
        if you only want pixels (matches imagecodecs's behavior).
    streaming : bool, default False
        If True and the source is a path / file-like with size > 4 MiB, use
        a background-thread chunked reader so file I/O can overlap with
        libjxl's decode work. Off by default because for typical NAS+APFS
        setups the kernel's prefetch on a single big ``read()`` is faster
        than the bg-thread chunked-read pipeline. Useful for very-slow
        storage or files larger than RAM where slurp would OOM.

    Returns
    -------
    JxlReader
        Use as a context manager. After construction, .shape, .dtype, .color
        are populated. Frames stream through .iter_frames() / iter(reader)
        / .read().
    """
    return JxlReader(
        src,
        numthreads=numthreads,
        keep_orientation=keep_orientation,
        coalesce=coalesce,
        parse_color=parse_color,
        streaming=streaming,
    )


def open_http(
    url: str,
    *,
    timeout: float = 60.0,
    headers: dict[str, str] | None = None,
    **kwargs,
) -> JxlReader:
    """Open a remote JPEG XL stream via HTTPS.

    libjxl wants the whole stream up front, so this issues a single
    GET and passes the bytes through to :class:`JxlReader`. For
    container formats with tile/strip random-access (TIFF, NDTiff)
    use the per-format ``HTTPDataSource`` path which only fetches
    the tiles the caller actually decodes.

    Any kwargs accepted by :func:`open` are forwarded.
    """
    from ._tiff_http import http_fetch_all

    data = http_fetch_all(url, timeout=timeout, headers=headers)
    return JxlReader(data, **kwargs)


def read(src: Any, **kwargs) -> np.ndarray:
    """Decode a JPEG XL source to a single ndarray (multi-frame -> stacked).

    Defaults to ``parse_color=False`` for the fast decode path. Pass
    ``parse_color=True`` to also populate color metadata.
    """
    return decode(src, **kwargs)


def write(
    dest: Any,
    arr: np.ndarray,
    *,
    color: str | ColorSpec | None = None,
    lossless: bool | None = None,
    quality: float | None = None,
    distance: float | None = None,
    effort: int = 5,
    decoding_speed: int = 0,
    numthreads: int | None = None,
    animation: bool = False,
    container: bool = False,
    intensity_target: float | None = None,
    icc_profile: bytes | None = None,
) -> bytes | None:
    """Encode `arr` as JPEG XL.

    `dest` may be a path, file-like, or None (in-memory: returns bytes).

    `intensity_target` (nits): brightness anchor written into the JXL
    basic-info. Required for linear-light HDR files to be recognized as
    HDR — set to the nit level corresponding to the peak encoded value
    (e.g. 1200 for an scRGB-style file where 1.0 = SDR diffuse white at
    100 nits and the file holds up to 12x SDR). Default (None) uses
    libjxl's per-transfer default.
    """
    return encode(
        arr,
        dest=dest,
        color=color,
        lossless=lossless,
        quality=quality,
        distance=distance,
        effort=effort,
        decoding_speed=decoding_speed,
        numthreads=numthreads,
        animation=animation,
        container=container,
        intensity_target=intensity_target,
        icc_profile=icc_profile,
    )


def iter_frames(
    src: Any,
    *,
    numthreads: int | None = None,
    keep_orientation: bool = False,
    coalesce: bool = True,
    parse_color: bool = False,
    streaming: bool = False,
) -> Iterator[np.ndarray]:
    """Yield decoded frames one at a time from `src`.

    Equivalent to ``open(src).iter_frames()`` but closes the reader when the
    generator is exhausted. Defaults to ``parse_color=False`` for speed —
    if you also want color/icc metadata, use ``open(...)`` directly.
    """
    reader = JxlReader(
        src,
        numthreads=numthreads,
        keep_orientation=keep_orientation,
        coalesce=coalesce,
        parse_color=parse_color,
        streaming=streaming,
    )
    try:
        yield from reader.iter_frames()
    finally:
        reader.close()
