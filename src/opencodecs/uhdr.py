"""Public Python API for the Ultra-HDR / ISO 21496-1 codec.

Encodes a single linear-light HDR raster (Display-P3 or Rec.2020) into
an Ultra-HDR JPEG (or HEIF / AVIF) containing an SDR base + a per-pixel
gain map. HDR-aware decoders (Chrome 116+, Safari 26+, Apple Photos,
libuhdr) composite to display headroom; older decoders see just the
SDR base -- which is the correctly-tonemapped fallback we want for
cross-browser HDR display.

Two encoders are exposed:

* :func:`encode` — wraps libuhdr's full pipeline (encode HDR raster
  directly). Slower but always tracks upstream.
* :func:`encode_native` — fused-Cython fast path: peak-normalize +
  sRGB OETF SDR base, gain-map kernel, JPEG encode the two layers in
  parallel, then hand the pre-encoded JPEGs to libuhdr's api-4 for
  container assembly. ~3-4× faster than :func:`encode` on a 2k² float
  HDR (M-series Mac: 30-40 ms vs 130 ms).

The native helpers (``compute_gain_map_u8`` / ``compute_sdr_base_u8``)
are also exported from :mod:`opencodecs.codecs._uhdr` if callers want
to compose the SDR base / gain-map / metadata individually and pass
them to :func:`encode_assembled`.

Examples
--------
Encode a linear-Display-P3 float HDR array as Ultra-HDR JPEG::

    import numpy as np
    import opencodecs
    # arr: HxWx3 or HxWx4 float, linear-light, 1.0 == 203 nits (SDR white).
    data = opencodecs.uhdr.encode_native(arr, gamut='display-p3')
    with open('out.jpg', 'wb') as f:
        f.write(data)

Decode back to fp16 HDR pixels::

    with open('out.jpg', 'rb') as f:
        info = opencodecs.uhdr.decode(f.read())
    hdr = info['hdr_fp16']   # (H, W, 4) fp16 RGBA, linear-light
"""

from __future__ import annotations

import numpy as np

# Backend is optional: opencodecs still imports when libuhdr / the Cython
# extension isn't available; calling any uhdr function then raises a
# clear error instead of an ImportError at import time.
try:
    from .codecs._uhdr import (
        encode,
        encode_assembled,
        decode,
        is_uhdr,
        libuhdr_version,
        UhdrError,
        compute_gain_map_u8 as _cython_gain_map,
        compute_sdr_base_u8 as _cython_sdr_base,
        apply_gainmap_fp32 as _cython_apply_gainmap,
        extract_layers as _cython_extract_layers,
        _srgb_eotf_lut_np,
    )
    _HAVE_BACKEND = True
except ImportError as _exc:  # pragma: no cover
    _HAVE_BACKEND = False
    _IMPORT_ERROR = _exc

    def _missing(*_a, **_kw):
        raise ImportError(
            "opencodecs.uhdr backend (libultrahdr Cython extension) is "
            f"not available: {_IMPORT_ERROR}. Install libultrahdr "
            "(macOS: `brew install libultrahdr`; Linux: build from "
            "https://github.com/google/libultrahdr) and reinstall opencodecs."
        )

    encode = encode_assembled = decode = is_uhdr = libuhdr_version = _missing  # type: ignore[assignment]
    _cython_gain_map = _cython_sdr_base = _srgb_eotf_lut_np = _missing  # type: ignore[assignment]
    _cython_apply_gainmap = _cython_extract_layers = _missing  # type: ignore[assignment]
    UhdrError = type("UhdrError", (Exception,), {})  # type: ignore[assignment]


__all__ = [
    "encode",
    "encode_native",
    "encode_assembled",
    "encode_to",
    "decode",
    "decode_native",
    "is_uhdr",
    "libuhdr_version",
    "UhdrError",
    "read",
    "write",
]


def encode_native(hdr, sdr=None, *,
                   gamut='display-p3', sdr_white_nits=1600.0,
                   quality=95, max_content_boost=None,
                   min_content_boost=1.0, gamma=1.0, parallel=True):
    """Fused-Cython fast-path Ultra-HDR encoder.

    Computes SDR base + gain map in fused Cython kernels, JPEG-encodes
    the two layers in parallel via :mod:`imagecodecs`, then hands the
    pre-encoded JPEGs to libuhdr's api-4 (:func:`encode_assembled`) for
    container assembly. ~3-4× faster than :func:`encode` on a 2k² float
    HDR raster.

    Parameters
    ----------
    hdr : ndarray
        ``(H, W, 3)`` float linear-light. ``1.0`` = SDR-reference white
        (203 nits). Values > 1.0 encode HDR headroom.
    sdr : ndarray, optional
        Pre-computed ``(H, W, 3)`` uint8 SDR base. ``None`` (default)
        runs :func:`compute_sdr_base_u8` to peak-normalize the HDR
        raster.
    gamut : {'display-p3', 'rec2020', 'srgb'}
        Color gamut tag written into the libuhdr container metadata.
    sdr_white_nits : float
        Peak luminance to map HDR ``1.0`` to in the gain-map metadata.
        ``1600.0`` is a reasonable default for content lit up to 8×
        SDR white.
    quality : int
        JPEG quality (1-100) for both the SDR base and the gain map.
    max_content_boost, min_content_boost, gamma
        Gain-map metadata. See :func:`compute_gain_map_u8`.
    parallel : bool
        Run the SDR JPEG encode, gain-map compute, and gain-map JPEG
        encode in parallel threads. Saves ~30-40% wall-clock on
        2-core+ machines.

    Returns
    -------
    bytes
        Ultra-HDR JPEG container (ISO 21496-1).
    """
    if not _HAVE_BACKEND:
        _missing()
    import imagecodecs
    import concurrent.futures as _cf

    hdr_arr = np.ascontiguousarray(hdr, dtype=np.float32)
    if hdr_arr.ndim != 3 or hdr_arr.shape[2] != 3:
        raise ValueError(
            f"hdr must be (H, W, 3); got {tuple(hdr_arr.shape)}")

    if sdr is None:
        peak = float(hdr_arr.max()) if hdr_arr.size else 1.0
        if peak <= 0.0:
            peak = 1.0
        sdr_arr = _cython_sdr_base(hdr_arr, peak=peak)
        if max_content_boost is None:
            mcb = max(peak * sdr_white_nits / 203.0,
                      min_content_boost + 1e-6)
        else:
            mcb = max_content_boost
    else:
        sdr_arr = np.ascontiguousarray(sdr, dtype=np.uint8)
        if sdr_arr.shape != hdr_arr.shape:
            raise ValueError(
                f"sdr shape {tuple(sdr_arr.shape)} must match hdr "
                f"{tuple(hdr_arr.shape)}")
        mcb = max_content_boost

    if parallel:
        ex = _cf.ThreadPoolExecutor(max_workers=3)
        try:
            base_fut = ex.submit(imagecodecs.jpeg_encode, sdr_arr, quality)
            gain_fut = ex.submit(_cython_gain_map, hdr_arr, sdr_arr,
                                 sdr_white_nits=sdr_white_nits,
                                 max_content_boost=mcb,
                                 min_content_boost=min_content_boost,
                                 gamma=gamma)
            gain_u8, metadata = gain_fut.result()
            gainmap_fut = ex.submit(imagecodecs.jpeg_encode, gain_u8, quality)
            base_jpeg = base_fut.result()
            gainmap_jpeg = gainmap_fut.result()
        finally:
            ex.shutdown(wait=False)
    else:
        gain_u8, metadata = _cython_gain_map(
            hdr_arr, sdr_arr,
            sdr_white_nits=sdr_white_nits,
            max_content_boost=mcb,
            min_content_boost=min_content_boost,
            gamma=gamma,
        )
        base_jpeg = imagecodecs.jpeg_encode(sdr_arr, quality)
        gainmap_jpeg = imagecodecs.jpeg_encode(gain_u8, quality)

    return encode_assembled(
        base_jpeg=base_jpeg,
        gainmap_jpeg=gainmap_jpeg,
        metadata=metadata,
        gamut=gamut,
    )


def decode_native(data, *, parallel=True, dtype=None,
                  display_boost=None) -> dict:
    """Fused-Cython fast-path Ultra-HDR decoder.

    Uses libuhdr to extract the compressed SDR base + gain-map JPEGs +
    metadata (no pixel decode), then decodes both JPEGs in parallel via
    :mod:`imagecodecs` (libjpeg-turbo SIMD, GIL released) and applies the
    gain map to fp32 HDR in a Cython kernel that uses the same sRGB EOTF
    LUT + polynomial exp2 the encoder uses. Output is cast to ``dtype``
    (default ``float16`` to match :func:`decode`'s convention).

    Parameters
    ----------
    data : bytes-like
        Encoded ISO 21496-1 stream (Ultra-HDR JPEG).
    parallel : bool
        Decode the SDR base JPEG and the gain-map JPEG concurrently in
        a 2-worker thread pool. Both decoders release the GIL.
    dtype : numpy dtype, optional
        Output HDR dtype. ``None`` (default) → ``float16`` to match
        :func:`decode`'s ``hdr_fp16`` convention. Pass ``np.float32``
        if you want to skip the fp16 cast (saves ~2 ms on a 2k² raster
        and avoids the ~5e-4 fp16-quantisation error).
    display_boost : float, optional
        Target display headroom. ``None`` (default) requests the
        encoded raster's full HDR (``hdr_capacity_max``). Pass 1.0 to
        match libuhdr's default decode — the SDR-equivalent.

    Returns
    -------
    dict with keys ``hdr``, ``sdr_u8``, ``gainmap_u8``,
    ``gainmap_metadata``, ``width``, ``height``. ``hdr`` is a
    ``(H, W, 3)`` HDR raster in ``dtype`` (linear-light;
    1.0 = 203 nits / SDR white).
    """
    if not _HAVE_BACKEND:
        _missing()
    import imagecodecs
    import concurrent.futures as _cf

    info = _cython_extract_layers(data)
    base_jpeg = info["base_jpeg"]
    gainmap_jpeg = info["gainmap_jpeg"]
    metadata = info["gainmap_metadata"]

    if parallel:
        ex = _cf.ThreadPoolExecutor(max_workers=2)
        try:
            sdr_fut = ex.submit(imagecodecs.jpeg_decode, base_jpeg)
            gain_fut = ex.submit(imagecodecs.jpeg_decode, gainmap_jpeg)
            sdr_u8 = sdr_fut.result()
            gain_u8 = gain_fut.result()
        finally:
            ex.shutdown(wait=False)
    else:
        sdr_u8 = imagecodecs.jpeg_decode(base_jpeg)
        gain_u8 = imagecodecs.jpeg_decode(gainmap_jpeg)

    # imagecodecs returns (H, W) grayscale or (H, W, 3) RGB. Normalise
    # to a 3-D array with the channel axis last; the kernel expects
    # that shape.
    if sdr_u8.ndim == 2:
        sdr_u8 = sdr_u8[:, :, None]
    if gain_u8.ndim == 2:
        gain_u8 = gain_u8[:, :, None]

    # If the gain-map is smaller than the SDR base (libuhdr's
    # scale-factor saves bytes on the gain), upscale to match. Nearest-
    # neighbour is enough — the original spec lets decoders interpolate
    # any way they like and the gain map is heavily band-limited.
    sH, sW = int(sdr_u8.shape[0]), int(sdr_u8.shape[1])
    gH, gW = int(gain_u8.shape[0]), int(gain_u8.shape[1])
    if (gH, gW) != (sH, sW):
        ys = (np.arange(sH) * gH // sH).astype(np.int64)
        xs = (np.arange(sW) * gW // sW).astype(np.int64)
        gain_u8 = gain_u8[ys[:, None], xs[None, :]]

    hdr_f32 = _cython_apply_gainmap(
        sdr_u8, gain_u8, metadata, display_boost=display_boost,
    )
    if dtype is None:
        out = hdr_f32.astype(np.float16)
    else:
        out = hdr_f32.astype(np.dtype(dtype), copy=False)
    return {
        "hdr": out,
        "sdr_u8": sdr_u8,
        "gainmap_u8": gain_u8,
        "gainmap_metadata": metadata,
        "width": info["width"],
        "height": info["height"],
    }


def encode_to(fp, hdr, **kwargs) -> int:
    """Streaming variant of :func:`encode_native` — write Ultra-HDR
    bytes directly to a file-like ``fp`` (anything with a ``write(bytes)``
    method: an open file, ``io.BytesIO``, an HTTP upload streamer, …).
    Returns the number of bytes written.

    Forward-compatible alias: for now ``encode_to`` is essentially
    ``fp.write(encode_native(hdr, **kwargs))`` — the libuhdr api-4 path
    we use for container assembly doesn't expose a streaming writer
    today. The function exists so callers can adopt the streaming API
    now and pick up any future libuhdr streaming write-out without
    changing their code.
    """
    data = encode_native(hdr, **kwargs)
    fp.write(data)
    return len(data)


def read(path, **kwargs) -> dict:
    """Decode an Ultra-HDR file from disk. Returns the same dict as
    :func:`decode`."""
    with open(path, "rb") as f:
        return decode(f.read(), **kwargs)


def write(path, hdr, **kwargs) -> None:
    """Encode an Ultra-HDR file to disk. ``hdr`` must be an HxWx3 or
    HxWx4 float array (linear-light, 1.0 == SDR-reference white = 203
    nits). All other kwargs forwarded to :func:`encode`."""
    data = encode(hdr, **kwargs)
    with open(path, "wb") as f:
        f.write(data)
