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
        probe,
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
    probe = _missing  # type: ignore[assignment]
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
    "probe",
    "libuhdr_version",
    "UhdrError",
    "read",
    "write",
]


def encode_native(hdr, sdr=None, *,
                   gamut='display-p3', sdr_white_nits=1600.0,
                   quality=95, gain_quality=None, gain_scale=1,
                   lossless=False,
                   max_content_boost=None,
                   min_content_boost=1.0, gamma=1.0, parallel=True,
                   out=None):
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
        JPEG quality (1-100) for the SDR base layer. The base is what
        HDR-unaware viewers see, so it gets the marquee number.
    gain_quality : int, optional
        JPEG quality for the gain-map layer. The gain map is heavily
        band-limited (low-frequency content by construction), so
        running it at a lower quality than the base is essentially
        free visually but cuts ~30-50% off the gain-map bytes.
        ``None`` (default) tracks ``quality``.
    gain_scale : int
        Downsample factor for the gain-map raster, applied before its
        JPEG encode. ``1`` (default) keeps full resolution; ``2``
        halves both axes (quarter-area, ~4× faster gain JPEG encode,
        ~75% smaller gain bytes), ``4`` etc. Gain maps are heavily
        band-limited, so 2-4× downscale is visually negligible on
        natural content. Decoders upscale on apply.
    lossless : bool
        Encode the SDR base layer with libjpeg-turbo's predictive
        lossless mode (PSV=1, 4:4:4, no chroma subsampling). The
        gain map keeps its lossy encode (no perceptual benefit to
        making that lossless). Output is ~3-5× larger than the
        ``quality=95`` baseline-DCT default on natural images.

        Caveats — empirically verified on macOS 15 + HDR display:

        * macOS Preview, Chrome 116+, and any viewer that parses
          the MPF / XMP gain-map block directly will HDR-render the
          output normally.
        * libuhdr's reference ``uhdr_decode`` (and thus
          :func:`decode`) rejects the file — it enforces a strict
          ``JCS_YCbCr`` / ``JCS_GRAYSCALE`` colorspace check that
          lossless-mode JPEG (which writes ``JCS_RGB``) fails.
        * Apple's ``CGImageSource`` ``Headroom`` / ``ISOGainMap``
          aux-data accessors also miss the HDR signal (same root
          cause), even though Preview's actual renderer composites
          fine.
        * :func:`decode_native` reads it cleanly — it uses our own
          gain-application kernel without the colorspace gate.

        Use only when exact SDR-base-pixel preservation is a hard
        requirement (archival, scientific imaging) AND every
        consumer in your pipeline is either Preview / Chrome /
        ``decode_native``.
    max_content_boost, min_content_boost, gamma
        Gain-map metadata. See :func:`compute_gain_map_u8`.
    parallel : bool
        Run the SDR JPEG encode, gain-map compute, and gain-map JPEG
        encode in parallel threads. Saves ~30-40% wall-clock on
        2-core+ machines.
    out : file-like, optional
        If given (anything with ``write(buf)``: open file,
        ``io.BytesIO``, HTTP upload sink), the assembled container
        is streamed directly to ``out`` via a zero-copy memoryview
        over libuhdr's internal output buffer. The function then
        returns ``None`` instead of bytes. Saves one full-output-
        size malloc + memcpy and ~1× output-size peak memory on
        large encodes.

    Returns
    -------
    bytes, or None if ``out`` is given
        Ultra-HDR JPEG container (ISO 21496-1).
    """
    if not _HAVE_BACKEND:
        _missing()
    import imagecodecs
    import concurrent.futures as _cf

    if gain_quality is None:
        gain_quality = quality
    if gain_scale < 1 or (gain_scale & (gain_scale - 1)) != 0:
        raise ValueError(
            f"gain_scale must be a positive power of 2; got {gain_scale}")

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

    def _maybe_downscale_gain(gain_u8):
        # Stride decimation by an integer factor. Gain maps are heavily
        # band-limited by construction (smooth ratio of low-frequency
        # luminance signals), so a nearest-sample stride gives
        # visually-equivalent output to a box average — and is ~20×
        # faster on a 2k² uint8 raster (numpy's .mean() promotes to
        # float64 and runs multiple passes, eating ~100ms; stride
        # decimation is ~5ms). Cuts gain JPEG encode work by scale²
        # and gain bytes proportionally.
        if gain_scale == 1:
            return gain_u8
        return np.ascontiguousarray(gain_u8[::gain_scale, ::gain_scale])

    # Gain map = per-pixel HDR boost multiplier. JPEG default 4:2:0
    # chroma subsampling averages boost values over 2x2 blocks, which
    # halves the effective peak on sparse-bright-pixel content (e.g.
    # fluorescence dye spots) — empirically a 7.84 → 3.67 peak drop on
    # round-trip. ``subsampling='444'`` keeps per-pixel gain fidelity.
    # The base layer is photographic color with low chroma acuity; the
    # default 4:2:0 there is fine.

    # When lossless=True the SDR base goes through our own _jpeg.encode
    # so we can hit the TJPARAM_LOSSLESS path that imagecodecs doesn't
    # expose. The gain map stays lossy; there's no perceptual reason
    # to make it lossless and doing so would balloon the file.
    def _encode_base(arr):
        if lossless:
            from .codecs._jpeg import encode as _oc_jpeg_encode
            return _oc_jpeg_encode(arr, lossless=True)
        return imagecodecs.jpeg_encode(arr, quality)

    if parallel:
        ex = _cf.ThreadPoolExecutor(max_workers=3)
        try:
            base_fut = ex.submit(_encode_base, sdr_arr)
            gain_fut = ex.submit(_cython_gain_map, hdr_arr, sdr_arr,
                                 sdr_white_nits=sdr_white_nits,
                                 max_content_boost=mcb,
                                 min_content_boost=min_content_boost,
                                 gamma=gamma)
            gain_u8, metadata = gain_fut.result()
            gain_u8 = _maybe_downscale_gain(gain_u8)
            gainmap_fut = ex.submit(
                imagecodecs.jpeg_encode, gain_u8, gain_quality,
                subsampling='444')
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
        gain_u8 = _maybe_downscale_gain(gain_u8)
        base_jpeg = _encode_base(sdr_arr)
        gainmap_jpeg = imagecodecs.jpeg_encode(
            gain_u8, gain_quality, subsampling='444')

    return encode_assembled(
        base_jpeg=base_jpeg,
        gainmap_jpeg=gainmap_jpeg,
        metadata=metadata,
        gamut=gamut,
        out=out,
    )


def decode_native(data, *, parallel=True, dtype=None,
                  display_boost=None, scale=None) -> dict:
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
    scale : int / float / (num, denom), optional
        DCT-domain decode-time downsample factor passed to both the
        SDR base and gain-map JPEG decoders (via libjpeg-turbo's
        ``tj3SetScalingFactor``). The supported set is the 16
        rationals N/8 for N ∈ {1, 2, …, 16}. Integer ``N`` is
        interpreted as ``1/N``; floats snap to the closest supported
        ratio; ``(num, denom)`` is taken verbatim. ``None`` (default)
        decodes at full resolution. Useful for thumbnail / preview
        pipelines that want HDR pixels but not the megapixel base.
        See :func:`opencodecs.codecs._jpeg.supported_scaling_factors`
        for the full set.

    Returns
    -------
    dict with keys ``hdr``, ``sdr_u8``, ``gainmap_u8``,
    ``gainmap_metadata``, ``width``, ``height``. ``hdr`` is a
    ``(H, W, 3)`` HDR raster in ``dtype`` (linear-light;
    1.0 = 203 nits / SDR white).
    """
    if not _HAVE_BACKEND:
        _missing()
    import concurrent.futures as _cf

    info = _cython_extract_layers(data)
    base_jpeg = info["base_jpeg"]
    gainmap_jpeg = info["gainmap_jpeg"]
    metadata = info["gainmap_metadata"]

    # Use opencodecs' own _jpeg decoder when a DCT-domain scale is
    # requested — imagecodecs.jpeg_decode doesn't expose libjpeg-
    # turbo's tj3SetScalingFactor knob. At scale=None we keep using
    # imagecodecs.jpeg_decode to avoid changing the v0.1.7 baseline
    # (slightly different output-shape conventions for grayscale
    # input — both ultimately produce the same RGB raster).
    if scale is not None:
        from .codecs._jpeg import decode as _jpeg_decode
        def _decode(b):
            return _jpeg_decode(b, scale=scale)
    else:
        import imagecodecs as _ic
        def _decode(b):
            return _ic.jpeg_decode(b)

    if parallel:
        ex = _cf.ThreadPoolExecutor(max_workers=2)
        try:
            sdr_fut = ex.submit(_decode, base_jpeg)
            gain_fut = ex.submit(_decode, gainmap_jpeg)
            sdr_u8 = sdr_fut.result()
            gain_u8 = gain_fut.result()
        finally:
            ex.shutdown(wait=False)
    else:
        sdr_u8 = _decode(base_jpeg)
        gain_u8 = _decode(gainmap_jpeg)

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


def encode_to(fp, hdr, **kwargs) -> None:
    """Streaming variant of :func:`encode_native` — write Ultra-HDR
    bytes directly to a file-like ``fp`` (anything with a ``write(buf)``
    method: an open file, ``io.BytesIO``, an HTTP upload sink, …).

    Internally calls ``encode_native(hdr, out=fp, ...)`` so the
    assembled container is handed to ``fp.write()`` via a zero-copy
    memoryview over libuhdr's internal output buffer. No
    intermediate Python ``bytes`` allocation. Saves ~1× output-size
    peak memory on large encodes.
    """
    encode_native(hdr, out=fp, **kwargs)


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
