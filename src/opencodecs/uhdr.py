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

import struct

import numpy as np


# ---------------------------------------------------------------------------
# EXIF thumbnail embed/extract
# ---------------------------------------------------------------------------
#
# JPEG metadata sidecar that's NOT a sidecar: a small pre-encoded JPEG
# thumbnail lives inside an APP1 segment of the main file, indexed via
# the TIFF/EXIF "1st IFD" structure. Every photo app on every OS
# already knows how to read these — they're how phone galleries get
# instant previews of multi-megapixel JPEGs.
#
# Wire format we emit:
#
#   APP1 marker (FFE1)
#   segment length (2 bytes, big-endian, includes itself)
#   "Exif\0\0"               6 bytes  signature
#   TIFF header              8 bytes  "II*\0" little-endian, ifd0_offset=8
#   IFD-0                            empty marker IFD, points at IFD-1
#     entry count (2 bytes) = 0
#     ifd1 offset (4 bytes) = current position
#   IFD-1                            describes the thumbnail
#     entry count = 3
#     tag 0x0103 (Compression)         SHORT, value=6 (JPEG)
#     tag 0x0201 (JPEGInterchangeFormat)        LONG, value=thumb_offset
#     tag 0x0202 (JPEGInterchangeFormatLength)  LONG, value=thumb_size
#     next-IFD offset (4 bytes) = 0
#   <thumbnail JPEG bytes>
#
# All offsets in the TIFF block are relative to the start of the TIFF
# header (the "II*\0" byte after the "Exif\0\0" signature).

_EXIF_SIG = b"Exif\x00\x00"
_TIFF_HEADER_LE = b"II*\x00" + struct.pack("<I", 8)  # IFD-0 at offset 8


def _build_exif_thumbnail_segment(thumb_jpeg: bytes) -> bytes:
    """Wrap a pre-encoded thumbnail JPEG in an EXIF/TIFF APP1 segment.

    Returns the segment bytes including the FFE1 marker + length
    header — ready to splice into the main JPEG bytestream right
    after the SOI.
    """
    # IFD-0: 0 entries, but its trailing "next-IFD" field points at IFD-1.
    # 2 (count) + 4 (next-IFD offset) = 6 bytes.
    # IFD-1: 3 entries + 4-byte next-IFD-zero terminator.
    # Each IFD entry is 12 bytes: tag(2) + type(2) + count(4) + value(4).
    ifd0 = struct.pack("<HI", 0, 8 + 6)  # IFD-1 starts at offset 14
    ifd1_offset = 8 + 6                  # after TIFF header (8) + IFD-0 (6)
    ifd1_size = 2 + 3 * 12 + 4           # count + 3 entries + next-IFD
    thumb_offset = ifd1_offset + ifd1_size  # thumbnail bytes follow IFD-1
    ifd1 = struct.pack(
        "<H"                          # entry count
        "HHII"                        # Compression: SHORT, count=1, value=6
        "HHII"                        # JPEGInterchangeFormat: LONG, count=1
        "HHII"                        # JPEGInterchangeFormatLength: LONG, count=1
        "I",                          # next-IFD offset (0 = none)
        3,
        0x0103, 3, 1, 6,              # Compression = JPEG
        0x0201, 4, 1, thumb_offset,   # offset of thumbnail JPEG within TIFF
        0x0202, 4, 1, len(thumb_jpeg),
        0,
    )
    payload = _EXIF_SIG + _TIFF_HEADER_LE + ifd0 + ifd1 + thumb_jpeg
    # APP1 segment length includes itself (2 bytes) but not the FFE1
    # marker (also 2 bytes). Cap at 65533 — JPEG segment max is 65535
    # minus the length field. Thumbnail at 256² q80 is typically
    # ~10-25 KB so we have plenty of headroom; abort cleanly otherwise.
    seg_len = len(payload) + 2
    if seg_len > 0xFFFF:
        raise ValueError(
            f"thumbnail JPEG too large for a single APP1 segment "
            f"({seg_len} bytes, max 65535); use a smaller thumbnail_size")
    return b"\xff\xe1" + struct.pack(">H", seg_len) + payload


def _inject_app1_after_soi(jpeg_bytes: bytes, segment: bytes) -> bytes:
    """Splice a single APP1 segment in immediately after the SOI marker.

    Most JPEGs have an APP0 (JFIF) or APP1 segment right after SOI;
    we insert ours BEFORE those so existing parsers find the
    thumbnail's EXIF first. Decoders that walk APP1 markers in order
    return the first match.
    """
    if len(jpeg_bytes) < 2 or jpeg_bytes[:2] != b"\xff\xd8":
        raise ValueError(
            "injection target doesn't start with SOI (FFD8) marker; "
            "is this really a JPEG?")
    return b"\xff\xd8" + segment + jpeg_bytes[2:]


def _find_app1_exif_payload(jpeg_bytes: bytes) -> tuple[bytes, int] | None:
    """Locate the first APP1 segment whose payload starts with the EXIF
    signature. Returns (segment_payload_starting_at_TIFF_header,
    file_offset_of_TIFF_header) or None.
    """
    i = 0
    n = len(jpeg_bytes)
    if n < 2 or jpeg_bytes[0:2] != b"\xff\xd8":
        return None
    i = 2
    while i < n - 4:
        if jpeg_bytes[i] != 0xFF:
            return None  # malformed; bail
        marker = jpeg_bytes[i + 1]
        if marker == 0xD8 or marker == 0x01:
            i += 2; continue
        if marker == 0xD9 or marker == 0xDA:
            # EOI / SOS — past metadata zone, no thumbnail
            return None
        # Variable-length segment: bytes [i+2, i+4) is length-incl-itself
        seg_len = (jpeg_bytes[i + 2] << 8) | jpeg_bytes[i + 3]
        start = i + 4
        end = i + 2 + seg_len
        if marker == 0xE1 and jpeg_bytes[start:start + 6] == _EXIF_SIG:
            return jpeg_bytes[start + 6:end], start + 6
        i = end


def _extract_exif_thumbnail_bytes(jpeg_bytes: bytes) -> bytes | None:
    """Return the embedded EXIF thumbnail as raw JPEG bytes, or None
    if the file has no thumbnail.
    """
    found = _find_app1_exif_payload(jpeg_bytes)
    if found is None:
        return None
    tiff, _ = found
    if len(tiff) < 8:
        return None
    byte_order = tiff[:2]
    if byte_order == b"II":
        endian = "<"
    elif byte_order == b"MM":
        endian = ">"
    else:
        return None
    magic = struct.unpack(endian + "H", tiff[2:4])[0]
    if magic != 0x002A:
        return None
    ifd0_offset = struct.unpack(endian + "I", tiff[4:8])[0]
    if ifd0_offset + 2 > len(tiff):
        return None
    n0 = struct.unpack(endian + "H", tiff[ifd0_offset:ifd0_offset + 2])[0]
    ifd1_off_pos = ifd0_offset + 2 + n0 * 12
    if ifd1_off_pos + 4 > len(tiff):
        return None
    ifd1_offset = struct.unpack(
        endian + "I", tiff[ifd1_off_pos:ifd1_off_pos + 4])[0]
    if ifd1_offset == 0 or ifd1_offset + 2 > len(tiff):
        return None  # no IFD-1 (no thumbnail)
    n1 = struct.unpack(endian + "H", tiff[ifd1_offset:ifd1_offset + 2])[0]
    thumb_off = thumb_len = None
    for k in range(n1):
        entry = tiff[ifd1_offset + 2 + k * 12 : ifd1_offset + 2 + (k + 1) * 12]
        tag, typ, count, value = struct.unpack(endian + "HHII", entry)
        if tag == 0x0201:
            thumb_off = value
        elif tag == 0x0202:
            thumb_len = value
    if thumb_off is None or thumb_len is None:
        return None
    if thumb_off + thumb_len > len(tiff):
        return None
    return bytes(tiff[thumb_off:thumb_off + thumb_len])

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
    "read_thumbnail",
    "read_thumbnail_hdr",
    "read_thumbnail_bytes",
    "libuhdr_version",
    "UhdrError",
    "read",
    "write",
]


def encode_native(hdr, sdr=None, *,
                   gamut='display-p3', sdr_white_nits=1600.0,
                   quality=95, gain_quality=None, gain_scale=1,
                   sdr_subsampling=None, lossless=False,
                   thumbnail_size=None, thumbnail_quality=80,
                   thumbnail_hdr=True,
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
    sdr_subsampling : str, optional
        Chroma subsampling for the SDR base JPEG: ``'420'`` (default),
        ``'422'``, ``'444'``, ``'440'``, ``'411'``. 4:2:0 is the
        universal photographic default (halves chroma data,
        invisible on natural images). Pass ``'444'`` for non-
        photographic SDR bases — synthetic plots, scientific
        imagery with sharp boundaries, text overlays — where chroma
        bleed at high-contrast edges matters. Ignored when
        ``lossless=True`` (lossless mode forces 4:4:4). The gain
        map's chroma is independently locked to 4:4:4 to preserve
        per-pixel boost peaks (4:2:0 there would attenuate sparse-
        bright HDR content).
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
    thumbnail_size : int, optional
        If set, embed a square thumbnail of at most ``thumbnail_size``
        px on each side inside the file's APP1 EXIF segment. By
        default the thumbnail is itself a mini Ultra-HDR (SDR base
        + downsampled gain map + MPF box), so HDR-aware viewers
        preserve peak brightness when previewing — same boost as the
        main image. SDR-only EXIF readers see the SDR base layer as
        a plain JPEG and ignore the MPF block, so backward-compat is
        preserved. Adds ~25-35 KB to the file and ~3 ms to the
        encode. Reads via :func:`read_thumbnail` (uint8 SDR, ~1 ms)
        or :func:`read_thumbnail_hdr` (fp32 HDR, ~1.2 ms).
        ``None`` (default) embeds nothing. Both the SDR base and
        gain map use centered-stride decimation so the thumbnail
        coordinate system aligns with the full-res image (no half-
        pixel bias).
    thumbnail_quality : int
        JPEG quality (1-100) for the thumbnail's SDR base and gain
        map. Default 80; the thumbnail is a preview, not the
        primary product.
    thumbnail_hdr : bool
        ``True`` (default) embeds an Ultra-HDR thumbnail (preserves
        peak brightness in HDR-aware viewers). ``False`` embeds a
        plain SDR JPEG thumbnail — smaller (~20 KB instead of ~30
        KB) but renders at SDR-white (~200 nits) on HDR displays,
        not the main image's full HDR boost. Useful when targeting
        legacy EXIF readers that don't tolerate MPF segments in
        thumbnail bytes.
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

    # Chrome compatibility nudge: when log2(mcb) lands on a clean integer
    # (mcb is a power of 2, e.g. 8.0), libuhdr's metadata serializer
    # detects "all per-channel denominators = 1" and emits its compact
    # 37-byte form (useCommonDenominator flag set). Chrome's UHDR parser
    # does NOT handle that compact form — files render as plain SDR.
    # A 1-part-in-10^7 perturbation pushes log2(mcb) off the integer
    # boundary, libuhdr's float→rational continued-fraction algorithm
    # then produces non-trivial denominators, and the canonical 61-byte
    # long form gets emitted instead. Numerically invisible.
    import math as _math
    if mcb > 0:
        log2mcb = _math.log2(mcb)
        if abs(log2mcb - round(log2mcb)) < 1e-9:
            mcb = mcb * (1.0 + 1e-7)

    def _maybe_downscale_gain(gain_u8):
        # Stride decimation by an integer factor. Gain maps are heavily
        # band-limited by construction (smooth ratio of low-frequency
        # luminance signals), so a nearest-sample stride gives
        # visually-equivalent output to a box average — and is ~20×
        # faster on a 2k² uint8 raster (numpy's .mean() promotes to
        # float64 and runs multiple passes, eating ~100ms; stride
        # decimation is ~5ms). Cuts gain JPEG encode work by scale²
        # and gain bytes proportionally.
        # Centered offset so the decoder's bilinear upscale lands the
        # gain samples on the same image coordinates as the SDR base.
        if gain_scale == 1:
            return gain_u8
        off = gain_scale // 2
        return np.ascontiguousarray(
            gain_u8[off::gain_scale, off::gain_scale])

    # Gain map = per-pixel HDR boost multiplier. JPEG default 4:2:0
    # chroma subsampling averages boost values over 2x2 blocks, which
    # halves the effective peak on sparse-bright-pixel content (e.g.
    # fluorescence dye spots) — empirically a 7.84 → 3.67 peak drop on
    # round-trip. ``subsampling='440'`` keeps per-pixel gain fidelity.
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
        if sdr_subsampling is None:
            return imagecodecs.jpeg_encode(arr, quality)
        return imagecodecs.jpeg_encode(
            arr, quality, subsampling=sdr_subsampling)

    if parallel:
        ex = _cf.ThreadPoolExecutor(max_workers=3)
        try:
            base_fut = ex.submit(_encode_base, sdr_arr)
            gain_fut = ex.submit(_cython_gain_map, hdr_arr, sdr_arr,
                                 sdr_white_nits=sdr_white_nits,
                                 max_content_boost=mcb,
                                 min_content_boost=min_content_boost,
                                 gamma=gamma)
            gain_full, metadata = gain_fut.result()
            gain_u8 = _maybe_downscale_gain(gain_full)
            gainmap_fut = ex.submit(
                imagecodecs.jpeg_encode, gain_u8, gain_quality,
                subsampling='440')
            base_jpeg = base_fut.result()
            gainmap_jpeg = gainmap_fut.result()
        finally:
            ex.shutdown(wait=False)
    else:
        gain_full, metadata = _cython_gain_map(
            hdr_arr, sdr_arr,
            sdr_white_nits=sdr_white_nits,
            max_content_boost=mcb,
            min_content_boost=min_content_boost,
            gamma=gamma,
        )
        gain_u8 = _maybe_downscale_gain(gain_full)
        base_jpeg = _encode_base(sdr_arr)
        gainmap_jpeg = imagecodecs.jpeg_encode(
            gain_u8, gain_quality, subsampling='440')

    if thumbnail_size is not None and thumbnail_size > 0:
        # Centered-stride decimate the SDR base (and, for UHDR
        # thumbnails, the FULL-resolution gain map). Centered offset
        # (stride//2) puts each sample at the middle of its
        # stride×stride block — top-left stride would introduce a
        # half-stride coordinate bias against the full-res image.
        max_dim = max(sdr_arr.shape[0], sdr_arr.shape[1])
        if max_dim > thumbnail_size:
            stride = (max_dim + thumbnail_size - 1) // thumbnail_size
            off = stride // 2
            thumb_sdr_raster = np.ascontiguousarray(
                sdr_arr[off::stride, off::stride])
        else:
            stride = 1
            off = 0
            thumb_sdr_raster = sdr_arr
        thumb_sdr_jpeg = imagecodecs.jpeg_encode(
            thumb_sdr_raster, int(thumbnail_quality))
        if thumbnail_hdr:
            # Mini-UHDR thumbnail: stride-decimate the FULL-resolution
            # gain raster with the same centered offset, JPEG-encode at
            # 4:4:4 chroma (per-pixel boost fidelity in a 250-ish-px
            # raster has zero chroma-bleed room to give up), and ask
            # libuhdr's api-4 to wrap base + gain + the main image's
            # metadata into a self-contained Ultra-HDR JPEG. That goes
            # into the parent file's EXIF IFD-1 slot as the thumbnail.
            # Using gain_full (pre-_maybe_downscale_gain) avoids
            # compounding the gain_scale downsample with the thumbnail
            # stride.
            if stride > 1:
                thumb_gain_raster = np.ascontiguousarray(
                    gain_full[off::stride, off::stride])
            else:
                thumb_gain_raster = gain_full
            # Match shapes if SDR/gain edge alignment diverged on the
            # last row/col (e.g. odd dim + stride trimming).
            h = min(thumb_sdr_raster.shape[0], thumb_gain_raster.shape[0])
            w = min(thumb_sdr_raster.shape[1], thumb_gain_raster.shape[1])
            if thumb_gain_raster.shape[:2] != (h, w) or \
                    thumb_sdr_raster.shape[:2] != (h, w):
                thumb_sdr_raster = thumb_sdr_raster[:h, :w]
                thumb_gain_raster = thumb_gain_raster[:h, :w]
                thumb_sdr_jpeg = imagecodecs.jpeg_encode(
                    np.ascontiguousarray(thumb_sdr_raster),
                    int(thumbnail_quality))
            thumb_gain_jpeg = imagecodecs.jpeg_encode(
                thumb_gain_raster, int(thumbnail_quality),
                subsampling='440')
            thumb_blob = encode_assembled(
                base_jpeg=thumb_sdr_jpeg,
                gainmap_jpeg=thumb_gain_jpeg,
                metadata=metadata,
                gamut=gamut,
            )
            exif_seg = _build_exif_thumbnail_segment(thumb_blob)
        else:
            exif_seg = _build_exif_thumbnail_segment(thumb_sdr_jpeg)
    else:
        exif_seg = None

    if exif_seg is None:
        return encode_assembled(
            base_jpeg=base_jpeg,
            gainmap_jpeg=gainmap_jpeg,
            metadata=metadata,
            gamut=gamut,
            out=out,
        )

    # Thumbnail path: assemble to bytes (we need to splice the EXIF
    # segment into the bytestream), then either return or stream-write.
    blob = encode_assembled(
        base_jpeg=base_jpeg,
        gainmap_jpeg=gainmap_jpeg,
        metadata=metadata,
        gamut=gamut,
    )
    blob = _inject_app1_after_soi(blob, exif_seg)
    if out is None:
        return blob
    out.write(blob)
    return None


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
        # Parallel Cython nearest-neighbor upscale (~3 ms for 500→2000 at
        # 16 threads). Replaced np.repeat (~14 ms) and fancy-index (~40 ms).
        # At gain_scale>1 this was the dominant decode cost.
        from .codecs._uhdr import upscale_gainmap
        gain_u8 = upscale_gainmap(gain_u8, sH, sW)

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


def read_thumbnail_bytes(data) -> bytes | None:
    """Return the encoded JPEG thumbnail embedded in the file's APP1
    EXIF block, or ``None`` if no thumbnail is present.

    Only the APP1 segment is parsed — the main SDR base + gain-map
    JPEGs are not touched. Sub-millisecond on local files, can be
    served from the first ~5-10 KB of an HTTP-ranged read for remote
    archives.

    Embeds via :func:`encode_native` ``thumbnail_size=`` kwarg.
    """
    if isinstance(data, (bytes, bytearray, memoryview)):
        buf = bytes(data)
    else:
        buf = bytes(data)
    return _extract_exif_thumbnail_bytes(buf)


def read_thumbnail(data):
    """Return the embedded EXIF thumbnail as a decoded uint8 numpy
    array, or ``None`` if no thumbnail is present. Convenience wrapper
    around :func:`read_thumbnail_bytes` + ``imagecodecs.jpeg_decode``.

    The thumbnail may itself be an Ultra-HDR file (default for
    ``encode_native(thumbnail_size=N)``) — this function returns only
    the SDR base layer (libjpeg-turbo ignores the MPF box). Use
    :func:`read_thumbnail_hdr` for fp32 HDR pixels.
    """
    raw = read_thumbnail_bytes(data)
    if raw is None:
        return None
    import imagecodecs
    return imagecodecs.jpeg_decode(raw)


def read_thumbnail_hdr(data, *, dtype=None, display_boost=None):
    """Return the embedded EXIF thumbnail as a decoded HDR float array,
    or ``None`` if no thumbnail is present.

    If the embedded thumbnail is itself an Ultra-HDR JPEG (default for
    files written by :func:`encode_native` with ``thumbnail_size=``,
    ``thumbnail_hdr=True``), this decodes it through :func:`decode_native`
    — preserving the main image's HDR peak in the thumbnail. If the
    embedded thumbnail is a plain SDR JPEG (``thumbnail_hdr=False``
    files, or legacy files), the SDR pixels are returned as fp32
    scaled to ``[0, 1]``.

    Parameters
    ----------
    data : bytes-like
        The parent Ultra-HDR JPEG.
    dtype : numpy dtype, optional
        Output HDR dtype (default ``float32``). Forwarded to
        :func:`decode_native` for UHDR thumbnails.
    display_boost : float, optional
        Target headroom; forwarded to :func:`decode_native`. ``None``
        (default) requests the encoded raster's full HDR.

    Returns
    -------
    ndarray (H, W, 3) float (default fp32), or ``None``.
    """
    raw = read_thumbnail_bytes(data)
    if raw is None:
        return None
    import numpy as np
    if dtype is None:
        dtype = np.float32
    if is_uhdr(raw):
        out = decode_native(raw, dtype=dtype, display_boost=display_boost)
        return out["hdr"]
    import imagecodecs
    sdr = imagecodecs.jpeg_decode(raw)
    return (sdr.astype(np.dtype(dtype), copy=False) / 255.0).astype(
        np.dtype(dtype), copy=False)


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
