"""DICOMweb / WADO-RS client for streaming pixel frames from a PACS.

DICOMweb (https://www.dicomstandard.org/using/dicomweb) is the HTTP
RESTful protocol for talking to PACS servers (orthanc, dcm4chee,
Google Healthcare API, IDC, AWS HealthImaging, etc). This module
implements the read side of the spec:

  - **WADO-RS frame retrieval**: ``GET /studies/{study}/series/{series}/
    instances/{instance}/frames/{N}`` returns the encoded pixel bytes
    of frame ``N``, wrapped in a ``multipart/related`` envelope. The
    part's ``Content-Type`` carries the DICOM transfer syntax UID so
    we can dispatch the right decoder.

  - **QIDO-RS instance enumeration** (optional helper):
    ``GET /studies/{study}/series/{series}/instances`` returns a JSON
    listing of instances + their per-instance tag values.

What this module deliberately doesn't do:

  - Authentication: pass a custom ``headers={"Authorization": "..."}``
    dict to inject bearer tokens / API keys; an OAuth dance is out of
    scope.
  - STOW-RS (upload): the codec layer is read-mostly anyway.
  - DICOM dataset parsing of metadata: WADO-RS frames already gave us
    the pixel payload + transfer syntax, no need to walk a full DICOM
    P10 file.

Transfer-syntax dispatch
========================

For each frame we look up the transfer-syntax UID and decode through
the matching opencodecs codec module:

  ============================ =================================
  UID                           Decoder
  ============================ =================================
  1.2.840.10008.1.2             raw little-endian (implicit VR)
  1.2.840.10008.1.2.1           raw little-endian (explicit VR)
  1.2.840.10008.1.2.4.50/51     JPEG baseline / extended (libjpeg-
                                turbo via opencodecs.codecs._jpeg)
  1.2.840.10008.1.2.4.70        JPEG lossless (jpegsof3 — not built
                                yet in opencodecs)
  1.2.840.10008.1.2.4.80/81     JPEG-LS lossless / near-lossless
                                (opencodecs.codecs._charls)
  1.2.840.10008.1.2.4.90/91     JPEG 2000 lossless / lossy
                                (opencodecs.codecs._jpeg2k)
  1.2.840.10008.1.2.4.201/202/  HTJ2K lossless / RPCL / lossy
       203                      (opencodecs.codecs._openjph)
  1.2.840.10008.1.2.5           RLE lossless (built-in below — DICOM
                                Annex G PackBits-like)
  ============================ =================================

Anything else raises ``UnsupportedTransferSyntax``.
"""

from __future__ import annotations

import json
import re
import urllib.parse
import urllib.request
from typing import Any, Iterable

# Common DICOM transfer-syntax UIDs.
TS_IMPLICIT_VR_LE  = "1.2.840.10008.1.2"
TS_EXPLICIT_VR_LE  = "1.2.840.10008.1.2.1"
TS_JPEG_BASELINE_1 = "1.2.840.10008.1.2.4.50"
TS_JPEG_EXTENDED_4 = "1.2.840.10008.1.2.4.51"
TS_JPEG_LOSSLESS   = "1.2.840.10008.1.2.4.70"
TS_JPEGLS_LOSSLESS = "1.2.840.10008.1.2.4.80"
TS_JPEGLS_NEAR     = "1.2.840.10008.1.2.4.81"
TS_JPEG2K_LOSSLESS = "1.2.840.10008.1.2.4.90"
TS_JPEG2K_LOSSY    = "1.2.840.10008.1.2.4.91"
TS_HTJ2K_LOSSLESS  = "1.2.840.10008.1.2.4.201"
TS_HTJ2K_RPCL      = "1.2.840.10008.1.2.4.202"
TS_HTJ2K_LOSSY     = "1.2.840.10008.1.2.4.203"
TS_RLE_LOSSLESS    = "1.2.840.10008.1.2.5"


class DicomwebError(RuntimeError):
    """Raised on protocol-level DICOMweb errors (bad multipart, etc)."""


class UnsupportedTransferSyntax(DicomwebError):
    """Raised when the returned frame uses a transfer syntax we can't decode."""


def _parse_multipart(body: bytes, content_type: str) -> list[tuple[dict[str, str], bytes]]:
    """Minimal multipart/related parser.

    Returns a list of ``(headers, body_bytes)`` for each part.
    The boundary is extracted from the top-level ``Content-Type``.
    Per RFC 2046, parts are separated by ``--<boundary>`` lines.
    """
    m = re.search(r'boundary="?([^";]+)"?', content_type)
    if not m:
        raise DicomwebError(
            f"missing multipart boundary in Content-Type: {content_type!r}"
        )
    boundary = b"--" + m.group(1).encode("ascii")
    # RFC 2046: leading CRLF before the first boundary is optional;
    # split on the boundary token and discard the preamble (split[0])
    # and the closing marker (last element, starts with b"--").
    parts = body.split(boundary)
    if len(parts) < 2:
        raise DicomwebError("multipart body has no parts")
    result: list[tuple[dict[str, str], bytes]] = []
    for part in parts[1:]:
        # Closing marker: starts with "--" (i.e. boundary + "--").
        if part.startswith(b"--"):
            break
        # Each part begins with a leading CRLF before the headers and
        # is terminated by a trailing CRLF before the next boundary.
        part = part.lstrip(b"\r\n")
        if not part:
            continue
        # Headers end at the first blank line.
        sep = b"\r\n\r\n"
        if sep not in part:
            sep = b"\n\n"
        head, _, payload = part.partition(sep)
        # Strip the trailing CRLF before the next boundary marker.
        payload = payload.rstrip(b"\r\n")
        headers: dict[str, str] = {}
        for line in head.split(b"\r\n") if b"\r\n" in head else head.split(b"\n"):
            line = line.strip()
            if not line:
                continue
            if b":" not in line:
                continue
            k, _, v = line.partition(b":")
            headers[k.strip().decode("ascii", errors="replace").lower()] = (
                v.strip().decode("ascii", errors="replace")
            )
        result.append((headers, payload))
    return result


def _extract_transfer_syntax(part_headers: dict[str, str]) -> str:
    """Pull the transfer syntax UID out of the part's Content-Type."""
    ct = part_headers.get("content-type", "")
    m = re.search(r'transfer-syntax="?([0-9.]+)"?', ct)
    if not m:
        raise DicomwebError(
            f"missing transfer-syntax in part Content-Type: {ct!r}"
        )
    return m.group(1)


def decode_frame(
    part_bytes: bytes,
    transfer_syntax: str,
    *,
    rows: int | None = None,
    columns: int | None = None,
    bits_allocated: int | None = None,
    samples_per_pixel: int = 1,
    pixel_representation: int = 0,
):
    """Decode a single frame given its transfer-syntax UID.

    For container-format transfer syntaxes (JPEG, JPEG-LS, JPEG-2000,
    HTJ2K) the codec parses the embedded header, so ``rows`` /
    ``columns`` / ``bits_allocated`` are only consulted for the raw
    syntaxes (implicit/explicit VR LE, RLE Lossless).
    """
    import numpy as np

    if transfer_syntax in (
        TS_JPEG_BASELINE_1, TS_JPEG_EXTENDED_4
    ):
        from opencodecs.codecs import _jpeg
        return _jpeg.decode(part_bytes)

    if transfer_syntax in (TS_JPEGLS_LOSSLESS, TS_JPEGLS_NEAR):
        from opencodecs.codecs import _charls
        return _charls.decode(part_bytes)

    if transfer_syntax in (TS_JPEG2K_LOSSLESS, TS_JPEG2K_LOSSY):
        from opencodecs.codecs import _jpeg2k
        return _jpeg2k.decode(part_bytes)

    if transfer_syntax in (
        TS_HTJ2K_LOSSLESS, TS_HTJ2K_RPCL, TS_HTJ2K_LOSSY
    ):
        from opencodecs.codecs import _openjph
        return _openjph.decode(part_bytes)

    if transfer_syntax in (TS_IMPLICIT_VR_LE, TS_EXPLICIT_VR_LE):
        if rows is None or columns is None or bits_allocated is None:
            raise DicomwebError(
                "raw transfer syntax requires rows/columns/bits_allocated"
            )
        dtype = _dtype_for(bits_allocated, pixel_representation)
        n = rows * columns * samples_per_pixel
        if len(part_bytes) < n * dtype().itemsize:
            raise DicomwebError(
                f"raw frame is short: got {len(part_bytes)} bytes, "
                f"need {n * dtype().itemsize}"
            )
        arr = np.frombuffer(part_bytes[:n * dtype().itemsize], dtype=dtype)
        if samples_per_pixel == 1:
            return arr.reshape(rows, columns)
        return arr.reshape(rows, columns, samples_per_pixel)

    if transfer_syntax == TS_RLE_LOSSLESS:
        if rows is None or columns is None or bits_allocated is None:
            raise DicomwebError(
                "RLE Lossless requires rows/columns/bits_allocated"
            )
        return _rle_lossless_decode(
            part_bytes, rows, columns, bits_allocated,
            samples_per_pixel, pixel_representation,
        )

    raise UnsupportedTransferSyntax(
        f"transfer syntax {transfer_syntax!r} not supported by opencodecs"
    )


def _dtype_for(bits_allocated: int, pixel_representation: int):
    import numpy as np
    if bits_allocated == 8:
        return np.int8 if pixel_representation else np.uint8
    if bits_allocated == 16:
        return np.int16 if pixel_representation else np.uint16
    if bits_allocated == 32:
        return np.int32 if pixel_representation else np.uint32
    raise DicomwebError(f"unsupported bits_allocated={bits_allocated}")


def _rle_lossless_decode(
    data: bytes, rows: int, columns: int, bits_allocated: int,
    samples_per_pixel: int, pixel_representation: int,
):
    """DICOM Annex G "RLE Lossless" — header table of segment offsets
    followed by PackBits-style segments, one per (sample, byte plane).

    Layout:
      uint32 num_segments
      uint32 offset_segment_1
      ...    (15 slots total; unused entries are 0)
      segment_1 bytes
      ...

    Each segment decodes via the PackBits scheme (n>=0: copy next n+1,
    n<0: repeat next byte 1-n times, n=-128: skip).
    """
    import numpy as np
    import struct

    if len(data) < 64:
        raise DicomwebError("RLE frame too short to hold header")
    num_segments = struct.unpack_from("<I", data, 0)[0]
    if num_segments < 1 or num_segments > 15:
        raise DicomwebError(f"RLE: invalid num_segments={num_segments}")
    offsets = list(struct.unpack_from(f"<{num_segments}I", data, 4))
    offsets.append(len(data))  # sentinel for the last segment's end
    seg_bytes: list[bytes] = []
    for i in range(num_segments):
        seg = _packbits_decode(data[offsets[i]:offsets[i + 1]],
                               rows * columns)
        seg_bytes.append(seg)

    dtype = _dtype_for(bits_allocated, pixel_representation)
    bytes_per_sample = dtype().itemsize
    # DICOM RLE: planes are stored MSB-first per sample, then by sample.
    # For samples_per_pixel=1, bits_allocated=8: num_segments=1.
    # For samples_per_pixel=1, bits_allocated=16: num_segments=2,
    # segment 0 = MSB byte plane, segment 1 = LSB.
    if num_segments != samples_per_pixel * bytes_per_sample:
        raise DicomwebError(
            f"RLE: num_segments={num_segments} != "
            f"samples_per_pixel*bytes_per_sample="
            f"{samples_per_pixel * bytes_per_sample}"
        )
    pixels = rows * columns
    out = np.zeros(pixels * samples_per_pixel * bytes_per_sample, np.uint8)
    seg_idx = 0
    for s in range(samples_per_pixel):
        for b in range(bytes_per_sample):
            # MSB-first: segment for byte-position b corresponds to
            # byte-offset (bytes_per_sample - 1 - b) within the sample.
            byte_pos = bytes_per_sample - 1 - b
            plane = np.frombuffer(seg_bytes[seg_idx], dtype=np.uint8)
            if plane.size < pixels:
                raise DicomwebError(
                    f"RLE: short segment {seg_idx}: got {plane.size}, "
                    f"need {pixels}"
                )
            out_view = out.reshape(pixels, samples_per_pixel,
                                   bytes_per_sample)
            out_view[:, s, byte_pos] = plane[:pixels]
            seg_idx += 1
    arr = np.frombuffer(out.tobytes(), dtype=dtype)
    if samples_per_pixel == 1:
        return arr.reshape(rows, columns)
    return arr.reshape(rows, columns, samples_per_pixel)


def _packbits_decode(seg: bytes, expected_size: int) -> bytes:
    """PackBits decode (Apple Macintosh format, also used in DICOM RLE
    and TIFF compression=32773). Stops at expected_size if reached."""
    out = bytearray()
    i = 0
    n = len(seg)
    while i < n and len(out) < expected_size:
        b = seg[i]
        i += 1
        if b == 0x80:  # -128: no-op
            continue
        if b < 0x80:  # 0..127: copy next b+1 bytes literally
            count = b + 1
            out.extend(seg[i:i + count])
            i += count
        else:  # 129..255: replicate next byte (1 - signed b) times
            count = 257 - b
            if i < n:
                out.extend(bytes([seg[i]]) * count)
                i += 1
    return bytes(out)


class DicomwebClient:
    """Minimal WADO-RS / QIDO-RS client.

    Parameters
    ----------
    base_url
        Root of the DICOMweb endpoint, e.g.
        ``https://server.example/dicomweb``. The client appends
        ``/studies/...`` paths to this; no trailing slash needed.
    headers
        Extra HTTP headers (typically auth, e.g.
        ``{"Authorization": "Bearer ..."}``).
    timeout
        Per-request timeout in seconds.
    """

    def __init__(
        self,
        base_url: str,
        *,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.headers = dict(headers or {})
        self.timeout = float(timeout)

    # ------------------------------------------------------------------
    # WADO-RS
    # ------------------------------------------------------------------

    def get_frame(
        self,
        study_uid: str,
        series_uid: str,
        instance_uid: str,
        frame: int = 1,
        *,
        accept: str = "multipart/related;type=application/octet-stream",
        rows: int | None = None,
        columns: int | None = None,
        bits_allocated: int | None = None,
        samples_per_pixel: int = 1,
        pixel_representation: int = 0,
    ):
        """Fetch and decode one frame; returns a numpy ndarray.

        ``rows`` / ``columns`` / ``bits_allocated`` are only consulted
        for the raw / RLE transfer syntaxes; container syntaxes parse
        their own headers.
        """
        url = (
            f"{self.base_url}/studies/{study_uid}/series/{series_uid}"
            f"/instances/{instance_uid}/frames/{frame}"
        )
        body, content_type = self._http_get(url, accept=accept)
        parts = _parse_multipart(body, content_type)
        if not parts:
            raise DicomwebError("server returned no frame parts")
        # WADO-RS frames endpoint returns one part per requested frame.
        part_headers, part_body = parts[0]
        ts = _extract_transfer_syntax(part_headers)
        return decode_frame(
            part_body, ts,
            rows=rows, columns=columns,
            bits_allocated=bits_allocated,
            samples_per_pixel=samples_per_pixel,
            pixel_representation=pixel_representation,
        )

    # ------------------------------------------------------------------
    # QIDO-RS
    # ------------------------------------------------------------------

    def list_instances(
        self, study_uid: str, series_uid: str,
    ) -> list[dict[str, Any]]:
        """Returns the JSON instance list for a given series."""
        url = (
            f"{self.base_url}/studies/{study_uid}/series/{series_uid}/"
            f"instances"
        )
        body, _ = self._http_get(url, accept="application/dicom+json")
        return json.loads(body.decode("utf-8"))

    # ------------------------------------------------------------------
    # HTTP plumbing
    # ------------------------------------------------------------------

    def _http_get(self, url: str, *, accept: str) -> tuple[bytes, str]:
        req = urllib.request.Request(url, headers={**self.headers,
                                                   "Accept": accept})
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return resp.read(), resp.headers.get("Content-Type", "")
