"""DICOMweb client tests.

We don't have a live PACS endpoint in CI, so the tests focus on:

  - multipart/related parsing of synthetic responses
  - RLE Lossless decode (DICOM Annex G PackBits-like layout)
  - Transfer-syntax dispatch when given a synthesized response body
    that wraps a known-codec encoded payload
  - Error handling for unsupported transfer syntaxes
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

dw = pytest.importorskip("opencodecs._dicomweb")
parse_multipart = dw._parse_multipart
extract_ts = dw._extract_transfer_syntax
decode_frame = dw.decode_frame
DicomwebError = dw.DicomwebError
UnsupportedTransferSyntax = dw.UnsupportedTransferSyntax


def _build_multipart(parts: list[tuple[str, bytes]], boundary: str = "X") -> tuple[bytes, str]:
    """Build a synthetic multipart/related body."""
    crlf = b"\r\n"
    bnd = boundary.encode("ascii")
    out = b""
    for transfer_syntax, body in parts:
        out += b"--" + bnd + crlf
        out += (
            f"Content-Type: application/octet-stream; "
            f'transfer-syntax="{transfer_syntax}"'
        ).encode("ascii") + crlf
        out += crlf
        out += body + crlf
    out += b"--" + bnd + b"--" + crlf
    content_type = (
        f'multipart/related; type="application/octet-stream"; boundary="{boundary}"'
    )
    return out, content_type


# ---------------------------------------------------------------------------
# multipart parser
# ---------------------------------------------------------------------------


def test_parse_multipart_single_part():
    body, ct = _build_multipart([("1.2.840.10008.1.2.1", b"raw-pixels")])
    parts = parse_multipart(body, ct)
    assert len(parts) == 1
    headers, payload = parts[0]
    assert payload == b"raw-pixels"
    assert extract_ts(headers) == "1.2.840.10008.1.2.1"


def test_parse_multipart_multiple_parts():
    body, ct = _build_multipart([
        ("1.2.840.10008.1.2.4.80", b"jpegls-frame"),
        ("1.2.840.10008.1.2.4.90", b"j2k-frame"),
    ])
    parts = parse_multipart(body, ct)
    assert len(parts) == 2
    assert parts[0][1] == b"jpegls-frame"
    assert parts[1][1] == b"j2k-frame"
    assert extract_ts(parts[0][0]) == "1.2.840.10008.1.2.4.80"
    assert extract_ts(parts[1][0]) == "1.2.840.10008.1.2.4.90"


def test_parse_multipart_rejects_missing_boundary():
    with pytest.raises(DicomwebError):
        parse_multipart(b"data", "application/octet-stream")


# ---------------------------------------------------------------------------
# Raw / explicit-VR LE transfer syntax
# ---------------------------------------------------------------------------


def test_decode_frame_raw_explicit_vr_le_u8():
    arr = np.arange(16 * 24, dtype=np.uint8).reshape(16, 24)
    back = decode_frame(
        arr.tobytes(), "1.2.840.10008.1.2.1",
        rows=16, columns=24, bits_allocated=8,
        samples_per_pixel=1, pixel_representation=0,
    )
    np.testing.assert_array_equal(back, arr)


def test_decode_frame_raw_u16():
    arr = (np.arange(16 * 24, dtype=np.uint16) * 17).reshape(16, 24)
    back = decode_frame(
        arr.tobytes(), "1.2.840.10008.1.2.1",
        rows=16, columns=24, bits_allocated=16,
        samples_per_pixel=1, pixel_representation=0,
    )
    np.testing.assert_array_equal(back, arr)


def test_decode_frame_raw_signed():
    arr = np.array([[-1, 0, 1], [127, -128, 64]], dtype=np.int8)
    back = decode_frame(
        arr.tobytes(), "1.2.840.10008.1.2.1",
        rows=2, columns=3, bits_allocated=8,
        samples_per_pixel=1, pixel_representation=1,
    )
    np.testing.assert_array_equal(back, arr)


# ---------------------------------------------------------------------------
# RLE Lossless (DICOM Annex G)
# ---------------------------------------------------------------------------


def _encode_packbits(data: bytes) -> bytes:
    """A simple literal-only PackBits encoder; good enough to round-
    trip through the decoder for testing."""
    out = bytearray()
    i = 0
    while i < len(data):
        chunk = data[i:i + 128]
        out.append(len(chunk) - 1)
        out.extend(chunk)
        i += len(chunk)
    return bytes(out)


def _encode_rle_lossless(arr: np.ndarray, samples_per_pixel: int) -> bytes:
    """Build a DICOM RLE Lossless payload from a uint8/uint16 array."""
    bytes_per_sample = arr.dtype.itemsize
    num_segments = samples_per_pixel * bytes_per_sample
    if samples_per_pixel == 1:
        flat = arr.ravel()
    else:
        flat = arr.reshape(-1, samples_per_pixel)
    segments = []
    if bytes_per_sample == 1:
        if samples_per_pixel == 1:
            segments.append(_encode_packbits(flat.tobytes()))
        else:
            for s in range(samples_per_pixel):
                segments.append(_encode_packbits(flat[:, s].tobytes()))
    else:
        # MSB-first per sample, then by sample.
        view = arr.view(np.uint8).reshape(-1, samples_per_pixel,
                                           bytes_per_sample)
        for s in range(samples_per_pixel):
            for b in reversed(range(bytes_per_sample)):
                segments.append(_encode_packbits(view[:, s, b].tobytes()))
    # Header: num_segments + 15 offset slots, all little-endian u32.
    header = bytearray(struct.pack("<I", num_segments))
    offsets = [0] * 15
    running = 64
    for i, seg in enumerate(segments):
        offsets[i] = running
        running += len(seg)
    header += struct.pack("<15I", *offsets)
    return bytes(header) + b"".join(segments)


def test_rle_lossless_u8_grayscale():
    arr = np.arange(16 * 24, dtype=np.uint8).reshape(16, 24)
    payload = _encode_rle_lossless(arr, samples_per_pixel=1)
    back = decode_frame(
        payload, "1.2.840.10008.1.2.5",
        rows=16, columns=24, bits_allocated=8,
        samples_per_pixel=1, pixel_representation=0,
    )
    np.testing.assert_array_equal(back, arr)


def test_rle_lossless_u16_grayscale():
    arr = (np.arange(8 * 12, dtype=np.uint16) * 257).reshape(8, 12)
    payload = _encode_rle_lossless(arr, samples_per_pixel=1)
    back = decode_frame(
        payload, "1.2.840.10008.1.2.5",
        rows=8, columns=12, bits_allocated=16,
        samples_per_pixel=1, pixel_representation=0,
    )
    np.testing.assert_array_equal(back, arr)


def test_rle_lossless_rgb():
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(6, 10, 3), dtype=np.uint8)
    payload = _encode_rle_lossless(arr, samples_per_pixel=3)
    back = decode_frame(
        payload, "1.2.840.10008.1.2.5",
        rows=6, columns=10, bits_allocated=8,
        samples_per_pixel=3, pixel_representation=0,
    )
    np.testing.assert_array_equal(back, arr)


# ---------------------------------------------------------------------------
# Container-syntax dispatch
# ---------------------------------------------------------------------------


def test_decode_frame_dispatches_to_jpegls():
    """JPEG-LS transfer syntax must round-trip via the _charls codec."""
    pytest.importorskip("opencodecs.codecs._charls")
    from opencodecs.codecs import _charls
    arr = (np.arange(48 * 64, dtype=np.uint16) * 17 % 4000).reshape(48, 64)
    enc = _charls.encode(arr.astype(np.uint16))
    back = decode_frame(enc, "1.2.840.10008.1.2.4.80")
    np.testing.assert_array_equal(back, arr.astype(np.uint16))


def test_decode_frame_dispatches_to_htj2k():
    """HTJ2K (Part-15) routes to the OpenJPH codec."""
    pytest.importorskip("opencodecs.codecs._openjph")
    from opencodecs.codecs import _openjph
    arr = (np.arange(32 * 48, dtype=np.uint16) * 7 % 1024).reshape(32, 48)
    enc = _openjph.encode(arr.astype(np.uint16))
    back = decode_frame(enc, "1.2.840.10008.1.2.4.201")
    np.testing.assert_array_equal(back, arr.astype(np.uint16))


def test_decode_frame_rejects_unknown_transfer_syntax():
    with pytest.raises(UnsupportedTransferSyntax):
        decode_frame(b"x", "9.9.9")


# ---------------------------------------------------------------------------
# Client wiring (no live server; just construction)
# ---------------------------------------------------------------------------


def test_client_constructs_with_auth_header():
    c = dw.DicomwebClient(
        "https://example/dicomweb/",
        headers={"Authorization": "Bearer abc"},
        timeout=5.0,
    )
    # base_url should strip the trailing slash for clean URL joining.
    assert c.base_url == "https://example/dicomweb"
    assert c.headers["Authorization"] == "Bearer abc"
    assert c.timeout == 5.0
