"""DICOMweb client tests. Covered:

  - multipart/related parsing of synthetic responses
  - RLE Lossless decode (DICOM Annex G PackBits-like layout)
  - Transfer-syntax dispatch when given a synthesized response body
    that wraps a known-codec encoded payload
  - Error handling for unsupported transfer syntaxes
  - The whole client over HTTP, against the local DICOMweb server in
    _dicomweb_server.py (QIDO-RS search, WADO-RS frames, content
    negotiation, errors), with no network access
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


# ---------------------------------------------------------------------------
# End to end over HTTP, against the local DICOMweb server in
# _dicomweb_server.py: a synthetic study whose frames were written by
# reference encoders (imagecodecs, pydicom).
# ---------------------------------------------------------------------------

AS_STORED = 'multipart/related; type="application/octet-stream"; transfer-syntax=*'


@pytest.fixture(scope="module")
def study():
    pytest.importorskip("imagecodecs")
    pytest.importorskip("pydicom")
    from _dicomweb_server import synthetic_study
    return synthetic_study()


@pytest.fixture(scope="module")
def served(study):
    from _dicomweb_server import dicomweb_server
    with dicomweb_server(study) as (base, requests):
        yield base, requests


def _qido(url):
    import json
    import urllib.request
    req = urllib.request.Request(url, headers={"Accept": "application/dicom+json"})
    with urllib.request.urlopen(req, timeout=10) as resp:
        return json.loads(resp.read())


def _value(element, tag):
    return element[tag]["Value"][0]


def _pixel_options(instance_json):
    return dict(
        rows=_value(instance_json, "00280010"),
        columns=_value(instance_json, "00280011"),
        bits_allocated=_value(instance_json, "00280100"),
        samples_per_pixel=_value(instance_json, "00280002"),
        pixel_representation=_value(instance_json, "00280103"),
    )


def _walk(base):
    """Every (study, series, instance JSON) found by QIDO-RS alone."""
    client = dw.DicomwebClient(base, timeout=10)
    for study_json in _qido(f"{base}/studies"):
        study_uid = _value(study_json, "0020000D")
        for series_json in _qido(f"{base}/studies/{study_uid}/series"):
            series_uid = _value(series_json, "0020000E")
            for inst in client.list_instances(study_uid, series_uid):
                yield study_uid, series_uid, inst


def test_qido_walk_finds_every_instance(study, served):
    base, _ = served
    found = {_value(i, "00080018"): _value(i, "00083002") for *_, i in _walk(base)}
    expected = {i.uid: i.transfer_syntax for s in study.series for i in s.instances}
    assert found == expected


def _modules_for(ts):
    return {
        dw.TS_JPEG_BASELINE_1: "opencodecs.codecs._jpeg",
        dw.TS_JPEGLS_LOSSLESS: "opencodecs.codecs._charls",
        dw.TS_JPEG2K_LOSSLESS: "opencodecs.codecs._jpeg2k",
        dw.TS_HTJ2K_LOSSLESS: "opencodecs.codecs._openjph",
    }.get(ts)


def test_every_stored_syntax_decodes_to_the_reference(study, served):
    """Each frame as stored, fetched and decoded by the client."""
    base, _ = served
    client = dw.DicomwebClient(base, timeout=10)
    checked = set()
    for study_uid, series_uid, inst_json in _walk(base):
        inst = study.instance(_value(inst_json, "00080018"))
        module = _modules_for(inst.transfer_syntax)
        if module is not None:
            try:
                __import__(module)
            except ImportError:
                continue
        for n in range(1, inst.pixels.shape[0] + 1):
            frame = client.get_frame(study_uid, series_uid, inst.uid, n,
                                     accept=AS_STORED, **_pixel_options(inst_json))
            np.testing.assert_array_equal(
                frame, inst.expected[n - 1],
                err_msg=f"{inst.transfer_syntax} frame {n}")
        checked.add(inst.transfer_syntax)
    # Raw and RLE need no optional codec, so they are always exercised.
    assert {dw.TS_EXPLICIT_VR_LE, dw.TS_RLE_LOSSLESS} <= checked


def test_default_accept_asks_for_uncompressed_frames(study, served):
    """The client's default Accept names no transfer syntax, which PS3.18
    reads as Explicit VR Little Endian: a conforming server transcodes, so
    the caller gets raw bytes and must supply the pixel geometry."""
    base, _ = served
    client = dw.DicomwebClient(base, timeout=10)
    study_uid, series_uid, inst_json = next(
        t for t in _walk(base) if _value(t[2], "00083002") == dw.TS_JPEGLS_LOSSLESS)
    inst = study.instance(_value(inst_json, "00080018"))
    frame = client.get_frame(study_uid, series_uid, inst.uid, 1,
                             **_pixel_options(inst_json))
    np.testing.assert_array_equal(frame, inst.pixels[0])
    with pytest.raises(DicomwebError, match="rows/columns/bits_allocated"):
        client.get_frame(study_uid, series_uid, inst.uid, 1)


def test_iter_frames_over_http_keeps_request_order(study, served):
    base, _ = served
    client = dw.DicomwebClient(base, timeout=10)
    study_uid, series_uid, inst_json = next(
        t for t in _walk(base) if _value(t[2], "00280008") > 1)
    inst = study.instance(_value(inst_json, "00080018"))
    order = [3, 1, 5, 2, 4]
    frames = list(client.iter_frames(
        study_uid, series_uid, inst.uid, order, max_frame_bytes=1 << 20,
        numthreads=3, accept=AS_STORED, **_pixel_options(inst_json)))
    for n, frame in zip(order, frames):
        np.testing.assert_array_equal(frame, inst.pixels[n - 1])
    with pytest.raises(DicomwebError, match="max_response_bytes"):
        list(client.iter_frames(study_uid, series_uid, inst.uid, [1],
                                max_frame_bytes=64, accept=AS_STORED,
                                **_pixel_options(inst_json)))


def test_client_headers_reach_the_server(served):
    base, requests = served
    client = dw.DicomwebClient(base, headers={"Authorization": "Bearer t0ken"})
    study_uid = _value(_qido(f"{base}/studies")[0], "0020000D")
    series_uid = _value(_qido(f"{base}/studies/{study_uid}/series")[0], "0020000E")
    del requests[:]
    client.list_instances(study_uid, series_uid)
    assert requests[-1]["headers"].get("Authorization") == "Bearer t0ken"
    assert requests[-1]["headers"].get("Accept") == "application/dicom+json"


def test_missing_instance_and_frame_are_http_errors(study, served):
    import urllib.error
    base, _ = served
    client = dw.DicomwebClient(base, timeout=10)
    series = study.series[0]
    inst = series.instances[0]
    with pytest.raises(urllib.error.HTTPError) as err:
        client.get_frame(study.uid, series.uid, "2.25.1", 1, accept=AS_STORED)
    assert err.value.code == 404
    with pytest.raises(urllib.error.HTTPError) as err:
        client.get_frame(study.uid, series.uid, inst.uid, 99, accept=AS_STORED)
    assert err.value.code == 404
