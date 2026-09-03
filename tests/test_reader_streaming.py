"""Byte savings for the readers that reach their data by offset.

MRC already had this. DICOM and NRRD did not, despite both putting
native pixel data at a fixed offset behind a header, which is exactly
the shape range requests are for. The assertion in each case is what the
server actually sent, because "the URL opened" is not the claim.

All three now reach storage through ``core._io_helpers.open_read_at``,
so this file is also the test that the shared plumbing works for each.
"""

from __future__ import annotations

import numpy as np
import pytest

from _range_http_server import range_http_server

from opencodecs._dicom import DicomFile
from opencodecs._mrc_writer import encode_mrc
from opencodecs._nrrd import NrrdFile


def write_nrrd(path, a):
    sizes = " ".join(str(s) for s in a.shape[::-1])
    head = (f"NRRD0004\ntype: short\ndimension: {a.ndim}\nsizes: {sizes}\n"
            f"encoding: raw\nendian: little\n\n").encode()
    path.write_bytes(head + a.astype("<i2").tobytes(order="F"))
    return len(head) + a.nbytes


def build_dicom(rows, cols, frames, fill=7):
    """A minimal explicit-VR file with native multi-frame Pixel Data."""
    import struct
    body = b""
    for tag, vr, value in (
        ((0x0028, 0x0002), b"US", struct.pack("<H", 1)),
        ((0x0028, 0x0008), b"IS", str(frames).encode().ljust(2)),
        ((0x0028, 0x0010), b"US", struct.pack("<H", rows)),
        ((0x0028, 0x0011), b"US", struct.pack("<H", cols)),
        ((0x0028, 0x0100), b"US", struct.pack("<H", 16)),
        ((0x0028, 0x0103), b"US", struct.pack("<H", 0)),
    ):
        body += struct.pack("<HH", *tag) + vr + struct.pack("<H", len(value)) + value
    pixels = np.zeros((frames, rows, cols), dtype="<u2")
    for f in range(frames):
        pixels[f] = f + fill
    raw = pixels.tobytes()
    body += (struct.pack("<HH", 0x7FE0, 0x0010) + b"OW" + b"\x00\x00"
             + struct.pack("<I", len(raw)) + raw)
    return b"\x00" * 128 + b"DICM" + body, pixels


# --------------------------------------------------------------------

def test_nrrd_reads_only_its_volume_over_http(tmp_path):
    a = (np.arange(64 * 256 * 256, dtype="i2") % 300).reshape(64, 256, 256)
    total = write_nrrd(tmp_path / "v.nrrd", a)

    with range_http_server(tmp_path) as (url, tracker):
        with NrrdFile(f"{url}/v.nrrd") as f:
            assert f.shape == (64, 256, 256)
            header_only = tracker.bytes_served
            got = f.asarray()
        served = tracker.bytes_served
    assert np.array_equal(got, a)
    # Reading the header must not pull the volume.
    assert header_only < total // 10, (
        f"parsing the header moved {header_only} of {total} bytes")
    # And reading the volume must not pull it twice.
    assert served < total * 1.5


def test_nrrd_header_costs_one_block(tmp_path):
    a = np.zeros((32, 256, 256), dtype="i2")
    total = write_nrrd(tmp_path / "v.nrrd", a)
    with range_http_server(tmp_path) as (url, tracker):
        with NrrdFile(f"{url}/v.nrrd") as f:
            _ = f.shape, f.dtype
        served = tracker.bytes_served
    assert served <= 64 * 1024
    assert served < total // 10


def test_dicom_reads_one_frame_not_the_series(tmp_path):
    """Frame 40 of a series should cost one frame, not forty."""
    blob, pixels = build_dicom(128, 128, 48)
    (tmp_path / "s.dcm").write_bytes(blob)
    frame_bytes = 128 * 128 * 2

    with range_http_server(tmp_path) as (url, tracker):
        with DicomFile(f"{url}/s.dcm") as d:
            assert d.n_frames == 48
            got = d.frame(40)
        served = tracker.bytes_served
    assert np.array_equal(got, pixels[40])
    assert served < len(blob) // 2, (
        f"one {frame_bytes}-byte frame moved {served} of {len(blob)} bytes")


def test_dicom_header_parse_does_not_read_the_pixels(tmp_path):
    blob, _ = build_dicom(256, 256, 32)
    (tmp_path / "s.dcm").write_bytes(blob)
    with range_http_server(tmp_path) as (url, tracker):
        with DicomFile(f"{url}/s.dcm") as d:
            assert (d.rows, d.columns, d.n_frames) == (256, 256, 32)
        served = tracker.bytes_served
    assert served < len(blob) // 4, (
        f"header parse moved {served} of {len(blob)} bytes")


@pytest.mark.parametrize("frames", [1, 8])
def test_http_and_local_dicom_agree(tmp_path, frames):
    blob, pixels = build_dicom(16, 24, frames)
    (tmp_path / "s.dcm").write_bytes(blob)
    with range_http_server(tmp_path) as (url, _):
        with DicomFile(f"{url}/s.dcm") as remote:
            over_http = remote.asarray()
    with DicomFile(str(tmp_path / "s.dcm")) as local:
        on_disk = local.asarray()
    assert np.array_equal(over_http, on_disk)
    expected = pixels[0] if frames == 1 else pixels
    assert np.array_equal(over_http, expected)


def test_http_and_local_nrrd_agree(tmp_path):
    a = (np.arange(4 * 32 * 32, dtype="i2") % 91).reshape(4, 32, 32)
    write_nrrd(tmp_path / "v.nrrd", a)
    with range_http_server(tmp_path) as (url, _):
        with NrrdFile(f"{url}/v.nrrd") as remote:
            over_http = remote.asarray()
    with NrrdFile(str(tmp_path / "v.nrrd")) as local:
        on_disk = local.asarray()
    assert np.array_equal(over_http, on_disk)
    assert np.array_equal(over_http, a)


def test_compressed_nrrd_still_works_over_http(tmp_path):
    """gzip has no random access, so this one legitimately reads it all."""
    import gzip
    a = (np.arange(2 * 16 * 16, dtype="i2")).reshape(2, 16, 16)
    head = (b"NRRD0004\ntype: short\ndimension: 3\nsizes: 16 16 2\n"
            b"encoding: gzip\nendian: little\n\n")
    (tmp_path / "c.nrrd").write_bytes(
        head + gzip.compress(a.astype("<i2").tobytes(order="F")))
    with range_http_server(tmp_path) as (url, _):
        with NrrdFile(f"{url}/c.nrrd") as f:
            assert np.array_equal(f.asarray(), a)
