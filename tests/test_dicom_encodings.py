"""The four transfer syntaxes that decide how a dataset is encoded.

Every DICOM file names its encoding in the File Meta group, and the
reader has to actually act on it. These are the cases where it did not:

* **Explicit VR big-endian** (1.2.840.10008.1.2.2), retired but still in
  older archives. Rows and Columns were unpacked little-endian whatever
  the file said, and 16-bit samples came back byte-swapped.
* **Deflated** (1.2.840.10008.1.2.1.99). The dataset is a raw deflate
  stream, so it has no random access at all: inflating a 64 KiB prefix
  gave zlib a truncated stream, and the offsets it produced indexed the
  inflated buffer while the pixel reads went to the file.
* **Implicit VR little-endian** (1.2.840.10008.1.2). No VR is stored, so
  a US value has to be recognized from a data dictionary. Guessing
  "decimal string first" reads Columns 48, stored as ``b"\\x30\\x00"``,
  as the ASCII "0" it happens to spell, and hands back an empty image.
* Undefined-length **sequences** whose items are themselves of undefined
  length. Their contents are ordinary elements, not more items; walking
  them as items desynchronizes everything after the sequence, and the
  file then parses cleanly and reports no Pixel Data.

pydicom is the cross-check where it is installed, because a fixture
written by the same assumptions as the reader validates nothing. The
synthetic builders below are the fallback so these still run without it.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

from opencodecs._dicom import DicomError, DicomFile

TS_IMPLICIT = "1.2.840.10008.1.2"
TS_EXPLICIT_LE = "1.2.840.10008.1.2.1"
TS_EXPLICIT_BE = "1.2.840.10008.1.2.2"
TS_DEFLATED = "1.2.840.10008.1.2.1.99"

_LONG_VRS = (b"OB", b"OW", b"SQ", b"UN", b"UT")


class Encoding:
    """How to lay out one dataset: explicit or not, and which byte order."""

    def __init__(self, ts, explicit=True, bo="<", deflate=False):
        self.ts, self.explicit, self.bo, self.deflate = ts, explicit, bo, deflate

    def element(self, tag, vr, value):
        if not self.explicit:
            return (struct.pack(self.bo + "HH", *tag)
                    + struct.pack(self.bo + "I", len(value)) + value)
        head = struct.pack(self.bo + "HH", *tag) + vr
        if vr in _LONG_VRS:
            head += b"\x00\x00" + struct.pack(self.bo + "I", len(value))
        else:
            head += struct.pack(self.bo + "H", len(value))
        return head + value

    def us(self, tag, value):
        return self.element(tag, b"US", struct.pack(self.bo + "H", value))


def explicit_le():
    return Encoding(TS_EXPLICIT_LE)


def implicit_le():
    return Encoding(TS_IMPLICIT, explicit=False)


def explicit_be():
    return Encoding(TS_EXPLICIT_BE, bo=">")


def deflated():
    return Encoding(TS_DEFLATED, deflate=True)


def _meta(ts):
    """The File Meta group, always explicit VR little-endian."""
    uid = ts.encode()
    if len(uid) % 2:
        uid += b"\x00"
    body = struct.pack("<HH", 0x0002, 0x0010) + b"UI" \
        + struct.pack("<H", len(uid)) + uid
    length = struct.pack("<HH", 0x0002, 0x0000) + b"UL" \
        + struct.pack("<H", 4) + struct.pack("<I", len(body))
    return b"\x00" * 128 + b"DICM" + length + body


def sequence(enc, undefined_items):
    """A Procedure Code Sequence of undefined length, with one item."""
    inner = enc.element((0x0008, 0x0100), b"SH", b"CODE01")
    if undefined_items:
        item = (struct.pack(enc.bo + "HHI", 0xFFFE, 0xE000, 0xFFFFFFFF)
                + inner
                + struct.pack(enc.bo + "HHI", 0xFFFE, 0xE00D, 0))
    else:
        item = struct.pack(enc.bo + "HHI", 0xFFFE, 0xE000, len(inner)) + inner
    head = struct.pack(enc.bo + "HH", 0x0008, 0x1032)
    if enc.explicit:
        head += b"SQ" + b"\x00\x00"
    head += struct.pack(enc.bo + "I", 0xFFFFFFFF)
    return head + item + struct.pack(enc.bo + "HHI", 0xFFFE, 0xE0DD, 0)


def build_dicom(enc, pixels, *, extra=b""):
    """One complete file holding ``pixels`` as native Pixel Data."""
    frames = pixels.shape[0] if pixels.ndim == 3 else 1
    rows, cols = pixels.shape[-2:]
    signed = pixels.dtype.kind == "i"
    body = (enc.us((0x0028, 0x0002), 1)
            + enc.element((0x0028, 0x0008), b"IS", str(frames).encode().ljust(2))
            + enc.us((0x0028, 0x0010), rows)
            + enc.us((0x0028, 0x0011), cols)
            + enc.us((0x0028, 0x0100), pixels.dtype.itemsize * 8)
            + enc.us((0x0028, 0x0103), 1 if signed else 0)
            + extra)
    raw = pixels.astype(enc.bo + pixels.dtype.str[1:]).tobytes()
    body += enc.element((0x7FE0, 0x0010), b"OW", raw)
    if enc.deflate:
        co = zlib.compressobj(9, zlib.DEFLATED, -zlib.MAX_WBITS)
        body = co.compress(body) + co.flush()
    return _meta(enc.ts) + body


ENCODINGS = [("explicit LE", explicit_le), ("implicit LE", implicit_le),
             ("explicit BE", explicit_be), ("deflated", deflated)]


# --------------------------------------------------------------------
# every encoding, on our own fixtures
# --------------------------------------------------------------------

@pytest.mark.parametrize("name,make", ENCODINGS, ids=[n for n, _ in ENCODINGS])
def test_every_transfer_syntax_decodes(name, make):
    a = (np.arange(3 * 8 * 12) % 4000).astype("u2").reshape(3, 8, 12)
    with DicomFile(build_dicom(make(), a)) as d:
        assert (d.rows, d.columns, d.n_frames) == (8, 12, 3)
        assert np.array_equal(d.asarray(), a)


@pytest.mark.parametrize("name,make", ENCODINGS, ids=[n for n, _ in ENCODINGS])
@pytest.mark.parametrize("undefined_items", [False, True],
                         ids=["defined-item", "undefined-item"])
def test_a_sequence_before_pixel_data_stays_in_step(name, make,
                                                    undefined_items):
    enc = make()
    a = (np.arange(2 * 6 * 6) % 900).astype("u2").reshape(2, 6, 6)
    blob = build_dicom(enc, a, extra=sequence(enc, undefined_items))
    with DicomFile(blob) as d:
        assert np.array_equal(d.asarray(), a)


@pytest.mark.parametrize("name,make", ENCODINGS, ids=[n for n, _ in ENCODINGS])
def test_signed_samples_survive_every_encoding(name, make):
    a = (np.arange(2 * 4 * 5) - 20).astype("i2").reshape(2, 4, 5)
    with DicomFile(build_dicom(make(), a)) as d:
        assert d.dtype.kind == "i"
        assert np.array_equal(d.asarray(), a)


def test_big_endian_dtype_says_so():
    """The reported dtype has to describe the array actually returned."""
    a = (np.arange(4 * 6) + 1000).astype("u2").reshape(4, 6)
    with DicomFile(build_dicom(explicit_be(), a)) as d:
        assert d.dtype.byteorder == ">"
        assert d.asarray().dtype.byteorder == ">"
        assert np.array_equal(d.asarray(), a)


def test_implicit_vr_columns_are_not_read_as_a_decimal_string():
    """Columns 48 is b"\\x30\\x00", which spells "0" in ASCII.

    A string-first guess parses that as zero and returns a (rows, 0)
    array, which has a plausible shape and no pixels in it.
    """
    a = np.arange(4 * 48, dtype="u2").reshape(4, 48)
    with DicomFile(build_dicom(implicit_le(), a)) as d:
        assert d.columns == 48
        assert d.asarray().shape == (4, 48)


def test_deflated_files_larger_than_the_header_prefix():
    """The prefix read is 64 KiB; deflate cannot be resumed from it."""
    # Noise, so the deflated stream really is bigger than the prefix.
    # A ramp compresses to a few kilobytes and tests nothing.
    a = np.random.default_rng(0).integers(0, 65535, (400, 400), dtype="u2")
    blob = build_dicom(deflated(), a)
    assert len(blob) > DicomFile._PREFIX, "fixture must exceed the prefix"
    with DicomFile(blob) as d:
        assert np.array_equal(d.asarray(), a)


def test_pixel_data_inside_a_sequence_is_not_the_image():
    """An Icon Image Sequence carries Pixel Data of its own.

    Taking the first Pixel Data element found would return the icon,
    which has a plausible shape and is not the acquisition.
    """
    enc = explicit_le()
    icon = np.full((4, 4), 9, dtype="u2")
    inner = (enc.us((0x0028, 0x0010), 4) + enc.us((0x0028, 0x0011), 4)
             + enc.element((0x7FE0, 0x0010), b"OW", icon.tobytes()))
    item = struct.pack("<HHI", 0xFFFE, 0xE000, len(inner)) + inner
    seq = (struct.pack("<HH", 0x0088, 0x0200) + b"SQ" + b"\x00\x00"
           + struct.pack("<I", 0xFFFFFFFF) + item
           + struct.pack("<HHI", 0xFFFE, 0xE0DD, 0))
    a = (np.arange(16 * 20) % 700).astype("u2").reshape(16, 20)
    with DicomFile(build_dicom(enc, a, extra=seq)) as d:
        assert (d.rows, d.columns) == (16, 20)
        assert np.array_equal(d.asarray(), a)


# --------------------------------------------------------------------
# pydicom writes, we read
# --------------------------------------------------------------------

PYDICOM_CASES = [
    ("explicit LE", "1.2.840.10008.1.2.1", False),
    ("implicit LE", "1.2.840.10008.1.2", False),
    ("explicit BE", "1.2.840.10008.1.2.2", False),
    ("deflated", "1.2.840.10008.1.2.1.99", False),
    ("explicit LE + SQ", "1.2.840.10008.1.2.1", True),
    ("implicit LE + SQ", "1.2.840.10008.1.2", True),
    ("explicit BE + SQ", "1.2.840.10008.1.2.2", True),
    ("deflated + SQ", "1.2.840.10008.1.2.1.99", True),
]


@pytest.mark.parametrize("name,ts,with_seq", PYDICOM_CASES,
                         ids=[c[0] for c in PYDICOM_CASES])
@pytest.mark.parametrize("signed", [False, True], ids=["unsigned", "signed"])
def test_matches_pydicom(tmp_path, name, ts, with_seq, signed):
    pydicom = pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset

    ds = Dataset()
    ds.preamble = b"\x00" * 128
    ds.file_meta = FileMetaDataset()
    ds.file_meta.TransferSyntaxUID = ts
    ds.file_meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    ds.file_meta.MediaStorageSOPInstanceUID = "1.2.3.4"
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.Rows, ds.Columns, ds.NumberOfFrames = 32, 24, 3
    ds.BitsAllocated = ds.BitsStored = 16
    ds.HighBit = 15
    ds.PixelRepresentation = 1 if signed else 0
    if with_seq:
        item = Dataset()
        item.CodeValue = "CODE01"
        nested = Dataset()
        nested.CodeValue = "NESTED"
        item.ConceptNameCodeSequence = [nested]
        ds.ProcedureCodeSequence = [item]

    order = ">" if ts == "1.2.840.10008.1.2.2" else "<"
    a = (np.arange(3 * 32 * 24) % 4000 - (3000 if signed else 0))
    a = a.astype(order + ("i2" if signed else "u2")).reshape(3, 32, 24)
    ds.PixelData = a.tobytes()
    path = tmp_path / "case.dcm"
    # The transfer syntax in the File Meta group already says how to
    # encode; pydicom 3 deprecates being told a second time.
    ds.save_as(str(path), enforce_file_format=False,
               little_endian=ts != "1.2.840.10008.1.2.2",
               implicit_vr=ts == "1.2.840.10008.1.2")

    ref = pydicom.dcmread(str(path)).pixel_array
    with DicomFile(str(path)) as d:
        got = d.asarray()
    assert got.shape == ref.shape
    assert np.array_equal(got, ref)


def test_truncated_pixel_data_is_reported(tmp_path):
    a = np.arange(4 * 6, dtype="u2").reshape(4, 6)
    blob = build_dicom(explicit_le(), a)
    path = tmp_path / "cut.dcm"
    path.write_bytes(blob[:-10])
    with DicomFile(str(path)) as d:
        with pytest.raises(DicomError, match="truncated"):
            d.asarray()
