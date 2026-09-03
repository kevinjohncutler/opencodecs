"""DICOM file reader.

The corpus carries the same MR slice in four transfer syntaxes, which is
the whole design of these tests: the picture is constant, so any
difference between decodes is our parser rather than the image. Explicit
VR, implicit VR, RLE and JPEG 2000 exercise two dataset-parsing modes and
both native and encapsulated Pixel Data.
"""

from __future__ import annotations

import pathlib
import struct

import numpy as np
import pytest

from opencodecs._dicom import DicomError, DicomFile

DATA = pathlib.Path(__file__).resolve().parent.parent / ".test_data" / "dicom"
MR_EXPLICIT = DATA / "MR_small.dcm"
MR_IMPLICIT = DATA / "MR_small_implicit.dcm"
MR_RLE = DATA / "MR_small_RLE.dcm"
MR_J2K = DATA / "MR_small_jp2klossless.dcm"
CT = DATA / "CT_small.dcm"

ALL_MR = [MR_EXPLICIT, MR_IMPLICIT, MR_RLE, MR_J2K]
needs_corpus = pytest.mark.skipif(
    not MR_EXPLICIT.is_file(),
    reason="run `python corpus/corpus.py fetch dicom_pydicom_variants`")


# --------------------------------------------------------------------
# the four-syntax invariant
# --------------------------------------------------------------------

@needs_corpus
@pytest.mark.parametrize("path", ALL_MR, ids=lambda p: p.stem)
def test_every_transfer_syntax_yields_the_same_image(path):
    """One slice, four encodings, one expected array.

    This is the strongest check available without a second
    implementation: explicit VR, implicit VR, RLE and JPEG 2000 differ in
    how the dataset is parsed and whether Pixel Data is native or
    encapsulated, and none of that should reach the pixels.
    """
    with DicomFile(str(MR_EXPLICIT)) as ref:
        expected = ref.asarray()
    with DicomFile(str(path)) as d:
        got = d.asarray()
    assert got.shape == expected.shape
    assert got.dtype == expected.dtype
    assert np.array_equal(got, expected)


@needs_corpus
def test_signed_data_in_an_unsigned_codestream():
    """The trap that a raw JPEG 2000 decode walks into.

    This file's header says PixelRepresentation 1 (signed) while the
    JPEG 2000 codestream stores unsigned samples with a DC level shift.
    Returning the codec's array untouched gives values exactly 32768 too
    high, which looks like a plausible image and is wrong everywhere.
    """
    with DicomFile(str(MR_J2K)) as d:
        assert d.pixel_representation == 1
        assert d.is_encapsulated
        got = d.asarray()
    assert got.dtype == np.dtype("int16")
    assert int(got.min()) == 127 and int(got.max()) == 2145


@needs_corpus
@pytest.mark.parametrize("path", ALL_MR + [CT], ids=lambda p: p.stem)
def test_matches_pydicom(path):
    pydicom = pytest.importorskip("pydicom")
    ref = pydicom.dcmread(str(path)).pixel_array
    with DicomFile(str(path)) as d:
        got = d.asarray()
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert np.array_equal(got, ref)


# --------------------------------------------------------------------
# header interpretation
# --------------------------------------------------------------------

@needs_corpus
def test_transfer_syntax_and_encapsulation_are_reported():
    expected = {
        "MR_small": ("1.2.840.10008.1.2.1", False),
        "MR_small_implicit": ("1.2.840.10008.1.2", False),
        "MR_small_RLE": ("1.2.840.10008.1.2.5", True),
        "MR_small_jp2klossless": ("1.2.840.10008.1.2.4.90", True),
    }
    for path in ALL_MR:
        ts, enc = expected[path.stem]
        with DicomFile(str(path)) as d:
            assert d.transfer_syntax == ts, path.name
            assert d.is_encapsulated is enc, path.name


@needs_corpus
def test_geometry_from_the_header():
    with DicomFile(str(MR_EXPLICIT)) as d:
        assert (d.rows, d.columns) == (64, 64)
        assert d.samples_per_pixel == 1
        assert d.bits_allocated == 16
        assert d.n_frames == 1
        assert d.shape == (64, 64)


@needs_corpus
def test_rescale_is_off_by_default_and_applied_on_request():
    """Rescale turns stored values into Hounsfield units on a CT.

    Off by default because it changes the dtype to float, and a caller
    reading a codec usually wants what the file stores.
    """
    with DicomFile(str(CT)) as d:
        slope, inter = d.rescale
        raw = d.asarray()
        scaled = d.asarray(rescale=True)
    assert raw.dtype == np.dtype("int16")
    if slope != 1.0 or inter != 0.0:
        assert np.allclose(scaled, raw * slope + inter)
    else:
        assert np.array_equal(scaled, raw)


@needs_corpus
def test_implicit_vr_numbers_are_read_without_a_vr():
    """Implicit VR carries no type, so Rows must still come back as 64."""
    with DicomFile(str(MR_IMPLICIT)) as d:
        assert d.transfer_syntax == "1.2.840.10008.1.2"
        assert (d.rows, d.columns, d.bits_allocated) == (64, 64, 16)


@needs_corpus
def test_frame_and_iteration_agree():
    with DicomFile(str(MR_EXPLICIT)) as d:
        assert np.array_equal(d.frame(0), d.asarray())
        assert len(list(iter(d))) == d.n_frames
        with pytest.raises(IndexError):
            d.frame(5)


# --------------------------------------------------------------------
# refusals
# --------------------------------------------------------------------

def test_not_dicom_raises():
    with pytest.raises(DicomError, match="not a DICOM file"):
        DicomFile(b"\x00" * 64 + b"nothing here")


def test_too_short_raises():
    with pytest.raises(DicomError, match="too short"):
        DicomFile(b"\x01\x02")


@needs_corpus
def test_truncated_pixel_data_raises():
    """A file cut short must not return whatever bytes happen to remain."""
    blob = MR_EXPLICIT.read_bytes()
    with pytest.raises(DicomError, match="truncated"):
        DicomFile(blob[:-2000]).asarray()


def test_preamble_is_optional():
    """Datasets pulled off a network stream have no 128-byte preamble.

    Built here rather than fetched: the corpus files all have one, and
    the point is the path that triggers when the magic is absent.
    """
    body = b""
    for tag, vr, value in (
        ((0x0028, 0x0010), b"US", struct.pack("<H", 2)),   # Rows
        ((0x0028, 0x0011), b"US", struct.pack("<H", 3)),   # Columns
        ((0x0028, 0x0100), b"US", struct.pack("<H", 8)),   # BitsAllocated
        ((0x7FE0, 0x0010), b"OB", bytes(range(6))),        # PixelData
    ):
        body += struct.pack("<HH", *tag) + vr
        if vr in (b"OB",):
            body += b"\x00\x00" + struct.pack("<I", len(value))
        else:
            body += struct.pack("<H", len(value))
        body += value
    d = DicomFile(body)
    assert d.transfer_syntax == "1.2.840.10008.1.2"        # the default
    assert d.shape == (2, 3)
    assert np.array_equal(d.asarray(), np.arange(6, dtype="u1").reshape(2, 3))
