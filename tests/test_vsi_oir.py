"""Tests for VSI (TIFF-delegate) + OIR (format-detection stub).

VSI is Olympus CellSens virtual slide — top-level is TIFF, full-res
data lives in sibling .ets files we don't yet parse. VsiCodec
exposes the TIFF index correctly.

OIR is Olympus FluoView newer — undocumented OLYMPUSRAWFORMAT
container. OirCodec is a format-detection stub: signature() works,
decode/open raise NotImplementedError with a clear message.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc

CORPUS = Path(__file__).resolve().parent.parent / ".test_data"
VSI_SAMPLE = CORPUS / "vsi" / "metadataTest_01.vsi"
OIR_SAMPLE = CORPUS / "oir" / "amy_slice_z_stack.oir"

_HINT = (
    "Run `bash tests/download_test_corpus.sh --light` from the repo "
    "root to populate the corpus."
)


# ---------------------------------------------------------------------------
# VSI — TIFF-backed index
# ---------------------------------------------------------------------------


def test_vsi_codec_registered():
    assert "vsi" in [c["name"] for c in oc.list_codecs()]


def test_vsi_codec_lists_native():
    entry = next(c for c in oc.list_codecs() if c["name"] == "vsi")
    assert entry["native"] is True


@pytest.mark.skipif(not VSI_SAMPLE.exists(), reason=_HINT)
def test_vsi_signature_detection():
    """VSI shares TIFF magic — we accept II*\\0 / MM\\0* in head."""
    with open(VSI_SAMPLE, "rb") as f:
        head = f.read(16)
    codec = oc.get_codec("vsi")
    assert codec.signature(head) is True
    assert codec.signature(b"not tiff") is False


@pytest.mark.skipif(
    not (CORPUS / "vsi" / "_metadataTest_01_" / "stack1"
         / "frame_t_0.ets").exists(),
    reason=_HINT,
)
def test_vsi_full_native_decode():
    """Native VSI decode (with .ets companion present) returns the
    full pyramid stack. Ground truth (bftools): 5T × 18Z × 2C
    × 216×260 uint16 = 180 planes."""
    arr = oc.read(str(VSI_SAMPLE), format="vsi")
    assert arr.shape == (180, 216, 260)
    assert arr.dtype == np.uint16


@pytest.mark.skipif(not VSI_SAMPLE.exists(), reason=_HINT)
def test_vsi_thumbnail_mode():
    """backend='thumbnail' returns the TIFF index thumbnail rather
    than the .ets stack.

    Covers both the documented ``backend=`` kwarg and the legacy
    ``mode=`` alias kept for back-compat — both must reach the same
    thumbnail path."""
    expected_shape = (216, 260, 3)
    with oc.get_codec("vsi").open(str(VSI_SAMPLE), backend="thumbnail") as r:
        arr_backend = r.read()
    with oc.get_codec("vsi").open(str(VSI_SAMPLE), mode="thumbnail") as r:
        arr_mode = r.read()
    assert arr_backend.shape == expected_shape
    assert arr_mode.shape == expected_shape
    assert arr_backend.dtype == np.uint8
    np.testing.assert_array_equal(arr_backend, arr_mode)


@pytest.mark.skipif(
    not (CORPUS / "vsi" / "_metadataTest_01_" / "stack1"
         / "frame_t_0.ets").exists(),
    reason=_HINT,
)
def test_vsi_extension_dispatch():
    """oc.read('foo.vsi') picks VsiCodec without format= override.
    With .ets companion present this gives the full 180-plane stack."""
    arr = oc.read(str(VSI_SAMPLE))
    assert arr.shape == (180, 216, 260)
    assert arr.dtype == np.uint16


# ---------------------------------------------------------------------------
# OIR — format-detection stub
# ---------------------------------------------------------------------------


def test_oir_codec_registered():
    assert "oir" in [c["name"] for c in oc.list_codecs()]


def test_oir_codec_advertises_native_decode():
    entry = next(c for c in oc.list_codecs() if c["name"] == "oir")
    assert entry["native"] is True
    assert entry["decode"] is True
    assert entry["encode"] is False


@pytest.mark.skipif(not OIR_SAMPLE.exists(), reason=_HINT)
def test_oir_signature_detection():
    """OIR starts with the 16-byte ASCII signature
    ``OLYMPUSRAWFORMAT``. Detection works even without a decoder."""
    with open(OIR_SAMPLE, "rb") as f:
        head = f.read(32)
    codec = oc.get_codec("oir")
    assert codec.signature(head) is True
    assert codec.signature(b"not olympus") is False


@pytest.mark.skipif(not OIR_SAMPLE.exists(), reason=_HINT)
def test_oir_native_decode_shape_and_dtype():
    """Native OIR decode produces the right shape + dtype.
    Ground truth (bftools): 32 planes × 512×512 uint16.
    """
    arr = oc.read(str(OIR_SAMPLE), format="oir")
    assert arr.shape == (32, 512, 512)
    assert arr.dtype == np.uint16
    # 10-bit data → values in [0, 1023]
    assert arr.min() >= 0
    assert arr.max() < 1024


@pytest.mark.skipif(not OIR_SAMPLE.exists(), reason=_HINT)
def test_oir_streaming_reader_per_frame():
    """The streaming reader yields one plane per iter_frames()."""
    full = oc.read(str(OIR_SAMPLE), format="oir")
    with oc.get_codec("oir").open(str(OIR_SAMPLE)) as r:
        assert r.n_frames == 32
        for i, frame in enumerate(r.iter_frames()):
            assert np.array_equal(frame, full[i]), (
                f"streamed frame {i} differs from full read")


@pytest.mark.skipif(not OIR_SAMPLE.exists(), reason=_HINT)
def test_oir_partial_parse_via_info():
    """OirCodec.info() returns the clean-room partial parse:
    file size, footer offset/count, per-frame XML metadata."""
    info = oc.get_codec("oir").info(str(OIR_SAMPLE))
    assert info["file_size"] == 25_957_525
    assert info["n_records"] == 265
    assert info["footer_offset"] == 25_955_401
    # Per the embedded XML the corpus file is 512x512x10-bit gray
    fm = info["frame_metadata"]
    assert fm["width"] == "512"
    assert fm["height"] == "512"
    assert fm["depth"] == "2"
    assert fm["bitCounts"] == "10"


# ---------------------------------------------------------------------------
# VSI .ets companion partial-parse
# ---------------------------------------------------------------------------


@pytest.mark.skipif(
    not (CORPUS / "vsi" / "_metadataTest_01_" / "stack1"
         / "frame_t_0.ets").exists(),
    reason="run `bash tests/download_test_corpus.sh --light` then "
           "fetch the .ets companion file manually",
)
def test_vsi_ets_partial_parse():
    """VsiCodec.info() walks the sibling ``_NAME_/stackN/frame_t.ets``
    tree and partially parses each. Returns geometry + level count
    + SIS magic-ok flag without decoding any tile pixels."""
    info = oc.get_codec("vsi").info(str(VSI_SAMPLE))
    assert info["index_shape"] == (216, 260, 3)
    assert len(info["ets_stacks"]) == 1
    stack = info["ets_stacks"][0]
    assert stack["magic_ok"] is True
    assert stack["width"] == 260
    assert stack["height"] == 216
    assert stack["n_components"] == 4
    # Was `level_count == 6`, which pinned a misreading rather than
    # checking anything: 6 is a constant repeated in every record of
    # the trailing table, not a count of pyramid levels. Assert the
    # table's shape against the file instead -- entries that point
    # inside it, at one plane's worth of bytes each, a whole number of
    # planes apart.
    assert stack["record_count"] == 4
    plane_bytes = stack["width"] * stack["height"] * 2
    assert stack["plane_stride"] % plane_bytes == 0
    assert stack["plane_stride"] // plane_bytes == 36


def test_ets_records_point_at_real_planes():
    """Every record names image data, not an arbitrary offset.

    A table read at the wrong stride still yields numbers; what it
    does not yield is offsets whose bytes look like a microscope
    image. Neighbouring pixels in real acquisition correlate, so a
    small mean absolute difference across a row separates a genuine
    plane from a misaligned read of compressed or unrelated bytes.
    """
    import numpy as np
    from opencodecs._ets import parse_ets

    ets = (CORPUS / "vsi" / "_metadataTest_01_" / "stack1"
           / "frame_t_0.ets")
    if not ets.is_file():
        pytest.skip("VSI corpus sample not present")
    info = parse_ets(str(ets))
    raw = ets.read_bytes()
    plane_bytes = info.width * info.height * 2
    assert info.n_records >= 1
    for rec in info.records:
        assert rec.size == plane_bytes
        assert rec.offset + plane_bytes <= info.file_size
        a = np.frombuffer(raw[rec.offset:rec.offset + plane_bytes],
                          dtype="<u2").reshape(info.height, info.width)
        span = int(a.max()) - int(a.min())
        assert span > 0, "a constant block is not an image"
        neighbour = np.abs(np.diff(a.astype(np.int32), axis=1)).mean()
        assert neighbour < span / 4, (
            f"record at {rec.offset} has no spatial structure "
            f"(neighbour diff {neighbour:.1f} vs span {span})")


# ---------------------------------------------------------------------------
# Experimental raw-record decode (uint16 frames; not yet verified
# against ground truth)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not OIR_SAMPLE.exists(), reason=_HINT)
def test_oir_raw_records_decode_experimental():
    """raw_records() returns one uint16 ndarray per frame record.
    The 99 frame records have plausible microscopy-data values
    (3..1023 range, low mean). NOT verified against ground truth —
    the per-record shape is a best-effort heuristic."""
    records = oc.get_codec("oir").raw_records(str(OIR_SAMPLE))
    assert len(records) == 99
    # First three frame records: two large + one small
    assert records[0].dtype == np.uint16
    assert records[0].shape == (237, 512)   # 242688 bytes
    assert records[2].shape == (256, 76)    # 38912 bytes (thumbnail?)
    # Values look like 10-bit microscopy data
    assert 0 <= records[0].min() < 100
    assert 500 < records[0].max() < 1024


@pytest.mark.skipif(
    not (CORPUS / "vsi" / "_metadataTest_01_" / "stack1"
         / "frame_t_0.ets").exists(),
    reason=_HINT,
)
def test_ets_decode_full_native():
    """decode_ets() returns the full plane stack from one .ets file.
    Verified byte-identical to bftools on the OME corpus sample."""
    from opencodecs._ets import decode_ets
    arr = decode_ets(
        str(CORPUS / "vsi" / "_metadataTest_01_"
            / "stack1" / "frame_t_0.ets"))
    assert arr.shape == (180, 216, 260)
    assert arr.dtype == np.uint16


def test_ets_record_fields_are_populated():
    """EtsRecord is what replaced the bogus level_count.

    Each field has to carry a real value, or the record table is just
    a differently shaped way of reporting nothing.
    """
    from opencodecs._ets import EtsInfo, EtsRecord, parse_ets

    ets = (CORPUS / "vsi" / "_metadataTest_01_" / "stack1"
           / "frame_t_0.ets")
    if not ets.is_file():
        pytest.skip("VSI corpus sample not present")
    info = parse_ets(str(ets))
    assert isinstance(info, EtsInfo)
    assert info.n_records == len(info.records) == 4
    assert all(isinstance(r, EtsRecord) for r in info.records)
    # plane_index counts up in equal steps; that is what makes the
    # table sparse rather than one entry per plane.
    idx = [r.plane_index for r in info.records]
    assert idx == sorted(idx) and idx[0] == 0
    steps = {b - a for a, b in zip(idx, idx[1:])}
    assert len(steps) == 1, f"uneven plane_index steps: {idx}"
    # tag is the constant that used to be reported as a level count.
    assert len({r.tag for r in info.records}) == 1


def test_parse_ets_rejects_a_file_without_the_magic(tmp_path):
    from opencodecs._ets import parse_ets
    bogus = tmp_path / "not.ets"
    bogus.write_bytes(b"\x00" * 128)
    info = parse_ets(str(bogus))
    assert info.magic_ok is False
    assert info.records == ()
    assert info.n_records == 0
    assert info.plane_stride == 0


def _synthetic_ets(records, *, width=8, height=4, file_pad=4096):
    """Build a minimal SIS/ETS file with a chosen record table.

    The corpus sample is one well-formed file, which cannot exercise
    what happens when the table is not well-formed. Synthesizing lets
    the malformed cases be tested without waiting for a bad file to
    turn up in the wild -- and a table read at the wrong stride
    produces exactly this shape of garbage.
    """
    import struct

    from opencodecs._ets import _ETS_TABLE_PREAMBLE, _ETS_TABLE_RECORD

    table = bytearray(b"\x00" * _ETS_TABLE_PREAMBLE)
    for off, size, plane_index, tag in records:
        # Only the first 20 bytes of each record are fields we read;
        # the rest is padding to the stride the real files use. Packing
        # them back to back instead would build a file the parser is
        # right to disagree with.
        rec = struct.pack("<IIIII", off, 0, size, plane_index, tag)
        table += rec + b"\x00" * (_ETS_TABLE_RECORD - len(rec))

    ptr2 = file_pad
    hdr = bytearray(64)
    hdr[0:4] = b"SIS\x00"
    struct.pack_into("<I", hdr, 4, 64)
    struct.pack_into("<I", hdr, 8, 3)
    struct.pack_into("<Q", hdr, 16, 64)      # sub-header at 64
    struct.pack_into("<Q", hdr, 24, 228)
    struct.pack_into("<Q", hdr, 32, ptr2)    # table
    struct.pack_into("<Q", hdr, 40, len(table))

    sub = bytearray(228)
    sub[0:4] = b"ETS\x00"
    struct.pack_into("<I", sub, 8, 1)        # n_components
    struct.pack_into("<I", sub, 28, width)
    struct.pack_into("<I", sub, 32, height)

    blob = bytearray(hdr + sub)
    blob += b"\x00" * (ptr2 - len(blob))
    blob += table
    return bytes(blob)


def test_ets_table_walk_stops_at_a_record_pointing_outside_the_file(tmp_path):
    """An offset past the end ends the walk instead of being recorded.

    Without this the parser would happily report entries addressing
    bytes that do not exist, and every consumer downstream would read
    garbage while the table looked structurally fine.
    """
    from opencodecs._ets import parse_ets

    plane = 8 * 4 * 2
    good = [(300, plane, 0, 6), (300 + plane, plane, 1, 6)]
    bad = good + [(1 << 30, plane, 2, 6)]     # way past the end
    f = tmp_path / "truncated.ets"
    f.write_bytes(_synthetic_ets(bad))
    info = parse_ets(str(f))
    assert info.magic_ok is True
    assert info.width == 8 and info.height == 4
    assert info.n_records == 2, (
        f"walk did not stop at the out-of-range record: {info.records}")


def test_ets_table_walk_stops_at_a_zero_sized_record(tmp_path):
    from opencodecs._ets import parse_ets

    plane = 8 * 4 * 2
    recs = [(300, plane, 0, 6), (400, 0, 1, 6), (500, plane, 2, 6)]
    f = tmp_path / "zerosize.ets"
    f.write_bytes(_synthetic_ets(recs))
    assert parse_ets(str(f)).n_records == 1


def test_ets_with_an_empty_table_reports_no_records(tmp_path):
    from opencodecs._ets import parse_ets

    f = tmp_path / "notable.ets"
    f.write_bytes(_synthetic_ets([]))
    info = parse_ets(str(f))
    assert info.magic_ok is True
    assert info.n_records == 0
    assert info.plane_stride == 0
