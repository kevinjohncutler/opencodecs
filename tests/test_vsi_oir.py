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
    assert stack["level_count"] == 6
    assert stack["n_components"] == 4


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
