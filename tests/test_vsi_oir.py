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


@pytest.mark.skipif(not VSI_SAMPLE.exists(), reason=_HINT)
def test_vsi_decode_via_tiff_reader():
    """The VSI top-level container is a TIFF. Our TIFF reader
    decodes the thumbnail / overview image directly."""
    arr = oc.read(str(VSI_SAMPLE), format="vsi")
    # Olympus metadataTest_01.vsi is a 216x260x3 RGB thumbnail
    assert arr.ndim == 3
    assert arr.shape[2] == 3
    assert arr.dtype == np.uint8


@pytest.mark.skipif(not VSI_SAMPLE.exists(), reason=_HINT)
def test_vsi_extension_dispatch():
    """oc.read('foo.vsi') picks VsiCodec without format= override."""
    arr = oc.read(str(VSI_SAMPLE))
    assert arr.dtype == np.uint8


# ---------------------------------------------------------------------------
# OIR — format-detection stub
# ---------------------------------------------------------------------------


def test_oir_codec_registered():
    assert "oir" in [c["name"] for c in oc.list_codecs()]


def test_oir_codec_advertises_no_decode():
    entry = next(c for c in oc.list_codecs() if c["name"] == "oir")
    assert entry["native"] is False
    assert entry["decode"] is False
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
def test_oir_decode_raises_clearly():
    """OIR decode raises NotImplementedError with a clear message —
    not a confusing low-level error. Users get the message even
    via the top-level oc.read API."""
    with pytest.raises(NotImplementedError, match="OIR"):
        oc.read(str(OIR_SAMPLE), format="oir")


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
    assert stack["width"] == 216
    assert stack["height"] == 260
    assert stack["level_count"] == 6
    assert stack["n_components"] == 4
