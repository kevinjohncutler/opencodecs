"""Real-world corpus tests for vendor microscopy formats.

Covers:
  * **CZI** (smaller idr0011 plate sample, ~43 MB): exercises CZI
    parser against a different ZEN version than the 505 MB Axioscan.
  * **NDTiff** v3 (~1.5 MB): real Micro-Manager test layout.
  * **ND2** (~13 MB): Nikon NIS-Elements binary container — wraps the
    ``nd2`` Python package.
  * **LIF** (~230 KB): Leica LAS-X confocal — wraps ``readlif``.
  * **OIB** (~25 MB): Olympus FluoView — wraps ``oiffile``.

Each codec test:
  1. Verifies the codec is registered (assuming its delegate library
     is importable).
  2. Decodes the corpus file and validates shape/dtype.
  3. Roundtrips via the streaming Reader (``iter_frames``).
  4. Validates the magic-byte signature() detection.

Tests are skipif-gated on (a) the corpus file's presence and
(b) the delegate Python library being installed.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc

CORPUS = Path(__file__).resolve().parent.parent / ".test_data"
_HINT = (
    "Run `bash tests/download_test_corpus.sh --light` from the repo "
    "root to populate the corpus."
)


# ---------------------------------------------------------------------------
# Small CZI (idr0011 plate)
# ---------------------------------------------------------------------------

CZI_SMALL = CORPUS / "czi" / "idr0011_plate1_scene1.czi"


@pytest.mark.skipif(not CZI_SMALL.exists(), reason=_HINT)
def test_czi_idr0011_opens_and_has_frames():
    """The idr0011 well CZI has multi-channel, multi-Z, multi-position
    content. Verify the CZI parser handles a ZEN dialect different
    from our existing Axioscan corpus."""
    with oc.get_codec("czi").open(str(CZI_SMALL)) as r:
        assert r.n_frames > 1
        frame0 = next(r.iter_frames())
        assert frame0.dtype in (np.uint8, np.uint16)
        assert frame0.ndim >= 2


# ---------------------------------------------------------------------------
# NDTiff v3 (Micro-Manager test layout)
# ---------------------------------------------------------------------------

NDTIFF_DIR = CORPUS / "ndtiff"


@pytest.mark.skipif(
    not (NDTIFF_DIR / "NDTiff.index").exists()
    or not (NDTIFF_DIR / "NDTiff3.2_monochrome_NDTiffStack.tif").exists(),
    reason=_HINT,
)
def test_ndtiff_v3_opens_and_reads():
    from opencodecs._ndtiff import NDTiffDataset
    ds = NDTiffDataset(str(NDTIFF_DIR))
    assert ds.n_frames > 0
    arr = ds.read()
    assert arr.ndim == 3   # (n_frames, H, W)
    assert arr.dtype in (np.uint8, np.uint16)


# ---------------------------------------------------------------------------
# ND2 (Nikon NIS-Elements) — needs `nd2` Python package
# ---------------------------------------------------------------------------

ND2_SAMPLE = CORPUS / "nd2" / "MeOh_high_fluo_007.nd2"


@pytest.mark.skipif(not ND2_SAMPLE.exists(), reason=_HINT)
def test_nd2_codec_registered():
    pytest.importorskip("nd2")
    assert "nd2" in [c["name"] for c in oc.list_codecs()]


@pytest.mark.skipif(not ND2_SAMPLE.exists(), reason=_HINT)
def test_nd2_decode_basic():
    """Real Nikon ND2 from the OME mirror. Validates shape + dtype
    + signature."""
    pytest.importorskip("nd2")
    arr = oc.read(str(ND2_SAMPLE), format="nd2")
    # Sample is a (T=13, Y=600, X=800) uint16 timelapse
    assert arr.shape == (13, 600, 800)
    assert arr.dtype == np.uint16


@pytest.mark.skipif(not ND2_SAMPLE.exists(), reason=_HINT)
def test_nd2_streaming_iter_frames():
    pytest.importorskip("nd2")
    with oc.get_codec("nd2").open(str(ND2_SAMPLE)) as r:
        assert r.n_frames == 13
        frames = list(r.iter_frames())
    assert len(frames) == 13
    assert all(f.shape == (600, 800) for f in frames)
    assert all(f.dtype == np.uint16 for f in frames)


@pytest.mark.skipif(not ND2_SAMPLE.exists(), reason=_HINT)
def test_nd2_signature_detection():
    """ND2 magic is ``0xDABCD83E`` in the first 4 bytes."""
    pytest.importorskip("nd2")
    codec = oc.get_codec("nd2")
    with open(ND2_SAMPLE, "rb") as f:
        head = f.read(16)
    assert codec.signature(head) is True
    assert codec.signature(b"not nd2") is False


# ---------------------------------------------------------------------------
# LIF (Leica LAS-X) — needs `readlif`
# ---------------------------------------------------------------------------

LIF_SAMPLE = CORPUS / "lif" / "PR2729_frameOrderCombinedScanTypes.lif"


@pytest.mark.skipif(not LIF_SAMPLE.exists(), reason=_HINT)
def test_lif_codec_registered():
    pytest.importorskip("readlif")
    assert "lif" in [c["name"] for c in oc.list_codecs()]


@pytest.mark.skipif(not LIF_SAMPLE.exists(), reason=_HINT)
def test_lif_decode_multi_mosaic():
    """The PR2729 LIF has 4 mosaic tiles, 3 z, 2 t, 2 channels.
    readlif's as_array() can fail on inhomogeneous mosaics — our
    LifReader falls back to per-plane stitching."""
    pytest.importorskip("readlif")
    arr = oc.read(str(LIF_SAMPLE), format="lif")
    # Squeezed shape: (M=4, Z=3, T=2, C=2, Y=64, X=64)
    assert arr.shape == (4, 3, 2, 2, 64, 64)
    assert arr.dtype == np.uint8


@pytest.mark.skipif(not LIF_SAMPLE.exists(), reason=_HINT)
def test_lif_reader_image_navigation():
    pytest.importorskip("readlif")
    from opencodecs._lif_codec import LifReader
    with LifReader(str(LIF_SAMPLE)) as r:
        assert r.n_images >= 1
        names = r.image_names
        assert isinstance(names, list)
        assert len(names) == r.n_images


@pytest.mark.skipif(not LIF_SAMPLE.exists(), reason=_HINT)
def test_lif_signature_detection():
    """LIF magic is ``0x70 0x00 0x00 0x00`` in the first 4 bytes."""
    pytest.importorskip("readlif")
    codec = oc.get_codec("lif")
    with open(LIF_SAMPLE, "rb") as f:
        head = f.read(16)
    assert codec.signature(head) is True
    assert codec.signature(b"\x00\x00\x00\x70 wrong") is False


# ---------------------------------------------------------------------------
# OIB (Olympus FluoView) — needs `oiffile`
# ---------------------------------------------------------------------------

OIB_SAMPLE = CORPUS / "oib" / "imagesc_71616_60x.oib"


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_oib_codec_registered():
    pytest.importorskip("oiffile")
    assert "oib" in [c["name"] for c in oc.list_codecs()]


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_oib_decode_basic():
    """Real Olympus FluoView OIB. Verify shape + dtype."""
    pytest.importorskip("oiffile")
    arr = oc.read(str(OIB_SAMPLE), format="oib")
    # OIB sample is (C=2, T=6, Y=1024, X=1024) uint16
    assert arr.ndim == 4
    assert arr.dtype == np.uint16
    assert arr.shape[-1] == 1024
    assert arr.shape[-2] == 1024


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_oib_signature_detection():
    """OIB is a Microsoft Compound File Binary container; magic is
    ``\\xD0\\xCF\\x11\\xE0\\xA1\\xB1\\x1A\\xE1``."""
    pytest.importorskip("oiffile")
    codec = oc.get_codec("oib")
    with open(OIB_SAMPLE, "rb") as f:
        head = f.read(16)
    assert codec.signature(head) is True
    assert codec.signature(b"not OLE") is False


# ---------------------------------------------------------------------------
# Format detection via top-level oc.read() (no format=)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not ND2_SAMPLE.exists(), reason=_HINT)
def test_extension_dispatch_finds_nd2():
    """oc.read("foo.nd2") should pick the ND2 codec by extension
    without needing format=."""
    pytest.importorskip("nd2")
    # Don't pass format=; codec is chosen by .nd2 extension.
    arr = oc.read(str(ND2_SAMPLE))
    assert arr.shape == (13, 600, 800)


@pytest.mark.skipif(not LIF_SAMPLE.exists(), reason=_HINT)
def test_extension_dispatch_finds_lif():
    pytest.importorskip("readlif")
    arr = oc.read(str(LIF_SAMPLE))
    assert arr.dtype == np.uint8


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_extension_dispatch_finds_oib():
    pytest.importorskip("oiffile")
    arr = oc.read(str(OIB_SAMPLE))
    assert arr.dtype == np.uint16
