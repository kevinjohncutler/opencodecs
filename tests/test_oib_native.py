"""Tests for the native OIB / OLE2 parser.

The native parser is the *default* backend for OibCodec — oiffile
serves as fallback for OIF directory variants. Native decode is
byte-identical to oiffile on standard OIB files and uses the same
DataSource abstraction as our TIFF / ND2 readers (so HTTP partial
reads work transparently).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc

CORPUS = Path(__file__).resolve().parent.parent / ".test_data"
OIB_SAMPLE = CORPUS / "oib" / "imagesc_71616_60x.oib"

_HINT = (
    "Run `bash tests/download_test_corpus.sh --light` from the repo "
    "root to populate the OIB corpus."
)


# ---------------------------------------------------------------------------
# OLE2 parser
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_ole2_lists_oib_streams():
    """Our OLE2 parser finds every stream in the OIB container.
    The corpus file has 33 streams (12 TIFFs + 12 .pty metadata +
    a few thumbnails / LUTs / ROIs + OibInfo.txt + main .oif)."""
    from opencodecs._ole2 import OleReader
    with OleReader(str(OIB_SAMPLE)) as ole:
        streams = ole.list_streams()
    assert len(streams) >= 30
    # OibInfo.txt is the OIB manifest — must be present
    assert "OibInfo.txt" in streams


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_ole2_first_stream_is_tiff():
    """Stream00001 of an OIB is a TIFF frame; verify it starts with
    the TIFF magic ``II*\\0`` (or ``MM*\\0`` for big-endian)."""
    from opencodecs._ole2 import OleReader
    with OleReader(str(OIB_SAMPLE)) as ole:
        data = ole.read_stream("Stream00001")
    assert data[:4] in (b"II*\x00", b"MM\x00*")


# ---------------------------------------------------------------------------
# OIB layout + decode
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_native_oib_layout():
    """Parser reads OibInfo.txt + main .oif and finds every (C, Z, T)
    frame's TIFF stream."""
    from opencodecs._oib_native import OibFileParser
    p = OibFileParser(str(OIB_SAMPLE))
    L = p.layout
    assert L.width == 1024
    assert L.height == 1024
    assert L.n_channels == 2
    assert L.n_z == 6
    assert L.n_t == 1
    # All 12 (C, Z) frames must map to streams
    assert len(L.frames) == 12


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_native_oib_decode_matches_oiffile():
    """Full decode of the corpus OIB must be byte-identical to what
    the oiffile package returns. Axis convention is (C, Z, H, W) to
    match FluoView / oiffile."""
    pytest.importorskip("oiffile")
    import oiffile
    from opencodecs._oib_native import OibNativeReader

    with OibNativeReader(str(OIB_SAMPLE)) as r:
        ours = r.read()
    ref = oiffile.imread(str(OIB_SAMPLE))
    assert ours.shape == ref.shape
    assert np.array_equal(ours, ref)


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_native_oib_per_frame_decode():
    """read_frame(c, z, t) returns a 2D ndarray matching the
    corresponding slice of the full stack."""
    from opencodecs._oib_native import OibFileParser
    p = OibFileParser(str(OIB_SAMPLE))
    L = p.layout
    full = p.read_all()
    # Sample a few (c, z) frames
    for c in (0, 1):
        for z in (0, 3, 5):
            frame = p.read_frame(c, z, 0)
            assert frame.shape == (L.height, L.width)
            # Compare to the corresponding slice
            slc = full[c, z]
            assert np.array_equal(frame, slc), (
                f"per-frame (c={c}, z={z}) disagrees with read_all")


# ---------------------------------------------------------------------------
# Codec adapter: native + delegate backends
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_oib_codec_default_uses_native():
    """Default open() yields the native reader."""
    from opencodecs._oib_native import OibNativeReader
    with oc.get_codec("oib").open(str(OIB_SAMPLE)) as r:
        assert isinstance(r, OibNativeReader)


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_oib_codec_force_native():
    from opencodecs._oib_native import OibNativeReader
    with oc.get_codec("oib").open(str(OIB_SAMPLE), backend="native") as r:
        assert isinstance(r, OibNativeReader)


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_oib_codec_force_delegate():
    pytest.importorskip("oiffile")
    from opencodecs._oib_codec import OibReader
    with oc.get_codec("oib").open(str(OIB_SAMPLE), backend="oiffile") as r:
        assert isinstance(r, OibReader)


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_oib_codec_oc_read_decodes_natively():
    """oc.read(path.oib) uses the native backend and matches the
    delegate's output byte-for-byte."""
    pytest.importorskip("oiffile")
    import oiffile
    ours = oc.read(str(OIB_SAMPLE), format="oib")
    ref = oiffile.imread(str(OIB_SAMPLE))
    assert np.array_equal(ours, ref)


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_oib_codec_lists_native():
    entry = next(c for c in oc.list_codecs() if c["name"] == "oib")
    assert entry["native"] is True


@pytest.mark.skipif(not OIB_SAMPLE.exists(), reason=_HINT)
def test_oib_codec_info_returns_layout_without_decode():
    """``codec.info(path)`` reads OibInfo.txt + the main .oif INI
    via the OLE2 container and returns layout — no per-frame TIFF
    is decoded."""
    info = oc.get_codec("oib").info(str(OIB_SAMPLE))
    layout = info["layout"]
    assert info["n_frames"] > 0
    assert info["n_streams"] >= info["n_frames"]
    assert layout["width"] > 0 and layout["height"] > 0
    assert layout["axis_order"]
    assert layout["dtype"] in ("uint8", "uint16", "uint32")
    assert layout["shape"][-2:] == (layout["height"], layout["width"])
