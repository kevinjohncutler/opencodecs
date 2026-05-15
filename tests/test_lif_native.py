"""Tests for the native LIF parser.

The native parser is experimental and not the registered LIF codec
yet — readlif via LifCodec remains the production path. These tests
verify the binary-parsing + base-case decode story, and document
the frame-order limitation that prevents adopting native as default.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CORPUS = Path(__file__).resolve().parent.parent / ".test_data"
LIF_SAMPLE = CORPUS / "lif" / "PR2729_frameOrderCombinedScanTypes.lif"

_HINT = (
    "Run `bash tests/download_test_corpus.sh --light` from the repo "
    "root to populate the LIF corpus."
)


@pytest.mark.skipif(not LIF_SAMPLE.exists(), reason=_HINT)
def test_native_lif_binary_parse():
    """The container parser identifies the XML header + every memory
    block. Independent of XML semantics."""
    from opencodecs._lif_native import LifFileParser
    p = LifFileParser(str(LIF_SAMPLE))
    assert p.xml.startswith("<LMSDataContainerHeader")
    # Corpus PR2729 has 2 memory blocks (one placeholder + one real)
    assert len(p.blocks) == 2
    sizes = sorted(b.size for b in p.blocks.values())
    # 64x64 pixels × 48 frames (2 t × 4 m × 3 z × 2 c) × 1 byte/sample
    assert 196_608 in sizes


@pytest.mark.skipif(not LIF_SAMPLE.exists(), reason=_HINT)
def test_native_lif_xml_image_discovery():
    """The XML walker finds the single Image element + maps it to
    its non-empty memory block."""
    from opencodecs._lif_native import LifFileParser
    p = LifFileParser(str(LIF_SAMPLE))
    assert len(p.images) == 1
    img = p.images[0]
    assert img.name == "TileScan_001"
    assert img.dtype == np.uint8
    assert img.n_channels == 2
    # axis_order is fastest-first; expect X (BytesInc=1) then Y (BytesInc=64).
    labels = [a[0] for a in img.axis_order]
    assert labels[0] == "X"
    assert labels[1] == "Y"


@pytest.mark.skipif(not LIF_SAMPLE.exists(), reason=_HINT)
def test_native_lif_shape_matches_metadata():
    """Reader shape is derived purely from XML dimensions + channel
    info. Compare against the known geometry of PR2729 (T=2, M=4,
    Z=3, C=2, Y=64, X=64)."""
    from opencodecs._lif_native import LifNativeReader
    with LifNativeReader(str(LIF_SAMPLE)) as r:
        assert r.shape == (2, 4, 3, 2, 64, 64)
        assert r.dtype == np.uint8
        assert r.n_images == 1


@pytest.mark.skipif(not LIF_SAMPLE.exists(), reason=_HINT)
def test_native_lif_first_frame_matches_readlif():
    """The single absolute-base-case frame (t=0, m=0, z=0, c=0)
    matches readlif exactly. This verifies the binary parser + XML
    metadata interpretation + reshape are correct in the simplest
    possible configuration — the rest of the corpus file exercises
    LIF features (per-frame FlipX/FlipY, FrameOrder permutations)
    that the native parser doesn't yet honor."""
    pytest.importorskip("readlif")
    from opencodecs._lif_native import LifNativeReader
    from readlif.reader import LifFile

    native = LifNativeReader(str(LIF_SAMPLE))
    arr = native.read()
    lf = LifFile(str(LIF_SAMPLE))
    im = lf.get_image(0)

    ours = arr[0, 0, 0, 0]
    ref = np.asarray(im.get_frame(0, 0, 0, m=0))
    assert np.array_equal(ours, ref), (
        "native vs readlif disagreement on (t=0, m=0, z=0, c=0)"
    )


@pytest.mark.skipif(not LIF_SAMPLE.exists(), reason=_HINT)
def test_native_lif_via_data_source_matches_path():
    """Opening through a FileDataSource (the same primitive used by
    HTTPDataSource for remote LIFs) reproduces the same pixels as
    opening through a path. Confirms read_at-based parsing/decode is
    byte-equivalent to the legacy whole-file read."""
    from opencodecs._lif_native import LifNativeReader
    from opencodecs._tiff_http import FileDataSource
    arr_path = LifNativeReader(str(LIF_SAMPLE)).read()
    ds = FileDataSource(str(LIF_SAMPLE))
    arr_ds = LifNativeReader(ds).read()
    assert np.array_equal(arr_path, arr_ds)
    assert arr_ds.shape == arr_path.shape
    assert arr_ds.dtype == arr_path.dtype


@pytest.mark.xfail(
    reason="LIF frame-order XML attributes (LAS-X tile-scan workflows) "
           "override natural memory-stride ordering for mosaic / Z / T "
           "axes. The native parser doesn't honor them yet — readlif "
           "does. Until we parse FrameOrder + ScanOrder, m>=1 mosaic "
           "slices won't match readlif on overriding files.",
    strict=False,
)
@pytest.mark.skipif(not LIF_SAMPLE.exists(), reason=_HINT)
def test_native_lif_full_match_xfail():
    """Documents the frame-order limitation — all 48 frames should
    match readlif. Will start passing once FrameOrder parsing lands."""
    pytest.importorskip("readlif")
    from opencodecs._lif_native import LifNativeReader
    from readlif.reader import LifFile

    native = LifNativeReader(str(LIF_SAMPLE))
    arr = native.read()
    lf = LifFile(str(LIF_SAMPLE))
    im = lf.get_image(0)
    for t in range(2):
        for m in range(4):
            for z in range(3):
                for c in range(2):
                    ours = arr[t, m, z, c]
                    ref = np.asarray(im.get_frame(z, t, c, m=m))
                    assert np.array_equal(ours, ref), (
                        f"diverged at (t={t},m={m},z={z},c={c})"
                    )
