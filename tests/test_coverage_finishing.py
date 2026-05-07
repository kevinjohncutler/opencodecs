"""Final touches to push coverage close to 100%. Each test in this
file targets a specific remaining missing line.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import opencodecs as oc


# ---------------------------------------------------------------------------
# __init__.py 75 — write() without format, dest not a path
# ---------------------------------------------------------------------------


def test_write_without_format_to_non_path_dest_raises():
    """oc.write(BytesIO(), arr) without format=... must raise ValueError."""
    import io
    bio = io.BytesIO()
    with pytest.raises(ValueError, match="format"):
        oc.write(bio, b"hello")


# ---------------------------------------------------------------------------
# core/color.py 118 — non-str non-ColorSpec input
# ---------------------------------------------------------------------------


def test_color_parse_unsupported_type_raises():
    from opencodecs.core.color import parse_color
    with pytest.raises(TypeError, match="color must be str or ColorSpec"):
        parse_color(42)


# ---------------------------------------------------------------------------
# zarr.py 137 — JxlCodec(lossless=False, animation=True) repr branch
# ---------------------------------------------------------------------------


def test_zarr_jxlcodec_repr_lossy_animation():
    pytest.importorskip("numcodecs")
    if not oc.has_codec("jxl"):
        pytest.skip("jxl not available")
    from opencodecs.zarr import JxlCodec
    s = repr(JxlCodec(lossless=False, distance=2.0, animation=True))
    assert "animation" in s
    assert "distance" in s


def test_zarr_jxlcodec_repr_with_color_includes_color():
    """Constructing with a color string should put the color in repr."""
    pytest.importorskip("numcodecs")
    if not oc.has_codec("jxl"):
        pytest.skip("jxl not available")
    from opencodecs.zarr import JxlCodec
    s = repr(JxlCodec(lossless=True, color="display-p3"))
    assert "display-p3" in s


# ---------------------------------------------------------------------------
# _czi_reader.py 515 — __len__
# ---------------------------------------------------------------------------


_LAB_CZI = (
    "/Volumes/HiprDrive/2024_02_02_GNE_synthetic_community/"
    "2024_02_02_GNEPanelTest_slide1_B1_GNE0001_cellmix01_200nMENC_"
    "20nMCOMP_quarterpower_fov_4_561.czi"
)


@pytest.mark.skipif(not os.path.isfile(_LAB_CZI), reason="lab CZI not mounted")
def test_czi_reader_len():
    from opencodecs._czi_reader import CziReader
    with CziReader(_LAB_CZI) as r:
        assert len(r) == r.n_frames
        assert len(r) > 0


# ---------------------------------------------------------------------------
# _czi_reader.py 593-594 — imread convenience function
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.isfile(_LAB_CZI), reason="lab CZI not mounted")
def test_czi_imread_convenience():
    from opencodecs._czi_reader import imread
    arr = imread(_LAB_CZI)
    assert arr.size > 0


# ---------------------------------------------------------------------------
# _czi_reader.py 507-510 — subblock_metadata_bytes for sub-blocks with
# non-empty inline metadata (CZI with mosaic position info typically has
# small XML per tile)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not os.path.isfile(_LAB_CZI), reason="lab CZI not mounted")
def test_czi_subblock_metadata_returns_bytes_for_all():
    """Every index returns bytes — the lab CZI happens to have empty
    inline metadata for every sub-block (no per-tile XML), so this
    exercises the meta_size <= 0 branch."""
    from opencodecs._czi_reader import CziReader
    with CziReader(_LAB_CZI) as r:
        for i in range(r.n_frames):
            m = r.subblock_metadata_bytes(i)
            assert isinstance(m, bytes)
