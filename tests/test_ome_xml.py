"""OME-XML + write_ome_tiff round-trip tests.

Verifies that opencodecs-emitted OME-TIFF files round-trip through
``tifffile`` (the reference OME-XML parser) with correct axes,
shape, and physical metadata.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from opencodecs._ome_xml import (
    Channel, Plane, make_ome_xml, write_ome_tiff,
)
from opencodecs._tiff_writer import TiffWriter

tifffile = pytest.importorskip("tifffile")


# ---------------------------------------------------------------------------
# make_ome_xml (pure XML generation)
# ---------------------------------------------------------------------------


def test_make_ome_xml_minimal_is_well_formed():
    import xml.etree.ElementTree as ET
    xml = make_ome_xml(size_x=64, size_y=48, dtype=np.uint8)
    # Strip the XML declaration so etree can parse what's left.
    root = ET.fromstring(xml.split("?>", 1)[1])
    # Namespaced tag — Clark notation.
    assert root.tag.endswith("OME")


def test_make_ome_xml_dtype_mapping():
    for np_dtype, expected in [
        (np.uint8, "uint8"), (np.int16, "int16"), (np.uint32, "uint32"),
        (np.float32, "float"), (np.float64, "double"),
    ]:
        xml = make_ome_xml(size_x=16, size_y=16, dtype=np_dtype)
        assert f'Type="{expected}"' in xml


def test_make_ome_xml_rejects_size_c_channel_mismatch():
    with pytest.raises(ValueError):
        make_ome_xml(
            size_x=16, size_y=16, size_c=2, dtype=np.uint8,
            channels=[Channel(name="x")],  # only 1, but size_c=2
        )


def test_make_ome_xml_channel_color_int_or_tuple():
    xml = make_ome_xml(
        size_x=16, size_y=16, size_c=2, dtype=np.uint8,
        channels=[
            Channel(name="A", color=(255, 0, 255, 0)),     # green-ish ARGB
            Channel(name="B", color=-65536),               # packed int
        ],
    )
    # The first channel's color is 0xFF00FF00 = -16711936 in signed int32.
    assert 'Color="-16711936"' in xml
    assert 'Color="-65536"' in xml


def test_make_ome_xml_physical_sizes():
    xml = make_ome_xml(
        size_x=16, size_y=16, dtype=np.uint8,
        physical_size_x_um=0.108,
        physical_size_y_um=0.108,
        physical_size_z_um=0.5,
    )
    assert 'PhysicalSizeX="0.108"' in xml
    assert "PhysicalSizeXUnit" in xml


def test_make_ome_xml_planes():
    xml = make_ome_xml(
        size_x=16, size_y=16, size_z=2, size_c=1, dtype=np.uint8,
        planes=[
            Plane(the_c=0, the_z=0, the_t=0,
                  position_x_um=10.5, exposure_time_seconds=0.05),
            Plane(the_c=0, the_z=1, the_t=0, position_x_um=10.5),
        ],
    )
    assert 'TheC="0"' in xml
    assert 'TheZ="0"' in xml
    assert 'TheZ="1"' in xml
    assert 'ExposureTime="0.05"' in xml


# ---------------------------------------------------------------------------
# write_ome_tiff (round-trip via tifffile)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("axes, shape", [
    ("YX",     (64, 96)),
    ("ZYX",    (5, 64, 96)),
    ("ZCYX",   (3, 2, 64, 96)),
    ("CZYX",   (2, 3, 64, 96)),
    ("TCYX",   (4, 2, 64, 96)),
    ("TCZYX",  (2, 3, 4, 64, 96)),
])
def test_write_ome_tiff_round_trip(axes, shape):
    arr = np.random.default_rng(0).integers(
        0, 4000, size=shape, dtype=np.uint16,
    )
    buf = io.BytesIO()
    write_ome_tiff(buf, arr, axes=axes)
    with tifffile.TiffFile(io.BytesIO(buf.getvalue())) as tf:
        assert tf.is_ome
        s = tf.series[0]
        assert s.axes == axes
        assert s.shape == shape
        np.testing.assert_array_equal(s.asarray(), arr)


def test_write_ome_tiff_with_channels_and_physical_size():
    arr = np.random.default_rng(0).integers(
        0, 4000, size=(3, 2, 64, 96), dtype=np.uint16,
    )
    channels = [
        Channel(name="DAPI", excitation_wavelength_nm=405,
                emission_wavelength_nm=460),
        Channel(name="GFP",  excitation_wavelength_nm=488,
                emission_wavelength_nm=520),
    ]
    buf = io.BytesIO()
    write_ome_tiff(
        buf, arr, axes="ZCYX",
        channels=channels,
        physical_size_um=(0.108, 0.108, 0.5),
    )
    with tifffile.TiffFile(io.BytesIO(buf.getvalue())) as tf:
        assert tf.is_ome
        np.testing.assert_array_equal(tf.series[0].asarray(), arr)
        # XML inspection: channel names + emission propagate.
        meta = tf.ome_metadata
        assert "DAPI" in meta
        assert "GFP" in meta
        # Physical size strings appear in the XML.
        assert "0.108" in meta
        assert "0.5" in meta


def test_write_ome_tiff_bigtiff():
    arr = np.random.default_rng(0).integers(
        0, 4000, size=(2, 3, 4, 64, 96), dtype=np.uint16,
    )
    buf = io.BytesIO()
    write_ome_tiff(buf, arr, axes="TCZYX", bigtiff=True)
    with tifffile.TiffFile(io.BytesIO(buf.getvalue())) as tf:
        assert tf.is_bigtiff
        assert tf.is_ome
        np.testing.assert_array_equal(tf.series[0].asarray(), arr)


def test_write_ome_tiff_rejects_invalid_axes():
    arr = np.zeros((4, 4), dtype=np.uint8)
    with pytest.raises(ValueError):
        write_ome_tiff(io.BytesIO(), arr, axes="QR")  # YX missing
    with pytest.raises(ValueError):
        write_ome_tiff(io.BytesIO(), arr, axes="YY")  # duplicate
    with pytest.raises(ValueError):
        write_ome_tiff(io.BytesIO(),
                       np.zeros((4, 4, 4), dtype=np.uint8),
                       axes="YX")   # length mismatch


def test_write_ome_tiff_path(tmp_path):
    """write_ome_tiff via a path string (not just file-like)."""
    arr = np.random.default_rng(1).integers(
        0, 4000, size=(3, 64, 96), dtype=np.uint16,
    )
    p = tmp_path / "out.ome.tif"
    write_ome_tiff(p, arr, axes="ZYX")
    with tifffile.TiffFile(str(p)) as tf:
        assert tf.is_ome
        s = tf.series[0]
        assert s.axes == "ZYX"
        np.testing.assert_array_equal(s.asarray(), arr)
