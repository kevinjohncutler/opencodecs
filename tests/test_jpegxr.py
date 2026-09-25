"""JPEG XR decoding (the optional ``_jpegxr`` extension) and CZI compression 4.

Streams are written by imagecodecs, a separate binding over jxrlib, and our
decode must match its decode of the same stream exactly: every supported
sample type, gray, color and alpha, sizes that are not multiples of the
16-pixel macroblock, lossless and lossy. CZI sub-blocks compressed this way
(most Zeiss slide scans) must read like their uncompressed equivalents.
"""
from __future__ import annotations

import pathlib
import sys

import numpy as np
import pytest

jxr = pytest.importorskip("opencodecs.codecs._jpegxr")
imagecodecs = pytest.importorskip("imagecodecs")
if not getattr(getattr(imagecodecs, "JPEGXR", None), "available", False):
    pytest.skip("imagecodecs lacks JPEG XR", allow_module_level=True)

sys.path.insert(0, str(pathlib.Path(__file__).parent))

from _czi_fixture import mosaic_czi_bytes  # noqa: E402

from opencodecs._czi_reader import CziError, CziPyramidReader, CziReader  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[1]
AXIOSCAN = ROOT / ".test_data" / "czi" / "ome_axioscan_pyramid.czi"


def _image(dtype, shape, seed):
    rs = np.random.RandomState(seed)
    if dtype == np.float32:
        return rs.rand(*shape).astype(dtype)
    return rs.randint(0, np.iinfo(dtype).max, shape).astype(dtype)


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.float32])
@pytest.mark.parametrize("shape", [(37, 53), (1504, 2048), (17, 31, 3), (33, 65, 4)])
@pytest.mark.parametrize("level", [None, 0.9])
def test_decode_matches_imagecodecs(dtype, shape, level):
    if dtype == np.float32 and len(shape) == 3 and shape[2] == 4:
        pytest.skip("imagecodecs does not write float RGBA JPEG XR")
    stream = imagecodecs.jpegxr_encode(_image(dtype, shape, 1), level=level)
    expected = imagecodecs.jpegxr_decode(stream)
    got = jxr.decode(stream, bgr=False)
    assert got.dtype == expected.dtype and got.shape == expected.shape
    np.testing.assert_array_equal(got, expected)


def test_lossless_round_trip_and_info():
    image = _image(np.uint16, (61, 77), 2)
    stream = imagecodecs.jpegxr_encode(image)
    np.testing.assert_array_equal(jxr.decode(stream), image)
    info = jxr.info(stream)
    assert (info["width"], info["height"], info["bits_per_pixel"]) == (77, 61, 16)


def test_out_and_channel_order():
    rgb = _image(np.uint8, (40, 24, 3), 3)
    stream = imagecodecs.jpegxr_encode(rgb)
    stored_bgr = jxr.info(stream)["bgr"]
    native = jxr.decode(stream)
    as_rgb = jxr.decode(stream, bgr=False)
    as_bgr = jxr.decode(stream, bgr=True)
    np.testing.assert_array_equal(as_rgb, rgb)
    np.testing.assert_array_equal(as_bgr, rgb[..., ::-1])
    np.testing.assert_array_equal(native, as_bgr if stored_bgr else as_rgb)
    out = np.zeros(rgb.nbytes, np.uint8)
    got = jxr.decode(stream, out=out, bgr=False)
    assert got.base is out or np.shares_memory(got, out)
    np.testing.assert_array_equal(got, rgb)
    with pytest.raises(ValueError):
        jxr.decode(stream, out=np.zeros(rgb.nbytes + 1, np.uint8))


def test_malformed_streams_raise():
    with pytest.raises(jxr.JpegXrError):
        jxr.decode(b"not a jpeg xr stream at all")
    with pytest.raises(jxr.JpegXrError):
        jxr.decode(b"")


# ---- CZI compression 4 ----

def _mosaic(dtype, seed):
    rs = np.random.RandomState(seed)
    tiles = []
    for r in range(3):
        for c in range(4):
            t = rs.randint(0, np.iinfo(dtype).max, (48, 40)).astype(dtype)
            tiles.append((t, (r * 44, c * 36)))  # 4 px overlaps
    return tiles


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_czi_jpegxr_tiles_read_like_their_pixels(dtype):
    tiles = _mosaic(dtype, 4)
    jxr_bytes = mosaic_czi_bytes(tiles, compression=4)
    raw_bytes = mosaic_czi_bytes(tiles, compression=0)
    with CziReader(buffer=jxr_bytes) as r:
        assert {e.compression for e in r.entries} == {4}
        stack = r.read()
        for i, (t, _) in enumerate(tiles):
            np.testing.assert_array_equal(np.squeeze(r[i]), t)
            np.testing.assert_array_equal(stack[i], t)
    with CziPyramidReader(CziReader(buffer=jxr_bytes)) as a, \
            CziPyramidReader(CziReader(buffer=raw_bytes)) as b:
        np.testing.assert_array_equal(a.read_region(0), b.read_region(0))
        np.testing.assert_array_equal(a.read_region(0, y=(5, 101), x=(30, 141)),
                                      b.read_region(0, y=(5, 101), x=(30, 141)))


def test_czi_jpegxr_payload_of_the_wrong_size_raises():
    tiles = _mosaic(np.uint16, 5)
    data = bytearray(mosaic_czi_bytes(tiles, compression=4))
    other = imagecodecs.jpegxr_encode(_image(np.uint16, (16, 16), 6))
    with CziReader(buffer=bytes(data)) as r:
        view, _ = r._pixel_data_view(r.entries[0])
        with pytest.raises(CziError):
            CziReader._decode_payload(other, r.entries[0].pixel_type, 4,
                                      r.entries[0].stored_shape)
        with pytest.raises(CziError):
            CziReader._decode_payload(b"\x00" * 64, r.entries[0].pixel_type, 4,
                                      r.entries[0].stored_shape)
        assert len(view) > 0


@pytest.mark.skipif(not AXIOSCAN.exists(), reason="corpus slide not downloaded")
def test_real_axioscan_tiles_and_overlaps_match_czifile():
    """JPEG XR tiles of a real slide decode as czifile decodes them, and an
    overlapping region composes with the higher mosaic index on top."""
    czifile = pytest.importorskip("czifile")
    with czifile.CziFile(str(AXIOSCAN)) as ref, \
            CziPyramidReader(CziReader(AXIOSCAN)) as p:
        level = p.levels[0]
        index = p._level_index(level)
        entries = level.reader
        by_position = {e.file_position: e for e in entries}
        decoded = {}
        for sb in ref.subblocks():
            e = by_position.get(sb.directory_entry.file_position)
            if e is not None:
                decoded[e.file_position] = np.squeeze(sb.data())
        # A crop across a four-tile corner, where overlaps decide pixels.
        y0, y1, x0, x1 = 5000, 7500, 3000, 5500
        hits, _ = index.query(y0, y1, x0, x1)
        assert len(hits) >= 4
        expected = np.zeros((y1 - y0, x1 - x0), np.uint16)
        for i in sorted(hits, key=lambda i: entries[i].mosaic_index):
            b = index.bounds[i]
            tile = decoded[entries[i].file_position]
            iy0, iy1, ix0, ix1 = max(b[0], y0), min(b[1], y1), max(b[2], x0), min(b[3], x1)
            expected[iy0 - y0:iy1 - y0, ix0 - x0:ix1 - x0] = \
                tile[iy0 - b[0]:iy1 - b[0], ix0 - b[2]:ix1 - b[2]]
        np.testing.assert_array_equal(p.read_region(0, y=(y0, y1), x=(x0, x1)), expected)
