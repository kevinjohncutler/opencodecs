"""TIFF segments decoded, un-predicted and placed in one native call.

Every file here is written by tifffile, the reference writer, and must
read back as the array that was written. The layouts span what the fused
path takes on (every byte-stream compression, predictors 1 to 3, integer
and float samples, one or three samples per pixel, tiles with cropped
edges, strips with a short last strip) and what it hands back to the
general path (separate planes, byte-swapped files, malformed segments).
"""
from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc
from opencodecs import _tiff_codec

tifffile = pytest.importorskip("tifffile")

COMPRESSIONS = [None, "zlib", "deflate", "zstd", "lzw", "packbits"]


def _image(dtype, shape):
    rs = np.random.default_rng(20260928)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    base = (yy * 7 + xx * 3) % 251
    if len(shape) == 3:
        base = np.stack([base + 17 * c for c in range(shape[2])], axis=-1)
    dt = np.dtype(dtype)
    if dt.kind == "f":
        return (base / 7.0 + rs.normal(0, 0.01, base.shape)).astype(dt)
    info = np.iinfo(dt)
    noise = rs.integers(-3, 4, base.shape)
    return np.clip(base * (max(1, info.max // 512)) + noise + (info.min // 4),
                   info.min, info.max).astype(dt)


def _write(path, arr, *, compression, predictor, layout, **extra):
    kw = dict(compression=compression, predictor=predictor, **extra)
    if layout == "tiled":
        kw["tile"] = (64, 48)
    else:
        kw["rowsperstrip"] = 37
    if arr.ndim == 3:
        kw["photometric"] = "rgb"
    tifffile.imwrite(path, arr, **kw)


def _no_general_path(monkeypatch):
    def refuse(*a, **k):
        raise AssertionError("fell back to the per-segment path")
    monkeypatch.setattr(_tiff_codec.TiffPage, "_decode_segment_pixels", refuse)


@pytest.mark.parametrize("compression", COMPRESSIONS)
@pytest.mark.parametrize("dtype,predictor", [
    ("u1", None), ("u2", None), ("u2", "horizontal"), ("i2", "horizontal"),
    ("u1", "horizontal"), ("u4", "horizontal"), ("f4", None),
    ("f4", "floatingpoint"), ("f8", "floatingpoint"), ("f2", "floatingpoint"),
])
@pytest.mark.parametrize("layout", ["tiled", "strips"])
@pytest.mark.parametrize("samples", [1, 3])
def test_fused_path_matches_what_was_written(tmp_path, monkeypatch, compression,
                                              dtype, predictor, layout, samples):
    if compression is None and predictor is not None:
        pytest.skip("tifffile writes a predictor only with compression")
    if compression is None and layout == "strips":
        pytest.skip("uncompressed strips take the direct copy path")
    shape = (301, 211, 3) if samples == 3 else (301, 211)
    arr = _image(dtype, shape)
    path = tmp_path / "img.tif"
    _write(path, arr, compression=compression, predictor=predictor, layout=layout)
    _no_general_path(monkeypatch)
    for numthreads in (1, None, 4):
        got = oc.read(str(path), numthreads=numthreads)
        assert got.dtype == arr.dtype
        np.testing.assert_array_equal(got, arr)


@pytest.mark.parametrize("compression", ["zlib", "lzw"])
def test_separate_planes_and_big_endian_still_read(tmp_path, compression):
    """Layouts the fused call does not take go down the general path."""
    arr = _image("u2", (150, 97, 3))
    planar = tmp_path / "planar.tif"
    # tifffile takes separate planes as (samples, rows, columns).
    tifffile.imwrite(planar, np.moveaxis(arr, -1, 0), tile=(32, 32),
                     compression=compression, planarconfig="separate",
                     photometric="rgb")
    expected = np.moveaxis(tifffile.imread(planar), 0, -1)
    np.testing.assert_array_equal(expected, arr)
    np.testing.assert_array_equal(oc.read(str(planar)), arr)
    big = tmp_path / "big.tif"
    tifffile.imwrite(big, arr[..., 0], tile=(32, 32), compression=compression,
                     predictor="horizontal", byteorder=">")
    np.testing.assert_array_equal(oc.read(str(big)), arr[..., 0])
    bigf = tmp_path / "bigf.tif"
    f = arr[..., 0].astype("f4") / 3
    tifffile.imwrite(bigf, f, tile=(32, 32), compression=compression,
                     predictor="floatingpoint", byteorder=">")
    np.testing.assert_array_equal(oc.read(str(bigf)), f)


def test_a_corrupt_segment_raises_the_same_error_either_way(tmp_path):
    arr = _image("u2", (256, 256))
    path = tmp_path / "bad.tif"
    tifffile.imwrite(path, arr, tile=(64, 64), compression="zlib")
    with tifffile.TiffFile(path) as tf:
        offset = tf.pages[0].dataoffsets[5]
    raw = bytearray(path.read_bytes())
    raw[offset + 2:offset + 40] = b"\x55" * 38
    path.write_bytes(bytes(raw))
    errors = []
    for numthreads in (1, 4):
        with pytest.raises(Exception) as err:
            oc.read(str(path), numthreads=numthreads)
        errors.append(type(err.value))
    assert errors[0] is errors[1]


def test_bytes_source_reads_like_a_file(tmp_path):
    arr = _image("u2", (200, 190, 3))
    path = tmp_path / "rgb.tif"
    _write(path, arr, compression="zstd", predictor="horizontal", layout="tiled")
    np.testing.assert_array_equal(oc.read(path.read_bytes(), format="tiff"), arr)
    with open(path, "rb") as fh:
        np.testing.assert_array_equal(oc.read(fh, format="tiff"), arr)
