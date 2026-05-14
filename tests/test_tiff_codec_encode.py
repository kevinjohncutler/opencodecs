"""TiffCodec.encode integration tests — high-level codec adapter wiring.

These tests exercise the ``codec.encode(arr, dest=...)`` path on the
unified Codec API. The underlying writer is exhaustively covered in
test_tiff_writer.py; this file just verifies the public adapter is
hooked up properly (bytes path, file path, kwargs forwarding,
multi-page via open_writer, can_encode flag).
"""

from __future__ import annotations

import io
import tempfile
from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc


@pytest.fixture
def codec():
    return oc.get_codec("tiff")


def test_can_encode_is_true(codec):
    assert codec.can_encode is True


def test_list_codecs_reports_encode_true():
    entry = next(t for t in oc.list_codecs() if t["name"] == "tiff")
    assert entry["encode"] is True


def test_encode_to_bytes_roundtrip(codec):
    arr = np.arange(64 * 64, dtype=np.uint16).reshape(64, 64)
    enc = codec.encode(arr)
    assert isinstance(enc, (bytes, bytearray))
    assert len(enc) > 0
    dec = codec.decode(enc).squeeze()
    assert np.array_equal(dec, arr)


def test_encode_to_path_returns_none(codec, tmp_path):
    arr = np.arange(64 * 64, dtype=np.uint8).reshape(64, 64)
    out = tmp_path / "out.tif"
    result = codec.encode(arr, dest=str(out))
    assert result is None
    assert out.exists() and out.stat().st_size > 0
    dec = codec.decode(str(out)).squeeze()
    assert np.array_equal(dec, arr)


def test_encode_to_filelike(codec):
    arr = np.random.default_rng(0).integers(
        0, 256, (32, 32), dtype=np.uint8
    )
    buf = io.BytesIO()
    codec.encode(arr, dest=buf)
    assert buf.tell() > 0
    buf.seek(0)
    dec = codec.decode(buf.getvalue()).squeeze()
    assert np.array_equal(dec, arr)


@pytest.mark.parametrize("compression", ["none", "deflate", "lzw", "zstd"])
def test_encode_compression_dispatch(codec, compression):
    arr = np.tile(np.arange(256, dtype=np.uint8), (256, 4))  # 1 KB tile pattern
    enc = codec.encode(arr, compression=compression)
    dec = codec.decode(enc).squeeze()
    assert np.array_equal(dec, arr), f"compression={compression} failed"


def test_encode_tiled(codec):
    arr = np.arange(512 * 512, dtype=np.uint16).reshape(512, 512)
    enc = codec.encode(arr, compression="zstd", tile=(256, 256))
    dec = codec.decode(enc).squeeze()
    assert np.array_equal(dec, arr)


def test_encode_rgb(codec):
    rng = np.random.default_rng(42)
    rgb = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    enc = codec.encode(rgb, compression="deflate", photometric="rgb")
    dec = codec.decode(enc)
    # decode squeezes singleton page dim
    assert np.array_equal(dec, rgb)


def test_encode_bigtiff(codec):
    arr = np.arange(128 * 128, dtype=np.uint16).reshape(128, 128)
    enc = codec.encode(arr, bigtiff=True, compression="zstd")
    # BigTIFF magic = 43 at bytes 2-3 (LE)
    assert enc[2:4] == b"\x2b\x00"
    dec = codec.decode(enc).squeeze()
    assert np.array_equal(dec, arr)


def test_encode_predictor(codec):
    # Predictor=2 requires a uint integer dtype
    arr = np.arange(256 * 256, dtype=np.uint16).reshape(256, 256)
    enc = codec.encode(arr, compression="deflate", predictor=2)
    dec = codec.decode(enc).squeeze()
    assert np.array_equal(dec, arr)


def test_multi_page_via_open_writer(codec, tmp_path):
    """open_writer() returns a TiffWriter for multi-page output."""
    out = tmp_path / "multi.tif"
    frames = [
        np.full((64, 64), i, dtype=np.uint16) for i in range(5)
    ]
    with codec.open_writer(str(out)) as w:
        for fr in frames:
            w.write_page(fr, compression="zstd")
    # Read back
    with codec.open(str(out)) as r:
        assert r.n_frames == 5
        for i, page_arr in enumerate(r.iter_frames()):
            assert np.array_equal(page_arr.squeeze(), frames[i])


def test_encode_float32(codec):
    rng = np.random.default_rng(7)
    arr = rng.standard_normal((128, 128)).astype(np.float32)
    enc = codec.encode(arr, compression="zstd")
    dec = codec.decode(enc).squeeze()
    assert np.array_equal(dec, arr)
