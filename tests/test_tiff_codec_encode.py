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


@pytest.mark.parametrize("compression", ["deflate", "lzw", "zstd"])
def test_encode_predictor_rgb(codec, compression):
    """Regression: predictor=2 on RGB used to fail at decode time because
    frombuffer() returned a read-only view that the predictor kernel
    couldn't mutate. Fixed in _undo_predictor by copying to a writable
    buffer when needed."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (128, 128, 3), dtype=np.uint8)
    enc = codec.encode(
        arr, compression=compression, predictor=2, photometric="rgb",
    )
    dec = codec.decode(enc)
    assert np.array_equal(dec, arr), f"compression={compression} failed"


def test_encode_predictor_rgba(codec):
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, (128, 128, 4), dtype=np.uint8)
    enc = codec.encode(
        arr, compression="deflate", predictor=2, photometric="rgb",
    )
    dec = codec.decode(enc)
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


# ---------------------------------------------------------------------------
# LZW encoder: opencodecs' own implementation (3rdparty/oc_tifflzw)
#
# The encoder and decoder must agree on TIFF's code-width transition, and
# the encoder has to stay compatible with every other TIFF reader. The
# width rule is the easy thing to get wrong: the decoder widens when its
# own next free code hits (1 << width) - 1, but its table lags the
# encoder's by one entry, so the encoder's threshold is 1 << width. Get
# that off by one and short inputs still round-trip while anything long
# enough to reach 511 codes silently corrupts, which is why the cases
# below deliberately span the 9->10, 10->11, 11->12 and table-full
# boundaries.
# ---------------------------------------------------------------------------

import numpy as np
import pytest

from opencodecs.codecs import _tiff


def _lzw_cases():
    rng = np.random.default_rng(0)
    x = np.linspace(0, 40, 300_000)
    photo = (np.clip(np.sin(x) * 0.5 + 0.5 + 0.01 * rng.standard_normal(300_000),
                     0, 1) * 255).astype(np.uint8)
    return {
        "empty": b"",
        "one byte": b"\x00",
        "two bytes": b"\xff\x00",
        # long runs: fills the table fast and forces repeated CLEARs
        "zeros 100k": bytes(100_000),
        "ones 64k": b"\xff" * 65_536,
        # crosses every code-width boundary then overflows the table
        "alternating runs": (b"A" * 300 + b"B" * 300) * 400,
        "all byte values": bytes(range(256)) * 400,
        "photo-like": photo.tobytes(),
        # worst case for LZW: never compresses, resets the table constantly
        "incompressible": rng.integers(0, 256, 300_000, dtype=np.uint8).tobytes(),
    }


@pytest.mark.parametrize("name", list(_lzw_cases()))
def test_lzw_round_trip_through_our_decoder(name):
    data = _lzw_cases()[name]
    encoded = _tiff.lzw_encode(data)
    assert bytes(_tiff.lzw_decode(encoded, len(data))) == data


@pytest.mark.parametrize("name", list(_lzw_cases()))
def test_lzw_output_is_readable_by_imagecodecs(name):
    """Independent decoder, so this is what actually pins the wire format."""
    imagecodecs = pytest.importorskip("imagecodecs")
    data = _lzw_cases()[name]
    assert bytes(imagecodecs.lzw_decode(_tiff.lzw_encode(data))) == data


def test_lzw_empty_input_is_a_valid_stream():
    """Must still emit CLEAR + EOI rather than nothing at all."""
    encoded = _tiff.lzw_encode(b"")
    assert len(encoded) > 0
    assert bytes(_tiff.lzw_decode(encoded, 0)) == b""


def test_lzw_compresses_repetitive_data():
    encoded = _tiff.lzw_encode(bytes(100_000))
    assert len(encoded) < 1000


def test_lzw_bounded_expansion_on_incompressible_data():
    """LZW expands worst case; the growth must stay within the bound the
    writer allocates from oc_tifflzw_encode_bound."""
    rng = np.random.default_rng(1)
    data = rng.integers(0, 256, 200_000, dtype=np.uint8).tobytes()
    encoded = _tiff.lzw_encode(data)
    assert len(encoded) < len(data) * 1.5 + 64
    assert bytes(_tiff.lzw_decode(encoded, len(data))) == data


def test_lzw_round_trips_through_a_real_tiff_file():
    import io
    tifffile = pytest.importorskip("tifffile")
    from opencodecs import TiffWriter

    rng = np.random.default_rng(2)
    arr = rng.integers(0, 256, (512, 512), dtype=np.uint8)
    arr[:256] = 7                      # half compressible, half not
    buf = io.BytesIO()
    with TiffWriter(buf) as tw:
        tw.write_page(arr, compression="lzw")
    buf.seek(0)
    assert np.array_equal(tifffile.imread(buf), arr)
