"""PNG row streams use native scanlines and independent PNG fixtures."""

import io
import struct
import zlib

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._png_codec import PngCodec


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
def test_rows_roundtrip_independently(dtype, channels):
    imagecodecs = pytest.importorskip("imagecodecs")
    shape = (67, 91) if channels == 1 else (67, 91, channels)
    image = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    codec = PngCodec()
    encoded = codec.encode_rows(iter(image), shape=image.shape, dtype=image.dtype)
    np.testing.assert_array_equal(imagecodecs.png_decode(encoded), image)
    updates = list(codec.decode_rows(imagecodecs.png_encode(image)))
    assert len(updates) == shape[0]
    assert all(update.x_start == 0 and update.x_step == 1 for update in updates)
    np.testing.assert_array_equal(np.stack([update.pixels for update in updates]), image)


@pytest.mark.parametrize("height,width", [(71, 93), (1, 1), (2, 3), (3, 2), (9, 1)])
def test_adam7_updates_have_explicit_coordinates(height, width):
    imagecodecs = pytest.importorskip("imagecodecs")
    image = np.arange(height * width * 3, dtype=np.uint8).reshape(height, width, 3)
    # Minimal unfiltered Adam7 fixture, independently validated by libpng.
    # This does not use the PNG implementation under test to encode a fixture.
    raw = bytearray()
    for x, y, dx, dy in [(0, 0, 8, 8), (4, 0, 8, 8), (0, 4, 4, 8),
                         (2, 0, 4, 4), (0, 2, 2, 4), (1, 0, 2, 2),
                         (0, 1, 1, 2)]:
        if x >= width:
            continue
        for row in range(y, height, dy):
            raw.append(0)
            raw.extend(image[row, x::dx].tobytes())
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data +
                struct.pack(">I", zlib.crc32(kind + data)))
    encoded = (b"\x89PNG\r\n\x1a\n" +
               chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 1)) +
               chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    np.testing.assert_array_equal(imagecodecs.png_decode(encoded), image)
    result = np.empty_like(image)
    coverage = np.zeros((height, width), dtype=np.uint8)
    updates = list(PngCodec().decode_rows(encoded))
    for update in updates:
        result[update.row, update.x_start::update.x_step] = update.pixels
        coverage[update.row, update.x_start::update.x_step] += 1
    np.testing.assert_array_equal(result, imagecodecs.png_decode(encoded))
    np.testing.assert_array_equal(coverage, 1)
    assert len({id(update.pixels) for update in updates}) == len(updates)


class ShortStream(io.BytesIO):
    def read(self, count=-1):
        return super().read(min(count, 19))


@pytest.mark.parametrize("bits,color_type", [(1, 0), (2, 0), (4, 0), (4, 3)])
def test_packed_and_palette_rows_match_independent_decoder(bits, color_type):
    imagecodecs = pytest.importorskip("imagecodecs")
    height, width = 7, 19
    levels = np.arange(height * width, dtype="u1").reshape(height, width) % (1 << bits)
    raw = bytearray()
    for row in levels:
        bit_rows = ((row[:, None] >> np.arange(bits - 1, -1, -1)) & 1).reshape(-1)
        raw.append(0)
        raw.extend(np.packbits(bit_rows).tobytes())
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))
    encoded = b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, bits, color_type, 0, 0, 0))
    if color_type == 3:
        palette = np.arange((1 << bits) * 3, dtype="u1").reshape(-1, 3) * 5
        encoded += chunk(b"PLTE", palette.tobytes())
    encoded += chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b"")
    # Rows keep the color type exactly as whole-image decode and libpng
    # do: sub-byte gray stays (H, W), a palette without tRNS is RGB.
    reference = imagecodecs.png_decode(encoded)
    assert reference.shape == ((height, width) if color_type == 0 else (height, width, 3))
    rows = list(PngCodec().decode_rows(encoded))
    np.testing.assert_array_equal(np.stack([row.pixels for row in rows]), reference)

    def write(self, value):
        return super().write(memoryview(value)[:17])


def test_row_stream_short_io_and_ownership():
    imagecodecs = pytest.importorskip("imagecodecs")
    image = np.arange(64 * 128, dtype=np.uint16).reshape(64, 128)
    dest = ShortStream()
    PngCodec().encode_rows(image, shape=image.shape, dtype=image.dtype, dest=dest)
    assert not dest.closed
    np.testing.assert_array_equal(imagecodecs.png_decode(dest.getvalue()), image)
    source = ShortStream(dest.getvalue())
    np.testing.assert_array_equal(np.stack([u.pixels for u in PngCodec().decode_rows(source)]), image)
    assert not source.closed


def test_decoder_backpressure_and_early_close():
    imagecodecs = pytest.importorskip("imagecodecs")
    image = np.random.default_rng(93).integers(0, 256, (1024, 1024, 3), dtype=np.uint8)
    encoded = imagecodecs.png_encode(image)
    source = io.BytesIO(encoded)
    iterator = PngCodec().decode_rows(source)
    first = next(iterator)
    np.testing.assert_array_equal(first.pixels, image[0])
    assert source.tell() < len(encoded) // 2
    position = source.tell()
    iterator.close()
    assert source.tell() == position
    assert not source.closed


def test_row_count_errors_and_truncation():
    image = np.zeros((16, 32), dtype=np.uint8)
    codec = PngCodec()
    with pytest.raises(ValueError, match="not enough"):
        codec.encode_rows(image[:-1], shape=image.shape, dtype=image.dtype)
    with pytest.raises(ValueError, match="too many"):
        codec.encode_rows(image, shape=(15, 32), dtype=image.dtype)
    encoded = codec.encode_rows(image, shape=image.shape, dtype=image.dtype)
    with pytest.raises((EOFError, RuntimeError)):
        list(codec.decode_rows(encoded[:-1]))


def test_png_row_sink_errors_propagate():
    class Broken:
        def write(self, data):
            raise OSError("sink stopped")
    with pytest.raises(OSError, match="sink stopped"):
        PngCodec().encode_rows([np.zeros(32, dtype=np.uint8)], shape=(1, 32),
                               dtype=np.uint8, dest=Broken())
