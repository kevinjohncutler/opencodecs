"""tifffile_patch PNG and WebP adapters keep imagecodecs' signatures.

The WebP adapter defaulted to ``lossless=False``, so
``tifffile.imwrite(compression="webp")`` under ``patched()`` wrote lossy
tiles where plain tifffile (imagecodecs, lossless by default) writes exact
ones. It also dropped ``method``, ``numthreads``, ``hasalpha`` and
``index``, and the PNG adapter dropped ``strategy`` and ``filter``. The
reference throughout is imagecodecs and plain tifffile.
"""

from __future__ import annotations

import io
import struct
import zlib

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
imagecodecs = pytest.importorskip("imagecodecs")
pytest.importorskip("opencodecs.codecs._webp")
pytest.importorskip("opencodecs.codecs._png")

import opencodecs.tifffile_patch as patch  # noqa: E402
from opencodecs.codecs import _webp  # noqa: E402
from _tifffile_guard import requires_patchable_tifffile  # noqa: E402


def _textured(shape=(64, 96), channels=3, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[:shape[0], :shape[1]]
    planes = [x * 2, y * 3, x + y, (x * y) % 256][:channels]
    image = np.stack(planes, -1) + rng.integers(0, 20, shape + (channels,))
    return np.clip(image, 0, 255).astype(np.uint8)


def _same_libwebp():
    return _webp.version() == imagecodecs.webp_version()


def _idat(data):
    pos, out = 8, b""
    while pos < len(data):
        (length,) = struct.unpack(">I", data[pos:pos + 4])
        if data[pos + 4:pos + 8] == b"IDAT":
            out += data[pos + 8:pos + 8 + length]
        pos += 12 + length
    return out


def _filter_types(data, height, row_bytes):
    raw = zlib.decompress(_idat(data))
    return {raw[r * (row_bytes + 1)] for r in range(height)}


def test_webp_encode_defaults_to_lossless():
    image = _textured()
    encoded = patch.webp_encode(image)
    np.testing.assert_array_equal(imagecodecs.webp_decode(encoded), image)
    if _same_libwebp():
        assert encoded == imagecodecs.webp_encode(image)


@pytest.mark.parametrize("kwargs", [
    {"level": 50, "lossless": False, "method": 0},
    {"level": 50, "lossless": False, "method": 6},
    {"level": 90, "lossless": 1, "method": -1, "numthreads": 4},
    {"lossless": np.False_},
])
def test_webp_encode_forwards_every_option(kwargs):
    image = _textured()
    ours = patch.webp_encode(image, **kwargs)
    if _same_libwebp():
        assert ours == imagecodecs.webp_encode(image, **kwargs)
    else:
        assert ours == _webp.encode(image, **kwargs)


def test_webp_method_changes_the_bytes():
    image = _textured()
    assert patch.webp_encode(image, 50, lossless=False, method=0) != \
        patch.webp_encode(image, 50, lossless=False, method=6)


def test_encoders_refuse_unknown_options():
    image = _textured()
    with pytest.raises(TypeError):
        patch.webp_encode(image, delay=10)
    with pytest.raises(TypeError):
        patch.webp_encode(image, out=bytearray(1 << 20))
    with pytest.raises(TypeError):
        patch.png_encode(image, primaries=1)


def test_webp_decode_hasalpha_for_an_opaque_rgba_tile():
    # libwebp stores an all-opaque RGBA image without alpha. tifffile
    # passes hasalpha=True for four-sample images to get it back.
    tile = _textured(channels=4)
    tile[..., 3] = 255
    blob = imagecodecs.webp_encode(tile, lossless=True)
    assert imagecodecs.webp_decode(blob).shape[-1] == 3
    assert patch.webp_decode(blob).shape[-1] == 3
    got = patch.webp_decode(blob, hasalpha=True)
    np.testing.assert_array_equal(got, imagecodecs.webp_decode(blob, hasalpha=True))
    np.testing.assert_array_equal(got, tile)


@requires_patchable_tifffile
def test_tifffile_webp_default_is_exact_and_matches_plain_tifffile():
    image = _textured()
    plain = io.BytesIO()
    tifffile.imwrite(plain, image, compression="webp", tile=(32, 32),
                     photometric="rgb")
    patched = io.BytesIO()
    with patch.patched():
        tifffile.imwrite(patched, image, compression="webp", tile=(32, 32),
                         photometric="rgb")
        back = tifffile.imread(io.BytesIO(patched.getvalue()))
    np.testing.assert_array_equal(back, image)
    np.testing.assert_array_equal(tifffile.imread(io.BytesIO(patched.getvalue())), image)
    if _same_libwebp():
        assert patched.getvalue() == plain.getvalue()


@requires_patchable_tifffile
def test_tifffile_webp_rgba_with_opaque_tiles():
    image = _textured(channels=4)
    image[..., 3] = 255
    image[:32, :32, 3] = 9          # one tile keeps its alpha
    buffer = io.BytesIO()
    with patch.patched():
        tifffile.imwrite(buffer, image, compression="webp", tile=(32, 32),
                         photometric="rgb", extrasamples=[2])
        back = tifffile.imread(io.BytesIO(buffer.getvalue()))
    np.testing.assert_array_equal(back, image)


def test_png_encode_forwards_strategy_and_filter():
    image = _textured()
    rows = image.shape[1] * 3
    assert _filter_types(patch.png_encode(image, filter=16), 64, rows) == {1}
    assert _filter_types(patch.png_encode(image, filter="paeth"), 64, rows) == {4}
    flat = np.zeros((256, 256), np.uint8)
    fixed = patch.png_encode(flat, 6, strategy=4)
    # RFC 1951: BTYPE (bits 1-2 of the first deflate byte) 01 is fixed Huffman.
    assert (_idat(fixed)[2] >> 1) & 3 == 1
    assert len(patch.png_encode(flat, 6, strategy=2)) >= 256 * 257 // 8
    for encoded in (fixed, patch.png_encode(flat, 6, strategy=2)):
        np.testing.assert_array_equal(imagecodecs.png_decode(encoded), flat)


@requires_patchable_tifffile
def test_tifffile_png_compressionargs_reach_the_encoder():
    # Smooth, so deflate's fixed-Huffman block beats a stored one.
    image = np.add.outer(np.arange(64), np.arange(96)).astype(np.uint8)
    buffer = io.BytesIO()
    with patch.patched():
        tifffile.imwrite(buffer, image, compression="png", tile=(32, 32),
                         compressionargs={"level": 6, "strategy": 4, "filter": 16})
        back = tifffile.imread(io.BytesIO(buffer.getvalue()))
    np.testing.assert_array_equal(back, image)
    with tifffile.TiffFile(io.BytesIO(buffer.getvalue())) as tif:
        page = tif.pages[0]
        fh = tif.filehandle
        fh.seek(page.dataoffsets[0])
        tile = fh.read(page.databytecounts[0])
    assert _filter_types(tile, 32, 32) == {1}
    assert (_idat(tile)[2] >> 1) & 3 == 1
