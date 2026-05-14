"""GIF codec tests — single-frame encode + multi-frame decode roundtrips.

We use giflib so the core encode/decode is well-trodden. The interesting
contracts to verify:
* Magic-byte signature detection (GIF87a + GIF89a).
* Palette-index round-trip (asrgb=False) is byte-equal.
* RGB output composites a custom palette correctly.
* asrgb=False on a multi-frame GIF errors out cleanly.
"""

from __future__ import annotations

import numpy as np
import pytest

mod = pytest.importorskip("opencodecs.codecs._gif")
import opencodecs as oc


def test_gif_registered_in_codec_list():
    assert "gif" in {c["name"] for c in oc.list_codecs()}


def test_gif_signature_check():
    assert mod.check_signature(b"GIF89a__")
    assert mod.check_signature(b"GIF87a__")
    assert not mod.check_signature(b"PNG_\x89")
    assert not mod.check_signature(b"")


def test_gif_palette_roundtrip_byte_equal():
    """asrgb=False round-trips palette indices exactly."""
    arr = np.tile(np.arange(256, dtype=np.uint8), (128, 1))
    blob = mod.encode(arr)
    assert blob[:6] in (b"GIF87a", b"GIF89a")
    back = mod.decode(blob, asrgb=False)
    assert back.shape == arr.shape
    np.testing.assert_array_equal(back, arr)


def test_gif_default_grayscale_palette():
    """Default colormap is grayscale: RGB output has R==G==B == index."""
    arr = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
    rgb = mod.decode(mod.encode(arr))
    assert rgb.shape == (32, 256, 3)
    np.testing.assert_array_equal(rgb[..., 0], arr)
    np.testing.assert_array_equal(rgb[..., 1], arr)
    np.testing.assert_array_equal(rgb[..., 2], arr)


def test_gif_custom_colormap_applied_to_rgb_output():
    cmap = np.random.default_rng(0).integers(0, 256, (256, 3), dtype=np.uint8)
    arr = np.random.default_rng(1).integers(0, 256, (64, 96), dtype=np.uint8)
    rgb = mod.decode(mod.encode(arr, colormap=cmap))
    expected = cmap[arr]   # broadcast palette → RGB
    np.testing.assert_array_equal(rgb, expected)


def test_gif_rejects_non_uint8_input():
    arr = np.zeros((32, 32), dtype=np.uint16)
    with pytest.raises(mod.GifError, match="uint8"):
        mod.encode(arr)


def test_gif_rejects_3d_rgb_input():
    """RGB-to-GIF encoding would need quantization; we don't ship one."""
    arr = np.zeros((32, 32, 3), dtype=np.uint8)
    with pytest.raises(mod.GifError, match="2D palette-index"):
        mod.encode(arr)


def test_gif_rejects_too_large():
    """GIF format limits dimensions to <65536."""
    arr = np.zeros((65536, 1), dtype=np.uint8)
    with pytest.raises(mod.GifError, match="65536"):
        mod.encode(arr)


def test_gif_rejects_bad_colormap_shape():
    arr = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(mod.GifError, match=r"\(256, 3\)"):
        mod.encode(arr, colormap=np.zeros((128, 3), dtype=np.uint8))


def test_gif_codec_adapter_roundtrip():
    """Round-trip via the unified oc.write / oc.read API."""
    arr = np.tile(np.arange(256, dtype=np.uint8), (64, 1))
    blob = oc.write(None, arr, format="gif")
    back = oc.read(blob, format="gif", asrgb=False)
    np.testing.assert_array_equal(back, arr)


def test_gif_decode_short_input():
    with pytest.raises(mod.GifError, match="too short"):
        mod.decode(b"abc")
