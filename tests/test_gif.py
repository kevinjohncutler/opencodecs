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


# ---------------------------------------------------------------------------
# Streaming: GifReader + GifWriter
# ---------------------------------------------------------------------------


def test_gif_streaming_decode_single_frame():
    arr = np.tile(np.arange(256, dtype=np.uint8), (32, 1))
    blob = mod.encode(arr)
    r = mod.GifReader(blob)
    try:
        assert r.n_frames == 1
        assert r.shape == (32, 256, 3)
        frames = list(r.iter_frames())
        assert len(frames) == 1
        assert frames[0].shape == (32, 256, 3)
        np.testing.assert_array_equal(frames[0][..., 0], arr)
    finally:
        r.close()


def test_gif_streaming_multi_frame_roundtrip():
    """Encode 5 frames via GifWriter, decode via GifReader.iter_frames."""
    H, W, N = 48, 64, 5
    w = mod.GifWriter(width=W, height=H, loop=0)
    frames_in = []
    for f in range(N):
        a = np.full((H, W), f * 30, dtype=np.uint8)
        a[f*9:(f+1)*9, :] = 200
        frames_in.append(a)
        w.write_frame(a, delay_centiseconds=10)
    blob = w.close()
    assert blob[:6] in (b"GIF87a", b"GIF89a")

    r = mod.GifReader(blob)
    try:
        assert r.n_frames == N
        assert r.shape == (N, H, W, 3)
        # iter_frames
        out = list(r.iter_frames())
        assert len(out) == N
        for f in range(N):
            # Each frame should have the highlighted band at the right rows.
            band = out[f][f*9:(f+1)*9, :, 0]
            assert (band == 200).all(), f"frame {f}: highlight wrong"
    finally:
        r.close()


def test_gif_reader_random_access():
    """``reader[i]`` returns the i-th frame; identical to iter_frames()[i]."""
    H, W, N = 32, 32, 4
    w = mod.GifWriter(width=W, height=H, loop=0)
    for f in range(N):
        a = np.full((H, W), f * 50, dtype=np.uint8)
        w.write_frame(a)
    blob = w.close()
    r = mod.GifReader(blob)
    try:
        seq = list(r.iter_frames())
        for i in range(N):
            np.testing.assert_array_equal(r[i], seq[i])
        # Negative index.
        np.testing.assert_array_equal(r[-1], seq[-1])
        # Out of range.
        with pytest.raises(IndexError):
            r[N]
    finally:
        r.close()


def test_gif_reader_read_returns_stack():
    H, W, N = 24, 32, 3
    w = mod.GifWriter(width=W, height=H, loop=-1)  # no Netscape loop ext
    for f in range(N):
        w.write_frame(np.full((H, W), f * 80, dtype=np.uint8))
    blob = w.close()
    r = mod.GifReader(blob)
    try:
        stack = r.read()
        assert stack.shape == (N, H, W, 3)
    finally:
        r.close()


def test_gif_codec_open_returns_streaming_reader():
    """The unified Codec.open(...) API yields a GifReader."""
    import opencodecs as oc
    arr = np.tile(np.arange(256, dtype=np.uint8), (24, 1))
    blob = oc.write(None, arr, format="gif")
    with oc.get_codec("gif").open(blob) as r:
        assert hasattr(r, "iter_frames")
        assert r.n_frames == 1
        frame, = list(r.iter_frames())
        assert frame.shape == (24, 256, 3)


def test_gif_codec_advertises_streaming():
    import opencodecs as oc
    gif = oc.get_codec("gif")
    assert gif.streaming_decode is True
    assert gif.multi_frame is True


def test_gif_writer_rejects_wrong_dimensions():
    w = mod.GifWriter(width=64, height=48)
    with pytest.raises(mod.GifError, match="doesn't match writer dimensions"):
        w.write_frame(np.zeros((47, 64), dtype=np.uint8))
    w.close()


def test_gif_writer_rejects_3d_input():
    w = mod.GifWriter(width=32, height=32)
    with pytest.raises(mod.GifError, match="2D"):
        w.write_frame(np.zeros((32, 32, 3), dtype=np.uint8))
    w.close()


def test_gif_writer_double_close_returns_same_bytes():
    w = mod.GifWriter(width=16, height=16, loop=-1)
    w.write_frame(np.zeros((16, 16), dtype=np.uint8))
    first = w.close()
    second = w.close()
    assert first == second


# ---------------------------------------------------------------------------
# decode_fast: opencodecs's custom LZW (oc_giflzw)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("shape", [
    (256, 256), (1, 1), (512, 1024), (100, 100), (333, 777),
])
def test_decode_fast_byte_equal_to_libgif(shape):
    """The custom LZW must produce byte-identical output to libgif's
    reference decoder for every test image. Tested across a range of
    shapes including 1x1, oddly-sized, and large."""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, shape, dtype=np.uint8)
    blob = mod.encode(arr)
    ref = mod.decode(blob, asrgb=False)
    fast = mod.decode_fast(blob, asrgb=False)
    np.testing.assert_array_equal(ref, fast)


def test_decode_fast_rgb_output_matches():
    arr = np.tile(np.arange(256, dtype=np.uint8), (64, 1))
    blob = mod.encode(arr)
    ref_rgb = mod.decode(blob, asrgb=True)
    fast_rgb = mod.decode_fast(blob, asrgb=True)
    np.testing.assert_array_equal(ref_rgb, fast_rgb)


def test_decode_fast_with_custom_colormap():
    """Custom palette → RGB compositing should still be correct."""
    rng = np.random.default_rng(0)
    cmap = rng.integers(0, 256, (256, 3), dtype=np.uint8)
    arr = rng.integers(0, 256, (96, 128), dtype=np.uint8)
    blob = mod.encode(arr, colormap=cmap)
    fast_rgb = mod.decode_fast(blob)
    expected = cmap[arr]
    np.testing.assert_array_equal(fast_rgb, expected)


def test_decode_fast_constant_image():
    """All-zero image — edge case for LZW state machine."""
    arr = np.zeros((64, 64), dtype=np.uint8)
    blob = mod.encode(arr)
    fast = mod.decode_fast(blob, asrgb=False)
    np.testing.assert_array_equal(fast, arr)


def test_decode_fast_constant_nonzero():
    arr = np.full((48, 96), 200, dtype=np.uint8)
    blob = mod.encode(arr)
    fast = mod.decode_fast(blob, asrgb=False)
    np.testing.assert_array_equal(fast, arr)
