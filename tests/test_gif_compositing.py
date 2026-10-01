"""GIF89a compositing: transparency, disposal, background and interlace.

Every fixture here is a GIF89a stream written byte by byte from the
specification (a literal-only LZW encoder, an explicit Graphic Control
Extension per frame), so the expected pixels come from the spec text,
not from a round trip through opencodecs:

* section 18: pixels not covered by an image show the background color;
* section 23: a pixel equal to the transparency index leaves the display
  unchanged; disposal 2 restores the frame's area to the background
  color, disposal 3 restores what was there before the frame;
* section 20 and Appendix E: interlaced rows arrive in four passes.

Where imagecodecs is installed, its ``gif_decode`` is checked as a
second, independent reference.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

mod = pytest.importorskip("opencodecs.codecs._gif")
import opencodecs as oc

W, H, BG = 7, 5, 9


def _palette():
    return np.random.default_rng(12345).integers(
        0, 256, (256, 3), dtype=np.uint8)


PAL = _palette()


def _lzw(indices):
    """Literal-only LZW at minimum code size 8 (9-bit codes).

    A clear code every 200 literals keeps the decoder's table below 512
    entries, so the code width never grows.
    """
    codes = []
    indices = [int(i) for i in np.asarray(indices).ravel()]
    for start in range(0, max(len(indices), 1), 200):
        codes.append(256)
        codes.extend(indices[start:start + 200])
    codes.append(257)
    bits = nbits = 0
    out = bytearray()
    for c in codes:
        bits |= c << nbits
        nbits += 9
        while nbits >= 8:
            out.append(bits & 0xFF)
            bits >>= 8
            nbits -= 8
    if nbits:
        out.append(bits & 0xFF)
    blocks = bytearray([8])
    for i in range(0, len(out), 255):
        chunk = out[i:i + 255]
        blocks += bytes([len(chunk)]) + chunk
    return bytes(blocks + b"\x00")


def _gce(disposal, trans=None, delay=5):
    flags = (disposal << 2) | (1 if trans is not None else 0)
    return (b"\x21\xf9\x04" + bytes([flags]) + struct.pack("<H", delay)
            + bytes([trans or 0]) + b"\x00")


def _image(left, top, w, h, idx, interlace=False, stream=None):
    idx = np.asarray(idx, np.uint8).reshape(h, w)
    if stream is None:
        stream = idx
    return (b"\x2c" + struct.pack("<HHHH", left, top, w, h)
            + bytes([0x40 if interlace else 0]) + _lzw(stream))


def _gif(frames, w=W, h=H, bg=BG):
    hdr = (b"GIF89a" + struct.pack("<HH", w, h) + bytes([0xF7, bg, 0])
           + PAL.tobytes())
    return hdr + b"".join(frames) + b"\x3b"


def _full(v):
    return _image(0, 0, W, H, np.full((H, W), v))


def _labels(rgb):
    """Map RGB pixels back to palette indices (the fixture palette has
    no duplicate colors among the indices used here)."""
    lut = {tuple(int(c) for c in PAL[i]): i for i in range(256)}
    rgb = np.asarray(rgb)
    flat = rgb.reshape(-1, 3)
    return np.array([lut[tuple(int(c) for c in px)] for px in flat]
                    ).reshape(rgb.shape[:-1])


def _all_rgb_paths(blob):
    """Every opencodecs RGB decode path, which must agree."""
    codec = oc.get_codec("gif")
    with codec.open(blob) as r:
        streamed = r.read()
        iterated = np.stack(list(r.iter_frames()))
        indexed = np.stack([r[i] for i in range(r.n_frames)])
    return {
        "codec.decode": codec.decode(blob),
        "module.decode": mod.decode(blob),
        "open.read": streamed,
        "iter_frames": iterated.reshape(np.shape(streamed)),
        "getitem": indexed.reshape(np.shape(streamed)),
    }


def _imagecodecs():
    try:
        import imagecodecs
    except ImportError:
        return None
    return imagecodecs


# Frame 1 is a 3x2 rectangle at (left=2, top=1).
F1 = np.array([[33, 22, 22], [22, 33, 22]])


def _expected(frames):
    return np.stack([np.asarray(f) for f in frames])


def test_transparent_index_leaves_pixel_unchanged():
    blob = _gif([
        _gce(1) + _full(11),
        _gce(1, trans=33) + _image(2, 1, 3, 2, F1),
        _gce(1) + _image(6, 4, 1, 1, [44]),
    ])
    f0 = np.full((H, W), 11)
    f1 = f0.copy()
    f1[1:3, 2:5] = np.where(F1 == 33, 11, F1)
    f2 = f1.copy()
    f2[4, 6] = 44
    want = _expected([f0, f1, f2])
    for name, got in _all_rgb_paths(blob).items():
        np.testing.assert_array_equal(_labels(got), want, err_msg=name)
    ic = _imagecodecs()
    if ic is not None:
        np.testing.assert_array_equal(_labels(ic.gif_decode(blob)), want)


def test_transparency_on_first_frame_shows_background():
    blob = _gif([_gce(0, trans=33) + _image(0, 0, 3, 2, F1)])
    want = np.full((H, W), BG)
    want[0:2, 0:3] = np.where(F1 == 33, BG, F1)
    for name, got in _all_rgb_paths(blob).items():
        np.testing.assert_array_equal(_labels(got), want, err_msg=name)
    np.testing.assert_array_equal(_labels(mod.decode_fast(blob)), want)


def test_disposal_2_restores_background():
    blob = _gif([
        _gce(1) + _full(11),
        _gce(2) + _image(2, 1, 3, 2, np.full((2, 3), 22)),
        _gce(1) + _image(6, 4, 1, 1, [44]),
    ])
    f0 = np.full((H, W), 11)
    f1 = f0.copy()
    f1[1:3, 2:5] = 22
    f2 = f0.copy()
    f2[1:3, 2:5] = BG
    f2[4, 6] = 44
    want = _expected([f0, f1, f2])
    for name, got in _all_rgb_paths(blob).items():
        np.testing.assert_array_equal(_labels(got), want, err_msg=name)
    ic = _imagecodecs()
    if ic is not None:
        np.testing.assert_array_equal(_labels(ic.gif_decode(blob)), want)


def test_disposal_3_restores_previous():
    blob = _gif([
        _gce(1) + _full(11),
        _gce(1) + _image(1, 0, 2, 2, np.full((2, 2), 55)),
        _gce(3) + _image(2, 1, 3, 2, np.full((2, 3), 22)),
        _gce(1) + _image(6, 4, 1, 1, [44]),
    ])
    f0 = np.full((H, W), 11)
    f1 = f0.copy()
    f1[0:2, 1:3] = 55
    f2 = f1.copy()
    f2[1:3, 2:5] = 22
    f3 = f1.copy()          # frame 2's area goes back to frame 1's state
    f3[4, 6] = 44
    want = _expected([f0, f1, f2, f3])
    for name, got in _all_rgb_paths(blob).items():
        np.testing.assert_array_equal(_labels(got), want, err_msg=name)
    ic = _imagecodecs()
    if ic is not None:
        np.testing.assert_array_equal(_labels(ic.gif_decode(blob)), want)


def test_disposal_3_after_disposal_2_restores_the_disposed_canvas():
    """Frame 0's disposal 2 restores the whole canvas to the background
    before frame 1, so frame 1's disposal 3 restores that background.
    imagecodecs 2026.8.16 does not: it leaves frame 1's pixels in frame
    2 ([9, 22, 22, 9, ...] on row 1), so it is not a reference here."""
    blob = _gif([
        _gce(2) + _full(11),
        _gce(3) + _image(1, 1, 2, 2, np.full((2, 2), 22)),
        _gce(1) + _image(6, 4, 1, 1, [44]),
    ])
    f0 = np.full((H, W), 11)
    f1 = np.full((H, W), BG)
    f1[1:3, 1:3] = 22
    f2 = np.full((H, W), BG)
    f2[4, 6] = 44
    want = _expected([f0, f1, f2])
    for name, got in _all_rgb_paths(blob).items():
        np.testing.assert_array_equal(_labels(got), want, err_msg=name)


def test_graphic_control_scope_is_the_next_image_only():
    """A GCE applies to the first image after it (GIF89a section 23)."""
    blob = _gif([
        _gce(1) + _full(11),
        _gce(2, trans=22) + _image(0, 0, 2, 1, [22, 66]),
        _image(0, 0, 2, 1, [22, 77]),     # no GCE: opaque, disposal 0
    ])
    f0 = np.full((H, W), 11)
    f1 = f0.copy()
    f1[0, 1] = 66
    f2 = f0.copy()          # frame 1 disposed to background first
    f2[0, :2] = BG
    f2[0, 0] = 22
    f2[0, 1] = 77
    want = _expected([f0, f1, f2])
    with oc.get_codec("gif").open(blob) as r:
        assert r.disposal == (1, 2, 0)
        assert r.transparent_index == (-1, 22, -1)
    for name, got in _all_rgb_paths(blob).items():
        np.testing.assert_array_equal(_labels(got), want, err_msg=name)


def test_single_subrectangle_frame_fills_background():
    blob = _gif([_gce(0) + _image(2, 1, 3, 2, np.full((2, 3), 22))])
    want = np.full((H, W), BG)
    want[1:3, 2:5] = 22
    for name, got in _all_rgb_paths(blob).items():
        assert got.shape == (H, W, 3), name
        np.testing.assert_array_equal(_labels(got), want, err_msg=name)
    np.testing.assert_array_equal(_labels(mod.decode_fast(blob)), want)
    ic = _imagecodecs()
    if ic is not None:
        np.testing.assert_array_equal(_labels(ic.gif_decode(blob)), want)


@pytest.mark.parametrize("h", [1, 2, 3, 5, 8, 11, 17])
def test_interlaced_rows_are_reordered(h):
    w = 3
    rows = np.arange(h)[:, None] * 10 % 250 + np.zeros((1, w), int)
    order = (list(range(0, h, 8)) + list(range(4, h, 8))
             + list(range(2, h, 4)) + list(range(1, h, 2)))
    blob = _gif([_gce(0) + _image(0, 0, w, h, rows, interlace=True,
                                  stream=rows[order])], w=w, h=h, bg=0)
    for name, got in _all_rgb_paths(blob).items():
        np.testing.assert_array_equal(_labels(got), rows, err_msg=name)
    np.testing.assert_array_equal(_labels(mod.decode_fast(blob)), rows)
    # Palette indices: our LZW path and libgif's must both deinterlace.
    np.testing.assert_array_equal(mod.decode_fast(blob, asrgb=False), rows)
    np.testing.assert_array_equal(mod.decode(blob, asrgb=False), rows)
    np.testing.assert_array_equal(mod._decode_indices_libgif(blob), rows)
    ic = _imagecodecs()
    if ic is not None:
        np.testing.assert_array_equal(_labels(ic.gif_decode(blob)), rows)


def _three_frames():
    return _gif([
        _gce(1) + _full(11),
        _gce(1, trans=33) + _image(2, 1, 3, 2, F1),
        _gce(1) + _image(6, 4, 1, 1, [44]),
    ])


def test_index_draws_one_frame_over_background():
    """``index=`` follows imagecodecs: that frame alone, no history."""
    blob = _three_frames()
    codec = oc.get_codec("gif")
    want1 = np.full((H, W), BG)
    want1[1:3, 2:5] = np.where(F1 == 33, BG, F1)
    np.testing.assert_array_equal(_labels(codec.decode(blob, index=1)),
                                  want1)
    np.testing.assert_array_equal(_labels(codec.decode(blob, index=-3)),
                                  np.full((H, W), 11))
    with pytest.raises(IndexError):
        codec.decode(blob, index=3)
    ic = _imagecodecs()
    if ic is not None:
        for i in range(3):
            np.testing.assert_array_equal(
                codec.decode(blob, index=i), ic.gif_decode(blob, index=i))


def test_palette_indices_are_canvas_sized_per_frame():
    """``asrgb=False`` returns every frame's indices on a zero canvas,
    the layout imagecodecs uses."""
    blob = _three_frames()
    codec = oc.get_codec("gif")
    got = codec.decode(blob, asrgb=False)
    want = np.zeros((3, H, W), np.uint8)
    want[0] = 11
    want[1, 1:3, 2:5] = F1
    want[2, 4, 6] = 44
    np.testing.assert_array_equal(got, want)
    np.testing.assert_array_equal(codec.decode(blob, asrgb=False, index=1),
                                  want[1])
    ic = _imagecodecs()
    if ic is not None:
        np.testing.assert_array_equal(got, ic.gif_decode(blob, asrgb=False))
        np.testing.assert_array_equal(
            codec.decode(blob, asrgb=False, index=1),
            ic.gif_decode(blob, asrgb=False, index=1))


def test_decode_out_parameter():
    blob = _three_frames()
    codec = oc.get_codec("gif")
    out = np.empty((3, H, W, 3), np.uint8)
    assert codec.decode(blob, out=out) is out
    np.testing.assert_array_equal(out, codec.decode(blob))


def test_index_by_position():
    """imagecodecs ``gif_decode(data, index=None, *, asrgb, out)`` takes
    ``index`` by position; the codec and the module do too."""
    blob = _three_frames()
    codec = oc.get_codec("gif")
    for i in (0, 1, 2, -1):
        np.testing.assert_array_equal(codec.decode(blob, i),
                                      codec.decode(blob, index=i))
        np.testing.assert_array_equal(mod.decode(blob, i),
                                      mod.decode(blob, index=i))
        np.testing.assert_array_equal(mod.decode(blob, i, asrgb=False),
                                      codec.decode(blob, index=i,
                                                   asrgb=False))
    ic = _imagecodecs()
    if ic is not None:
        np.testing.assert_array_equal(codec.decode(blob, 1),
                                      ic.gif_decode(blob, 1))


def test_transparent_first_frame_stays_rgb():
    """When the first frame uses its transparent index, imagecodecs
    returns a fourth channel that is 255 everywhere. opencodecs returns
    RGB; the three color channels agree with the spec and imagecodecs."""
    idx = np.full((H, W), 3)
    idx[0, :3] = 200
    blob = _gif([_gce(1, trans=200) + _image(0, 0, W, H, idx),
                 _gce(0) + _image(1, 1, 2, 2, np.full((2, 2), 7))])
    got = oc.get_codec("gif").decode(blob)
    want0 = np.where(idx == 200, BG, idx)
    want1 = want0.copy()
    want1[1:3, 1:3] = 7
    np.testing.assert_array_equal(_labels(got), _expected([want0, want1]))
    np.testing.assert_array_equal(_labels(mod.decode(blob, 0)), want0)
    ic = _imagecodecs()
    if ic is not None:
        ref = ic.gif_decode(blob)
        if ref.shape[-1] == 4:
            assert (ref[..., 3] == 255).all()
        np.testing.assert_array_equal(got, ref[..., :3])


def _every_path(blob):
    """Every opencodecs decode call, RGB and indices, as callables."""
    codec = oc.get_codec("gif")

    def opened():
        with codec.open(blob) as r:
            return r.read()

    return {
        "codec.decode": lambda: codec.decode(blob),
        "module.decode": lambda: mod.decode(blob),
        "module.decode index": lambda: mod.decode(blob, 0),
        "module.decode asrgb=False": lambda: mod.decode(blob, asrgb=False),
        "decode_fast": lambda: mod.decode_fast(blob),
        "decode_fast asrgb=False":
            lambda: mod.decode_fast(blob, asrgb=False),
        "open.read": opened,
    }


def test_frame_with_too_few_pixels_is_an_error():
    """GIF89a section 22: a frame's image data codes width * height
    pixels. An end code after 10 of 64 used to leave the rest of the
    raster as uninitialized memory, different on every call; libgif and
    imagecodecs report "Image EOF detected before image complete"."""
    blob = _gif([_image(0, 0, 8, 8, np.zeros(64), stream=np.arange(10))],
                w=8, h=8)
    for name, call in _every_path(blob).items():
        with pytest.raises(mod.GifError, match="EOF"):
            call()
            pytest.fail(name)
    ic = _imagecodecs()
    if ic is not None:
        with pytest.raises(Exception, match="EOF"):
            ic.gif_decode(blob)


def test_codes_past_the_last_pixel_are_ignored():
    """libgif decodes width * height pixels and skips the rest of the
    image data, and imagecodecs follows it; the module decode read such
    files through libgif before it moved to the shared reader."""
    idx = np.arange(70) % 50
    blob = _gif([_image(0, 0, 8, 8, np.zeros(64), stream=idx)], w=8, h=8)
    want = idx[:64].reshape(8, 8)
    for name, call in _every_path(blob).items():
        got = call()
        if got.ndim == 3:
            got = _labels(got)
        np.testing.assert_array_equal(got, want, err_msg=name)
    ic = _imagecodecs()
    if ic is not None:
        np.testing.assert_array_equal(_labels(ic.gif_decode(blob)), want)


def _raw_image(w, h, codes):
    """An image descriptor whose LZW data is the given 9-bit codes."""
    bits = nbits = 0
    out = bytearray()
    for c in codes:
        bits |= c << nbits
        nbits += 9
        while nbits >= 8:
            out.append(bits & 0xFF)
            bits >>= 8
            nbits -= 8
    if nbits:
        out.append(bits & 0xFF)
    return (b"\x2c" + struct.pack("<HHHH", 0, 0, w, h) + b"\x00\x08"
            + bytes([len(out)]) + bytes(out) + b"\x00")


def test_undefined_lzw_code_is_an_error():
    """GIF89a Appendix F: a code may name only a table entry already
    defined or the one being defined next (here 258). Code 300 used to
    be taken as the next entry, and the entries built after it pointed
    at uninitialized table slots, so later codes walked outside the
    table; fuzzed files crashed the interpreter. libgif and imagecodecs
    reject it."""
    blob = _gif([_raw_image(8, 8, [256, 1, 300, 2, 259, 1, 1, 257])],
                w=8, h=8)
    for name, call in _every_path(blob).items():
        with pytest.raises(mod.GifError):
            call()
            pytest.fail(name)
    ic = _imagecodecs()
    if ic is not None:
        with pytest.raises(Exception):
            ic.gif_decode(blob)


@pytest.mark.parametrize("fw, fh", [(9, 0), (0, 2)])
@pytest.mark.parametrize("first", [False, True])
def test_zero_size_frame_draws_nothing(fw, fh, first):
    """A frame of zero width or height codes no pixels, so it leaves the
    canvas as it was. libgif's DGifSlurp writes past its raster for one,
    so ``asrgb=False`` (then decoded through libgif) crashed the
    interpreter; every path now goes through the shared reader.
    imagecodecs raises GifError for such a file."""
    frames = [_gce(0) + _image(0, 0, fw, fh, np.zeros((fh, fw)))]
    if first:
        frames.insert(0, _gce(0) + _image(0, 0, 9, 2, np.full((2, 9), 5)))
    blob = _gif(frames, w=9, h=2)
    shown = np.full((2, 9), 5 if first else BG)
    want_rgb = _expected([shown] * len(frames)) if first else shown
    want_idx = np.zeros((len(frames), 2, 9), np.uint8)
    if first:
        want_idx[0] = 5
    else:
        want_idx = want_idx[0]
    codec = oc.get_codec("gif")
    paths = _every_path(blob)
    for name in ("codec.decode", "module.decode", "open.read"):
        np.testing.assert_array_equal(_labels(paths[name]()), want_rgb,
                                      err_msg=name)
    np.testing.assert_array_equal(paths["module.decode asrgb=False"](),
                                  want_idx)
    np.testing.assert_array_equal(codec.decode(blob, asrgb=False), want_idx)
    np.testing.assert_array_equal(_labels(mod.decode_fast(blob)),
                                  np.full((2, 9), 5 if first else BG))
    np.testing.assert_array_equal(
        _labels(mod.decode(blob, len(frames) - 1)), np.full((2, 9), BG))
    np.testing.assert_array_equal(
        codec.decode(blob, len(frames) - 1, asrgb=False),
        np.zeros((2, 9), np.uint8))
    with pytest.raises(mod.GifError, match="zero width or height"):
        mod._decode_indices_libgif(blob)
    ic = _imagecodecs()
    if ic is not None:
        with pytest.raises(Exception):
            ic.gif_decode(blob, asrgb=False)


def test_frame_past_the_screen_is_clipped():
    """GIF89a positions an image within the logical screen, the display
    area, so the part of a frame outside it is not shown. imagecodecs
    enlarges its canvas to hold the frame instead (a documented
    difference), so only the visible part is compared with it."""
    idx = (np.arange(20).reshape(4, 5) + 1)
    blob = _gif([_gce(0) + _image(3, 2, 5, 4, idx)], w=6, h=5)
    want = np.full((5, 6), BG)
    want[2:, 3:] = idx[:3, :3]
    for name, call in _every_path(blob).items():
        got = call()
        if got.ndim == 3:
            got = _labels(got)
        else:
            want_idx = np.zeros((5, 6), np.uint8)
            want_idx[2:, 3:] = idx[:3, :3]
            np.testing.assert_array_equal(got, want_idx, err_msg=name)
            continue
        np.testing.assert_array_equal(got, want, err_msg=name)
    ic = _imagecodecs()
    if ic is not None:
        ref = ic.gif_decode(blob)
        assert ref.shape[:2] != (5, 6)
        np.testing.assert_array_equal(_labels(ref[2:5, 3:6, :3]),
                                      idx[:3, :3])


def test_decode_out_must_match_like_imagecodecs():
    """``out`` follows imagecodecs: same dtype, C-contiguous, and the
    decoded shape apart from length-1 axes. A wrong one raised nothing
    and was filled by broadcasting or casting."""
    blob = _three_frames()
    codec = oc.get_codec("gif")
    want = codec.decode(blob)
    got = codec.decode(blob, out=np.empty((1,) + want.shape, np.uint8))
    np.testing.assert_array_equal(got, want)
    idx = codec.decode(blob, asrgb=False)
    out = np.empty(idx.shape, np.uint8)
    assert codec.decode(blob, asrgb=False, out=out) is out
    np.testing.assert_array_equal(out, idx)
    one = codec.decode(blob, 1)
    bad = [np.empty((2,) + one.shape, np.uint8),
           np.empty(one.shape, np.float32),
           np.empty((one.size,), np.uint8),
           np.empty((2 * one.shape[0],) + one.shape[1:], np.uint8)[::2]]
    ic = _imagecodecs()
    for o in bad:
        with pytest.raises(ValueError):
            codec.decode(blob, 1, out=o)
        if ic is not None:
            with pytest.raises(ValueError):
                ic.gif_decode(blob, 1, out=o)
