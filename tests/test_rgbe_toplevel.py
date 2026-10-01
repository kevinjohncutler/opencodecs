"""Top-level ``opencodecs.rgbe_*`` helpers and the ``header``/``rle`` options.

The top-level helpers used to be a second, pure-Python RGBE codec that
wrote a non-Radiance ``GAMMA=1.0`` header line and ignored the
resolution line's orientation. They now share the C codec, so:

* their bytes equal imagecodecs ``rgbe_encode`` byte for byte;
* every orientation of the Radiance resolution string decodes to the
  same image. The oriented files are built here by rearranging flat
  RGBE quadruples written by imagecodecs (``header=False``), following
  the Radiance file format's "Resolution String" rules: the first axis
  is the scanline order, ``+Y`` runs bottom to top, ``-X`` right to
  left, and an X-first string stores columns.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc
from opencodecs.codecs import _rgbe
from _ic_reference import skip_if_old_imagecodecs  # noqa: E402

pytestmark = skip_if_old_imagecodecs

H, W = 3, 9


def _image(h=H, w=W, seed=0):
    rng = np.random.default_rng(seed)
    return np.exp(rng.normal(0, 4, (h, w, 3))).astype(np.float32)


def _flat_quads(img):
    """Flat RGBE quadruples (H, W, 4) uint8, written by imagecodecs when
    it is installed, else by the codec under test."""
    try:
        import imagecodecs
        raw = imagecodecs.rgbe_encode(img, header=False)
    except ImportError:
        raw = _rgbe.encode(img, header=False)
    return np.frombuffer(bytes(raw), np.uint8).reshape(img.shape[:2] + (4,))


HEADER = b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\n\n"


def test_top_level_is_the_codec():
    img = _image()
    blob = oc.rgbe_encode(img)
    assert blob == oc.get_codec("rgbe").encode(img)
    assert b"GAMMA" not in blob
    assert blob.startswith(HEADER + b"-Y 3 +X 9\n")
    np.testing.assert_array_equal(oc.rgbe_decode(blob),
                                  oc.get_codec("rgbe").decode(blob))
    from opencodecs._rgbe import RgbeError
    assert RgbeError is _rgbe.RgbeError


@pytest.mark.parametrize("shape", [(3, 9), (1, 1), (5, 7), (4, 40), (2, 300)])
def test_top_level_bytes_match_imagecodecs(shape):
    imagecodecs = pytest.importorskip("imagecodecs")
    img = _image(*shape, seed=sum(shape))
    assert oc.rgbe_encode(img) == bytes(imagecodecs.rgbe_encode(img))


@pytest.mark.parametrize("resolution,arrange", [
    (b"-Y 3 +X 9", lambda q: q),
    (b"+Y 3 +X 9", lambda q: q[::-1]),
    (b"-Y 3 -X 9", lambda q: q[:, ::-1]),
    (b"+Y 3 -X 9", lambda q: q[::-1, ::-1]),
    (b"+X 9 -Y 3", lambda q: q.transpose(1, 0, 2)),
    (b"-X 9 +Y 3", lambda q: q.transpose(1, 0, 2)[::-1, ::-1]),
    (b"+X 9 +Y 3", lambda q: q.transpose(1, 0, 2)[:, ::-1]),
    (b"-X 9 -Y 3", lambda q: q.transpose(1, 0, 2)[::-1]),
])
def test_every_orientation_decodes_to_the_same_image(resolution, arrange):
    img = _image()
    quads = _flat_quads(img)
    want = oc.rgbe_decode(HEADER + b"-Y 3 +X 9\n" + quads.tobytes())
    blob = HEADER + resolution + b"\n" + np.ascontiguousarray(
        arrange(quads)).tobytes()
    np.testing.assert_array_equal(oc.rgbe_decode(blob), want)
    np.testing.assert_array_equal(oc.get_codec("rgbe").decode(blob), want)


def test_headerless_matches_imagecodecs():
    img = _image()
    raw = oc.get_codec("rgbe").encode(img, header=False)
    assert len(raw) == H * W * 4
    out = np.empty_like(img)
    got = oc.get_codec("rgbe").decode(raw, header=False, out=out)
    assert got is out
    with pytest.raises(ValueError, match="out="):
        oc.get_codec("rgbe").decode(raw, header=False)
    imagecodecs = pytest.importorskip("imagecodecs")
    assert raw == bytes(imagecodecs.rgbe_encode(img, header=False))
    np.testing.assert_array_equal(
        got, imagecodecs.rgbe_decode(raw, header=False,
                                     out=np.empty_like(img)))


def test_headerless_rle_round_trips():
    img = _image(4, 40)
    raw = oc.rgbe_encode(img, header=False, rle=True)
    assert raw[:4] == b"\x02\x02\x00\x28"     # an RLE scanline marker
    out = np.empty_like(img)
    np.testing.assert_array_equal(
        oc.rgbe_decode(raw, header=False, out=out),
        oc.rgbe_decode(oc.rgbe_encode(img)))


def test_flat_pixels_after_a_header():
    img = _image(4, 40)
    blob = oc.rgbe_encode(img, rle=False)
    head = HEADER + b"-Y 4 +X 40\n"
    assert blob.startswith(head)
    assert len(blob) == len(head) + 4 * 40 * 4
    np.testing.assert_array_equal(oc.rgbe_decode(blob),
                                  oc.rgbe_decode(oc.rgbe_encode(img)))
    imagecodecs = pytest.importorskip("imagecodecs")
    np.testing.assert_array_equal(imagecodecs.rgbe_decode(blob),
                                  oc.rgbe_decode(blob))


def test_imread_imwrite(tmp_path):
    img = _image()
    p = tmp_path / "x.hdr"
    oc.rgbe_imwrite(p, img)
    assert p.read_bytes() == oc.get_codec("rgbe").encode(img)
    np.testing.assert_array_equal(oc.rgbe_imread(p), oc.rgbe_decode(
        p.read_bytes()))


@pytest.mark.parametrize("head", [
    b"#?RADIANCE\n\n",
    b"#?RGBE\n\n",
    b"#?RADIANCE\n# a comment\nEXPOSURE=2.0\nGAMMA=1.0\n\n",
    b"#?RADIANCE\nFORMAT=32-bit_rle_rgbe\nFORMAT=other\n\n",
])
def test_header_without_format_line_is_read(head):
    """Radiance's reader (checkheader in header.c) takes a picture with
    no FORMAT line as its default format; earlier opencodecs top-level
    helpers read such files, so the shared codec must too. The pixels
    are flat RGBE quadruples written by imagecodecs."""
    img = _image(6, 13)
    quads = _flat_quads(img).tobytes()
    want = oc.rgbe_decode(HEADER + b"-Y 6 +X 13\n" + quads)
    blob = head + b"-Y 6 +X 13\n" + quads
    np.testing.assert_array_equal(oc.rgbe_decode(blob), want)
    np.testing.assert_array_equal(oc.get_codec("rgbe").decode(blob), want)
    imagecodecs = pytest.importorskip("imagecodecs")
    np.testing.assert_array_equal(
        want, imagecodecs.rgbe_decode(HEADER + b"-Y 6 +X 13\n" + quads))


@pytest.mark.parametrize("head", [
    b"#?RADIANCE\nFORMAT=32-bit_rle_cie\n\n",   # another pixel format
    b"#?RADIANCE\nFORMAT=ascii\n\n",
    b"NOTRADIANCE\n\n",                          # no magic, no FORMAT
])
def test_header_naming_no_rgbe_format_is_refused(head):
    img = _image(6, 13)
    blob = head + b"-Y 6 +X 13\n" + _flat_quads(img).tobytes()
    with pytest.raises(_rgbe.RgbeError):
        oc.get_codec("rgbe").decode(blob, header=True)
    with pytest.raises(_rgbe.RgbeError):
        oc.rgbe_decode(blob, header=True)


def _quads_to_float(quads):
    """Bruce Walter's rgbe2float, the reference the vendored reader
    follows: mantissa * 2**(exponent - 136), and zero for exponent 0."""
    q = np.asarray(quads, np.float64)
    scale = np.where(q[..., 3] > 0, np.exp2(q[..., 3] - 136.0), 0.0)
    return (q[..., :3] * scale[..., None]).astype(np.float32)


def _noisy_quads(h, w, seed=3):
    """RGBE quadruples with every channel noisy, the worst case for the
    run-length encoder (no runs, so each channel grows by its count
    bytes)."""
    rng = np.random.default_rng(seed)
    q = rng.integers(0, 256, (h, w, 4), dtype=np.uint8)
    q[..., 0] |= 0x80                     # a normalized mantissa
    q[..., 3] = rng.integers(100, 160, (h, w))
    return q


def _decoders():
    codec = oc.get_codec("rgbe")
    return {"module": _rgbe.decode, "codec": codec.decode,
            "top": oc.rgbe_decode}


@pytest.mark.parametrize("magic", [b"", b"#?RADIANCE\n"])
def test_header_without_magic_is_read_also_with_out(magic):
    """A header with a FORMAT line but no "#?" magic is one Bruce
    Walter's reader accepts. With out= given, header=None must still
    read it as a header; it used to be decoded as a bare pixel stream,
    header text and all."""
    quads = np.zeros((3, 9, 4), np.uint8)
    quads[0, 0] = [128, 236, 157, 129]
    want = _quads_to_float(quads)
    assert want[0, 0].tolist() == [1.0, 1.84375, 1.2265625]
    blob = (magic + b"FORMAT=32-bit_rle_rgbe\n\n-Y 3 +X 9\n"
            + quads.tobytes())
    for name, decode in _decoders().items():
        np.testing.assert_array_equal(decode(blob), want, err_msg=name)
        out = np.empty((3, 9, 3), np.float32)
        got = decode(blob, out=out)
        assert got is out, name
        np.testing.assert_array_equal(got, want, err_msg=name)
    imagecodecs = pytest.importorskip("imagecodecs")
    np.testing.assert_array_equal(
        imagecodecs.rgbe_decode(blob, out=np.empty((3, 9, 3), np.float32)),
        want)


@pytest.mark.parametrize("rle", [False, True])
def test_headerless_is_detected_when_no_header_parses(rle):
    """header=None with out= reads a bare pixel stream when the data
    holds no header, as imagecodecs does."""
    quads = _noisy_quads(4, 40)
    want = _quads_to_float(quads)
    raw = (quads.tobytes() if not rle
           else oc.rgbe_encode(want, header=False, rle=True))
    for name, decode in _decoders().items():
        got = decode(raw, out=np.empty((4, 40, 3), np.float32))
        np.testing.assert_array_equal(got, want, err_msg=name)


@pytest.mark.parametrize("line", [
    b"FORMAT= 32-bit_rle_rgbe",
    b"FORMAT=\t32-bit_rle_rgbe",
    b"FORMAT=32-bit_rle_rgbe  ",
    b"FORMAT=32-bit_rle_rgbe\r",
    b"FORMAT=  32-bit_rle_xyze",
])
def test_format_value_is_read_as_radiance_reads_it(line):
    """Radiance's formatval (header.c) skips whitespace after "FORMAT="
    and ends the value at the next whitespace; the earlier pure-Python
    helpers read these files too."""
    quads = _noisy_quads(6, 13)
    want = _quads_to_float(quads)
    blob = b"#?RADIANCE\n" + line + b"\n\n-Y 6 +X 13\n" + quads.tobytes()
    for name, decode in _decoders().items():
        np.testing.assert_array_equal(decode(blob), want, err_msg=name)


@pytest.mark.parametrize("shape", [(16, 1000), (64, 129), (3, 32767),
                                   (2, 8), (1, 7)])
def test_noisy_images_encode(shape):
    """Noisy scanlines make run-length encoding grow the data by a count
    byte per 128 literal bytes, past 4 bytes per pixel. The output buffer
    used to be sized at 4 bytes per pixel, so these raised RgbeError;
    the earlier pure-Python helpers wrote them."""
    quads = _noisy_quads(*shape)
    img = _quads_to_float(quads)
    # The encoder reproduces the quadruples exactly (flat, no header).
    assert oc.rgbe_encode(img, header=False, rle=False) == quads.tobytes()
    blob = oc.rgbe_encode(img)
    if 8 <= shape[1] <= 32767:
        assert len(blob) > 4 * shape[0] * shape[1]
    np.testing.assert_array_equal(oc.rgbe_decode(blob), img)
    raw = oc.rgbe_encode(img, header=False, rle=True)
    np.testing.assert_array_equal(
        oc.rgbe_decode(raw, header=False,
                       out=np.empty(img.shape, np.float32)), img)
    imagecodecs = pytest.importorskip("imagecodecs")
    np.testing.assert_array_equal(imagecodecs.rgbe_decode(blob), img)


@pytest.mark.parametrize("case", ["junk", "short_out", "text_header"])
def test_bare_stream_must_fill_out_exactly(case):
    """A bare pixel stream read into ``out`` must be used up exactly:
    imagecodecs raises ValueError("not all input decoded") otherwise.
    Header text with neither the "#?" magic nor a FORMAT line is no
    header, and is caught the same way instead of being decoded as
    pixels (values near 1e36 before)."""
    quads = _noisy_quads(3, 9)
    flat = quads.tobytes()
    shape = (3, 9, 3)
    if case == "junk":
        data = flat + b"123456"
    elif case == "short_out":
        data, shape = flat, (2, 9, 3)
    else:
        data = b"SOFTWARE=x\n\n-Y 3 +X 9\n" + flat
    for name, decode in _decoders().items():
        with pytest.raises(ValueError, match="not all input decoded"):
            decode(data, out=np.empty(shape, np.float32))
            pytest.fail(name)
        with pytest.raises(ValueError, match="not all input decoded"):
            decode(data, header=False, out=np.empty(shape, np.float32))
            pytest.fail(name)
    # The exact stream still decodes, to Walter's reference values.
    np.testing.assert_array_equal(
        _rgbe.decode(flat, out=np.empty((3, 9, 3), np.float32)),
        _quads_to_float(quads))
    imagecodecs = pytest.importorskip("imagecodecs")
    with pytest.raises(ValueError, match="not all input decoded"):
        imagecodecs.rgbe_decode(data, out=np.empty(shape, np.float32))
