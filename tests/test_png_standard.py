"""PNG decode layout, tRNS, encode options and byte order against the spec.

Every fixture here is a PNG stream assembled by hand from the
specification (ISO/IEC 15948, W3C PNG): chunks, CRCs and zlib come from
the standard library, never from the encoder under test. Expected pixels
are computed from the spec as well, and imagecodecs (libpng) is a second,
independent reference where it is installed.
"""

from __future__ import annotations

import struct
import zlib

import numpy as np
import pytest

pytest.importorskip("opencodecs.codecs._png")

import opencodecs as oc  # noqa: E402
from opencodecs._png_codec import PngCodec  # noqa: E402
from opencodecs.codecs import _png  # noqa: E402

H, W = 5, 11  # odd width so sub-byte rows end in padding bits


def _chunk(kind: bytes, data: bytes) -> bytes:
    return (struct.pack(">I", len(data)) + kind + data
            + struct.pack(">I", zlib.crc32(kind + data)))


def _pack_rows(samples: np.ndarray, bit_depth: int) -> bytes:
    """Serialize (H, W*C) samples as unfiltered PNG scanlines."""
    raw = bytearray()
    for row in samples:
        raw.append(0)
        if bit_depth == 16:
            raw += row.astype(">u2").tobytes()
        elif bit_depth == 8:
            raw += row.astype("u1").tobytes()
        else:
            bits = ((row[:, None].astype("u1") >> np.arange(bit_depth - 1, -1, -1)) & 1)
            raw += np.packbits(bits.reshape(-1)).tobytes()
    return bytes(raw)


def _build_png(samples, bit_depth, color_type, *, plte=None, trns=None) -> bytes:
    height, width = samples.shape[:2]
    flat = samples.reshape(height, -1)
    out = b"\x89PNG\r\n\x1a\n" + _chunk(
        b"IHDR", struct.pack(">IIBBBBB", width, height, bit_depth, color_type, 0, 0, 0))
    if plte is not None:
        out += _chunk(b"PLTE", np.asarray(plte, "u1").tobytes())
    if trns is not None:
        out += _chunk(b"tRNS", trns)
    return out + _chunk(b"IDAT", zlib.compress(_pack_rows(flat, bit_depth))) + _chunk(b"IEND", b"")


def _decode_both_ways(data):
    """Whole-image decode, and the row decoder reassembled, must agree."""
    codec = PngCodec()
    image = codec.decode(data)
    rows = list(codec.decode_rows(data))
    assert len(rows) == image.shape[0]
    np.testing.assert_array_equal(np.stack([r.pixels for r in rows]), image)
    return image


def _check_reference(data, expected):
    image = _decode_both_ways(data)
    assert image.dtype == expected.dtype
    assert image.shape == expected.shape
    np.testing.assert_array_equal(image, expected)
    try:
        import imagecodecs
    except ImportError:
        return
    reference = imagecodecs.png_decode(data)
    assert reference.shape == expected.shape
    assert reference.dtype == expected.dtype
    np.testing.assert_array_equal(reference, expected)


RNG = np.random.default_rng(15948)


@pytest.mark.parametrize("bit_depth", [1, 2, 4, 8, 16])
def test_gray_stays_gray_and_sub_byte_is_scaled(bit_depth):
    # PNG 13.12: a b-bit sample v scales to v * (2**8 - 1) / (2**b - 1);
    # for 1, 2 and 4 bits that is exact in integers. Gray stays (H, W).
    levels = RNG.integers(0, 1 << bit_depth, (H, W))
    data = _build_png(levels, bit_depth, 0)
    if bit_depth == 16:
        expected = levels.astype(np.uint16)
    elif bit_depth == 8:
        expected = levels.astype(np.uint8)
    else:
        expected = (levels * 255 // ((1 << bit_depth) - 1)).astype(np.uint8)
    _check_reference(data, expected)


@pytest.mark.parametrize("bit_depth", [1, 2, 4, 8, 16])
def test_gray_trns_adds_alpha(bit_depth):
    # PNG 11.3.2.1: tRNS on a gray image names one fully transparent value.
    levels = RNG.integers(0, 1 << bit_depth, (H, W))
    key = int(levels[0, 0])
    data = _build_png(levels, bit_depth, 0, trns=struct.pack(">H", key))
    if bit_depth == 16:
        gray, full, dtype = levels, 65535, np.uint16
    else:
        gray = (levels * 255 // ((1 << bit_depth) - 1))
        full, dtype = 255, np.uint8
    alpha = np.where(levels == key, 0, full)
    expected = np.stack([gray, alpha], -1).astype(dtype)
    _check_reference(data, expected)
    assert (expected[..., 1] == 0).any() and (expected[..., 1] == full).any()


@pytest.mark.parametrize("bit_depth", [1, 2, 4, 8])
def test_palette_without_trns_is_rgb(bit_depth):
    entries = 1 << bit_depth
    palette = RNG.integers(0, 256, (entries, 3)).astype("u1")
    index = RNG.integers(0, entries, (H, W))
    data = _build_png(index, bit_depth, 3, plte=palette)
    _check_reference(data, palette[index])


@pytest.mark.parametrize("bit_depth", [1, 2, 4, 8])
def test_palette_trns_gives_rgba_with_real_alpha(bit_depth):
    # PNG 11.3.2.1: tRNS on an indexed image holds alpha for the first
    # entries; entries past its end are opaque. The decoder used to
    # drop this and return alpha 255 everywhere.
    entries = 1 << bit_depth
    palette = RNG.integers(0, 256, (entries, 3)).astype("u1")
    n_alpha = max(1, entries - 1)
    alpha_table = np.full(entries, 255, "u1")
    alpha_table[:n_alpha] = RNG.integers(0, 255, n_alpha)
    alpha_table[0] = 0
    index = RNG.integers(0, entries, (H, W))
    index[0, :2] = 0, entries - 1
    data = _build_png(index, bit_depth, 3, plte=palette,
                trns=alpha_table[:n_alpha].tobytes())
    expected = np.concatenate([palette[index], alpha_table[index][..., None]], -1)
    _check_reference(data, expected)
    assert len(np.unique(expected[..., 3])) > 1


@pytest.mark.parametrize("bit_depth", [8, 16])
def test_truecolor_trns_adds_alpha(bit_depth):
    top = 1 << bit_depth
    rgb = RNG.integers(0, top, (H, W, 3))
    rgb[1, 2] = rgb[3, 4] = rgb[0, 0]
    key = rgb[0, 0]
    data = _build_png(rgb, bit_depth, 2, trns=struct.pack(">HHH", *map(int, key)))
    alpha = np.where((rgb == key).all(-1), 0, top - 1)
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    expected = np.concatenate([rgb, alpha[..., None]], -1).astype(dtype)
    _check_reference(data, expected)


@pytest.mark.parametrize("bit_depth", [8, 16])
@pytest.mark.parametrize("color_type,channels", [(0, 1), (2, 3), (4, 2), (6, 4)])
def test_plain_color_types_unchanged(bit_depth, color_type, channels):
    shape = (H, W) if channels == 1 else (H, W, channels)
    samples = RNG.integers(0, 1 << bit_depth, shape)
    dtype = np.uint16 if bit_depth == 16 else np.uint8
    _check_reference(_build_png(samples, bit_depth, color_type), samples.astype(dtype))


def test_decode_out_matches_new_layout():
    palette = RNG.integers(0, 256, (4, 3)).astype("u1")
    index = RNG.integers(0, 4, (H, W))
    data = _build_png(index, 2, 3, plte=palette, trns=b"\x00\x80")
    out = np.empty((H, W, 4), np.uint8)
    assert PngCodec().decode(data, out=out) is out
    np.testing.assert_array_equal(out[..., :3], palette[index])
    with pytest.raises(ValueError):
        PngCodec().decode(data, out=np.empty((H, W, 3), np.uint8))


# ---------------------------------------------------------------------------
# Encode options
# ---------------------------------------------------------------------------


def _idat(data: bytes) -> bytes:
    """Concatenated IDAT payloads (one zlib stream), parsed by hand."""
    pos, out = 8, bytearray()
    while pos < len(data):
        length, kind = struct.unpack(">I4s", data[pos:pos + 8])
        if kind == b"IDAT":
            out += data[pos + 8:pos + 8 + length]
        pos += 12 + length
    return bytes(out)


def _filter_types(data: bytes, height: int, row_bytes: int) -> set:
    raw = zlib.decompress(_idat(data))
    assert len(raw) == height * (row_bytes + 1)
    return {raw[r * (row_bytes + 1)] for r in range(height)}


def _gradient():
    rng = np.random.default_rng(0)
    y, x = np.mgrid[:64, :96]
    image = np.stack([x * 2, y * 3, x + y], -1) + rng.integers(0, 8, (64, 96, 3))
    return np.clip(image, 0, 255).astype(np.uint8)


# PNG filter type bytes: 0 None, 1 Sub, 2 Up, 3 Average, 4 Paeth.
@pytest.mark.parametrize("choice,allowed", [
    ("none", {0}), ("off", {0}), ("sub", {1}), ("up", {2}),
    ("avg", {3}), ("paeth", {4}), ("fast", {0, 1, 2}),
    ("all", {0, 1, 2, 3, 4}),
])
def test_codec_forwards_filter_choice(choice, allowed):
    image = _gradient()
    encoded = PngCodec().encode(image, filter_choice=choice)
    # The Codec layer used to drop filter_choice and always write "fast".
    assert encoded == _png.encode(image, filter_choice=choice)
    used = _filter_types(encoded, image.shape[0], image.shape[1] * 3)
    assert used <= allowed
    if len(allowed) == 1:
        assert used == allowed
    np.testing.assert_array_equal(PngCodec().decode(encoded), image)


# imagecodecs PNG.FILTER values are libpng's PNG_FILTER_* bits.
@pytest.mark.parametrize("value,allowed", [
    (0, {0}), (8, {0}), (16, {1}), (32, {2}), (64, {3}), (128, {4}),
    (56, {0, 1, 2}), (248, {0, 1, 2, 3, 4}),
])
def test_imagecodecs_filter_alias(value, allowed):
    image = _gradient()
    encoded = PngCodec().encode(image, filter=value)
    assert _filter_types(encoded, image.shape[0], image.shape[1] * 3) <= allowed
    rows = PngCodec().encode_rows(iter(image), shape=image.shape, dtype=image.dtype,
                                  filter=value)
    assert _filter_types(rows, image.shape[0], image.shape[1] * 3) <= allowed
    np.testing.assert_array_equal(PngCodec().decode(encoded), image)


def test_imagecodecs_filter_enum_values_match():
    imagecodecs = pytest.importorskip("imagecodecs")
    image = _gradient()
    for name in ("NO", "NONE", "SUB", "UP", "AVG", "PAETH", "FAST", "ALL"):
        value = getattr(imagecodecs.PNG.FILTER, name)
        assert PngCodec().encode(image, filter=value) == PngCodec().encode(image, filter=int(value))


@pytest.mark.parametrize("name", ["no", "NO", "none", "sub", "Up", "avg",
                                  "paeth", "fast", "all"])
def test_imagecodecs_filter_names(name):
    # imagecodecs takes PNG.FILTER member names as strings too; "no"
    # (PNG.FILTER.NO, every row unfiltered) was missing here.
    imagecodecs = pytest.importorskip("imagecodecs")
    image = _gradient()
    ours = PngCodec().encode(image, filter=name)
    theirs = imagecodecs.png_encode(image, filter=name)
    value = int(getattr(imagecodecs.PNG.FILTER, name.upper()))
    assert ours == PngCodec().encode(image, filter=value)
    rows = image.shape[1] * 3
    if bin(value).count("1") <= 1:
        # A single allowed filter (or none) fixes every row's type.
        assert _filter_types(ours, 64, rows) == _filter_types(theirs, 64, rows)
    else:
        assert _filter_types(ours, 64, rows) <= {0, 1, 2, 3, 4}


@pytest.mark.parametrize("value", [1, 2, 3, 4, 5, 7, "best"])
def test_filter_values_imagecodecs_refuses_are_refused(value):
    # imagecodecs 2026.8.16 validates filter= against PNG.FILTER, so the
    # libpng filter type numbers 1-4 are not filter settings there either.
    image = _gradient()
    with pytest.raises(ValueError) as caught:
        PngCodec().encode(image, filter=value)
    assert isinstance(caught.value, _png.PngError)
    imagecodecs = pytest.importorskip("imagecodecs")
    with pytest.raises((ValueError, KeyError)):
        imagecodecs.png_encode(image, filter=value)


def test_strategy_names_and_range_follow_imagecodecs():
    imagecodecs = pytest.importorskip("imagecodecs")
    image = np.zeros((256, 256), np.uint8)
    for name, value in (("default", 0), ("FILTERED", 1), ("huffman_only", 2),
                        ("rle", 3), ("Fixed", 4)):
        assert getattr(imagecodecs.PNG.STRATEGY, name.upper()) == value
        ours = PngCodec().encode(image, strategy=name)
        assert ours == PngCodec().encode(image, strategy=value)
        assert ours == PngCodec().encode(image, strategy=np.int64(value))
        np.testing.assert_array_equal(imagecodecs.png_decode(ours), image)
    assert _first_block_type(PngCodec().encode(image, strategy="fixed")) == 1
    for bad in (-1, 5, "lzw"):
        with pytest.raises((ValueError, KeyError)):
            imagecodecs.png_encode(image, strategy=bad)
        with pytest.raises(ValueError) as caught:
            PngCodec().encode(image, strategy=bad)
        assert isinstance(caught.value, _png.PngError)
        with pytest.raises(ValueError):
            PngCodec().encode_rows(iter(image), shape=image.shape,
                                   dtype=image.dtype, strategy=bad)


def test_numpy_integer_level():
    image = _gradient()
    assert PngCodec().encode(image, level=np.int64(9)) == PngCodec().encode(image, level=9)


def test_decode_rejects_unknown_options():
    # imagecodecs.png_decode defines only out=; anything else was dropped.
    encoded = PngCodec().encode(_gradient())
    with pytest.raises(TypeError):
        PngCodec().decode(encoded, index=1)
    np.testing.assert_array_equal(PngCodec().decode(encoded, numthreads=4),
                                  _gradient())


def test_filter_option_errors():
    image = _gradient()
    with pytest.raises(TypeError, match="not both"):
        PngCodec().encode(image, filter=16, filter_choice="sub")
    with pytest.raises(ValueError, match="unknown filter_choice"):
        PngCodec().encode(image, filter_choice="best")
    # Earlier releases raised PngError here; both still catch it.
    with pytest.raises(_png.PngError, match="unknown filter_choice"):
        PngCodec().encode(image, filter_choice="best")
    with pytest.raises(ValueError, match="bitmask"):
        PngCodec().encode(image, filter=3)
    with pytest.raises(TypeError, match="unexpected option"):
        PngCodec().encode(image, filters=16)


def _first_block_type(data: bytes) -> int:
    stream = _idat(data)
    # RFC 1950: two header bytes; RFC 1951 3.2.3: BFINAL is bit 0 and
    # BTYPE bits 1-2 of the first deflate byte (01 fixed, 10 dynamic).
    return (stream[2] >> 1) & 3


@pytest.mark.parametrize("rows", [False, True])
def test_strategy_is_honored(rows):
    # The build compresses through libdeflate, which has no strategy
    # setting, so strategy= did nothing. An explicit strategy now goes
    # through zlib with that strategy.
    image = np.zeros((512, 512), np.uint8)

    def encode(**kw):
        if rows:
            return PngCodec().encode_rows(iter(image), shape=image.shape,
                                          dtype=image.dtype, **kw)
        return PngCodec().encode(image, **kw)

    default = encode()
    fixed = encode(strategy=4)          # Z_FIXED
    huffman = encode(strategy=2)        # Z_HUFFMAN_ONLY
    assert _first_block_type(fixed) == 1
    # Without LZ77 every filtered byte costs at least one bit:
    # 512 rows * 513 bytes / 8.
    assert len(huffman) >= 512 * 513 // 8
    assert len(default) < 4096
    for encoded in (default, fixed, huffman, encode(strategy=3), encode(strategy=0)):
        np.testing.assert_array_equal(PngCodec().decode(encoded), image)


def test_strategy_matches_imagecodecs_sizes():
    imagecodecs = pytest.importorskip("imagecodecs")
    image = np.zeros((256, 256), np.uint8)
    # Z_HUFFMAN_ONLY and Z_FIXED are fully determined by zlib for an
    # all-zero image with no filtering choice to make.
    for strategy in (2, 4):
        ours = PngCodec().encode(image, strategy=strategy, filter_choice="off")
        theirs = imagecodecs.png_encode(image, strategy=strategy, filter=0)
        np.testing.assert_array_equal(imagecodecs.png_decode(ours), image)
        assert abs(len(ours) - len(theirs)) <= 16


def _stored_samples(encoded, shape):
    """Read the samples back out of an unfiltered 16-bit stream."""
    raw = zlib.decompress(_idat(encoded))
    height = shape[0]
    rows = np.frombuffer(raw, "u1").reshape(height, -1)
    assert (rows[:, 0] == 0).all()
    return rows[:, 1:].copy().view(">u2").reshape(shape)


@pytest.mark.parametrize("shape", [(4, 5), (4, 5, 3)])
def test_big_endian_uint16_stores_values(shape):
    # PNG 7.1: samples are stored most significant byte first, so the
    # numeric value 1 must be the bytes 00 01 whatever the array's byte
    # order. Reinterpreting the bytes of a '>u2' array would store 256.
    values = np.arange(np.prod(shape), dtype="=u2").reshape(shape) * 257 + 1
    big = values.astype(">u2")
    encoded = PngCodec().encode(big, filter_choice="off")
    np.testing.assert_array_equal(_stored_samples(encoded, shape), values)
    np.testing.assert_array_equal(PngCodec().decode(encoded), values)
    assert encoded == PngCodec().encode(values, filter_choice="off")
    rows = PngCodec().encode_rows(iter(big), shape=shape, dtype=big.dtype,
                                  filter_choice="off")
    assert rows == encoded


def test_numthreads_is_accepted_on_encode():
    """oc.write(..., numthreads=N) worked for PNG in 0.4.0 and must keep
    working: the package's encoders take numthreads, and PNG has no
    threads to give it to, so the bytes do not depend on it."""
    imagecodecs = pytest.importorskip("imagecodecs")
    image = _gradient()
    expected = PngCodec().encode(image)
    assert PngCodec().encode(image, numthreads=4) == expected
    assert oc.write(None, image, format="png", numthreads=2) == expected
    np.testing.assert_array_equal(imagecodecs.png_decode(expected), image)


def test_encode_out_none_is_accepted_and_a_buffer_is_refused():
    """0.4.0 accepted out= (imagecodecs defines it) and dropped it; out=None
    must still work, and a real buffer is refused rather than left unwritten."""
    imagecodecs = pytest.importorskip("imagecodecs")
    image = _gradient()
    encoded = PngCodec().encode(image, out=None)
    assert encoded == PngCodec().encode(image)
    np.testing.assert_array_equal(imagecodecs.png_decode(encoded), image)
    for out in (bytearray(1 << 16), 1 << 16):
        with pytest.raises(TypeError, match="out="):
            PngCodec().encode(image, out=out)


def test_encode_rows_accepts_numthreads_like_encode():
    """PngCodec.encode took numthreads and encode_rows raised TypeError."""
    imagecodecs = pytest.importorskip("imagecodecs")
    image = _gradient()
    expected = PngCodec().encode_rows(iter(image), shape=image.shape,
                                      dtype=image.dtype)
    got = PngCodec().encode_rows(iter(image), shape=image.shape,
                                 dtype=image.dtype, numthreads=4)
    assert got == expected
    np.testing.assert_array_equal(imagecodecs.png_decode(got), image)
    with pytest.raises(TypeError):
        PngCodec().encode_rows(iter(image), shape=image.shape,
                               dtype=image.dtype, quality=3)
