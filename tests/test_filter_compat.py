"""Filter codecs against the definitions they implement, and imagecodecs.

Each fix in the 0.5.0 compatibility round gets two kinds of pin here: a
reference built by hand from the published definition (TIFF predictor 2
on unsigned words, the TIFF row bitstream, GenICam and GigE Vision
packed pixels, the HDF5 shuffle filter, netCDF-C's quantize modes), and,
where imagecodecs is installed, byte equality with it. A round trip
through our own code proves neither.
"""

from __future__ import annotations

import math
import numpy as np
import pytest

import opencodecs as oc


def _imagecodecs():
    return pytest.importorskip("imagecodecs")


# ---------------------------------------------------------------------------
# delta / xor: bit patterns, byte order, ndarray and bytes input
# ---------------------------------------------------------------------------

FLOATS = ["<f2", ">f2", "<f4", ">f4", "<f8", ">f8"]
INTS = ["u1", "i1", "<u2", ">u2", "<i2", ">i2", "<u4", ">i4", "<u8", ">i8"]


def _bits(a):
    """``a``'s samples as native unsigned integers of the same width."""
    return a.astype(a.dtype.newbyteorder("=")).view(f"u{a.dtype.itemsize}")


def _reference(a, axis, op):
    """Predictor 2 (or XOR) on the bit patterns, by hand, in ``a``'s dtype."""
    bits = _bits(a)
    out = bits.copy()
    moved_in = np.moveaxis(bits, axis, 0)
    moved_out = np.moveaxis(out, axis, 0)
    for i in range(1, moved_in.shape[0]):
        moved_out[i] = op(moved_in[i], moved_in[i - 1])
    return out.view(a.dtype.newbyteorder("=")).astype(a.dtype).tobytes()


def _sample(dtype, shape, seed=0):
    rng = np.random.default_rng(seed)
    dtype = np.dtype(dtype)
    if dtype.kind == "f":
        a = (rng.standard_normal(shape) * 1e3).astype(dtype)
        a.flat[0] = 1.0
        a.flat[-1] = -0.0
        return a
    info = np.iinfo(dtype)
    return rng.integers(info.min, info.max, shape, dtype=dtype.newbyteorder("="),
                        endpoint=True).astype(dtype)


@pytest.mark.parametrize("name,op", [("delta", np.subtract), ("xor", np.bitwise_xor)])
@pytest.mark.parametrize("dtype", FLOATS + INTS)
@pytest.mark.parametrize("axis", [-1, 0])
def test_predictor_is_defined_on_bit_patterns(name, op, dtype, axis):
    """libtiff's predictor 2 differences BitsPerSample-wide unsigned words
    whatever the sample format; float delta used to subtract float values,
    which rounds, and float XOR raised TypeError."""
    a = _sample(dtype, (5, 9))
    codec = oc.get_codec(name)
    enc = codec.encode(a, axis=axis)
    assert enc == _reference(a, axis, op)
    back = codec.decode(enc, dtype=a.dtype, shape=a.shape, axis=axis)
    assert back.dtype == a.dtype                    # byte order kept
    assert back.tobytes() == a.tobytes()            # bit exact


def test_float_delta_is_lossless_where_float_arithmetic_is_not():
    """The audit's case: float cumsum of float differences turned the
    third value, 1.0, into 0.0."""
    a = np.array([1.0, 1e8, 1.0, 3.3, 1e-8], np.float32)
    codec = oc.get_codec("delta")
    back = codec.decode(codec.encode(a), dtype=np.float32)
    assert back.tobytes() == a.tobytes()


# Float delta streams written by opencodecs 0.4.0 (differences of float
# values), with what 0.4.0's own decoder returned for them, as
# little-endian bytes: (dtype, shape, axis, dist, stream, decoded).
_DELTA_040 = [
    ("<f4", (2, 5), -1, 1,
     "0000803f20bcbe4c20bcbecc33331340333353c0cdcccc3d666626c000001841cff7dfc0"
     "00e07f47",
     "0000803f20bcbe4c0000000033331340000080bfcdcccc3d000020c00000e0400010833a"
     "00e07f47"),
    (">f8", (2, 5), 0, 1,
     "3ff00000000000004197d784000000003ff0000000000000400a6666600000003e45798e"
     "e0000000bfeccccccc000000c197d7840a0000004018000000000000c00a645a16440000"
     "40effbfffffffaa2",
     "000000000000f03f0000000084d79741000000000000f03f0000006066660a40000000e0"
     "8e79453e000000a09999b93f00000000000004c00000000000001c40000000e04d62503f"
     "0000000000fcef40"),
    ("<f2", (2, 5), -1, 2,
     "003c007c000000fc00bc662e00c1e6460141ff7b",
     "003c007c003c007e0000662e00c100470018ff7b"),
]


@pytest.mark.parametrize("dtype,shape,axis,dist,stream,want", _DELTA_040)
def test_float_delta_reads_streams_from_0_4_0(dtype, shape, axis, dist, stream, want):
    """0.4.0 stored float differences, which the bit-pattern decoder reads
    as garbage with no error (the stream has no header). legacy_float=True
    returns the values 0.4.0's decoder returned, bit for bit, in the
    requested byte order (0.4.0 returned native order)."""
    codec = oc.get_codec("delta")
    want = np.frombuffer(bytes.fromhex(want), np.dtype(dtype).newbyteorder("<"))
    want = want.reshape(shape)
    raw = bytes.fromhex(stream)
    kw = dict(dtype=dtype, shape=shape, axis=axis, dist=dist)
    for out in (None, np.empty(shape, dtype)):
        got = codec.decode(raw, legacy_float=True, out=out, **kw)
        assert got.dtype == np.dtype(dtype) and got.shape == shape
        assert _same_bits_any_nan(got.astype(want.dtype), want)
    # The same rule by hand: a running float sum of each lane.
    diffs = np.frombuffer(raw, dtype).reshape(shape).astype(want.dtype)
    lanes = np.moveaxis(diffs, axis, -1).copy()
    with np.errstate(invalid="ignore", over="ignore"):
        for k in range(dist, lanes.shape[-1]):
            lanes[..., k] = lanes[..., k] + lanes[..., k - dist]
    assert _same_bits_any_nan(np.moveaxis(lanes, -1, axis), want)


def test_float_delta_legacy_flag_is_delta_and_float_only():
    codec = oc.get_codec("delta")
    with pytest.raises(ValueError, match="float"):
        codec.decode(bytes(8), dtype="u2", legacy_float=True)
    with pytest.raises(TypeError, match="legacy_float"):
        oc.get_codec("xor").decode(bytes(8), dtype="f4", legacy_float=True)
    a = np.array([1.5, 2.25, -3.0], np.float32)
    assert codec.decode(codec.encode(a), dtype="f4", legacy_float=False).tobytes() == a.tobytes()


@pytest.mark.parametrize("name", ["delta", "xor"])
@pytest.mark.parametrize("dtype", FLOATS + INTS)
@pytest.mark.parametrize("shape,axis", [((13,), -1), ((7, 11), 0), ((3, 5, 7), 1)])
def test_predictor_matches_imagecodecs(name, dtype, shape, axis):
    ic = _imagecodecs()
    a = _sample(dtype, shape, seed=len(shape))
    ref = getattr(ic, f"{name}_encode")(a, axis=axis)
    codec = oc.get_codec(name)
    assert codec.encode(a, axis=axis) == ref.tobytes()
    # imagecodecs' ndarray output carries dtype and shape; bytes need them.
    for got in (codec.decode(ref, axis=axis),
                codec.decode(ref.tobytes(), dtype=a.dtype, shape=a.shape, axis=axis),
                codec.decode(ref.tobytes(), dtype=a.dtype, shape=a.shape, axis=axis,
                             out=np.empty_like(a))):
        assert got.dtype == a.dtype and got.shape == a.shape
        assert got.tobytes() == a.tobytes()
    ours = np.frombuffer(codec.encode(a, axis=axis), a.dtype).reshape(a.shape)
    assert getattr(ic, f"{name}_decode")(ours, axis=axis).tobytes() == a.tobytes()


@pytest.mark.parametrize("name", ["delta", "xor"])
@pytest.mark.parametrize("dtype", [">u2", ">i2", ">i4", ">u8", ">f4"])
def test_big_endian_decode(name, dtype):
    """Decode without out= handed big-endian samples to the native kernel
    and raised "Big-endian buffer not supported on little-endian compiler"."""
    a = _sample(dtype, (4, 6))
    codec = oc.get_codec(name)
    enc = codec.encode(a)
    for out in (None, np.empty_like(a)):
        got = codec.decode(enc, dtype=dtype, shape=a.shape, out=out)
        assert got.dtype == np.dtype(dtype)
        np.testing.assert_array_equal(got, a)


@pytest.mark.parametrize("name", ["delta", "xor"])
def test_ndarray_source_gives_dtype_and_shape(name):
    """An encoded ndarray used to decode as flat uint8."""
    a = (np.arange(40, dtype="u2") * 1000).reshape(5, 8)
    codec = oc.get_codec(name)
    stored = np.frombuffer(codec.encode(a), a.dtype).reshape(a.shape)
    got = codec.decode(stored)
    assert got.dtype == a.dtype and got.shape == a.shape
    np.testing.assert_array_equal(got, a)


@pytest.mark.parametrize("name,op", [("delta", np.subtract), ("xor", np.bitwise_xor)])
def test_bytes_input_is_uint8(name, op):
    """Encode of bytes made a one-element 'S' array and raised."""
    raw = bytes(range(0, 250, 7))
    codec = oc.get_codec(name)
    enc = codec.encode(raw)
    assert enc == _reference(np.frombuffer(raw, np.uint8), -1, op)
    assert codec.decode(enc).tobytes() == raw


@pytest.mark.parametrize("name", ["delta", "xor"])
def test_predictor_rejects_what_it_does_not_implement(name):
    codec = oc.get_codec(name)
    a = np.arange(8, dtype=np.uint8)
    with pytest.raises(TypeError, match="bogus"):
        codec.encode(a, bogus=1)
    with pytest.raises(TypeError, match="bogus"):
        codec.decode(a.tobytes(), bogus=1)
    with pytest.raises(ValueError):
        codec.encode(a, dist=0)
    with pytest.raises(ValueError):
        codec.decode(a.tobytes(), dist=0)
    with pytest.raises(ValueError):
        codec.encode(np.zeros(4, np.complex64))


@pytest.mark.parametrize("name", ["delta", "xor"])
def test_predictor_zero_dimensional_raises_value_error(name):
    """A 0-d array has no axis; this raised a bare IndexError from the
    slicing. imagecodecs raises ValueError("invalid axis")."""
    codec = oc.get_codec(name)
    a = np.array(3.0, np.float32)
    with pytest.raises(ValueError, match="axis"):
        codec.encode(a)
    with pytest.raises(ValueError, match="axis"):
        codec.decode(a.tobytes(), dtype=np.float32, shape=())
    with pytest.raises(ValueError, match="axis"):
        codec.encode(np.zeros((2, 3), np.uint8), axis=2)
    ic = pytest.importorskip("imagecodecs")
    with pytest.raises(ValueError):
        getattr(ic, f"{name}_encode")(a)


@pytest.mark.parametrize("name", ["delta", "xor"])
def test_predictor_bool_raises_value_error(name):
    """0.4.0 xor-encoded and xor-decoded bool as bytes, and delta decode
    summed bool as logical or; imagecodecs raises ValueError for a bool
    array in all four functions, and so does this."""
    codec = oc.get_codec(name)
    a = np.array([True, False, True])
    with pytest.raises(ValueError, match="bool"):
        codec.encode(a)
    with pytest.raises(ValueError, match="bool"):
        codec.decode(bytes([1, 1, 0]), dtype=bool)
    with pytest.raises(ValueError, match="bool"):
        codec.decode(a)
    ic = pytest.importorskip("imagecodecs")
    for func in ("encode", "decode"):
        with pytest.raises(ValueError):
            getattr(ic, f"{name}_{func}")(a)


# ---------------------------------------------------------------------------
# packints
# ---------------------------------------------------------------------------


def _msb(values, width):
    """TIFF FillOrder 1: the first sample in the high bits of byte 0."""
    bits = "".join(format(int(v), f"0{width}b") for v in values)
    bits += "0" * (-len(bits) % 8)
    return bytes(int(bits[i:i + 8], 2) for i in range(0, len(bits), 8))


def _lsb(values, width):
    """GenICam PFNC lsb-first: sample k occupies bits k*width.. of the
    little-endian bit string."""
    acc = sum(int(v) << (k * width) for k, v in enumerate(values))
    return acc.to_bytes((len(values) * width + 7) // 8, "little")


def _gige(values, width):
    """GigE Vision Mono10Packed / Mono12Packed, from the spec's byte table."""
    low = width - 8
    out = bytearray()
    for p0, p1 in zip(values[0::2], values[1::2]):
        p0, p1 = int(p0), int(p1)
        lo = (1 << low) - 1
        out += bytes([p0 >> low, ((p1 & lo) << 4) | (p0 & lo), p1 >> low])
    return bytes(out)


@pytest.mark.parametrize("width,dtype,values,want", [
    (16, "u2", [0x0102, 0x0A0B], "01020a0b"),
    (32, "u4", [0x01020304, 0x0A0B0C0D], "010203040a0b0c0d"),
    (64, "u8", [0x0102030405060708], "0102030405060708"),
])
def test_packints_whole_byte_widths_are_msb_first(width, dtype, values, want):
    """An MSB-first bitstream of 16/32/64-bit samples is big-endian bytes on
    every machine, as 24-bit and other widths already were. imagecodecs
    copies these widths in memory order instead; that difference is kept
    on purpose and documented."""
    p = oc.get_codec("packints")
    a = np.array(values, dtype)
    assert p.encode(a, bitspersample=width).hex() == want
    assert p.encode(a, bitspersample=width) == _msb(values, width)
    np.testing.assert_array_equal(
        p.decode(bytes.fromhex(want), dtype=dtype, bitspersample=width), a)


@pytest.mark.parametrize("width", [1, 2, 3, 5, 7, 9, 12, 15, 24, 31, 33])
@pytest.mark.parametrize("rows,runlen", [(3, 5), (4, 7), (2, 1), (1, 13), (3, 8)])
def test_packints_runlen_pads_each_row(width, rows, runlen):
    """TIFF 6.0: "leaving no unused bits except at the end of a row". The
    runlen keyword was swallowed and a continuous stream came back."""
    rng = np.random.default_rng(width * 100 + runlen)
    dtype = "u1" if width <= 8 else "u2" if width <= 16 else "u4" if width <= 32 else "u8"
    a = rng.integers(0, 1 << width, (rows, runlen), dtype="u8").astype(dtype)
    want = b"".join(_msb(row, width) for row in a)
    p = oc.get_codec("packints")
    assert p.encode(a, bitspersample=width, runlen=runlen) == want
    got = p.decode(want, dtype=dtype, bitspersample=width, runlen=runlen)
    np.testing.assert_array_equal(got, a.ravel())
    out = np.empty(a.shape, dtype)
    assert p.decode(want, dtype=dtype, bitspersample=width, runlen=runlen,
                    shape=a.shape, out=out) is out
    np.testing.assert_array_equal(out, a)


@pytest.mark.parametrize("width", [1, 4, 9, 10, 11, 12, 13, 14, 16, 20])
@pytest.mark.parametrize("n", [1, 2, 7, 64])
def test_packints_lsb_first(width, n):
    rng = np.random.default_rng(width + n)
    dtype = "u1" if width <= 8 else "u2" if width <= 16 else "u4"
    a = rng.integers(0, 1 << width, n, dtype="u4").astype(dtype)
    p = oc.get_codec("packints")
    want = _lsb(a, width)
    assert p.encode(a, bitspersample=width, bitorder="<") == want
    np.testing.assert_array_equal(
        p.decode(want, dtype=dtype, bitspersample=width, bitorder="<", n_elements=n), a)


@pytest.mark.parametrize("width", [10, 12])
def test_packints_gige_paired(width):
    rng = np.random.default_rng(width)
    a = rng.integers(0, 1 << width, 64, dtype="u2")
    p = oc.get_codec("packints")
    want = _gige(a, width)
    assert p.encode(a, bitspersample=width, bitorder=">") == want
    np.testing.assert_array_equal(
        p.decode(want, dtype="u2", bitspersample=width, bitorder=">"), a)


def test_packints_camera_layouts_by_hand():
    """Pixels (0x000, 0x12C), (0x258, 0x384) worked through each spec's byte
    table: PFNC Mono12p B1 = p1[3:0] << 4 | p0[11:8]; GigE Mono12Packed
    B0 = p0[11:4], B1 = p1[3:0] << 4 | p0[3:0]."""
    a = np.array([0x000, 0x12C, 0x258, 0x384], "u2")
    p = oc.get_codec("packints")
    assert p.encode(a, bitspersample=12, bitorder="<").hex() == "00c012584238"
    assert p.encode(a, bitspersample=12, bitorder=">").hex() == "00c012254838"
    assert p.encode(a, bitspersample=12).hex() == "00012c258384"


@pytest.mark.parametrize("width", list(range(1, 33)))
def test_packints_matches_imagecodecs(width):
    ic = _imagecodecs()
    rng = np.random.default_rng(width)
    dtype = "u1" if width <= 8 else "u2" if width <= 16 else "u4"
    p = oc.get_codec("packints")
    cases = [dict(), dict(runlen=5), dict(runlen=7)]
    if 9 <= width <= 14:
        cases.append(dict(bitorder="<"))
    if width in (10, 12):
        cases.append(dict(bitorder=">"))
    for kw in cases:
        a = rng.integers(0, 1 << width, 70, dtype="u4").astype(dtype)
        ref = ic.packints_encode(a, width, **kw)
        if width % 8 == 0 and width != 24 and "bitorder" not in kw:
            # imagecodecs copies whole-byte widths in memory order.
            assert p.encode(a, bitspersample=width, **kw) == \
                a.astype(f">u{width // 8}").tobytes()
            continue
        assert p.encode(a, bitspersample=width, **kw) == ref
        np.testing.assert_array_equal(
            p.decode(ref, dtype=dtype, bitspersample=width, **kw),
            ic.packints_decode(ref, dtype, width, **kw))


def test_packints_rejects_what_it_does_not_implement():
    p = oc.get_codec("packints")
    a = np.zeros(6, "u2")
    with pytest.raises(TypeError, match="bogus"):
        p.encode(a, bitspersample=12, bogus=1)
    with pytest.raises(TypeError, match="bogus"):
        p.decode(bytes(9), dtype="u2", bitspersample=12, bogus=1)
    with pytest.raises(ValueError, match="bitorder"):
        p.encode(a, bitspersample=12, bitorder="little")
    with pytest.raises(ValueError, match="runlen"):
        p.encode(a, bitspersample=12, bitorder="<", runlen=3)
    with pytest.raises(ValueError, match="10 or 12"):
        p.encode(a, bitspersample=11, bitorder=">")
    with pytest.raises(ValueError, match="even"):
        p.encode(a[:5], bitspersample=12, bitorder=">")
    # imagecodecs drops a trailing partial run; that loses samples.
    with pytest.raises(ValueError, match="runs"):
        p.encode(np.zeros(7, "u2"), bitspersample=12, runlen=3)
    # Decode raises for it too, whole-byte rows (5 * 16 bits) included.
    with pytest.raises(ValueError, match="runs"):
        p.decode(bytes(14), dtype="u2", bitspersample=16, runlen=5, n_elements=7)
    with pytest.raises(ValueError, match="runs"):
        p.decode(bytes(14), dtype="u2", bitspersample=12, runlen=3, n_elements=7)


@pytest.mark.parametrize("width,dtype,bad", [
    (12, "u2", 4096), (12, "u2", 0xFFFF), (1, "u1", 2), (8, "u2", 256),
    (16, "u4", 1 << 16), (33, "u8", 1 << 33)])
@pytest.mark.parametrize("bitorder", [None, "<", ">"])
def test_packints_rejects_samples_wider_than_the_width(width, dtype, bad, bitorder):
    """imagecodecs, and this codec before, kept only the low bits of an
    out-of-range sample: 4096 at 12 bits packed to 0x000."""
    if bitorder == ">" and width not in (10, 12):
        pytest.skip("paired layout is 10 or 12 bits")
    p = oc.get_codec("packints")
    a = np.array([1, bad], dtype)
    with pytest.raises(ValueError, match="does not fit"):
        p.encode(a, bitspersample=width, bitorder=bitorder)
    top = np.array([1, (1 << width) - 1], dtype)    # the largest that fits
    assert p.encode(top, bitspersample=width, bitorder=bitorder) is not None


def _unpack_by_bits(raw, width, n, lsb_first):
    """Samples from a bit string expanded one byte per bit (small inputs)."""
    bits = np.unpackbits(np.frombuffer(raw, np.uint8),
                         bitorder="little" if lsb_first else "big")
    bits = bits[:n * width].reshape(n, width).astype(object)
    order = range(width) if lsb_first else range(width - 1, -1, -1)
    weights = np.array([1 << k for k in order], dtype=object)
    return (bits * weights).sum(axis=1)


@pytest.mark.parametrize("width", [3, 12, 31, 57, 58, 63, 64])
@pytest.mark.parametrize("lsb_first", [True, False])
def test_packints_decode_across_blocks(width, lsb_first):
    """Decode works in blocks of samples now (the LSB-first path built a
    samples-by-bits matrix of the whole stream: 1.9 GB for one 4096 x 4096
    12-bit frame). Pinned across block boundaries, at the 57-bit edge of
    the 64-bit window, and on the NumPy path the MSB-first stream takes
    when the output is not contiguous."""
    from opencodecs import _packints_codec as pc
    n = 2 * pc._PACK_BLOCK + 37
    rng = np.random.default_rng(width)
    dtype = "u1" if width <= 8 else "u2" if width <= 16 else "u4" if width <= 32 else "u8"
    a = rng.integers(0, 1 << width, n, dtype="u8", endpoint=False).astype(dtype)
    if width == 64:
        a = rng.integers(0, 1 << 63, n, dtype="u8") * np.uint64(2) + np.uint64(1)
    a[:3] = [0, (1 << width) - 1, 1 << (width - 1)]
    p = oc.get_codec("packints")
    kw = dict(bitorder="<") if lsb_first else {}
    raw = p.encode(a, bitspersample=width, **kw)
    head = raw[:(1000 * width + 7) // 8]
    assert list(_unpack_by_bits(head, width, 1000, lsb_first)) == [int(v) for v in a[:1000]]
    got = p.decode(raw, dtype=dtype, bitspersample=width, n_elements=n, **kw)
    assert np.array_equal(got, a)
    got = pc._unpack_stream(raw, np.dtype(dtype), width, n, lsb_first)
    assert np.array_equal(got, a)
    if not lsb_first:
        out = np.zeros((n, 2), dtype)[:, 0]          # strided: NumPy path
        assert p.decode(raw, dtype=dtype, bitspersample=width, n_elements=n,
                        out=out) is out
        assert np.array_equal(out, a)


def test_packints_lsb_decode_memory_is_bounded():
    import tracemalloc
    n = 1 << 22                                      # a 2048 x 2048 frame
    a = np.random.default_rng(0).integers(0, 4096, n, dtype="u2")
    p = oc.get_codec("packints")
    raw = p.encode(a, bitspersample=12, bitorder="<")
    tracemalloc.start()
    try:
        got = p.decode(raw, dtype="u2", bitspersample=12, bitorder="<")
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert np.array_equal(got, a)
    assert peak < a.nbytes + (24 << 20), peak        # was about 60x the output


@pytest.mark.parametrize("dtype,width", [("f4", 12), ("f2", 12), ("f8", 4),
                                         ("u1", 12), ("u2", 17), ("i4", 33),
                                         ("?", 2)])
def test_packints_decode_rejects_dtypes_that_cannot_hold_the_samples(dtype, width):
    """A float dtype took the integer values and a narrow integer one kept
    their low bits, without a word; imagecodecs raises for these."""
    p = oc.get_codec("packints")
    with pytest.raises(ValueError, match="integer type"):
        p.decode(bytes(16), dtype=dtype, bitspersample=width, n_elements=2)


def test_packints_decode_takes_bool_for_one_bit():
    """tifffile asks for bool bilevel samples, as imagecodecs allows."""
    p = oc.get_codec("packints")
    got = p.decode(b"\xa0", dtype="?", bitspersample=1)
    np.testing.assert_array_equal(got, [1, 0, 1, 0, 0, 0, 0, 0])
    assert got.dtype == np.bool_
    np.testing.assert_array_equal(
        p.decode(b"\x0a\x0b", dtype="i2", bitspersample=12, n_elements=1), [0x0A0])


# ---------------------------------------------------------------------------
# bitshuffle / byteshuffle
# ---------------------------------------------------------------------------


def test_bitshuffle_ndarray_source_gives_itemsize():
    """Decode of an ndarray defaulted to itemsize=1 and returned wrong bytes."""
    a = (np.arange(48, dtype="i4") * 12345).reshape(6, 8)
    codec = oc.get_codec("bitshuffle")
    stored = np.frombuffer(codec.encode(a), a.dtype).reshape(a.shape)
    got = codec.decode(stored)
    assert isinstance(got, np.ndarray)
    assert got.dtype == a.dtype and got.shape == a.shape
    np.testing.assert_array_equal(got, a)
    assert bytes(codec.decode(stored.tobytes(), itemsize=4)) == a.tobytes()
    with pytest.raises(TypeError, match="bogus"):
        codec.decode(stored, bogus=1)


def test_bitshuffle_matches_imagecodecs():
    ic = _imagecodecs()
    a = (np.arange(48, dtype="i4") * 12345).reshape(6, 8)
    ref = ic.bitshuffle_encode(a)
    codec = oc.get_codec("bitshuffle")
    assert codec.encode(a) == ref.tobytes()
    np.testing.assert_array_equal(codec.decode(ref), a)


def _hdf5_shuffle(a):
    """H5Z_FILTER_SHUFFLE: byte j of every element, for j = 0, 1, ..."""
    return a.view(np.uint8).reshape(a.size, a.dtype.itemsize).T.tobytes()


@pytest.mark.parametrize("shape", [(37,), (5, 7), (3, 4, 5)])
def test_byteshuffle_is_the_hdf5_shuffle(shape):
    """The whole-buffer transpose, for every shape: not imagecodecs' per-row
    byteshuffle, which agrees only for 1-D input."""
    a = np.arange(np.prod(shape), dtype="<f4").reshape(shape) * 1.5
    codec = oc.get_codec("byteshuffle")
    enc = codec.encode(a)
    assert enc == _hdf5_shuffle(a)
    numcodecs = pytest.importorskip("numcodecs")
    assert enc == bytes(numcodecs.Shuffle(elementsize=4).encode(a))
    got = codec.decode(np.frombuffer(enc, a.dtype).reshape(a.shape))
    assert got.dtype == a.dtype and got.shape == a.shape
    np.testing.assert_array_equal(got, a)


@pytest.mark.parametrize("keyword", ["axis", "dist", "delta", "reorder", "bogus"])
def test_byteshuffle_raises_on_imagecodecs_row_keywords(keyword):
    """These were accepted and ignored, returning the whole-buffer shuffle."""
    codec = oc.get_codec("byteshuffle")
    a = np.arange(12, dtype="u2").reshape(3, 4)
    with pytest.raises(TypeError, match=keyword):
        codec.encode(a, **{keyword: 0})
    with pytest.raises(TypeError, match=keyword):
        codec.decode(a.tobytes(), itemsize=2, **{keyword: 0})


# ---------------------------------------------------------------------------
# quantize
# ---------------------------------------------------------------------------


def _values(dtype, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(n) * 10.0 ** rng.uniform(-20, 20, n)
    v = v.astype(dtype)
    v[:6] = np.array([0.0, -0.0, np.inf, -np.inf, 1.0, 2.5]).astype(dtype)
    return v


def _same_bits(a, b):
    return np.array_equal(a.view(f"u{a.dtype.itemsize}"), b.view(f"u{b.dtype.itemsize}"))


def _same_bits_any_nan(a, b):
    """Bits equal except that a NaN may have any sign and payload: an
    arithmetic NaN (inf + -inf) is positive on Arm and negative on x86,
    so the NaN 0.4.0 produced depends on where it ran."""
    nan = np.isnan(a)
    if not np.array_equal(nan, np.isnan(b)):
        return False
    return _same_bits(np.where(nan, 0, a).astype(a.dtype), np.where(nan, 0, b).astype(b.dtype))


def _q(a, mode, nsd):
    return np.frombuffer(oc.get_codec("quantize").encode(a, mode, nsd), a.dtype)


# The nsd netCDF-C's nc_def_var_quantize accepts: BitRound bits, then
# BitGroom and Granular BitRound digits (NC_QUANTIZE_MAX_*_NSB / _NSD).
LIMITS = {"<f4": (23, 6), "<f8": (52, 15), "<f2": (10, 2)}

_NC_FILL = 9.9692099683868690e+36
_BIT_PER_DGT = math.log(10) / math.log(2)


def _netcdf_c(a, mode, nsd):
    """netCDF-C's nc4_convert_type quantize loops, one value at a time.

    A transliteration of libsrc4/nc4var.c: every mode skips the fill
    value, +-0 and NaN. Infinities are skipped as well; that is where
    this codec departs from the C code, whose BitGroom sets the dropped
    bits of an odd-indexed infinity (making it NaN) and whose Granular
    BitRound converts log10 of it to int, which C leaves undefined.
    """
    width = a.dtype.itemsize * 8
    mbits = {16: 10, 32: 23, 64: 52}[width]
    full = (1 << width) - 1
    fill = None if width == 16 else float(a.dtype.type(_NC_FILL))
    bits = [int(b) for b in a.view(f"u{a.dtype.itemsize}")]
    out = []
    for idx, (x, u) in enumerate(zip(a.tolist(), bits)):
        if x == fill or x == 0.0 or math.isnan(x) or math.isinf(x):
            out.append(u)
            continue
        if mode == "granularbr":
            mnt, xpn = math.frexp(x)
            lg = math.log10(abs(mnt))
            dgt = math.floor(xpn * (math.log(2) / math.log(10)) + lg) + 1
            qnt = math.floor(_BIT_PER_DGT * (dgt - nsd))
            keep = abs(math.floor(xpn - _BIT_PER_DGT * lg) - qnt) - 1
        elif mode == "bitgroom":
            keep = math.ceil(nsd * _BIT_PER_DGT) + 1
        else:
            keep = nsd
        zro = (full << (mbits - keep)) & full
        one = ~zro & full
        hshv = one & (zro >> 1)
        if mode == "bitgroom":
            u = u & zro if idx % 2 == 0 else u | one
        else:
            u = ((u + hshv) & full) & zro
        out.append(u)
    return np.array(out, f"u{a.dtype.itemsize}").view(a.dtype)


def _specials(dtype):
    """Values netCDF-C leaves alone, NaN payloads included, at even and odd
    positions, around ordinary values."""
    dt = np.dtype(dtype)
    w = dt.itemsize * 8
    m = {16: 10, 32: 23, 64: 52}[w]
    exp = ((1 << (w - m - 1)) - 1) << m             # all exponent bits
    sign, quiet = 1 << (w - 1), 1 << (m - 1)
    nans = [exp | 1, sign | exp | 1, exp | quiet, exp | quiet | 1,
            sign | exp | quiet | 3, exp | (quiet >> 1)]
    raw = np.array(nans, f"u{dt.itemsize}").view(dt)
    with np.errstate(over="ignore"):
        vals = np.array([0.0, -0.0, np.inf, -np.inf, 1.0, -0.0, 2.5, 0.0,
                         -1.75, np.inf, 3.25, -np.inf], dt)
        if w != 16:
            vals = np.concatenate([vals, np.array([_NC_FILL, 7.0, _NC_FILL], dt)])
    return np.concatenate([vals, raw, raw[::-1], np.array([1.5], dt)])


@pytest.mark.parametrize("dtype", ["<f4", "<f8", "<f2"])
def test_quantize_follows_netcdf_c(dtype):
    """Against the C loops: -0.0 at an odd index became the negative
    subnormal 0x80000fff in BitGroom (a bit-pattern zero test), and NaN
    payloads were rounded (0x7f800001 to +inf in BitRound)."""
    bits, digits = LIMITS[dtype]
    with np.errstate(over="ignore"):
        ordinary = _values(dtype, n=4000, seed=3)
    a = np.concatenate([_specials(dtype), ordinary, _specials(dtype)])
    for nsd in range(1, bits + 1):
        assert _same_bits(_q(a, "bitround", nsd), _netcdf_c(a, "bitround", nsd)), nsd
    for nsd in range(1, digits + 1):
        for mode in ("bitgroom", "granularbr"):
            assert _same_bits(_q(a, mode, nsd), _netcdf_c(a, mode, nsd)), (mode, nsd)


@pytest.mark.parametrize("dtype", ["<f4", "<f8"])
def test_quantize_matches_imagecodecs(dtype):
    """imagecodecs alters zeros, NaN and infinities where netCDF-C does
    not (test above); on every other value the bits agree."""
    ic = _imagecodecs()
    a = _values(dtype)
    a[6] = _NC_FILL                                  # netCDF fill value
    a = a[np.isfinite(a) & (a != 0)]
    bits, digits = LIMITS[dtype]
    for nsd in range(1, bits + 1):
        assert _same_bits(_q(a, "bitround", nsd), ic.quantize_encode(a, "bitround", nsd)), nsd
    for nsd in range(1, digits + 1):
        for mode in ("bitgroom", "granularbr", "gbr"):
            assert _same_bits(_q(a, mode, nsd), ic.quantize_encode(a, mode, nsd)), nsd
    a = _values(dtype)
    for nsd in range(12):
        assert _same_bits(_q(a, "scale", nsd), ic.quantize_encode(a, "scale", nsd)), nsd


def test_quantize_scale_by_hand():
    """nsd=1: 10**-1 rounds up to the power-of-two grid 1/16, and C's round
    takes halves away from zero."""
    a = np.array([1234.5678, 0.00123456, -98.7654321, 3.14159265, 0.03125, -0.03125],
                 np.float32)
    want = np.array([19753 / 16, 0.0, -1580 / 16, 50 / 16, 1 / 16, -1 / 16], np.float32)
    assert _same_bits(_q(a, "scale", 1), want)


def test_quantize_bitgroom_and_bitround_by_hand():
    """netCDF-C: BitRound adds the half of the dropped bits then masks;
    BitGroom keeps ceil(nsd * log2(10)) + 1 bits, shaving even-indexed
    values and setting the dropped bits of odd-indexed ones."""
    a = np.array([np.pi, np.e, -np.pi, 0.0], np.float32)
    u = a.view("u4")
    keep = 8                                         # nsd=2: ceil(6.64) + 1
    shave = np.uint32((0xFFFFFFFF << (23 - keep)) & 0xFFFFFFFF)
    want = u.copy()
    want[0::2] &= shave
    want[1] |= ~shave                                # odd, nonzero
    assert _same_bits(_q(a, "bitgroom", 2), want.view("f4"))
    half = np.uint32(1 << (23 - keep - 1))
    assert _same_bits(_q(a, "bitround", keep), ((u + half) & shave).view("f4"))


@pytest.mark.parametrize("mode", ["bitround", "bitgroom", "granularbr"])
@pytest.mark.parametrize("dtype", ["<f4", "<f8", ">f4"])
def test_quantize_keeps_zeros_nan_and_infinities(mode, dtype):
    """The verifier's cases: [1.0, -0.0] in BitGroom at nsd=3 gave
    0x80000fff (float32) and -1.09e-311 (float64); 0x7f800001 in BitRound
    at nsd=7 gave +inf; quiet NaN payloads changed."""
    nan = np.array([0x7F800001, 0xFFC00001, 0x7FC00000, 0x7FC00000], "u4").view("f4")
    with np.errstate(invalid="ignore"):             # signaling NaN cast
        a = np.concatenate([np.array([1.0, -0.0, 0.0, -0.0, np.inf, np.inf,
                                      -np.inf, -np.inf], "f4"), nan]).astype(dtype)
    if dtype == "<f8":
        a[8:] = np.array([0x7FF0000000000001, 0xFFF8000000000001,
                          0x7FF8000000000000, 0x7FF8000000000000], "u8").view("f8")
    got = _q(a, mode, 7 if mode == "bitround" else 3)
    assert _same_bits(got[1:], a[1:])
    assert got[0] == 1.0


@pytest.mark.parametrize("dtype,mode,bad,good", [
    ("<f4", "bitround", [0, 24, -1], [1, 23]),
    ("<f8", "bitround", [0, 53], [1, 52]),
    ("<f2", "bitround", [0, 11], [1, 10]),
    ("<f4", "bitgroom", [0, 7, 9], [1, 6]),
    ("<f4", "granularbr", [0, 7], [1, 6]),
    ("<f8", "bitgroom", [0, 16], [1, 15]),
    ("<f8", "gbr", [0, 16], [1, 15]),
    ("<f2", "bitgroom", [0, 3], [1, 2]),
    ("<f4", "scale", [-1], [0, 30]),
    ("<f4", "nsd", [0], [1, 17]),
])
def test_quantize_nsd_range(dtype, mode, bad, good):
    """nc_def_var_quantize (libhdf5/hdf5var.c) rejects nsd <= 0, BitRound
    beyond the mantissa bits and BitGroom or Granular BitRound beyond 6
    (float) or 15 (double) digits. bitgroom nsd=0 changed 2.5 to
    2.9999998, and nsd past the limit returned the data unchanged."""
    a = np.array([2.5, 1.1, -3.75], dtype)
    q = oc.get_codec("quantize")
    for nsd in bad:
        with pytest.raises(ValueError, match="nsd"):
            q.encode(a, mode, nsd)
        with pytest.raises(ValueError, match="nsd"):
            q.decode(a, mode, nsd)
    for nsd in good:
        q.encode(a, mode, nsd)
        q.decode(a, mode, nsd)


def test_quantize_nsd_must_be_whole():
    """nsd=7.9 was truncated to 7 without a word."""
    q = oc.get_codec("quantize")
    a = np.array([2.5, 1.1], np.float32)
    for kw in (dict(nsd=7.9), dict(bitspersample=7.5)):
        with pytest.raises(ValueError, match="whole"):
            q.encode(a, "bitround", **kw)
    with pytest.raises(TypeError, match="integer"):
        q.encode(a, "bitround", "7")
    assert q.encode(a, "bitround", 7.0) == q.encode(a, "bitround", np.int64(7))


def test_quantize_bitround_skips_the_netcdf_fill_value():
    a = np.array([9.9692099683868690e+36, 1.2345], np.float32)
    got = _q(a, "bitround", 3)
    assert got[0] == a[0] and got[1] != a[1]


def test_quantize_nsd_is_computed_in_float64():
    """The scale was computed in float32 for float32 input, so 1234.5678 at
    one digit came out 999.99994, and 4 to 5 percent of random float32
    values missed the correctly rounded result."""
    q = oc.get_codec("quantize")
    a = np.array([1234.5678, 0.00123456, -98.7654321, 3.14159265], np.float32)
    got = np.frombuffer(q.encode(a, mode="nsd", nsd=1), np.float32)
    np.testing.assert_array_equal(got, np.array([1000, 0.001, -100, 3], np.float32))
    rng = np.random.default_rng(5)
    v = (rng.standard_normal(5000) * 10.0 ** rng.uniform(-6, 6, 5000)).astype(np.float32)
    for nsd in (1, 2, 3, 4):
        got = np.frombuffer(q.encode(v, mode="nsd", nsd=nsd), np.float32)
        want = np.array([float(f"{float(x):.{nsd - 1}e}") for x in v], np.float32)
        assert np.count_nonzero(got != want) == 0, nsd


def _nearest_of_type(d, dtype):
    """The value of float ``dtype`` nearest the Fraction ``d`` (ties to
    even), in exact arithmetic, or None past the overflow threshold. No
    float64 is formed on the way, so nothing is rounded twice."""
    from fractions import Fraction
    info = np.finfo(dtype)
    mant = info.nmant
    a = abs(d)
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    e = max(e, int(info.minexp))
    ulp = Fraction(2) ** (e - mant)
    q = a / ulp
    k = q.numerator // q.denominator
    rem = q - k
    if rem > Fraction(1, 2) or (rem == Fraction(1, 2) and k % 2):
        k += 1
    mag = k * ulp
    if mag > Fraction(float(info.max)):
        return None
    return dtype.type(float(-mag if d < 0 else mag))


def _nsd_reference(a, nsd):
    """Each value correctly rounded to ``nsd`` digits by Python's own
    float formatting (exact binary value, ties to even), then the value of
    ``a``'s dtype nearest that decimal, found in exact arithmetic; a value
    whose rounding overflows the type is kept."""
    from fractions import Fraction
    want = a.copy()
    native = a.dtype.newbyteorder("=")
    for i, x in enumerate(a.tolist()):
        if x == 0 or not np.isfinite(x):
            continue
        r = _nearest_of_type(Fraction(f"{x:.{nsd - 1}e}"), native)
        if r is not None:
            want[i] = r
    return want


@pytest.mark.parametrize("dtype,top,digits", [
    ("<f8", 300, 16), ("<f4", 38, 9), ("<f2", 4, 5), (">f8", 30, 8)])
def test_quantize_nsd_is_correctly_rounded(dtype, top, digits):
    """The scale 10**(nsd - 1 - floor(log10|x|)) is inexact once it is a
    negative power of ten, so multiplying and dividing by it missed the
    correctly rounded decimal for 5 to 7 percent of float64 values
    (1175103902.8858647 at one digit gave 999999999.9999999) and for some
    large float32 ones. Pinned against Python's float formatting over the
    whole exponent range of each type, plus random bit patterns."""
    rng = np.random.default_rng(len(dtype) + top)
    dt = np.dtype(dtype)
    fmax = float(np.finfo(dt).max)
    mags = 10.0 ** rng.uniform(-top, top, 3000)
    a = rng.choice([-1.0, 1.0], 3000) * mags * rng.uniform(1, 10, 3000)
    a = np.clip(a, -fmax, fmax).astype(dt)
    raw = rng.integers(0, 1 << (8 * dt.itemsize) - 1, 1000, dtype="u8",
                       endpoint=True).astype(f"u{dt.itemsize}").view(dt.newbyteorder("="))
    special = [2.5, 0.125, 1.5, 15, 25, 105, 1e22, 1e23, 9.5e-5, fmax, -fmax,
               float(np.finfo(dt).smallest_subnormal), np.inf, -np.inf,
               np.nan, 0.0, -0.0]
    special = np.array([x for x in special if abs(x) <= fmax or not np.isfinite(x)], dt)
    a = np.concatenate([a, raw.astype(dt), special])
    q = oc.get_codec("quantize")
    for nsd in range(1, digits + 1):
        got = np.frombuffer(q.encode(a, "nsd", nsd), dt)
        assert _same_bits(got, _nsd_reference(a, nsd)), nsd


def test_quantize_nsd_examples():
    q = oc.get_codec("quantize")
    a = np.array([1175103902.8858647, 2.5, 3.5, 1e23 * 1.07], np.float64)
    got = np.frombuffer(q.encode(a, "nsd", 1), np.float64)
    np.testing.assert_array_equal(got, [1e9, 2.0, 4.0, 1e23])
    f = np.array([-825137024, np.inf, -np.inf, 1.0], np.float32)
    got = np.frombuffer(q.encode(f, "nsd", 4), np.float32)
    assert got[0] == np.float32(-825100000)       # -825100032
    assert _same_bits(got[1:], f[1:])
    # Rounding f32 max to 4 digits gives 3.403e38, past the type: kept.
    m = np.array([np.finfo(np.float32).max], np.float32)
    assert _same_bits(np.frombuffer(q.encode(m, "nsd", 4), np.float32), m)
    # The float64 nearest 7.038531e-26 is halfway between two float32
    # values; casting it rounded again, to 7.0385313e-26, though the
    # float32 nearest the decimal is 7.0385307e-26 (which so stays put).
    f = np.array([7.0385307e-26, 7.0385313e-26, -7.0385313e-26], np.float32)
    got = np.frombuffer(q.encode(f, "nsd", 7), np.float32)
    want = np.float32(7.0385307e-26)
    assert _same_bits(got, np.array([want, want, -want], np.float32))
    assert _same_bits(got, _nsd_reference(f, 7))
    # 17 digits name every float64: nothing changes.
    r = np.random.default_rng(0).standard_normal(100)
    assert _same_bits(np.frombuffer(q.encode(r, "nsd", 17), np.float64), r)


def test_quantize_decode_checks_its_parameters():
    """decode is the identity, but a bad mode or nsd used to be accepted
    and ignored."""
    q = oc.get_codec("quantize")
    a = np.array([1.5, 2.25], np.float32)
    np.testing.assert_array_equal(q.decode(a, "bitround", 3), a)
    np.testing.assert_array_equal(q.decode(a, mode="nsd", nsd=2), a)
    np.testing.assert_array_equal(q.decode(a, bitspersample=3), a)
    with pytest.raises(ValueError, match="mode"):
        q.decode(a, "bogus")
    with pytest.raises(ValueError, match="nsd"):
        q.decode(a, "bitround", -5)
    with pytest.raises(ValueError, match="disagree"):
        q.decode(a, nsd=2, bitspersample=3)


def test_quantize_api():
    q = oc.get_codec("quantize")
    a = np.linspace(-3, 3, 11, dtype=np.float32)
    by_position = q.encode(a, "bitround", 7)
    assert q.encode(a, mode="bitround", nsd=7) == by_position
    assert q.encode(a, bitspersample=7) == by_position
    assert q.encode(a, "BITROUND", 7) == by_position
    assert q.encode(a, 3, 7) == by_position                 # NC_QUANTIZE_BITROUND
    with pytest.raises(ValueError, match="disagree"):
        q.encode(a, nsd=6, bitspersample=7)
    with pytest.raises(TypeError, match="bogus"):
        q.encode(a, "bitround", 7, bogus=1)
    with pytest.raises(ValueError, match="mode"):
        q.encode(a, "bitshave", 7)
    np.testing.assert_array_equal(q.decode(np.frombuffer(by_position, a.dtype)),
                                  np.frombuffer(by_position, a.dtype))
