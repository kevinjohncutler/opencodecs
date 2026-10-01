"""TIFF predictor 2 on every sample width, against independent references.

TIFF 6.0 section 14 defines Predictor=2 as horizontal differencing, and
libtiff (tif_predict.c, horDiff8/16/32/64 and horAcc8/16/32/64) runs it
on each sample's storage word as an unsigned integer of the same width,
with wraparound, whatever the SampleFormat. So 64-bit integers are
legal, and floating-point samples are differenced as bit patterns, never
subtracted as floats. tifffile reads both that way.

The written bytes are checked against a reference encoder written here
from that definition, the reader against files built here byte by byte
(no opencodecs code involved), and both against tifffile.
"""
from __future__ import annotations

import io
import struct
import zlib

import numpy as np
import pytest

tifffile = pytest.importorskip("tifffile")
from opencodecs._tiff_codec import TiffStream, _HAVE_BACKEND  # noqa: E402
from opencodecs._tiff_writer import TiffWriter, TiffWriterError  # noqa: E402

pytestmark = pytest.mark.skipif(not _HAVE_BACKEND, reason="native TIFF backend not built")

DTYPES = ["u1", "i1", "u2", "i2", "u4", "i4", "u8", "i8", "f2", "f4", "f8"]


def _values(dtype, shape, seed=7):
    """Random bit patterns, with float specials (signed zero, infinities,
    NaN payloads) where the dtype is floating point. Every comparison is
    made on the unsigned storage word, so NaNs compare exactly."""
    dtype = np.dtype(dtype)
    n = int(np.prod(shape))
    bits = np.random.default_rng(seed).integers(0, 256, (n, dtype.itemsize), dtype=np.uint8)
    values = bits.reshape(-1).view(dtype).reshape(shape).copy()
    special = {2: [0, 0x8000, 0x7C00, 0xFC00, 0x7E51, 0x7C21],
               4: [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7FC01234, 0x7F801234],
               8: [0, 0x8000000000000000, 0x7FF0000000000000, 0xFFF0000000000000,
                   0x7FF8000000001234, 0x7FF0000000001234]}
    if dtype.kind == "f":
        word = np.dtype(f"{dtype.byteorder}u{dtype.itemsize}".replace("=", ""))
        values.view(word).flat[:6] = special[dtype.itemsize]
    return values


def _bits(a):
    """Native unsigned integers holding each sample's bits."""
    word = np.dtype(f"u{a.dtype.itemsize}").newbyteorder(a.dtype.byteorder)
    return a.view(word).astype(np.dtype(f"u{a.dtype.itemsize}"))


def _assert_bits_equal(actual, expected):
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(_bits(actual), _bits(expected))


def _spec_predictor2(values, byte_order):
    """Predictor 2 stored bytes as TIFF 6.0 and libtiff define them.

    ``values`` is (rows, cols, samples). Each sample's storage word, read as
    an unsigned integer of its width, minus the same sample of the pixel to
    its left, modulo 2**bits; the first pixel of each row is kept.
    """
    word = np.dtype(f"u{values.dtype.itemsize}")
    bits = _bits(values)
    diff = bits.copy()
    diff[:, 1:] = bits[:, 1:] - bits[:, :-1]   # unsigned: wraps modulo 2**bits
    return diff.astype(word.newbyteorder(byte_order)).tobytes()


def _hand_built_tiff(values, byte_order, compression=8):
    """A one-strip, Predictor=2 classic TIFF built without opencodecs,
    deflate (8) or uncompressed (1)."""
    rows, cols, spp = values.shape
    dtype = values.dtype
    fmt = {"u": 1, "i": 2, "f": 3}[dtype.kind]
    strip = _spec_predictor2(values, byte_order)
    if compression == 8:
        strip = zlib.compress(strip)
    bo = byte_order
    shorts = lambda v: struct.pack(f"{bo}{len(v)}H", *v)  # noqa: E731
    # Header, then out-of-line tag values, then the strip, then the IFD.
    body = bytearray(struct.pack(f"{bo}2sHI", b"II" if bo == "<" else b"MM", 42, 0))
    bps_off = len(body); body += shorts([dtype.itemsize * 8] * spp) + b"\0\0"
    sf_off = len(body); body += shorts([fmt] * spp) + b"\0\0"
    strip_off = len(body); body += strip
    if len(body) % 2:
        body += b"\0"

    def entry(tag, typ, count, value):
        if typ == 3 and count == 1:
            return struct.pack(f"{bo}HHIH2x", tag, typ, count, value)
        return struct.pack(f"{bo}HHII", tag, typ, count, value)

    def short_array(tag, off, values_):
        if spp == 1:
            return entry(tag, 3, 1, values_[0])
        return entry(tag, 3, spp, off)

    entries = [
        entry(256, 4, 1, cols),
        entry(257, 4, 1, rows),
        short_array(258, bps_off, [dtype.itemsize * 8]),
        entry(259, 3, 1, compression),
        entry(262, 3, 1, 2 if spp == 3 else 1),
        entry(273, 4, 1, strip_off),
        entry(277, 3, 1, spp),
        entry(278, 4, 1, rows),
        entry(279, 4, 1, len(strip)),
        entry(284, 3, 1, 1),
        entry(317, 3, 1, 2),                          # Predictor: horizontal
        short_array(339, sf_off, [fmt]),
    ]
    ifd_off = len(body)
    body += struct.pack(f"{bo}H", len(entries)) + b"".join(entries) + struct.pack(f"{bo}I", 0)
    body[4:8] = struct.pack(f"{bo}I", ifd_off)
    return bytes(body)


def _strips(encoded):
    """Each stored strip or tile of page 0, decompressed with zlib."""
    with tifffile.TiffFile(io.BytesIO(encoded)) as tf:
        page = tf.pages[0]
        assert page.predictor == 2
        out = []
        for off, n in zip(page.dataoffsets, page.databytecounts):
            out.append(zlib.decompress(encoded[off:off + n]))
    return out


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("byte_order", ["<", ">"])
@pytest.mark.parametrize("spp", [1, 3])
def test_writer_stores_unsigned_word_differences(dtype, byte_order, spp):
    """The stored bytes are libtiff's: for floats, differences of bit patterns."""
    values = _values(dtype, (6, 9, spp))
    source = values if spp == 3 else values[:, :, 0]
    sink = io.BytesIO()
    with TiffWriter(sink, byte_order=byte_order) as w:
        w.write_page(source, compression="deflate", predictor=2, rows_per_strip=6)
    encoded = sink.getvalue()
    (strip,) = _strips(encoded)
    assert strip == _spec_predictor2(values, byte_order)
    _assert_bits_equal(tifffile.imread(io.BytesIO(encoded)), source)
    with TiffStream(encoded) as r:
        _assert_bits_equal(r.read(), source)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("byte_order", ["<", ">"])
@pytest.mark.parametrize("spp", [1, 3])
def test_reader_undoes_predictor2_on_hand_built_files(dtype, byte_order, spp):
    """Files made here from the definition read back bit for bit, on the
    fused native path (native byte order) and the general one (swapped)."""
    values = _values(dtype, (5, 11, spp), seed=11)
    encoded = _hand_built_tiff(values, byte_order)
    expected = values if spp == 3 else values[:, :, 0]
    _assert_bits_equal(tifffile.imread(io.BytesIO(encoded)), expected)
    with TiffStream(encoded) as r:
        page = r.page(0)
        if byte_order == "<" and np.little_endian or byte_order == ">" and not np.little_endian:
            assert page._fused_segment_codec() is not None
        _assert_bits_equal(r.read(), expected)


@pytest.mark.parametrize("dtype", ["u8", "i8"])
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("spp", [1, 3])
def test_reader_reads_tifffile_64bit_predictor2(dtype, tiled, spp):
    """tifffile writes Predictor=2 on 64-bit integers, as libtiff does."""
    shape = (37, 45, spp) if spp == 3 else (37, 45)
    values = _values(dtype, shape, seed=3)
    sink = io.BytesIO()
    tifffile.imwrite(sink, values, compression="zlib", predictor=2,
                     tile=(16, 32) if tiled else None, rowsperstrip=None if tiled else 7,
                     photometric="rgb" if spp == 3 else "minisblack")
    with TiffStream(sink.getvalue()) as r:
        page = r.page(0)
        assert page.predictor == 2
        _assert_bits_equal(r.read(), values)
        _assert_bits_equal(page.asarray(numthreads=1), values)


class _ForwardOnly:
    def __init__(self):
        self.buffer = bytearray()

    def write(self, data):
        self.buffer.extend(data)
        return len(data)

    def flush(self):
        pass


@pytest.mark.parametrize("dtype", ["f2", "f4", "f8", "u8", "i8"])
@pytest.mark.parametrize("byte_order", ["<", ">"])
@pytest.mark.parametrize("streaming", [False, True])
def test_writer_tiled_predictor2_verifies_and_matches_tifffile(dtype, byte_order, streaming):
    """Tiles, parallel encode, lossless verification and forward-only output."""
    values = _values(dtype, (35, 47, 3), seed=5)
    sink = _ForwardOnly() if streaming else io.BytesIO()
    options = dict(compression="deflate", predictor=2, tile=(16, 32), n_workers=4, verify=True)
    with TiffWriter(sink, byte_order=byte_order, streaming=streaming,
                    spool_threshold=128 if streaming else None) as w:
        if streaming:
            w.write_stream(iter([values]), total_pages=1, **options)
        else:
            w.write_page(values, **options)
    encoded = bytes(sink.buffer) if streaming else sink.getvalue()
    _assert_bits_equal(tifffile.imread(io.BytesIO(encoded)), values)
    with TiffStream(encoded) as r:
        _assert_bits_equal(r.read(), values)
    # Every tile is padded to 16 x 32 and differenced as the definition says.
    padded = np.zeros((48, 64, 3), dtype=values.dtype)
    padded[:35, :47] = values
    tiles = _strips(encoded)
    assert len(tiles) == 3 * 2
    for i, tile in enumerate(tiles):
        ty, tx = divmod(i, 2)
        ref = padded[ty * 16:(ty + 1) * 16, tx * 32:(tx + 1) * 32]
        assert tile == _spec_predictor2(ref, byte_order)


def _write(values, options, *, streaming=False, byte_order="<"):
    sink = _ForwardOnly() if streaming else io.BytesIO()
    with TiffWriter(sink, byte_order=byte_order, streaming=streaming,
                    spool_threshold=128 if streaming else None) as w:
        if streaming:
            w.write_stream(iter([values]), total_pages=1, **options)
        else:
            w.write_page(values, **options)
    return bytes(sink.buffer) if streaming else sink.getvalue()


@pytest.mark.parametrize("compression", ["none", "jpeg2000", "webp", "lerc"])
@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_writer_refuses_predictor2_where_it_is_not_written(compression, verify, streaming):
    """libtiff ignores Predictor on uncompressed data and image codecs take
    none, so the writer raises rather than writing something else. tifffile
    raises ValueError for uncompressed data and JPEG 2000 and WebP; it
    applies a predictor before LERC, which this writer does not implement.
    With verify=True an uncompressed write used to store the samples
    undifferenced under Predictor=2."""
    values = _values("u1", (32, 32))
    if compression != "lerc":
        with pytest.raises(ValueError, match="predictor"):
            tifffile.imwrite(io.BytesIO(), values, predictor=2,
                             compression=None if compression == "none" else compression)
    with pytest.raises(TiffWriterError, match="cannot use predictor 2"):
        _write(values, dict(compression=compression, predictor=2, verify=verify),
               streaming=streaming)


def test_verification_leaves_the_segment_it_checks_unchanged():
    """The verifier decodes an uncompressed segment straight from the
    buffer that is then written; undoing the predictor must not touch it."""
    values = _values("u2", (6, 9, 1), seed=2)
    stored = _spec_predictor2(values, "<")
    encoded = np.frombuffer(stored, dtype="<u2").copy()
    with TiffWriter(io.BytesIO()) as w:
        w._verify_segment_pixels(values.astype("<u2"), encoded, 1, 2)
    assert encoded.tobytes() == stored


@pytest.mark.parametrize("dtype", ["u1", "i2", "u4", "f4", "u8", "f8"])
@pytest.mark.parametrize("byte_order", ["<", ">"])
@pytest.mark.parametrize("spp", [1, 3])
def test_reader_undoes_predictor2_on_uncompressed_files(dtype, byte_order, spp):
    """Earlier versions wrote uncompressed predictor 2 files, and this reader
    still undoes the predictor there, as tifffile does where it can."""
    values = _values(dtype, (5, 11, spp), seed=13)
    encoded = _hand_built_tiff(values, byte_order, compression=1)
    expected = values if spp == 3 else values[:, :, 0]
    with TiffStream(encoded) as r:
        _assert_bits_equal(r.read(), expected)
        _assert_bits_equal(r.page(0).asarray(numthreads=1), expected)


@pytest.mark.parametrize("dtype", ["u1", "u2", "f4", "i8"])
@pytest.mark.parametrize("compression,predictor", [("none", 1), ("deflate", 1),
                                                   ("deflate", 2), ("zstd", 2)])
@pytest.mark.parametrize("tiled", [False, True])
@pytest.mark.parametrize("streaming", [False, True])
def test_writer_stores_separate_sample_planes(dtype, compression, predictor, tiled, streaming):
    """planar_config=2 stores plane after plane (TIFF 6.0 section 8); it used
    to tag the page separate while storing the samples interleaved."""
    values = _values(dtype, (35, 47, 3), seed=17)
    options = dict(compression=compression, predictor=predictor, planar_config=2,
                   verify=True, n_workers=4)
    if tiled:
        options["tile"] = (16, 32)
    else:
        options["rows_per_strip"] = 9
    encoded = _write(values, options, streaming=streaming, byte_order=">")
    with tifffile.TiffFile(io.BytesIO(encoded)) as tf:
        page = tf.pages[0]
        assert page.planarconfig == 2
        _assert_bits_equal(page.asarray(), np.moveaxis(values, 2, 0))
    with TiffStream(encoded) as r:
        _assert_bits_equal(r.read(), values)
    if compression == "deflate" and predictor == 2:
        # Every segment holds one plane, differenced as the definition says.
        segments = _strips(encoded)
        if tiled:
            padded = np.zeros((48, 64, 3), dtype=values.dtype)
            padded[:35, :47] = values
            refs = [padded[ty * 16:(ty + 1) * 16, tx * 32:(tx + 1) * 32, c:c + 1]
                    for c in range(3) for ty in range(3) for tx in range(2)]
        else:
            refs = [values[y:y + 9, :, c:c + 1] for c in range(3) for y in range(0, 35, 9)]
        assert len(segments) == len(refs)
        for segment, ref in zip(segments, refs):
            assert segment == _spec_predictor2(ref, ">")


def test_writer_refuses_unknown_planar_config():
    with pytest.raises(TiffWriterError, match="planar_config"):
        _write(_values("u1", (8, 8, 3)), dict(planar_config=3))


def test_writer_refuses_webp_sample_planes():
    """WebP holds RGB or RGBA pixels, so a one-sample plane cannot be stored."""
    with pytest.raises(TiffWriterError, match="planar_config=2"):
        _write(_values("u1", (16, 16, 3)), dict(compression="webp", planar_config=2))


@pytest.mark.parametrize("dtype", ["u1", "i2", "u4", "u8", "f2", "f4", "f8"])
@pytest.mark.parametrize("flag", [True, np.True_, False, np.False_])
@pytest.mark.parametrize("streaming", [False, True])
def test_writer_bool_predictor_matches_tifffile(dtype, flag, streaming):
    """predictor=True used to pass as the integer 1 and write no predictor.
    tifffile picks 3 for float and 2 for integers up to 32 bits, refuses
    True for 64-bit integers (this writer uses 2 there, which libtiff
    defines), and treats False as no predictor."""
    values = _values(dtype, (19, 37, 3), seed=23)
    encoded = _write(values, dict(compression="deflate", predictor=flag, rows_per_strip=7,
                                  verify=True), streaming=streaming)
    expected_tag = (3 if values.dtype.kind == "f" else 2) if flag else 1
    if flag and dtype == "u8":
        with pytest.raises(ValueError, match="predictor"):
            tifffile.imwrite(io.BytesIO(), values, predictor=bool(flag), compression="zlib")
    else:
        reference = io.BytesIO()
        tifffile.imwrite(reference, values, predictor=bool(flag), compression="zlib",
                         photometric="rgb")
        with tifffile.TiffFile(io.BytesIO(reference.getvalue())) as tf:
            assert tf.pages[0].predictor == expected_tag
    with tifffile.TiffFile(io.BytesIO(encoded)) as tf:
        page = tf.pages[0]
        assert page.predictor == expected_tag
        _assert_bits_equal(page.asarray(), values)
    with TiffStream(encoded) as r:
        _assert_bits_equal(r.read(), values)
    if expected_tag == 2:
        segments = _strips(encoded)
        refs = [values[y:y + 7] for y in range(0, 19, 7)]
        assert segments == [_spec_predictor2(ref, "<") for ref in refs]


@pytest.mark.parametrize("compression", ["none", "jpeg2000"])
def test_writer_refuses_bool_predictor_without_a_predictor_codec(compression):
    """True is a predictor request, so it raises where any predictor would,
    as tifffile does."""
    values = _values("u1", (32, 32))
    with pytest.raises(ValueError, match="predictor"):
        tifffile.imwrite(io.BytesIO(), values, predictor=True,
                         compression=None if compression == "none" else compression)
    with pytest.raises(TiffWriterError, match="cannot use predictor 2"):
        _write(values, dict(compression=compression, predictor=True))


@pytest.mark.parametrize("flag", [True, False, np.True_])
@pytest.mark.parametrize("streaming", [False, True])
def test_writer_refuses_bool_planar_config(flag, streaming):
    """True == 1 used to pass as chunky without saying so."""
    with pytest.raises(TiffWriterError, match="not a bool"):
        _write(_values("u1", (8, 8, 3)), dict(planar_config=flag), streaming=streaming)


def _patch_tag(encoded, code, value, *, count=False):
    """Overwrite one inline SHORT tag value of page 0 (or, with count=True,
    the count field of tag ``code``), leaving every other byte alone."""
    out = bytearray(encoded)
    with tifffile.TiffFile(io.BytesIO(encoded)) as tf:
        bo = tf.byteorder
        tag = tf.pages[0].tags[code]
        if count:
            struct.pack_into(f"{bo}I", out, tag.offset + 4, value)
        else:
            assert tag.count == 1 and tag.dtype == 3
            struct.pack_into(f"{bo}H", out, tag.valueoffset, value)
    return bytes(out)


@pytest.mark.parametrize("dtype", ["u1", "u2", "f4"])
@pytest.mark.parametrize("compression", [None, "zlib", "lzw", "zstd"])
@pytest.mark.parametrize("predictor", [False, True])
@pytest.mark.parametrize("layout", ["one strip", "strips", "one tile", "tiles"])
@pytest.mark.parametrize("byte_order", ["<", ">"])
def test_reader_reads_the_earlier_interleaved_planar_layout(dtype, compression, predictor,
                                                            layout, byte_order):
    """Earlier versions of the writer tagged planar_config=2 pages
    PlanarConfiguration=2 but stored interleaved samples, one run of
    strips or tiles. Built here by tifffile writing a chunky file and the
    tag being set to 2 afterwards. libtiff refuses such a file and
    tifffile raises or reads it wrong; this reader takes it as the
    interleaved data it holds. Single-strip and single-tile LZW pages used
    to come back wrong, and the others raised."""
    if predictor and compression is None:
        pytest.skip("tifffile writes no predictor without compression")
    values = _values(dtype, (40, 40, 3), seed=31)
    options = dict(compression=compression, predictor=predictor, byteorder=byte_order,
                   photometric="rgb", planarconfig="contig")
    if layout.endswith("tile"):
        options["tile"] = (48, 48)
    elif layout == "tiles":
        options["tile"] = (16, 32)
    else:
        options["rowsperstrip"] = 40 if layout == "one strip" else 7
    buf = io.BytesIO()
    tifffile.imwrite(buf, values, **options)
    encoded = _patch_tag(buf.getvalue(), 284, 2)
    with tifffile.TiffFile(io.BytesIO(encoded)) as tf:
        assert tf.pages[0].planarconfig == 2
    with TiffStream(encoded) as r:
        _assert_bits_equal(r.read(), values)
        _assert_bits_equal(r.page(0).asarray(numthreads=1), values)
        _assert_bits_equal(r.page(0).asarray(numthreads=4), values)


def test_reader_refuses_too_few_separate_plane_segments():
    """A PlanarConfiguration=2 page with fewer offsets than SamplesPerPixel
    runs used to have its planes shifted onto the wrong strips."""
    values = _values("u1", (14, 9, 3), seed=37)
    buf = io.BytesIO()
    tifffile.imwrite(buf, np.moveaxis(values, 2, 0), photometric="rgb",
                     planarconfig="separate", rowsperstrip=7)
    encoded = buf.getvalue()
    with TiffStream(encoded) as r:
        _assert_bits_equal(r.read(), values)
    # Six strips (two per plane) declared as four.
    short = _patch_tag(_patch_tag(encoded, 273, 4, count=True), 279, 4, count=True)
    with tifffile.TiffFile(io.BytesIO(short)) as tf:
        assert len(tf.pages[0].dataoffsets) == 4
    with TiffStream(short) as r, pytest.raises(ValueError, match="out of range"):
        r.read()


@pytest.mark.parametrize("byte_order", ["<", ">"])
@pytest.mark.parametrize("compression,predictor", [(None, False), ("zlib", 2)])
@pytest.mark.parametrize("tiled", [False, True])
def test_reader_returns_complex64_samples(byte_order, compression, predictor, tiled):
    """SampleFormat 6 with 64 bits is complex64 and used to come back as
    uint64. Built here by tifffile writing int64 words and SampleFormat
    being set to 6 afterwards. Without a predictor each float component is
    stored in the file's byte order. With predictor 2 libtiff 4.7 swaps and
    differences the whole 64-bit sample word (swabHorAcc64), so the native
    sample is that word; both were measured against libtiff writing and
    reading complex64 in both byte orders."""
    words = _values("i8", (9, 37), seed=41)
    stored = words.astype(words.dtype.newbyteorder(byte_order))
    if predictor:
        expected = words.view(np.complex64)
    else:
        expected = stored.view(np.dtype(np.complex64).newbyteorder(byte_order))
        expected = expected.astype(np.complex64)
    options = dict(compression=compression, predictor=predictor, byteorder=byte_order)
    if tiled:
        options["tile"] = (16, 16)
    buf = io.BytesIO()
    tifffile.imwrite(buf, stored, **options)
    encoded = _patch_tag(buf.getvalue(), 339, 6)
    with TiffStream(encoded) as r:
        out = r.read()
        assert out.dtype == np.complex64
        _assert_bits_equal(out, expected)
        _assert_bits_equal(r.page(0).asarray(numthreads=4), expected)


@pytest.mark.parametrize("byte_order", ["<", ">"])
@pytest.mark.parametrize("tiled", [False, True])
def test_reader_returns_complex_samples_as_tifffile_does(byte_order, tiled):
    """Uncompressed complex64 and complex128 from tifffile, each component
    in the file's byte order (libtiff 4.7 reads these files the same)."""
    rng = np.random.default_rng(43)
    for dtype in ("c8", "c16"):
        values = (rng.random((9, 37)) + 1j * rng.random((9, 37))).astype(dtype)
        buf = io.BytesIO()
        tifffile.imwrite(buf, values, byteorder=byte_order,
                         tile=(16, 16) if tiled else None)
        with TiffStream(buf.getvalue()) as r:
            out = r.read()
        assert out.dtype == values.dtype
        np.testing.assert_array_equal(out, values)


@pytest.mark.parametrize("predictor", [0, None, np.int64(0)])
@pytest.mark.parametrize("streaming", [False, True])
def test_writer_takes_zero_and_none_as_no_predictor(predictor, streaming):
    """tifffile writes no predictor for 0 or None; the writer used to raise."""
    values = _values("u2", (11, 13), seed=47)
    reference = io.BytesIO()
    tifffile.imwrite(reference, values, compression="zlib", predictor=predictor)
    encoded = _write(values, dict(compression="deflate", predictor=predictor),
                     streaming=streaming)
    for data in (reference.getvalue(), encoded):
        with tifffile.TiffFile(io.BytesIO(data)) as tf:
            assert tf.pages[0].predictor == 1
            _assert_bits_equal(tf.pages[0].asarray(), values)


_TIFFFILE_LERC_READ = """
import sys, tifffile
a = tifffile.imread(sys.argv[1])
sys.stdout.buffer.write(a.astype(a.dtype.newbyteorder("<")).tobytes())
"""


@pytest.mark.parametrize("dtype", ["u1", "u2", "i4", "f8"])
@pytest.mark.parametrize("planar_config", [1, 2])
@pytest.mark.parametrize("byte_order", ["<", ">"])
def test_lerc_big_endian_files_read_as_tifffile_reads_them(tmp_path, dtype, planar_config,
                                                          byte_order):
    """A LERC blob in a big-endian file holds each sample word in file
    order. The reader took those words as native values and returned
    wrong values; tifffile swaps them back. tifffile runs in its own
    process, because two LERC libraries cannot share one."""
    import subprocess
    import sys
    import opencodecs as oc
    if not oc.has_codec("lerc"):
        pytest.skip("opencodecs LERC backend not available")
    rng = np.random.default_rng(53)
    if dtype == "f8":
        values = rng.random((19, 23, 3)) * 1000
        # Swapped words that are NaN patterns are refused (next test).
        values[np.isnan(values.byteswap())] = 1.0
    else:
        values = rng.integers(0, np.iinfo(dtype).max, (19, 23, 3)).astype(dtype)
    encoded = _write(values, dict(compression="lerc", planar_config=planar_config,
                                  rows_per_strip=4, verify=True), byte_order=byte_order)
    with TiffStream(encoded) as r:
        np.testing.assert_array_equal(r.read(), values)
    path = tmp_path / "lerc.tif"
    path.write_bytes(encoded)
    proc = subprocess.run([sys.executable, "-c", _TIFFFILE_LERC_READ, str(path)],
                          capture_output=True)
    if proc.returncode and b"imagecodecs" in proc.stderr:
        pytest.skip("tifffile has no LERC decoder here")
    assert proc.returncode == 0, proc.stderr.decode()[-400:]
    theirs = np.frombuffer(proc.stdout, dtype=np.dtype(dtype).newbyteorder("<"))
    if planar_config == 2:
        theirs = np.moveaxis(theirs.reshape(3, 19, 23), 0, 2)
    np.testing.assert_array_equal(theirs.reshape(values.shape), values)


@pytest.mark.parametrize("planar_config", [1, 2])
@pytest.mark.parametrize("shape", [(19, 23), (19, 23, 3)])
def test_lerc_refuses_floats_it_cannot_store_in_a_swapped_file(planar_config, shape):
    """A file in the other byte order hands LERC byte-swapped floats; LERC
    does not keep NaN patterns, so the writer stored other values than it
    was given (2D float32), raised a LercError, or with planar_config=2
    wrote wrong values silently."""
    import sys
    import opencodecs as oc
    if not oc.has_codec("lerc"):
        pytest.skip("opencodecs LERC backend not available")
    native, swapped = ("<", ">") if sys.byteorder == "little" else (">", "<")
    values = np.random.default_rng(59).random(shape).astype("f4")
    assert np.isnan(values.byteswap()).any()
    with pytest.raises(TiffWriterError, match="NaN patterns"):
        _write(values, dict(compression="lerc", planar_config=planar_config,
                            rows_per_strip=4), byte_order=swapped)
    # The same values in a file of the machine's byte order are stored exactly.
    with TiffStream(_write(values, dict(compression="lerc", planar_config=planar_config,
                                        rows_per_strip=4, verify=True),
                           byte_order=native)) as r:
        _assert_bits_equal(r.read(), values)


def test_reader_refuses_complex_layouts_it_cannot_read():
    """libtiff has no predictor 2 for 128-bit samples; complex samples of
    another width have no numpy type. Both used to read as integers or
    raise an unrelated error."""
    from types import SimpleNamespace
    from opencodecs._tiff_codec import TiffPage

    def page(bps, predictor):
        values = {256: 2, 257: 1, 258: bps, 259: 1, 277: 1, 278: 1, 273: 0,
                  279: 2 * bps // 8, 317: predictor, 339: 6}
        return TiffPage(SimpleNamespace(_byte_order="<"), -1,
                        {tag: (0, 1, value) for tag, value in values.items()})

    raw = bytes(range(32))
    plain = page(128, 1)._decode_segment_pixels(raw, 0, 0)
    assert plain.dtype == np.complex128
    np.testing.assert_array_equal(plain, np.frombuffer(raw, "<c16").reshape(1, 2))
    with pytest.raises(NotImplementedError, match="predictor 2"):
        page(128, 2)._decode_segment_pixels(raw, 0, 0)
    with pytest.raises(NotImplementedError, match="complex"):
        page(32, 1)
