"""Byte-stream codecs against their specifications and imagecodecs.

Each test pins one behavior to a reference that is not opencodecs' own
code: imagecodecs 2026.8, the stdlib zlib and gzip modules, the xxhash
package, a third-party Snappy framing implementation, pydicom, or a
stream assembled by hand from the published format description.
"""

from __future__ import annotations

import gzip
import struct
import zlib

import numpy as np
import pytest

import opencodecs as oc


def _ic():
    return pytest.importorskip("imagecodecs")


A = b"hello world " * 5000
B = bytes(range(256)) * 300


# ---------------------------------------------------------------------------
# zstd: RFC 8878 section 3, data is one or more frames
# ---------------------------------------------------------------------------

_zstd = pytest.importorskip("opencodecs.codecs._zstd")


def _zstd_skippable(payload, nibble=0):
    """RFC 8878 section 3.1.2: magic 0x184D2A5?, 4-byte size, user data."""
    return struct.pack("<II", 0x184D2A50 | nibble, len(payload)) + payload


def test_zstd_concatenated_frames_decode_to_their_concatenation():
    ic = _ic()
    cat = ic.zstd_encode(A) + ic.zstd_encode(B)
    assert _zstd.decode(cat) == A + B
    assert oc.get_codec("zstd").decode(cat) == A + B
    assert _zstd.decode(_zstd.encode(A) + _zstd.encode(B, level=19)) == A + B


@pytest.mark.parametrize("where", ["before", "between", "after", "only"])
def test_zstd_skippable_frames_are_skipped(where):
    ic = _ic()
    skip = _zstd_skippable(b"metadata", nibble=7)
    fa, fb = ic.zstd_encode(A), ic.zstd_encode(B)
    data, want = {
        "before": (skip + fa, A),
        "between": (fa + skip + fb, A + B),
        "after": (fa + skip, A),
        "only": (skip + _zstd_skippable(b""), b""),
    }[where]
    assert _zstd.decode(data) == want


def test_zstd_multi_frame_into_caller_buffers():
    ic = _ic()
    cat = ic.zstd_encode(A) + _zstd_skippable(b"x") + ic.zstd_encode(B)
    assert _zstd.decode(cat, out=len(A) + len(B)) == A + B
    buf = bytearray(len(A) + len(B) + 10)
    assert bytes(_zstd.decode(cat, out=buf)) == A + B
    with pytest.raises(_zstd.ZstdError):
        _zstd.decode(cat, out=len(A))


def test_zstd_frame_without_content_size_among_sized_frames():
    """A streaming-written frame records no size; the grow path takes it."""
    from opencodecs.core.streaming import encode_chunks
    ic = _ic()
    unsized = b"".join(encode_chunks([B], codec="zstd"))
    assert ic.zstd_decode(unsized, out=len(B)) == B          # valid frame
    assert _zstd.decode(ic.zstd_encode(A) + unsized) == A + B


def test_zstd_streaming_decode_handles_several_frames():
    from opencodecs.core.streaming import decode_chunks
    ic = _ic()
    cat = _zstd_skippable(b"abc") + ic.zstd_encode(A) + ic.zstd_encode(B)
    pieces = [cat[i:i + 1000] for i in range(0, len(cat), 1000)]
    assert b"".join(decode_chunks(pieces, codec="zstd")) == A + B


def test_zstd_data_that_is_not_a_frame_still_raises():
    ic = _ic()
    with pytest.raises(_zstd.ZstdError, match="not a zstd frame"):
        _zstd.decode(b"definitely not zstd")
    with pytest.raises(_zstd.ZstdError, match="offset"):
        _zstd.decode(ic.zstd_encode(A) + b"trailing garbage")
    with pytest.raises(_zstd.ZstdError):
        _zstd.decode(ic.zstd_encode(A)[:-3])


def test_zstd_negative_levels_reach_libzstd():
    """Negative levels are libzstd's fast modes (zstd.h, ZSTD_minCLevel).

    imagecodecs clamps them to 0, its default; opencodecs passes them
    through, so the output is a real fast-mode frame, larger than the
    default's on compressible data, and imagecodecs still decodes it.
    """
    ic = _ic()
    rng = np.random.default_rng(1)
    data = np.cumsum(rng.integers(-2, 3, 1 << 20)).astype(np.uint8).tobytes()
    fast = _zstd.encode(data, level=-5)
    assert len(fast) > len(_zstd.encode(data))
    assert ic.zstd_decode(fast) == data
    assert ic.zstd_encode(data, level=-5) == ic.zstd_encode(data)


# ---------------------------------------------------------------------------
# LZ4 frames: concatenated and skippable frames (LZ4 Frame Format spec)
# ---------------------------------------------------------------------------

_lz4 = pytest.importorskip("opencodecs.codecs._lz4")


def _lz4_skippable(payload):
    return struct.pack("<II", 0x184D2A50, len(payload)) + payload


def test_lz4_concatenated_frames_all_decode():
    ic = _ic()
    cat = ic.lz4f_encode(A) + ic.lz4f_encode(B)
    assert _lz4.decode(cat) == A + B
    assert oc.get_codec("lz4").decode(cat) == A + B
    with_skip = _lz4_skippable(b"meta") + ic.lz4f_encode(A) + _lz4_skippable(b"") \
        + ic.lz4f_encode(B)
    assert _lz4.decode(with_skip) == A + B


def test_lz4_concatenated_frames_into_caller_buffers():
    ic = _ic()
    cat = ic.lz4f_encode(A) + ic.lz4f_encode(B)
    assert _lz4.decode(cat, out=len(A) + len(B)) == A + B
    buf = bytearray(len(A) + len(B))
    assert bytes(_lz4.decode(cat, out=buf)) == A + B
    with pytest.raises(_lz4.Lz4Error, match="too small"):
        _lz4.decode(cat, out=len(A) + 10)


def test_lz4_unsized_frames_concatenated():
    from opencodecs.core.streaming import encode_chunks
    ic = _ic()
    fa = b"".join(encode_chunks([A], codec="lz4"))
    fb = b"".join(encode_chunks([B], codec="lz4"))
    assert ic.lz4f_decode(fa) == A
    assert _lz4.decode(fa + fb) == A + B


def test_lz4_trailing_bytes_and_truncation_raise():
    ic = _ic()
    frame = ic.lz4f_encode(A)
    with pytest.raises(_lz4.Lz4Error):
        _lz4.decode(frame + b"not a frame")
    with pytest.raises(_lz4.Lz4Error):
        _lz4.decode(frame[:-5])


def test_lz4_bare_block_matches_imagecodecs():
    ic = _ic()
    assert _lz4.block_decode(ic.lz4_encode(A), len(A)) == A
    with pytest.raises(_lz4.Lz4Error):
        _lz4.block_decode(ic.lz4_encode(A), len(A) - 1)


def test_xxh32_matches_the_reference():
    # Published XXH32 values (the xxhash package's own examples).
    assert _lz4.xxh32(b"") == 0x02CC5D05
    assert _lz4.xxh32(b"a") == 0x550D7456
    assert _lz4.xxh32(b"abc") == 0x32D153FF
    xxhash = pytest.importorskip("xxhash")
    rng = np.random.default_rng(3)
    for n in (1, 3, 4, 15, 16, 17, 31, 32, 33, 1000, 65536):
        data = rng.integers(0, 256, n, dtype=np.uint8).tobytes()
        for seed in (0, 1, 0x9747B28C, 0xFFFFFFFF):
            assert _lz4.xxh32(data, seed) == xxhash.xxh32_intdigest(data, seed)


def test_lz4block_concatenated_streams_and_trailing_bytes():
    ic = _ic()

    def stream(data):
        packed = ic.lz4_encode(data)
        check = _lz4.xxh32(data, 0x9747B28C) & 0x0FFFFFFF
        return (b"LZ4Block" + bytes([0x26]) + struct.pack(
            "<iii", len(packed), len(data), check) + packed
            + b"LZ4Block" + bytes([0x16]) + bytes(12))

    a, b = A[:65536], B[:40000]
    assert _lz4.lz4block_decode(stream(a) + stream(b)) == a + b
    with pytest.raises(_lz4.Lz4Error, match="no block header"):
        _lz4.lz4block_decode(stream(a) + b"junk")


# ---------------------------------------------------------------------------
# deflate: raw= selects RFC 1951 (bare DEFLATE), default RFC 1950 (zlib)
# ---------------------------------------------------------------------------

_deflate = pytest.importorskip("opencodecs.codecs._deflate")


@pytest.mark.parametrize("level", [None, 1, 6, 9])
def test_deflate_raw_encode_is_bare_deflate(level):
    enc = _deflate.encode(A + B, level=level, raw=True)
    assert zlib.decompress(enc, -15) == A + B          # stdlib, wbits=-15
    assert enc[:1] != b"\x78"
    codec = oc.get_codec("deflate")
    assert zlib.decompress(codec.encode(A, level=level, raw=True), -15) == A
    assert zlib.decompress(codec.encode(A, level=level)) == A
    if _deflate.backend() == "libdeflate":
        ic = _ic()
        assert enc == ic.deflate_encode(A + B, level=level, raw=True)


def test_deflate_raw_decode_reads_bare_deflate():
    packer = zlib.compressobj(6, zlib.DEFLATED, -15)
    raw = packer.compress(A + B) + packer.flush()
    assert _deflate.decode(raw, raw=True) == A + B
    codec = oc.get_codec("deflate")
    assert codec.decode(raw, raw=True) == A + B
    assert codec.decode(raw, raw=True, out=len(A + B)) == A + B
    buf = bytearray(len(A + B))
    assert bytes(codec.decode(raw, raw=True, out=buf)) == A + B
    with pytest.raises(Exception):
        _deflate.decode(raw)                        # not a zlib stream
    with pytest.raises(Exception):
        _deflate.decode(zlib.compress(A), raw=True)  # not bare DEFLATE


def test_deflate_raw_is_refused_with_isal():
    codec = oc.get_codec("deflate")
    with pytest.raises(ValueError, match="raw=True"):
        codec.encode(A, raw=True, backend="isal")
    with pytest.raises(ValueError, match="raw=True"):
        codec.decode(b"", raw=True, backend="isal")


def test_deflate_zlib_fallback_paths():
    """The paths a build without libdeflate takes, checked by the stdlib."""
    raw = _deflate._fallback_encode(A, 6, 1)
    assert zlib.decompress(raw, -15) == A
    assert _deflate._fallback_raw_decode(raw) == A
    member = _deflate._fallback_encode(A, 9, 2)
    assert member[:10] == bytes.fromhex("1f8b08000000000002ff")
    assert gzip.decompress(member) == A
    with pytest.raises(_deflate.ZlibError):
        _deflate._fallback_raw_decode(raw[:-4])


# ---------------------------------------------------------------------------
# gzip: one deterministic RFC 1952 header on every platform
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level,xfl", [(None, 0), (0, 4), (1, 4), (6, 0),
                                       (8, 2), (9, 2), (12, 2)])
def test_gzip_header_is_fixed(level, xfl):
    out = oc.get_codec("gzip").encode(A, level=level)
    # ID1 ID2 CM=8 FLG=0 MTIME=0 XFL OS=255 (RFC 1952 section 2.3.1).
    assert out[:10] == bytes([0x1F, 0x8B, 8, 0, 0, 0, 0, 0, xfl, 255])
    assert gzip.decompress(out) == A
    if _deflate.backend() == "libdeflate":
        assert out == _ic().gzip_encode(A, level=level)


def test_gzip_still_decodes_every_member():
    member = oc.get_codec("gzip").encode(A)
    assert oc.get_codec("gzip").decode(member + gzip.compress(B)) == A + B


def test_omezarr_writer_gzip_chunks_are_deterministic():
    from opencodecs._omezarr_writer import _encode_chunk
    chunk = _encode_chunk(A, "gzip", None)
    assert chunk == oc.get_codec("gzip").encode(A)
    assert chunk[4:8] == bytes(4)                   # no timestamp


# ---------------------------------------------------------------------------
# LZW and PackBits: self-terminating streams need no size (TIFF 6.0 s9, s13)
# ---------------------------------------------------------------------------

_tiff = pytest.importorskip("opencodecs.codecs._tiff")


def _corpus():
    rng = np.random.default_rng(7)
    return {
        "zeros_1M": bytes(1 << 20),
        "runs": b"".join(bytes([v]) * int(n) for v, n in zip(
            rng.integers(0, 256, 2000), rng.integers(1, 400, 2000))),
        "text": b"The quick brown fox jumps over the lazy dog. " * 600,
        "random": rng.integers(0, 256, 4096, dtype=np.uint8).tobytes(),
    }


@pytest.mark.parametrize("name", list(_corpus()))
def test_lzw_and_packbits_decode_without_a_size(name):
    ic = _ic()
    data = _corpus()[name]
    assert _tiff.lzw_decode(ic.lzw_encode(data)) == data
    assert _tiff.packbits_decode(ic.packbits_encode(data)) == data
    assert _tiff.lzw_decode(ic.lzw_encode(data), len(data)) == data
    assert _tiff.packbits_decode(ic.packbits_encode(data), len(data)) == data


def test_lzw_guess_that_fills_exactly_is_not_a_truncation():
    """The decoder stops, successfully, once its buffer is full.

    For 1384 zero bytes imagecodecs writes 62 bytes, and a code ends
    exactly at 8 x 62 = 496 output bytes, the size the old guess picked:
    it returned 496 bytes and called that the strip.
    """
    ic = _ic()
    data = bytes(1384)
    enc = ic.lzw_encode(data)
    assert len(enc) == 62
    assert len(_tiff.lzw_decode(enc, 8 * len(enc))) == 8 * len(enc)
    assert _tiff.lzw_decode(enc) == data


def test_lzw_and_packbits_edge_inputs_match_imagecodecs():
    ic = _ic()
    assert _tiff.lzw_decode(b"") == ic.lzw_decode(b"") == b""
    assert _tiff.packbits_decode(b"") == ic.packbits_decode(b"") == b""
    for bad in (b"\x05ab", b"\xfe"):         # run past the end of input
        with pytest.raises(Exception):
            ic.packbits_decode(bad)
        with pytest.raises(Exception):
            _tiff.packbits_decode(bad)


@pytest.mark.parametrize("codec", ["lzw", "packbits"])
def test_segment_helpers_decode_without_a_size(codec):
    ic = _ic()
    from opencodecs.core.segment_compression import decode_segment
    from opencodecs.core.verification import verify_segment
    raw = np.zeros((512, 512), np.uint16).tobytes()
    enc = ic.lzw_encode(raw) if codec == "lzw" else ic.packbits_encode(raw)
    assert bytes(decode_segment(enc, codec)) == raw
    verify_segment(raw, enc, codec)


# ---------------------------------------------------------------------------
# Snappy framing format (framing_format.txt), which owns .sz
# ---------------------------------------------------------------------------

_snappy = pytest.importorskip("opencodecs.codecs._snappy")
STREAM_ID = b"\xff\x06\x00\x00sNaPpY"


def _masked(data):
    c = _snappy.crc32c(data)
    return (((c >> 15) | (c << 17)) + 0xA282EAD8) & 0xFFFFFFFF


def _chunk(kind, body):
    return bytes([kind]) + len(body).to_bytes(3, "little") + body


def _data_chunk(data, compressed):
    body = _ic().snappy_encode(data) if compressed else data
    return _chunk(0x00 if compressed else 0x01,
                  _masked(data).to_bytes(4, "little") + body)


def test_crc32c_check_value():
    # The CRC-32C check value (RFC 3720 appendix B.4; the CRC catalog).
    assert _snappy.crc32c(b"123456789") == 0xE3069283


def test_snappy_framed_hand_built_stream():
    stream = (STREAM_ID + _data_chunk(A[:60000], True)
              + _chunk(0xFE, bytes(5))                   # padding
              + _chunk(0x80, b"skippable")               # reserved, skippable
              + _data_chunk(B[:1000], False)
              + STREAM_ID                                # concatenation
              + _data_chunk(B[1000:65536 + 1000], True))
    want = A[:60000] + B[:1000] + B[1000:65536 + 1000]
    assert _snappy.framed_decode(stream) == want
    assert oc.get_codec("snappy_framed").decode(stream) == want


def test_snappy_framed_rejects_what_the_spec_rejects():
    good = STREAM_ID + _data_chunk(A[:5000], True)
    bad_crc = bytearray(good)
    bad_crc[14] ^= 1
    with pytest.raises(_snappy.SnappyError, match="checksum"):
        _snappy.framed_decode(bytes(bad_crc))
    assert _snappy.framed_decode(bytes(bad_crc), verify=False) == A[:5000]
    with pytest.raises(_snappy.SnappyError, match="unskippable"):
        _snappy.framed_decode(good + _chunk(0x02, b"x"))
    with pytest.raises(_snappy.SnappyError, match="stream identifier"):
        _snappy.framed_decode(_data_chunk(A[:10], False))
    with pytest.raises(_snappy.SnappyError, match="truncated"):
        _snappy.framed_decode(good[:-3])
    with pytest.raises(_snappy.SnappyError, match="65536"):
        _snappy.framed_decode(STREAM_ID + _data_chunk(bytes(65537), False))


@pytest.mark.parametrize("size", [0, 1, 65536, 65537, 300000])
def test_snappy_framed_encode_reads_back_by_spec(size):
    rng = np.random.default_rng(size)
    data = (bytes(size // 2)
            + rng.integers(0, 256, size - size // 2, dtype=np.uint8).tobytes())
    enc = _snappy.framed_encode(data)
    assert enc[:10] == STREAM_ID
    # Walk the chunks by hand: each holds <= 65536 bytes and a valid CRC.
    pos, out = 10, b""
    while pos < len(enc):
        kind, n = enc[pos], int.from_bytes(enc[pos + 1:pos + 4], "little")
        crc, body = enc[pos + 4:pos + 8], enc[pos + 8:pos + 4 + n]
        chunk = _ic().snappy_decode(body) if kind == 0 else body
        assert kind in (0, 1) and len(chunk) <= 65536
        assert int.from_bytes(crc, "little") == _masked(chunk)
        out += chunk
        pos += 4 + n
    assert out == data


def test_snappy_framed_against_cramjam():
    cramjam = pytest.importorskip("cramjam")
    data = A + B
    assert _snappy.framed_decode(bytes(cramjam.snappy.compress(data))) == data
    assert bytes(cramjam.snappy.decompress(_snappy.framed_encode(data))) == data


def test_sz_files_read_as_the_framing_format(tmp_path):
    path = tmp_path / "x.sz"
    path.write_bytes(STREAM_ID + _data_chunk(A[:3000], True))
    assert bytes(oc.read(str(path))) == A[:3000]
    raw = oc.get_codec("snappy").encode(A)
    assert raw == _ic().snappy_encode(A)            # raw block parity kept


# ---------------------------------------------------------------------------
# blosc2: c-blosc2's filter and split codes, imagecodecs' keywords
# ---------------------------------------------------------------------------

_blosc2 = pytest.importorskip("opencodecs.codecs._blosc2")
U16 = (np.arange(40000, dtype=np.uint16) * 7 % 1000).astype(np.uint16)


@pytest.mark.parametrize("shuffle", [None, True, False, 0, 1, 2, "noshuffle",
                                     "shuffle", "bitshuffle"])
# lz4 is left out of the byte comparison: imagecodecs 2026.8 bundles
# c-blosc2 3.x, whose shuffled lz4 blocks differ from 2.x's at some levels.
# Both decode either way (test_blosc2_lz4_cross_decodes).
@pytest.mark.parametrize("compressor,level", [("zstd", 1), ("zstd", 9),
                                              ("blosclz", 9)])
def test_blosc2_chunks_equal_imagecodecs(shuffle, compressor, level):
    ic = _ic()
    mine = _blosc2.encode(U16.tobytes(), typesize=2, shuffle=shuffle,
                          compressor=compressor, level=level)
    theirs = ic.blosc2_encode(U16, shuffle=shuffle, compressor=compressor,
                              level=level)
    assert mine == theirs
    assert ic.blosc2_decode(mine) == U16.tobytes()
    codec = oc.get_codec("blosc2")
    assert codec.encode(U16, shuffle=shuffle, compressor=compressor,
                        level=level) == theirs


@pytest.mark.parametrize("shuffle", [None, "noshuffle", "bitshuffle"])
@pytest.mark.parametrize("level", [1, 5, 9])
def test_blosc2_lz4_cross_decodes(shuffle, level):
    ic = _ic()
    mine = _blosc2.encode(U16, compressor="lz4", level=level, shuffle=shuffle)
    theirs = ic.blosc2_encode(U16, compressor="lz4", level=level,
                              shuffle=shuffle)
    assert ic.blosc2_decode(mine) == U16.tobytes()
    assert _blosc2.decode(theirs) == U16.tobytes()
    assert mine[21] == theirs[21]                    # same filter code


@pytest.mark.parametrize("splitmode", [None, 1, 2, 3, 4, "always", "never",
                                       "auto", "forward", False])
def test_blosc2_splitmode_equals_imagecodecs(splitmode):
    ic = _ic()
    mine = _blosc2.encode(U16, splitmode=splitmode, level=9)
    assert mine == ic.blosc2_encode(U16, splitmode=splitmode, level=9)


def test_blosc2_blocksize_and_threads():
    ic = _ic()
    big = np.tile(U16, 20)
    mine = _blosc2.encode(big, blocksize=16384)
    assert mine == ic.blosc2_encode(big, blocksize=16384)
    # Threads may store blocks in another order; the data is the same.
    threaded = _blosc2.encode(big, blocksize=16384, numthreads=4)
    assert ic.blosc2_decode(threaded) == big.tobytes()
    assert struct.unpack_from("<i", threaded, 8)[0] == 16384
    # Blosc header: nbytes, blocksize, cbytes as int32 at offset 4.
    nbytes, blocksize, cbytes = struct.unpack_from("<iii", mine, 4)
    assert (nbytes, blocksize, cbytes) == (big.nbytes, 16384, len(mine))


def test_blosc2_bitshuffle_filter_byte_and_aliases():
    for name, code in [("bit", 2), ("bitshuffle", 2), ("byte", 1),
                       ("none", 0), ("noshuffle", 0), (2, 2)]:
        chunk = _blosc2.encode(U16.tobytes(), typesize=2, shuffle=name)
        # c-blosc2 README_CHUNK_FORMAT: filters at bytes 16..21 of the
        # extended header; the shuffle slot is the last.
        assert chunk[21] == code, name
    with pytest.raises(ValueError):
        _blosc2.encode(b"abc", shuffle="sideways")
    with pytest.raises(ValueError):
        _blosc2.encode(b"abc", shuffle=7)
    with pytest.raises(ValueError):
        _blosc2.encode(b"abc", splitmode="sometimes")


# ---------------------------------------------------------------------------
# brotli: imagecodecs' default level and its mode / lgwin keywords
# ---------------------------------------------------------------------------

_brotli = pytest.importorskip("opencodecs.codecs._brotli")


def test_brotli_default_equals_imagecodecs():
    ic = _ic()
    data = A + B
    assert _brotli.encode(data) == ic.brotli_encode(data)
    assert _brotli.encode(data) == _brotli.encode(data, level=4)
    assert oc.get_codec("brotli").encode(data) == ic.brotli_encode(data)


@pytest.mark.parametrize("mode,lgwin", [(0, 22), (1, 16), (2, 24), ("text", 10),
                                        (None, 30), (None, 1)])
def test_brotli_mode_and_lgwin_equal_imagecodecs(mode, lgwin):
    ic = _ic()
    data = A + B
    mine = oc.get_codec("brotli").encode(data, level=5, mode=mode, lgwin=lgwin)
    ic_mode = {"text": 1}.get(mode, mode)
    assert mine == ic.brotli_encode(data, level=5, mode=ic_mode, lgwin=lgwin)
    assert ic.brotli_decode(mine) == data


def test_brotli_unknown_mode_raises():
    with pytest.raises(ValueError):
        _brotli.encode(b"abc", mode="poetry")


# ---------------------------------------------------------------------------
# DICOM RLE: interleaved (Planar Configuration 0) output, documented
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_dicomrle_layout_relative_to_imagecodecs_and_pydicom(dtype):
    ic = _ic()
    rng = np.random.default_rng(2)
    img = rng.integers(0, 50, (5, 7, 3)).astype(dtype)
    codec = oc.get_codec("dicomrle")
    enc = codec.encode(img)
    got = codec.decode(enc, shape=img.shape, dtype=dtype)
    assert np.array_equal(got, img)
    # imagecodecs returns the segment order: planar (C, H, W).
    planar = np.frombuffer(ic.dicomrle_decode(enc, dtype), dtype=dtype)
    assert np.array_equal(planar, got.transpose(2, 0, 1).ravel())
    rle = pytest.importorskip("pydicom.pixels.decoders.rle")
    bits = 8 * np.dtype(dtype).itemsize
    frame = rle._rle_decode_frame(enc, 5, 7, 3, bits)
    planes = np.frombuffer(bytes(frame), dtype=np.dtype(dtype).newbyteorder("<"))
    assert np.array_equal(planes, got.transpose(2, 0, 1).ravel())


# ---------------------------------------------------------------------------
# LZ4: many frames, and imagecodecs.lz4f_encode's keywords
# ---------------------------------------------------------------------------


def test_lz4_many_sized_frames_decode_in_one_pass():
    # imagecodecs records the content size in every frame (FLG bit 3),
    # the case that used to recurse once per frame: quadratic work, and
    # a C stack overflow (a crash, in a worker thread first) at about
    # ten thousand frames.
    import threading
    ic = _ic()
    pieces = [bytes([i % 251]) * (64 + i % 7) for i in range(20000)]
    frames = [ic.lz4f_encode(p) for p in pieces]
    assert all(f[4] & 0x08 for f in frames[:3])
    data = b"".join(frames)
    want = b"".join(pieces)
    assert _lz4.decode(data) == want
    got = []
    worker = threading.Thread(target=lambda: got.append(_lz4.decode(data)))
    worker.start()
    worker.join()
    assert got == [want]
    # A sized frame, an unsized one and a skippable one, in any order.
    from opencodecs.core.streaming import encode_chunks
    unsized = b"".join(encode_chunks([B], codec="lz4"))
    mixed = frames[0] + unsized + _lz4_skippable(b"x") + frames[1] + unsized
    assert _lz4.decode(mixed) == pieces[0] + B + pieces[1] + B


@pytest.mark.parametrize("kw", [
    {}, {"blocksizeid": 0}, {"blocksizeid": 4}, {"blocksizeid": 5},
    {"blocksizeid": 6}, {"blocksizeid": 7}, {"contentchecksum": True},
    {"contentchecksum": False}, {"blockchecksum": True},
    {"blocksizeid": 5, "contentchecksum": True, "blockchecksum": True},
])
def test_lz4_encode_keywords_equal_imagecodecs(kw):
    ic = _ic()
    data = (A + B) * 8                       # 3.6 MB: several blocks
    got = oc.get_codec("lz4").encode(data, **kw)
    assert got == ic.lz4f_encode(data, **kw)
    assert ic.lz4f_decode(got) == data
    # LZ4 Frame Format: FLG bit 4 block checksum, bit 2 content
    # checksum; BD bits 6-4 the block maximum size code.
    flg, bd = got[4], got[5]
    assert bool(flg & 0x10) == bool(kw.get("blockchecksum"))
    assert bool(flg & 0x04) == bool(kw.get("contentchecksum"))
    assert (bd >> 4) & 7 == (kw.get("blocksizeid") or 4)


@pytest.mark.parametrize("bsid", [1, 3, 8, -1])
def test_lz4_reserved_block_size_codes_raise(bsid):
    # The frame format reserves codes 0-3 in the header; imagecodecs
    # writes 1 and 3 into it unchecked.
    with pytest.raises(ValueError, match="blocksizeid"):
        _lz4.encode(A, blocksizeid=bsid)


# ---------------------------------------------------------------------------
# Snappy: .sz files that opencodecs 0.4.0 wrote as raw blocks
# ---------------------------------------------------------------------------

# oc.write("x.sz", b"hello snappy " * 100) under opencodecs 0.4.0: a raw
# Snappy block, equal to imagecodecs.snappy_encode of the same bytes.
LEGACY_SZ = bytes.fromhex(
    "940a3068656c6c6f20736e6170707920"
    + "fe0d00" * 20 + "0d0d")


def test_sz_files_from_040_still_read(tmp_path):
    payload = b"hello snappy " * 100
    assert LEGACY_SZ == _ic().snappy_encode(payload)
    path = tmp_path / "legacy.sz"
    path.write_bytes(LEGACY_SZ)
    assert bytes(oc.read(str(path))) == payload
    path.write_bytes(_ic().snappy_encode(A))
    assert bytes(oc.read(str(path))) == A
    path.write_bytes(b"\x00")                       # 0.4.0's empty file
    assert bytes(oc.read(str(path))) == b""
    codec = oc.get_codec("snappy_framed")
    buf = bytearray(len(payload))
    assert codec.decode(LEGACY_SZ, out=buf) is buf and buf == payload
    with pytest.raises(_snappy.SnappyError, match="stream identifier"):
        codec.decode(b"\x00 not snappy at all")


def test_sz_files_are_now_written_in_the_framing_format(tmp_path):
    path = tmp_path / "new.sz"
    oc.write(str(path), np.frombuffer(A, np.uint8))
    written = path.read_bytes()
    assert written[:10] == STREAM_ID
    cramjam = pytest.importorskip("cramjam")
    assert bytes(cramjam.snappy.decompress(written)) == A


# ---------------------------------------------------------------------------
# out= on encode (and gzip decode), and unknown keywords
# ---------------------------------------------------------------------------

_OUT_CODECS = {
    "zstd": "zstd", "lz4": "lz4f", "deflate": "deflate", "gzip": "gzip",
    "brotli": "brotli", "blosc2": "blosc2", "snappy": "snappy",
    "snappy_framed": None,
}


@pytest.mark.parametrize("name", list(_OUT_CODECS))
def test_encode_out_follows_imagecodecs(name):
    ic = _ic()
    codec = oc.get_codec(name)
    want = codec.encode(B)
    if _OUT_CODECS[name] is not None:
        icenc = getattr(ic, _OUT_CODECS[name] + "_encode")
        # imagecodecs' contract, shown on its own encoder: a roomy
        # buffer gets a memoryview of the written prefix, the bytearray
        # type gets a bytearray. (Some of its encoders want room for the
        # worst case, not just the result, hence the generous buffer.)
        roomy = bytearray(2 * len(B) + 1024)
        assert isinstance(icenc(B, out=roomy), memoryview)
        assert type(icenc(B, out=bytearray)) is bytearray
    roomy = bytearray(len(want) + 100)
    got = codec.encode(B, out=roomy)
    assert isinstance(got, memoryview) and bytes(got) == want
    assert bytes(roomy[:len(want)]) == want
    exact = bytearray(len(want))
    assert codec.encode(B, out=exact) is exact and exact == want
    npbuf = np.zeros(len(want) + 7, np.uint8)
    assert bytes(codec.encode(B, out=npbuf)) == want
    assert npbuf[:len(want)].tobytes() == want
    got = codec.encode(B, out=bytearray)
    assert type(got) is bytearray and got == want
    assert codec.encode(B, out=len(want)) == want
    with pytest.raises(ValueError):
        codec.encode(B, out=len(want) - 1)
    with pytest.raises(ValueError):
        codec.encode(B, out=bytearray(len(want) - 1))
    with pytest.raises(ValueError):
        codec.encode(B, out=bytes(len(want)))       # read-only
    with pytest.raises(ValueError, match="not both"):
        codec.encode(B, out=roomy, dest=io_sink())
    assert codec.decode(want) == B


def io_sink():
    import io
    return io.BytesIO()


# imagecodecs' decoder for each codec's output, for the ndarray out= check.
_DECODE_IC = {
    "zstd": "zstd", "lz4": "lz4f", "deflate": "deflate", "gzip": "gzip",
    "brotli": "brotli", "blosc2": "blosc2", "snappy": "snappy",
    "snappy_framed": None,
}


def _prefix_of(result, buffer, nbytes):
    """True when ``result`` is a uint8 ndarray view of ``buffer[:nbytes]``."""
    return (isinstance(result, np.ndarray) and result.dtype == np.uint8
            and result.shape == (nbytes,)
            and result.ctypes.data == buffer.ctypes.data)


@pytest.mark.parametrize("name", list(_OUT_CODECS))
def test_ndarray_out_returns_an_ndarray_like_imagecodecs(name):
    """A numpy out= that the result does not fill gives back an ndarray.

    imagecodecs returns ``out[:n]``, an ndarray view, where a bytearray
    or memoryview out= gives a memoryview; checked on its own codec
    first, then required of ours, on encode and on decode.
    """
    ic = _ic()
    codec = oc.get_codec(name)
    want = codec.encode(B)
    room = 2 * len(B) + 1024        # imagecodecs wants worst-case room
    if _OUT_CODECS[name] is not None:
        icbuf = np.zeros(room, np.uint8)
        icenc = getattr(ic, _OUT_CODECS[name] + "_encode")(B, out=icbuf)
        assert _prefix_of(icenc, icbuf, len(icenc))
    if _DECODE_IC[name] is not None:
        icbuf = np.zeros(len(B) + 9, np.uint8)
        icdec = getattr(ic, _DECODE_IC[name] + "_decode")(want, out=icbuf)
        assert _prefix_of(icdec, icbuf, len(B))
        assert icdec.tobytes() == B
    npbuf = np.zeros(room, np.uint8)
    got = codec.encode(B, out=npbuf)
    assert _prefix_of(got, npbuf, len(want)) and got.tobytes() == want
    exact = np.zeros(len(want), np.uint8)
    assert codec.encode(B, out=exact) is exact and exact.tobytes() == want
    npbuf = np.zeros(len(B) + 9, np.uint8)
    got = codec.decode(want, out=npbuf)
    assert _prefix_of(got, npbuf, len(B)) and got.tobytes() == B
    exact = np.zeros(len(B), np.uint8)
    assert codec.decode(want, out=exact) is exact and exact.tobytes() == B
    # bytearray and memoryview destinations still return a memoryview.
    for buf in (bytearray(len(B) + 9), memoryview(bytearray(len(B) + 9))):
        got = codec.decode(want, out=buf)
        assert isinstance(got, memoryview) and bytes(got) == B
    # Wider dtypes, which imagecodecs refuses, are written bytewise.
    wide = np.zeros(len(B) // 8 + 2, np.float64)
    got = codec.decode(want, out=wide)
    assert _prefix_of(got, wide, len(B)) and got.tobytes() == B
    assert wide.view(np.uint8)[:len(B)].tobytes() == B


def test_gzip_decode_out():
    ic = _ic()
    enc = ic.gzip_encode(A)
    codec = oc.get_codec("gzip")
    buf = bytearray(len(A) + 9)
    got = codec.decode(enc, out=buf)
    assert bytes(got) == A and bytes(buf[:len(A)]) == A
    assert bytes(ic.gzip_decode(enc, out=bytearray(len(A) + 9))) == A
    exact = bytearray(len(A))
    assert codec.decode(enc, out=exact) is exact and exact == A
    assert codec.decode(enc, out=len(A)) == A
    with pytest.raises(ValueError):
        codec.decode(enc, out=len(A) - 1)
    # Multi-member files still decode whole into a caller buffer.
    two = gzip.compress(A) + gzip.compress(B)
    buf = bytearray(len(A) + len(B))
    assert codec.decode(two, out=buf) is buf and buf == A + B


@pytest.mark.parametrize("name", list(_OUT_CODECS))
def test_unknown_keywords_raise(name):
    codec = oc.get_codec(name)
    enc = codec.encode(A)
    with pytest.raises(TypeError, match="bogus_keyword"):
        codec.encode(A, bogus_keyword=1)
    with pytest.raises(TypeError, match="bogus_keyword"):
        codec.decode(enc, bogus_keyword=1)
    with pytest.raises(TypeError, match="bogus_keyword"):
        oc.write(None, np.frombuffer(A, np.uint8), format=name,
                 bogus_keyword=1)


# ---------------------------------------------------------------------------
# blosc2: the native encoder's default typesize for typed buffers
# ---------------------------------------------------------------------------


def _blosc2_inputs():
    rng = np.random.default_rng(31)
    base = rng.integers(0, 40, 24000)
    return {
        "bytes": base.astype(np.uint8).tobytes(),
        "bytearray": bytearray(base.astype(np.uint8).tobytes()),
        "u1 1-D": base.astype(np.uint8),
        "u1 2-D": base.astype(np.uint8).reshape(120, 200),
        "u1 strided": base.astype(np.uint8)[::2],
        "bool 1-D": base.astype(bool),
        "bool 2-D": base.astype(bool).reshape(120, 200),
        "i1 1-D": base.astype(np.int8),
        "u2 1-D": base.astype(np.uint16),
        "u2 2-D": base.astype(np.uint16).reshape(120, 200),
        "f4 3-D": base.astype(np.float32).reshape(10, 12, 200),
        "c16 1-D": base.astype(np.complex128),
        "memoryview u1": memoryview(base.astype(np.uint8)),
        "memoryview u2": memoryview(base.astype(np.uint16)),
    }


@pytest.mark.parametrize("key", list(_blosc2_inputs()))
def test_blosc2_default_typesize_matches_imagecodecs(key):
    """The default typesize, and so the chunk, is imagecodecs' for every input.

    imagecodecs 2026.8.16 writes 8 for a flat run of unsigned bytes
    (bytes, bytearray, a contiguous 1-D uint8 or bool array) and the
    buffer's item size otherwise: 1 for a 2-D uint8, an int8 or a
    strided array.
    """
    ic = _ic()
    data = _blosc2_inputs()[key]
    want = ic.blosc2_encode(data)
    native = _blosc2.encode(data)
    adapter = oc.get_codec("blosc2").encode(data)
    # Blosc2 chunk header: byte 3 is the type size.
    assert native[3] == adapter[3] == want[3]
    assert native == adapter == want
    flat = data.tobytes() if isinstance(data, np.ndarray) else bytes(data)
    assert _blosc2.decode(native) == flat


@pytest.mark.parametrize("dtype", [np.uint16, np.float32, np.float64])
def test_blosc2_native_typesize_from_a_typed_buffer(dtype):
    ic = _ic()
    arr = np.arange(20000).astype(dtype)
    got = _blosc2.encode(arr)
    # Blosc2 chunk header: byte 3 is the type size.
    assert got[3] == arr.dtype.itemsize
    assert got == ic.blosc2_encode(arr)
    assert _blosc2.decode(got) == arr.tobytes()


# ---------------------------------------------------------------------------
# decode out=: the bytearray type, and a buffer the result fills exactly
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(_OUT_CODECS))
def test_decode_out_follows_imagecodecs(name):
    """``out=bytearray`` gives a new bytearray; an exact fill gives ``out``.

    imagecodecs' own decoder is checked first, then ours is held to it,
    for a bytearray, a memoryview and an ndarray destination.
    """
    ic = _ic()
    codec = oc.get_codec(name)
    want = codec.encode(B)
    icdec = (getattr(ic, _DECODE_IC[name] + "_decode")
             if _DECODE_IC[name] is not None else None)
    if icdec is not None:
        got = icdec(want, out=bytearray)
        assert type(got) is bytearray and got == B
        for exact in (bytearray(len(B)), memoryview(bytearray(len(B))),
                      np.zeros(len(B), np.uint8)):
            assert icdec(want, out=exact) is exact
    got = codec.decode(want, out=bytearray)
    assert type(got) is bytearray and got == B
    for exact in (bytearray(len(B)), memoryview(bytearray(len(B))),
                  np.zeros(len(B), np.uint8)):
        assert codec.decode(want, out=exact) is exact
        assert bytes(exact) == B
    # A larger bytearray or memoryview still returns a memoryview of the
    # written prefix, as imagecodecs does.
    roomy = bytearray(len(B) + 9)
    got = codec.decode(want, out=roomy)
    assert isinstance(got, memoryview) and bytes(got) == B
    if icdec is not None:
        assert isinstance(icdec(want, out=bytearray(len(B) + 9)), memoryview)


def test_deflate_raw_decode_out_bytearray_type():
    raw = _ic().deflate_encode(A, raw=True)
    got = oc.get_codec("deflate").decode(raw, raw=True, out=bytearray)
    assert type(got) is bytearray and got == A == zlib.decompress(raw, -15)


# ---------------------------------------------------------------------------
# LZ4 encode level: clamped to -1..12, as imagecodecs.lz4f_encode does
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [-2, -3, -9, -65538, -2**31, 13, 2**31 - 1])
def test_lz4_levels_outside_the_range_equal_imagecodecs(level):
    ic = _ic()
    data = (A + B) * 4
    got = oc.get_codec("lz4").encode(data, level=level)
    assert got == ic.lz4f_encode(data, level=level)
    edge = -1 if level < 0 else 12
    assert got == ic.lz4f_encode(data, level=edge)
    assert ic.lz4f_decode(got) == data


def test_lz4_stream_encoder_clamps_the_level_too():
    data = (A + B) * 4

    from opencodecs.core.streaming import encode_chunks

    def frame(level):
        return b"".join(encode_chunks([data], codec="lz4", level=level))

    assert frame(-5) == frame(-1)
    assert frame(40) == frame(12)
    assert _ic().lz4f_decode(frame(-5)) == data


# ---------------------------------------------------------------------------
# gzip encode on a build without the _deflate extension
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("level", [None, 0, 1, 6, 9, 12, -3])
def test_gzip_stdlib_fallback_writes_the_same_header(monkeypatch, level):
    from opencodecs import _gzip_codec
    monkeypatch.setattr(_gzip_codec, "_HAVE_BACKEND", False)
    out = oc.get_codec("gzip").encode(A, level=level)
    lvl = 6 if level is None else min(max(level, 0), 9)
    xfl = 4 if lvl < 2 else (2 if lvl >= 8 else 0)
    # ID1 ID2 CM=8 FLG=0 MTIME=0 XFL OS=255 (RFC 1952 section 2.3.1).
    assert out[:10] == bytes([0x1F, 0x8B, 8, 0, 0, 0, 0, 0, xfl, 255])
    assert gzip.decompress(out) == A
    body = zlib.compressobj(lvl, zlib.DEFLATED, -15)
    assert out[10:-8] == body.compress(A) + body.flush()
    assert out[-8:] == struct.pack("<II", zlib.crc32(A), len(A))
    # The extension's own zlib fallback writes the same bytes.
    assert out == _deflate._fallback_encode(A, lvl, 2)


def test_gzip_encodes_when_the_deflate_extension_is_missing(tmp_path):
    """A build without _deflate still writes .gz files and OME-Zarr gzip."""
    import subprocess
    import sys
    # Hide the extension from both ways opencodecs loads one: the
    # package's own spec_from_file_location loader and a plain import.
    script = (
        "import sys, gzip, importlib.util\n"
        "NAME = 'opencodecs.codecs._deflate'\n"
        "real = importlib.util.spec_from_file_location\n"
        "def spec(name, *a, **k):\n"
        "    if name == NAME:\n"
        "        raise ImportError('hidden for the test')\n"
        "    return real(name, *a, **k)\n"
        "importlib.util.spec_from_file_location = spec\n"
        "class Hide:\n"
        "    def find_spec(self, name, path=None, target=None):\n"
        "        if name == NAME:\n"
        "            raise ImportError('hidden for the test')\n"
        "sys.meta_path.insert(0, Hide())\n"
        "import numpy as np, opencodecs as oc\n"
        "from opencodecs import _gzip_codec\n"
        "assert not _gzip_codec._HAVE_BACKEND\n"
        "a = np.arange(5000, dtype=np.uint8)\n"
        "enc = oc.get_codec('gzip').encode(a)\n"
        "assert gzip.decompress(enc) == a.tobytes()\n"
        "assert enc[:10] == bytes.fromhex('1f8b08000000000000ff')\n"
        "from opencodecs._omezarr_writer import _encode_chunk\n"
        "assert _encode_chunk(a.tobytes(), 'gzip', None) == enc\n"
        f"oc.write({str(tmp_path / 't.gz')!r}, a)\n"
        f"assert gzip.decompress(open({str(tmp_path / 't.gz')!r}, 'rb').read())"
        " == a.tobytes()\n"
        "print('ok')\n"
    )
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, timeout=300)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().endswith("ok")
