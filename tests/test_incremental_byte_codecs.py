"""Independent parity and bounded output for genuine stateful byte sessions."""

import bz2
import gzip
import lzma
import zlib

import numpy as np
import pytest

from opencodecs.core.streaming import decode_chunks, encode_chunks


CODECS = ["deflate", "gzip", "bz2", "lzma", "brotli", "lz4", "zstd"]


def encode_reference(name, data):
    if name == "deflate":
        return zlib.compress(data)
    if name == "gzip":
        return gzip.compress(data)
    if name == "bz2":
        return bz2.compress(data)
    if name == "lzma":
        return lzma.compress(data)
    if name == "brotli":
        return pytest.importorskip("brotli").compress(data)
    if name == "lz4":
        return pytest.importorskip("imagecodecs").lz4f_encode(data)
    return pytest.importorskip("imagecodecs").zstd_encode(data)


def decode_reference(name, data):
    if name == "deflate":
        return zlib.decompress(data)
    if name == "gzip":
        return gzip.decompress(data)
    if name == "bz2":
        return bz2.decompress(data)
    if name == "lzma":
        return lzma.decompress(data)
    if name == "brotli":
        return pytest.importorskip("brotli").decompress(data)
    if name == "lz4":
        return pytest.importorskip("imagecodecs").lz4f_decode(data)
    return pytest.importorskip("imagecodecs").zstd_decode(data, out=4 * 1024 * 1024)


@pytest.mark.parametrize("name", CODECS)
@pytest.mark.parametrize("feed_size", [1, 137, 65536])
def test_incremental_decode_independent_stream(name, feed_size):
    raw = bytes(range(256)) * 113
    encoded = encode_reference(name, raw)
    chunks = (encoded[start:start + feed_size] for start in range(0, len(encoded), feed_size))
    result = list(decode_chunks(chunks, codec=name, chunk_size=101))
    assert all(0 < len(chunk) <= 101 for chunk in result)
    assert b"".join(result) == raw


@pytest.mark.parametrize("name", CODECS)
@pytest.mark.parametrize("empty", [False, True])
def test_incremental_encode_independent_decoder(name, empty):
    raw = b"" if empty else bytes(range(251)) * 1003
    chunks = (raw[start:start + 793] for start in range(0, len(raw), 793))
    result = list(encode_chunks(chunks, codec=name, chunk_size=127))
    assert all(0 < len(chunk) <= 127 for chunk in result)
    encoded = b"".join(result)
    assert decode_reference(name, encoded) == raw
    assert b"".join(decode_chunks([encoded], codec=name, chunk_size=137)) == raw


@pytest.mark.parametrize("name", CODECS)
def test_truncation_and_output_limit(name):
    raw = b"1234567890" * 1901
    encoded = encode_reference(name, raw)
    with pytest.raises((ValueError, RuntimeError, EOFError)):
        list(decode_chunks([encoded[:-1]], codec=name, chunk_size=97))
    with pytest.raises(ValueError, match="max_output"):
        list(decode_chunks([encoded], codec=name, max_output=len(raw) - 1))
    assert b"".join(decode_chunks([encoded], codec=name, max_output=len(raw))) == raw


@pytest.mark.parametrize("name", ["gzip", "bz2", "lzma", "lz4", "zstd"])
@pytest.mark.parametrize("feed_size", [1, 100000])
def test_concatenated_members(name, feed_size):
    a, b = b"first" * 399, b"second" * 512
    encoded = encode_reference(name, a) + encode_reference(name, b)
    chunks = (encoded[i:i + feed_size] for i in range(0, len(encoded), feed_size))
    assert b"".join(decode_chunks(chunks, codec=name, chunk_size=37)) == a + b


@pytest.mark.parametrize("name", CODECS)
def test_decoder_backpressure_and_early_close(name):
    raw = np.random.default_rng(91).integers(0, 256, 2**20, dtype=np.uint8).tobytes()
    encoded = encode_reference(name, raw)
    consumed = 0

    def source():
        nonlocal consumed
        for start in range(0, len(encoded), 4096):
            consumed += 1
            yield encoded[start:start + 4096]

    iterator = decode_chunks(source(), codec=name, chunk_size=2048)
    first = next(iterator)
    assert raw.startswith(first)
    assert consumed * 4096 < len(encoded)
    before = consumed
    iterator.close()
    assert consumed == before


def test_gzip_zero_padding_and_invalid_parameters():
    encoded = gzip.compress(b"abc") + bytes(7) + gzip.compress(b"def") + bytes(5)
    assert b"".join(decode_chunks([encoded], codec="gzip", chunk_size=1)) == b"abcdef"
    for size in (0, -1):
        with pytest.raises(ValueError):
            list(decode_chunks([], codec="gzip", chunk_size=size))
        with pytest.raises(ValueError):
            list(encode_chunks([], codec="gzip", chunk_size=size))


@pytest.mark.parametrize("name", ["gzip", "bz2", "lzma"])
def test_checksum_failure(name):
    encoded = bytearray(encode_reference(name, bytes(range(256)) * 199))
    # Corrupt compressed payload, leaving the final end marker intact.
    encoded[len(encoded) // 2] ^= 0x55
    with pytest.raises((ValueError, RuntimeError, OSError, EOFError, zlib.error, lzma.LZMAError)):
        list(decode_chunks([encoded], codec=name, chunk_size=93))


@pytest.mark.parametrize("name", ["deflate", "brotli"])
def test_single_stream_rejects_trailing_bytes(name):
    encoded = encode_reference(name, b"hello")
    with pytest.raises(ValueError, match="trailing"):
        list(decode_chunks([encoded + b"junk"], codec=name))


def test_native_state_closed_after_early_close_and_error(monkeypatch):
    import opencodecs.core.streaming as streaming

    closed = []
    original = streaming._decoder

    def tracked(name):
        real = original(name)

        class State:
            def process(self, data, size):
                return real.process(data, size)

            def close(self):
                closed.append(name)
                real.close()
        return State()

    monkeypatch.setattr(streaming, "_decoder", tracked)
    encoded = encode_reference("zstd", bytes(range(256)) * 100)
    iterator = decode_chunks([encoded], codec="zstd", chunk_size=10)
    next(iterator)
    iterator.close()
    assert closed == ["zstd"]
    with pytest.raises(ValueError, match="truncated"):
        list(decode_chunks([encoded[:-1]], codec="zstd"))
    assert closed == ["zstd", "zstd"]
