"""Explicit incremental byte-codec iterators alongside eager codec calls.

Compressed input is fed to persistent decoder state. Yielded bytes are owned
snapshots and can be retained by callers. Backpressure follows normal iterator
consumption: the next source chunk is not requested until the current chunk
has been processed. Close an abandoned iterator to release codec state.
"""

from __future__ import annotations

import bz2
import importlib
import lzma
import operator
import zlib


_ALIASES = {"zlib": "deflate", "zlibng": "deflate", "bzip2": "bz2", "xz": "lzma"}
_CODECS = {"deflate", "gzip", "bz2", "lzma", "brotli", "lz4", "zstd"}


def _name(codec):
    name = _ALIASES.get(codec.lower(), codec.lower())
    if name not in _CODECS:
        raise NotImplementedError(f"{codec}: incremental bytes are unsupported")
    return name


class _StdlibDecoder:
    def __init__(self, name):
        self.name = name
        if name in ("deflate", "gzip"):
            self.state = zlib.decompressobj(31 if name == "gzip" else 15)
        elif name == "bz2":
            self.state = bz2.BZ2Decompressor()
        else:
            self.state = lzma.LZMADecompressor()

    def process(self, data, size):
        state = self.state
        output = state.decompress(data, size)
        consumed = len(data)
        if self.name in ("gzip", "deflate") and not state.eof:
            consumed -= len(state.unconsumed_tail)
        return consumed, output, state.eof

    def close(self):
        self.state = None


def _decoder(name):
    if name in ("gzip", "deflate", "bz2", "lzma"):
        return _StdlibDecoder(name)
    return importlib.import_module(f"opencodecs.codecs._{name}").StreamDecoder()


def _pieces(chunks, size):
    for chunk in chunks:
        view = memoryview(chunk).cast("B")
        for start in range(0, len(view), size):
            yield view[start:start + size]


def decode_chunks(chunks, *, codec, chunk_size=65536, max_output=None):
    """Yield decoded byte chunks without retaining the complete input/output.

    ``max_output`` limits the total decoded bytes, independently of per-chunk
    size. Multiple members are accepted for gzip, bzip2, XZ, LZ4 and Zstandard.
    Deflate and Brotli accept exactly one stream and reject trailing data.
    Codec dictionaries and windows are additional native working memory.
    """
    name = _name(codec)
    chunk_size = operator.index(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if max_output is not None and max_output < 0:
        raise ValueError("max_output must be nonnegative")
    state = _decoder(name)
    complete = False
    seen = False
    total = 0
    try:
        for data in _pieces(chunks, chunk_size):
            if not data:
                continue
            seen = True
            while True:
                if complete:
                    if not data:
                        break
                    if name in ("deflate", "brotli"):
                        raise ValueError(f"{name}: trailing data after compressed stream")
                    if name == "gzip":
                        # gzip permits zero padding between/after members.
                        leading = 0
                        while leading < len(data) and data[leading] == 0:
                            leading += 1
                        data = data[leading:]
                        if not data:
                            break
                    state.close()
                    state = _decoder(name)
                    complete = False
                limit = chunk_size
                if max_output is not None:
                    limit = min(limit, max_output - total + 1)
                consumed, output, complete = state.process(data, limit)
                if complete and isinstance(state, _StdlibDecoder):
                    data = memoryview(state.state.unused_data)
                else:
                    data = data[consumed:]
                total += len(output)
                if max_output is not None and total > max_output:
                    raise ValueError("decoded output exceeds max_output")
                if output:
                    yield output
                if complete:
                    continue
                if not data and len(output) < limit:
                    break
                if not consumed and not output:
                    if data:
                        raise ValueError(f"{name}: decoder made no progress")
                    break
        if not seen or not complete:
            raise ValueError(f"{name}: truncated compressed stream")
    finally:
        state.close()


def encode_chunks(chunks, *, codec, chunk_size=65536, level=None):
    """Yield one encoded stream while consuming bounded pieces of raw input.

    The wire format matches the eager codec; compressed bytes need not be
    identical because streaming compression can choose different framing.
    No flush is injected at source-chunk boundaries. Finalization is required
    for a complete stream and happens when the iterator is exhausted.
    """
    name = _name(codec)
    chunk_size = operator.index(chunk_size)
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if name in ("gzip", "deflate"):
        state = zlib.compressobj(6 if level is None else int(level),
                                wbits=31 if name == "gzip" else 15)
    elif name == "bz2":
        state = bz2.BZ2Compressor(9 if level is None else int(level))
    elif name == "lzma":
        state = lzma.LZMACompressor(preset=6 if level is None else int(level))
    else:
        state = importlib.import_module(f"opencodecs.codecs._{name}").StreamEncoder(level=level)
    try:
        if name in ("gzip", "deflate", "bz2", "lzma"):
            for chunk in chunks:
                view = memoryview(chunk).cast("B")
                for offset in range(0, len(view), chunk_size):
                    output = state.compress(view[offset:offset + chunk_size])
                    for start in range(0, len(output), chunk_size):
                        yield output[start:start + chunk_size]
            output = state.flush()
            for start in range(0, len(output), chunk_size):
                yield output[start:start + chunk_size]
        else:
            for chunk in chunks:
                view = memoryview(chunk).cast("B")
                while view:
                    consumed, output, _ = state.process(view, chunk_size, False)
                    view = view[consumed:]
                    if output:
                        yield output
                    if not consumed and not output:
                        raise ValueError(f"{name}: encoder made no progress")
            finished = False
            while not finished:
                _, output, finished = state.process(b"", chunk_size, True)
                if output:
                    yield output
    finally:
        if hasattr(state, "close"):
            state.close()
