"""Owned row/pass updates and PNG sessions with bounded pixel storage."""

from dataclasses import dataclass
import io
import os

import numpy as np

from ._write_helpers import binary_destination, write_all


@dataclass(frozen=True)
class RowUpdate:
    """Final pixel samples for one ordinary row or one sparse interlace pass.

    Place pixels at ``image[row, x_start::x_step]``. An interlaced update need
    not contain a whole row. Each update owns its pixels, so advancing the
    iterator cannot overwrite a retained update. ``pass_index`` is zero-based.
    """
    row: int
    x_start: int
    x_step: int
    pass_index: int
    pixels: np.ndarray


class _RowSource:
    def __init__(self, src):
        self.error = None
        self.offset = 0
        self.owned = False
        self.view = None
        self.source = None
        if isinstance(src, (str, os.PathLike)):
            path = os.fspath(src)
            if isinstance(path, str) and path.startswith(("http://", "https://")):
                from .io import coerce_data_source
                self.source, self.owned, _ = coerce_data_source(path)
            else:
                self.source = open(path, "rb")
            self.owned = True
        elif hasattr(src, "read") or hasattr(src, "read_at"):
            self.source = src
        else:
            self.view = memoryview(src).cast("B")

    def read_exact(self, size):
        if self.view is not None:
            stop = self.offset + size
            if stop > len(self.view):
                raise EOFError("truncated PNG input")
            data = bytes(self.view[self.offset:stop])
            self.offset = stop
            return data
        chunks = []
        left = size
        while left:
            if hasattr(self.source, "read_at"):
                piece = self.source.read_at(self.offset, left)
            else:
                piece = self.source.read(left)
            if not piece or len(piece) > left:
                raise EOFError("PNG source returned a short or invalid read")
            chunks.append(piece)
            self.offset += len(piece)
            left -= len(piece)
        return bytes(chunks[0]) if len(chunks) == 1 else b"".join(chunks)

    def close(self):
        if self.owned and self.source is not None:
            self.source.close()
        if self.view is not None:
            self.view.release()
            self.view = None


class _RowDestination:
    def __init__(self, dest):
        self.dest = dest
        self.error = None

    def write(self, data):
        write_all(self.dest, data)


class PngRowReader:
    """Context-managed row iterator exposing shape, dtype and interlace mode."""

    def __init__(self, src):
        from ..codecs._png import RowDecoder
        self._source = _RowSource(src)
        self._decoder = None
        self.closed = False
        try:
            self._decoder = RowDecoder(self._source)
            info = self._decoder.info
            self.shape = info["shape"]
            self.dtype = info["dtype"]
            self.interlaced = info["interlaced"]
        except BaseException:
            self.close()
            raise

    def __iter__(self):
        return self

    def __next__(self):
        if self.closed:
            raise StopIteration
        try:
            item = self._decoder.read_row()
        except BaseException:
            self.close()
            raise
        if item is None:
            self.close()
            raise StopIteration
        return RowUpdate(*item)

    def close(self):
        if not self.closed:
            self.closed = True
            if self._decoder is not None:
                self._decoder.close()
            self._source.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        if hasattr(self, "closed"):
            self.close()


def decode_png_rows(src):
    """Read PNG through bounded scanlines, with explicit Adam7 pass updates.

    The returned context manager exposes shape, dtype and interlaced metadata.
    Ordinary images yield complete rows in order. Adam7 images yield sparse
    row updates in file pass order, with explicit horizontal coordinates.
    No full image is allocated internally. Native dictionary and metadata
    storage remain additional codec working memory.
    """
    return PngRowReader(src)


def encode_png_rows(rows, *, shape, dtype, dest=None, **options):
    """Write sequential PNG rows without retaining a source image.

    The declared shape is required before writing the header. File destinations
    receive compressed chunks immediately; dest=None intentionally retains the
    encoded result. Interlaced encoding is not part of this sequential contract.
    """
    from ..codecs._png import RowEncoder
    memory = io.BytesIO() if dest is None else None
    with binary_destination(memory if memory is not None else dest) as stream:
        encoder = RowEncoder(_RowDestination(stream), shape, dtype, **options)
        try:
            for row in rows:
                encoder.write_row(row)
            encoder.finish()
        finally:
            encoder.close()
    return memory.getvalue() if memory is not None else None
