"""Seekable source ownership shared by native container callbacks."""

import threading

from .io import coerce_data_source


class NativeSource:
    """Borrow a source or own a newly opened one, without reading its body.

    Native callbacks use a private cursor. Borrowed file objects keep their
    original position after each read and remain open after this bridge closes.
    A bridge belongs to one native operation at a time.
    """

    def __init__(self, source):
        self.position = 0
        self.error = None
        self.lock = threading.RLock()
        self.closed = False
        self._file = hasattr(source, "read") and hasattr(source, "seek")
        if self._file:
            self.source, self.owned = source, False
            position = source.tell()
            try:
                source.seek(0, 2)
                self.size = source.tell()
            finally:
                source.seek(position)
        elif hasattr(source, "read_at") and hasattr(source, "size"):
            self.source, self.owned, self.size = source, False, int(source.size)
        else:
            self.source, self.owned, self.size = coerce_data_source(source)
        if self.size <= 0:
            self.close()
            raise ValueError("native source requires a known positive size")

    def read_at(self, offset, size):
        with self.lock:
            if self.closed:
                raise ValueError("native source is closed")
            if offset < 0 or offset > self.size or size < 0:
                raise ValueError("native source range is invalid")
            size = min(size, self.size - offset, 65536)
            chunks = []
            remaining = size
            while remaining:
                if self._file:
                    position = self.source.tell()
                    try:
                        self.source.seek(offset)
                        data = self.source.read(remaining)
                    finally:
                        self.source.seek(position)
                else:
                    data = self.source.read_at(offset, remaining)
                if len(data) > remaining or not data:
                    raise EOFError("native source returned an invalid short read")
                chunks.append(data)
                offset += len(data)
                remaining -= len(data)
            if not chunks:
                return b""
            return chunks[0] if len(chunks) == 1 else b"".join(chunks)

    def reset(self):
        self.position = 0
        self.error = None

    def raise_error(self):
        if self.error is not None:
            raise self.error

    def close(self):
        with self.lock:
            if not self.closed:
                self.closed = True
                if self.owned:
                    self.source.close()
                if hasattr(self, "_avif_buffer"):
                    self._avif_buffer = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
