"""Explicit single-worker scratch ownership without a global pool."""
import math
import numpy as np


class ScratchBuffer:
    """Reuse one growable byte allocation across independent pieces.

    Views are borrowed until the next use. Copy a result before retaining it
    across calls. One worker owns each instance; concurrent use is unsupported.
    Growing replaces storage instead of resizing an exported buffer.
    """
    def __init__(self):
        self._buffer = bytearray()

    @property
    def capacity(self):
        return len(self._buffer)

    def bytes(self, size):
        if size < 0:
            raise ValueError('scratch size must be nonnegative')
        if size > self.capacity:
            self._buffer = bytearray(size)
        return memoryview(self._buffer)[:size]

    def array(self, shape, dtype):
        dtype = np.dtype(dtype)
        if dtype.hasobject:
            raise ValueError('object arrays cannot use byte scratch')
        count = math.prod(shape)
        return np.frombuffer(self.bytes(count * dtype.itemsize), dtype).reshape(shape)

    def clear(self):
        self._buffer = bytearray()
