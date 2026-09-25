"""Wire-format checksum validation, with a portable fallback."""
from functools import lru_cache
import struct


@lru_cache(maxsize=1)
def _crc32c_table():
    table = []
    for value in range(256):
        for _ in range(8):
            value = (value >> 1) ^ (0x82F63B78 if value & 1 else 0)
        table.append(value)
    return table


def crc32c(data):
    """Castagnoli cyclic redundancy check of contiguous bytes."""
    try:
        from ..codecs._bytetools import crc32c as native
    except ImportError:
        table = _crc32c_table()
        value = 0xFFFFFFFF
        for byte in memoryview(data).cast('B'):
            value = table[(value ^ byte) & 255] ^ (value >> 8)
        return value ^ 0xFFFFFFFF
    return native(data)


def strip_crc32c(data):
    """Validate a little-endian checksum trailer before exposing payload."""
    view = memoryview(data).cast('B')
    if len(view) < 4:
        raise ValueError('truncated CRC32C payload')
    payload = view[:-4]
    expected = struct.unpack_from('<I', view, len(view) - 4)[0]
    if crc32c(payload) != expected:
        raise ValueError('CRC32C checksum mismatch')
    return payload
