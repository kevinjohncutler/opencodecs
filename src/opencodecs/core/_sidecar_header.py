"""The small header the array-compressing codecs prefix to their blobs.

sz3, sperr and pcodec each turn an ndarray into an opaque compressed
blob that does not record its own shape or dtype, so each writes a
header in front. They arrived at the same layout separately -- magic,
dtype byte, ndim byte, padding, then fixed-width dimensions -- and
kept three copies of the code for it. One copy now, parameterized by
the two things that actually differ: the magic and how many dimension
slots the format reserves.
"""

from __future__ import annotations

import struct


def make_sidecar_header(magic: bytes, n_slots: int, error: type, label: str):
    """Build ``(header_len, pack, unpack)`` for one codec's header.

    ``magic`` is the 4-byte identifier, ``n_slots`` how many uint64
    dimensions are reserved, ``error`` the exception type to raise and
    ``label`` the name to put in its message. Each codec keeps its own
    magic and its own exception, which is the point: a sz3 blob handed
    to sperr should fail as a sperr error saying the magic is wrong,
    not as a generic one.
    """
    if len(magic) != 4:
        raise ValueError(f"{label}: magic must be 4 bytes, got {magic!r}")
    fmt = f"<4sBB2x{n_slots}Q"
    header_len = struct.calcsize(fmt)

    def pack(dtype_enum, ndim, shape) -> bytes:
        dims = [int(s) for s in shape]
        if len(dims) > n_slots:
            raise error(
                f"{label}: {len(dims)} dimensions exceeds the {n_slots} "
                f"this header reserves")
        dims += [0] * (n_slots - len(dims))
        return struct.pack(fmt, magic, int(dtype_enum) & 0xFF,
                           int(ndim) & 0xFF, *dims)

    def unpack(buf):
        if len(buf) < header_len:
            raise error(f"{label} blob too short to contain header")
        got_magic, dtype_enum, ndim, *dims = struct.unpack(
            fmt, bytes(buf[:header_len]))
        if got_magic != magic:
            raise error(f"{label} blob has wrong magic {got_magic!r}")
        return dtype_enum, ndim, dims

    return header_len, pack, unpack


__all__ = ["make_sidecar_header"]
