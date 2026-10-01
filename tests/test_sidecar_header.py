"""The header sz3, sperr and pcodec prefixed to their blobs up to 0.4.0.

Each wrote a small private header in front of the library's stream.
They now write the library's own stream instead (see
test_wrapper_streams.py), but blobs written before still have to
open, so the header code stays for reading. It is shared, parameterized
by magic and dimension count.

A header is a compatibility surface, so these check the BYTES against
the formats written out literally -- not against the shared
implementation, which would just be the new code marking its own work.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

import opencodecs as oc

# The layouts exactly as the three copies spelled them, before sharing.
LAYOUTS = {
    "sz3": ("<4sBB2x5Q", b"SZ3O", 48, "Sz3Error"),
    "sperr": ("<4sBB2x5Q", b"SPRR", 48, "SperrError"),
    "pcodec": ("<4sBB2x8Q", b"PCOO", 72, "PcodecError"),
}


def _mod(name):
    return pytest.importorskip(f"opencodecs.codecs._{name}")


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_header_length_is_unchanged(name):
    fmt, magic, length, _ = LAYOUTS[name]
    assert _mod(name)._HEADER_LEN == length
    assert struct.calcsize(fmt) == length


def test_sz3_packs_the_documented_bytes():
    m = _mod("sz3")
    got = m._pack_header(3, 2, [7, 9, 0, 0, 0])
    assert got == struct.pack(LAYOUTS["sz3"][0], b"SZ3O", 3, 2, 7, 9, 0, 0, 0)
    assert m._unpack_header(got) == (3, 2, [7, 9, 0, 0, 0])


def test_pcodec_packs_the_documented_bytes():
    m = _mod("pcodec")
    got = m._pack_header(5, 3, [4, 5, 6, 0, 0, 0, 0, 0])
    assert got == struct.pack(LAYOUTS["pcodec"][0], b"PCOO", 5, 3,
                              4, 5, 6, 0, 0, 0, 0, 0)


def test_sperr_keeps_its_positional_signature():
    """sperr's callers pass three dimensions, not a sequence."""
    m = _mod("sperr")
    got = m._pack_header(1, 3, 8, 9, 10)
    assert got == struct.pack(LAYOUTS["sperr"][0], b"SPRR", 1, 3, 8, 9, 10, 0, 0)
    assert m._unpack_header(got) == (1, 3, 8, 9, 10)


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_a_foreign_magic_raises_this_codecs_own_error(name):
    """Sharing the implementation must not share the exception.

    A sz3 blob handed to sperr should fail as a sperr error saying the
    magic is wrong, not as something generic from a helper module.
    """
    m = _mod(name)
    _, _, _, err_name = LAYOUTS[name]
    bad = b"XXXX" + bytes(m._HEADER_LEN - 4)
    with pytest.raises(Exception) as ei:
        m._unpack_header(bad)
    assert type(ei.value).__name__ == err_name
    assert "magic" in str(ei.value)


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_a_short_buffer_raises_rather_than_slicing_garbage(name):
    m = _mod(name)
    _, _, _, err_name = LAYOUTS[name]
    with pytest.raises(Exception) as ei:
        m._unpack_header(b"\x00" * 4)
    assert type(ei.value).__name__ == err_name
    assert "too short" in str(ei.value)


def test_too_many_dimensions_is_refused_not_truncated():
    """Silently dropping a dimension would corrupt the shape.

    The old copies packed a fixed argument list, so an over-long shape
    was a TypeError from struct. The shared one checks and says which
    limit was exceeded.
    """
    m = _mod("sz3")
    with pytest.raises(Exception, match="exceeds"):
        m._pack_header(3, 6, [1, 2, 3, 4, 5, 6])


# The dtype byte and dimension slots each format used for float32
# (4, 5, 6): sz3 stores (r1, r2, ...) fastest first with SZ_FLOAT = 0,
# sperr stores (dimx, dimy, dimz) with 1 meaning float32, pcodec stores
# the numpy shape with PCO_TYPE_F32 = 5.
LEGACY_FIELDS = {
    "sz3": (0, 3, [6, 5, 4, 0, 0]),
    "sperr": (1, 3, [6, 5, 4, 0, 0]),
    "pcodec": (5, 3, [4, 5, 6, 0, 0, 0, 0, 0]),
}


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_legacy_blobs_still_decode(name):
    """A 0.4.0 blob is this header, built here from the literal layout,
    followed by the library's own stream. It must decode to the shape
    and dtype the header records, with no arguments."""
    if not oc.has_codec(name):
        pytest.skip(f"{name} not built")
    a = np.sin(np.arange(4 * 5 * 6) / 3.0).reshape(4, 5, 6).astype(np.float32)
    codec = oc.get_codec(name)
    fmt, magic, _, _ = LAYOUTS[name]
    dtype_byte, ndim, dims = LEGACY_FIELDS[name]
    legacy = struct.pack(fmt, magic, dtype_byte, ndim, *dims) + bytes(codec.encode(a))
    out = np.asarray(codec.decode(legacy))
    assert out.shape == a.shape
    assert out.dtype == a.dtype
    if name == "pcodec":
        np.testing.assert_array_equal(out, a)
