"""The header sz3, sperr and pcodec prefix to their blobs.

Each compresses an ndarray to a blob that does not record its own
shape or dtype, so each writes a small header in front. They arrived
at the same layout separately and kept three copies of the code for
it; there is one now, parameterized by magic and dimension count.

A header is a compatibility surface, so these check the produced BYTES
against the formats written out literally -- not against the shared
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


@pytest.mark.parametrize("name", sorted(LAYOUTS))
def test_arrays_round_trip(name):
    if not oc.has_codec(name):
        pytest.skip(f"{name} not built")
    a = np.sin(np.arange(4 * 5 * 6) / 3.0).reshape(4, 5, 6).astype(np.float32)
    codec = oc.get_codec(name)
    out = np.asarray(codec.decode(codec.encode(a)))
    assert out.shape == a.shape
    assert out.dtype == a.dtype
