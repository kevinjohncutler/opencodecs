"""Truncated Rice streams must be refused, not read past the buffer.

The Rice decompressors read the first pixel unencoded, straight out of
the head of the input: four bytes for the int variant, two for short,
one for byte. cfitsio removed end-of-buffer checking from this code at
v3.08 to gain speed, documenting instead that callers must over-allocate
("a simple rule of thumb ... make it 1% larger"), which is a workable
contract for a library reading its own files and a poor one for a codec
handed arbitrary bytes.

Upstream cfitsio 4.7.0 has since added a `clen < 4` guard to
``fits_rdecomp``, but as of that release ``fits_rdecomp_short`` and
``fits_rdecomp_byte`` still have none: they read two bytes and one byte
respectively before any length is checked. Our vendored copy guards all
three (see 3rdparty/VENDOR.toml and docs/rice_bounds.md), and this test
is what keeps that true.
"""

from __future__ import annotations

import pytest

import opencodecs as oc

pytest.importorskip("opencodecs.codecs._rcomp")
from opencodecs.codecs import _rcomp  # noqa: E402


@pytest.mark.parametrize("bytes_per_pixel,minimum", [(4, 4), (2, 2), (1, 1)])
@pytest.mark.parametrize("length", [0, 1, 2, 3])
def test_short_input_is_refused_not_read_past(bytes_per_pixel, minimum, length):
    """Feed fewer bytes than the first pixel needs.

    Anything is acceptable except reading out of bounds, so the contract
    asserted here is "raises, or returns without crashing", and the real
    value is that this runs under a sanitizer build unchanged.
    """
    if length >= minimum:
        pytest.skip("not a truncated case for this pixel size")
    payload = bytes(length)
    with pytest.raises(Exception):
        _rcomp.decode_raw(payload, nelements=16, blocksize=32,
                          bytes_per_pixel=bytes_per_pixel)


@pytest.mark.parametrize("bytes_per_pixel", [4, 2, 1])
def test_empty_input_is_refused(bytes_per_pixel):
    with pytest.raises(Exception):
        _rcomp.decode_raw(b"", nelements=16, blocksize=32,
                          bytes_per_pixel=bytes_per_pixel)


def test_roundtrip_still_works():
    """The guards must not have cost us the ordinary path."""
    import numpy as np
    codec = oc.get_codec("rcomp")
    a = np.arange(1024, dtype=np.int32) % 517
    assert np.array_equal(codec.decode(codec.encode(a)), a)
