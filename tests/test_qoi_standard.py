"""QOI header colorspace byte, pinned against the QOI specification.

The specification (qoiformat.org) defines header byte 13 as 0 "sRGB with
linear alpha" or 1 "all channels linear", purely informative. opencodecs
writes 0 for RGB and RGBA, as the reference encoder does. imagecodecs
writes 1 for RGBA; that one byte is the only difference, and srgb=False
gives byte parity.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

pytest.importorskip("opencodecs.codecs._qoi")

import opencodecs as oc  # noqa: E402
from opencodecs._qoi_codec import QoiCodec  # noqa: E402
from _ic_reference import skip_if_old_imagecodecs  # noqa: E402

pytestmark = skip_if_old_imagecodecs


def _image(channels):
    rng = np.random.default_rng(channels)
    return rng.integers(0, 256, (13, 21, channels), dtype=np.uint8)


@pytest.mark.parametrize("channels", [3, 4])
def test_header_follows_spec(channels):
    image = _image(channels)
    encoded = QoiCodec().encode(image)
    magic, width, height, chans, colorspace = struct.unpack(">4sIIBB", encoded[:14])
    assert (magic, width, height, chans) == (b"qoif", 21, 13, channels)
    assert colorspace == 0
    linear = QoiCodec().encode(image, srgb=False)
    assert linear[13] == 1
    assert linear[:13] == encoded[:13] and linear[14:] == encoded[14:]
    np.testing.assert_array_equal(QoiCodec().decode(encoded), image)


@pytest.mark.parametrize("channels", [3, 4])
def test_matches_imagecodecs_except_informative_byte(channels):
    imagecodecs = pytest.importorskip("imagecodecs")
    image = _image(channels)
    ours = QoiCodec().encode(image)
    theirs = imagecodecs.qoi_encode(image)
    np.testing.assert_array_equal(imagecodecs.qoi_decode(ours), image)
    np.testing.assert_array_equal(QoiCodec().decode(theirs), image)
    if channels == 3:
        assert ours == theirs
    else:
        assert ours[13] == 0 and theirs[13] == 1
        assert ours[:13] == theirs[:13] and ours[14:] == theirs[14:]
        assert QoiCodec().encode(image, srgb=False) == theirs


def test_unknown_option_raises():
    with pytest.raises(TypeError, match="unexpected option"):
        QoiCodec().encode(_image(3), colorspace=1)


def test_decode_rejects_unknown_options():
    # imagecodecs.qoi_decode defines only out=; anything else was dropped.
    image = _image(4)
    encoded = QoiCodec().encode(image)
    with pytest.raises(TypeError):
        QoiCodec().decode(encoded, hasalpha=False)
    np.testing.assert_array_equal(QoiCodec().decode(encoded, numthreads=2), image)


def test_numthreads_is_accepted_on_encode():
    """oc.write(..., numthreads=N) worked for QOI in 0.4.0 and must keep
    working; QOI has no threads to give it to."""
    imagecodecs = pytest.importorskip("imagecodecs")
    image = _image(3)
    expected = QoiCodec().encode(image)
    assert QoiCodec().encode(image, numthreads=4) == expected
    assert oc.write(None, image, format="qoi", numthreads=2) == expected
    assert expected == imagecodecs.qoi_encode(image)


def test_encode_out_none_is_accepted_and_a_buffer_is_refused():
    """0.4.0 accepted out= (imagecodecs defines it) and dropped it; out=None
    must still work, and a real buffer is refused rather than left unwritten."""
    imagecodecs = pytest.importorskip("imagecodecs")
    image = _image(3)
    assert QoiCodec().encode(image, out=None) == imagecodecs.qoi_encode(image)
    for out in (bytearray(1 << 16), 1 << 16):
        with pytest.raises(TypeError, match="out="):
            QoiCodec().encode(image, out=out)
