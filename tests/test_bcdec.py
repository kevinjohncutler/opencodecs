"""BC/DDS texture decoder tests.

Cross-validates our bcdec-vendored decoder against imagecodecs's
bcn_decode (which uses the same upstream bcdec implementation but
binds it slightly differently). Both should produce pixel-equal
output for every supported format.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from opencodecs.codecs._bcdec import (
    decode_bc1, decode_bc2, decode_bc3, decode_bc4, decode_bc5, decode_bc7,
    decode_bc6h,
)

imagecodecs = pytest.importorskip("imagecodecs")


def _bc1_white_block():
    """A hand-built BC1 block with both endpoints = white (RGB565 0xFFFF)
    and all 16 indices = 0 → entire block is white."""
    return struct.pack("<HHI", 0xFFFF, 0x0000, 0x00000000)


def _bc1_solid_block(rgb565: int):
    """Single 4x4 block at a chosen RGB565 color, all indices = 0."""
    return struct.pack("<HHI", rgb565, 0x0000, 0x00000000)


def test_bc1_white_block_decodes_to_255():
    out = decode_bc1(_bc1_white_block(), width=4, height=4)
    assert out.shape == (4, 4, 4)
    assert out.dtype == np.uint8
    assert np.all(out[..., 0] == 255)
    assert np.all(out[..., 1] == 255)
    assert np.all(out[..., 2] == 255)
    assert np.all(out[..., 3] == 255)


def test_bc1_red_block_decodes_to_red():
    # RGB565: r=31, g=0, b=0 -> 0xF800
    out = decode_bc1(_bc1_solid_block(0xF800), width=4, height=4)
    # Expect pure red (after 565 -> 888 expansion: r = 31 << 3 | 31 >> 2 = 255)
    assert int(out[0, 0, 0]) == 255
    assert int(out[0, 0, 1]) == 0
    assert int(out[0, 0, 2]) == 0


@pytest.mark.parametrize("fmt_id,fn,block_size,channels", [
    (imagecodecs.BCN.FORMAT.BC1, decode_bc1, 8, 4),
    (imagecodecs.BCN.FORMAT.BC2, decode_bc2, 16, 4),
    (imagecodecs.BCN.FORMAT.BC3, decode_bc3, 16, 4),
    (imagecodecs.BCN.FORMAT.BC7, decode_bc7, 16, 4),
])
def test_decoder_matches_imagecodecs(fmt_id, fn, block_size, channels):
    """For each BC format, decode random block bytes through both
    opencodecs and imagecodecs.bcn_decode; outputs must be pixel-equal.
    """
    rng = np.random.default_rng(0)
    n_blocks_x = 8
    n_blocks_y = 4
    width = n_blocks_x * 4
    height = n_blocks_y * 4
    blocks = rng.integers(
        0, 256, size=n_blocks_x * n_blocks_y * block_size, dtype=np.uint8,
    ).tobytes()
    ours = fn(blocks, width=width, height=height)
    ic = imagecodecs.bcn_decode(
        blocks, format=fmt_id,
        shape=(height, width, channels),
    )
    # imagecodecs sometimes returns (H, W, 3) for BC1 without alpha;
    # our decoder always returns RGBA. Compare just RGB.
    if ic.shape[-1] == 3 and ours.shape[-1] == 4:
        np.testing.assert_array_equal(ours[..., :3], ic)
    else:
        np.testing.assert_array_equal(ours, ic)


def test_bc4_round_trip_random():
    """BC4 (single-channel) decoder produces a valid (H, W) array."""
    rng = np.random.default_rng(1)
    blocks = rng.integers(0, 256, size=2 * 2 * 8, dtype=np.uint8).tobytes()
    out = decode_bc4(blocks, width=8, height=8)
    assert out.shape == (8, 8)
    assert out.dtype == np.uint8


def test_bc5_round_trip_random():
    """BC5 (two-channel) decoder produces a valid (H, W, 2) array."""
    rng = np.random.default_rng(2)
    blocks = rng.integers(0, 256, size=2 * 2 * 16, dtype=np.uint8).tobytes()
    out = decode_bc5(blocks, width=8, height=8)
    assert out.shape == (8, 8, 2)
    assert out.dtype == np.uint8


def test_bc6h_float_output():
    """BC6H decoder produces (H, W, 3) float32 by default."""
    rng = np.random.default_rng(3)
    blocks = rng.integers(0, 256, size=2 * 2 * 16, dtype=np.uint8).tobytes()
    out = decode_bc6h(blocks, width=8, height=8)
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.float32


def test_bc6h_half_output():
    """format='half' returns float16."""
    rng = np.random.default_rng(4)
    blocks = rng.integers(0, 256, size=2 * 2 * 16, dtype=np.uint8).tobytes()
    out = decode_bc6h(blocks, width=8, height=8, format="half")
    assert out.shape == (8, 8, 3)
    assert out.dtype == np.float16


def test_non_multiple_of_4_rejected():
    """Width/height must be multiples of 4 — block geometry is fixed."""
    blocks = b"\x00" * 8
    for fn in (decode_bc1, decode_bc4):
        with pytest.raises(Exception):
            fn(blocks, width=5, height=4)
        with pytest.raises(Exception):
            fn(blocks, width=4, height=7)


def test_short_input_rejected():
    """Too-few bytes for the requested geometry must raise."""
    with pytest.raises(Exception):
        decode_bc1(b"\x00" * 4, width=8, height=8)   # need 4 blocks * 8 bytes
