"""BC3 alpha, BC4 and BC5 interpolants against the Khronos formulas.

The Khronos Data Format Specification 1.3 defines the BC3 alpha block
(section 18.4) and BC4 unsigned (section 19.1) with the same real-valued
interpolants: with endpoints a0 > a1, code k in 2..7 is
((8 - k) * a0 + (k - 1) * a1) / 7; otherwise codes 2..5 are
((6 - k) * a0 + (k - 1) * a1) / 5, code 6 is 0 and code 7 is 255. The
nearest 8-bit value is the rounded one (a multiple of 1/7 or 1/5 is
never exactly halfway). BC5 is two BC4 channels.

The expected values below are computed from those formulas, for every
endpoint pair and every code, not from any decoder.
"""

from __future__ import annotations

import numpy as np
import pytest

from opencodecs.codecs._bcdec import decode_bc3, decode_bc4, decode_bc5
from _ic_reference import skip_if_old_imagecodecs  # noqa: E402

pytestmark = skip_if_old_imagecodecs


def _spec_palette(a0, a1):
    """(N, 8) expected values for endpoint arrays a0, a1."""
    a0 = np.asarray(a0, np.int64)
    a1 = np.asarray(a1, np.int64)
    pal = np.zeros(a0.shape + (8,), np.int64)
    pal[..., 0] = a0
    pal[..., 1] = a1
    six = a0 > a1
    for k in range(2, 8):
        num = (8 - k) * a0 + (k - 1) * a1
        pal[..., k] = np.where(six, (2 * num + 7) // 14, pal[..., k])
    for k in range(2, 6):
        num = (6 - k) * a0 + (k - 1) * a1
        pal[..., k] = np.where(six, pal[..., k], (2 * num + 5) // 10)
    pal[..., 6] = np.where(six, pal[..., 6], 0)
    pal[..., 7] = np.where(six, pal[..., 7], 255)
    return pal


# Indices 0..7 then 0..7 again, 3 bits each, little-endian in 6 bytes.
_CODES = np.tile(np.arange(8), 2)
_IDX_BITS = sum(int(c) << (3 * i) for i, c in enumerate(_CODES))
_IDX_BYTES = np.frombuffer(_IDX_BITS.to_bytes(6, "little"), np.uint8)


def _alpha_blocks():
    """Every (a0, a1) pair as an 8-byte BC4 / BC3-alpha block."""
    a0, a1 = np.meshgrid(np.arange(256), np.arange(256), indexing="ij")
    a0, a1 = a0.ravel(), a1.ravel()
    blocks = np.empty((a0.size, 8), np.uint8)
    blocks[:, 0] = a0
    blocks[:, 1] = a1
    blocks[:, 2:] = _IDX_BYTES
    return a0, a1, blocks


def _block_pixels(img, n_blocks_x):
    """(H, W[, C]) image of 4x4 blocks to (n_blocks, 16[, C]) in block
    scan order."""
    h, w = img.shape[:2]
    rest = img.shape[2:]
    t = img.reshape(h // 4, 4, n_blocks_x, 4, *rest).swapaxes(1, 2)
    return t.reshape(-1, 16, *rest)


def test_bc4_every_endpoint_pair_rounds():
    a0, a1, blocks = _alpha_blocks()
    want = _spec_palette(a0, a1)[:, _CODES]
    got = decode_bc4(blocks.tobytes(), width=1024, height=1024)
    np.testing.assert_array_equal(_block_pixels(got, 256), want)


def test_bc3_alpha_every_endpoint_pair_rounds_like_bc4():
    a0, a1, alpha = _alpha_blocks()
    blocks = np.zeros((alpha.shape[0], 16), np.uint8)
    blocks[:, :8] = alpha
    blocks[:, 8:10] = (0x1F, 0xF8)   # a fixed BC1 color block
    got = decode_bc3(blocks.tobytes(), width=1024, height=1024)
    alpha_got = _block_pixels(got, 256)[..., 3]
    np.testing.assert_array_equal(alpha_got,
                                  _spec_palette(a0, a1)[:, _CODES])
    bc4 = decode_bc4(alpha.tobytes(), width=1024, height=1024)
    np.testing.assert_array_equal(got[..., 3], bc4)


def test_bc5_channels_round():
    rng = np.random.default_rng(5)
    a = rng.integers(0, 256, (64, 2, 2))
    blocks = np.zeros((64, 16), np.uint8)
    for ch in range(2):
        blocks[:, 8 * ch] = a[:, ch, 0]
        blocks[:, 8 * ch + 1] = a[:, ch, 1]
        blocks[:, 8 * ch + 2:8 * ch + 8] = _IDX_BYTES
    got = decode_bc5(blocks.tobytes(), width=32, height=32)
    px = _block_pixels(got, 8)
    for ch in range(2):
        want = _spec_palette(a[:, ch, 0], a[:, ch, 1])[:, _CODES]
        np.testing.assert_array_equal(px[..., ch], want)


def test_bc3_color_and_alpha_against_imagecodecs():
    """The color block is bit-identical to imagecodecs; the alpha is
    imagecodecs' truncated value or one more, and the one-more cases
    are exactly where the exact value's fraction is at least one half."""
    imagecodecs = pytest.importorskip("imagecodecs")
    a0, a1, alpha = _alpha_blocks()
    blocks = np.random.default_rng(6).integers(
        0, 256, (alpha.shape[0], 16), dtype=np.uint8)
    blocks[:, :8] = alpha
    data = blocks.tobytes()
    ours = decode_bc3(data, width=1024, height=1024)
    ic = imagecodecs.bcn_decode(data, format=imagecodecs.BCN.FORMAT.BC3,
                                shape=(1024, 1024, 4))
    np.testing.assert_array_equal(ours[..., :3], ic[..., :3])
    diff = ours[..., 3].astype(int) - ic[..., 3]
    assert diff.min() == 0 and diff.max() == 1
