"""backend="imageio": HEIF through Apple's ImageIO (macOS only).

Skipped off macOS. Checked: a file ImageIO decodes comes back in the CPU
path's layout, (H, W, 3) uint8, within a level of libheif for 4:4:4 and
exactly for lossless; files ImageIO should not take (gray, alpha, deep,
cropped, small-tile grids) are decoded by libheif with the same result as
backend=None; the hardware encoder's files decode on the CPU path within
a quality bound, and it refuses what it cannot write. Foreign decoders
read its encodes in test_foreign_readers.py.
"""
from __future__ import annotations

import sys

import numpy as np
import pytest

import opencodecs as oc
from opencodecs.backends import available
from opencodecs.core.codec import has_codec

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin" or not available("imageio"),
    reason="Apple ImageIO is macOS only")

APPLE = {"backend": "imageio"}


@pytest.fixture(scope="module")
def heif():
    if not has_codec("heif", op="encode"):
        pytest.skip("heif not built")
    return oc.get_codec("heif")


def _rgb(h, w, seed=0, noise=2.0):
    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (np.sin(yy / 17.0) + np.cos(xx / 23.0) + 2) * 55 + 10
    base = np.stack([np.roll(base, 9 * i, 1) for i in range(3)], -1)
    return np.clip(base + rs.normal(0, noise, base.shape), 0, 255).astype(np.uint8)


def _psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return np.inf if mse == 0 else 10 * np.log10(255 ** 2 / mse)


def test_lossless_file_decodes_exactly(heif):
    from opencodecs.backends._imageio import route
    x = _rgb(128, 192)
    blob = heif.encode(x)
    assert route(blob)[0] == "imageio"
    got = heif.decode(blob, **APPLE)
    assert got.shape == x.shape and got.dtype == np.uint8
    np.testing.assert_array_equal(got, x)


@pytest.mark.parametrize("level", [50, 90])
def test_lossy_444_within_one_level_of_libheif(heif, level):
    from opencodecs.backends._imageio import route
    blob = heif.encode(_rgb(256, 320), level=level)
    assert route(blob)[0] == "imageio"
    cpu = heif.decode(blob)
    got = heif.decode(blob, **APPLE)
    assert got.shape == cpu.shape and got.dtype == cpu.dtype
    assert np.abs(got.astype(int) - cpu).max() <= 1


def test_out_is_filled_in_place(heif):
    x = _rgb(64, 96)
    blob = heif.encode(x)
    out = np.empty_like(x)
    assert heif.decode(blob, out=out, **APPLE) is out
    np.testing.assert_array_equal(out, x)
    with pytest.raises(ValueError):
        heif.decode(blob, out=np.empty((64, 96, 4), np.uint8), **APPLE)


@pytest.mark.parametrize("make", [
    lambda: np.full((64, 64), 100, np.uint8),                         # gray
    lambda: np.dstack([_rgb(64, 64), np.full((64, 64), 9, np.uint8)]),  # alpha
    lambda: _rgb(64, 64).astype(np.uint16) * 4,                       # 10-bit
    lambda: _rgb(63, 95),                                             # cropped
], ids=["gray", "alpha", "10bit", "clap"])
def test_files_imageio_should_not_take_go_to_libheif(heif, make):
    from opencodecs.backends._imageio import route
    blob = heif.encode(make(), level=90)
    assert route(blob)[0] == "native"
    np.testing.assert_array_equal(heif.decode(blob, **APPLE), heif.decode(blob))


def test_small_tile_grid_goes_to_libheif(heif):
    """ImageIO's own large encodes are 512 x 512 grids, as an iPhone's."""
    from opencodecs.backends._imageio import route
    blob = heif.encode(_rgb(1024, 1280), level=80, **APPLE)
    path, reason = route(blob)
    assert path == "native" and "512" in reason
    np.testing.assert_array_equal(heif.decode(blob, **APPLE), heif.decode(blob))


@pytest.mark.parametrize("shape", [(48, 64), (256, 320), (1024, 1280)])
def test_hardware_encode_decodes_on_cpu(heif, shape):
    x = _rgb(*shape)
    blob = heif.encode(x, level=90, **APPLE)
    assert blob[4:12] == b"ftypheic"
    back = heif.decode(blob)
    assert back.shape == x.shape and back.dtype == np.uint8
    # 4:2:0 on saturated synthetic color: 33 dB at 48 x 64, measured.
    assert _psnr(back, x) >= 30


def test_hardware_encode_refuses_what_it_cannot_write(heif):
    x = _rgb(32, 32)
    with pytest.raises(ValueError, match="level"):
        heif.encode(x, **APPLE)                       # lossless default
    with pytest.raises(ValueError):
        heif.encode(x, level=90, lossless=True, **APPLE)
    with pytest.raises(ValueError):
        heif.encode(x[..., 0].copy(), level=90, **APPLE)
    with pytest.raises(ValueError):
        heif.encode(x.astype(np.uint16), level=90, **APPLE)
    with pytest.raises(ValueError):
        heif.encode(x, level=90, bit_depth=10, **APPLE)


def test_read_and_write_pass_the_backend(heif, tmp_path):
    x = _rgb(64, 96)
    path = tmp_path / "x.heic"
    oc.write(path, x, level=90, backend="imageio")
    assert _psnr(oc.read(path, backend="imageio"), x) >= 35
