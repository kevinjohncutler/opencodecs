"""backend="nvimgcodec": JPEG, JPEG 2000 and HTJ2K on an NVIDIA GPU.

Skipped unless CuPy, nvImageCodec and a CUDA device are all present. The
contract checked here: the GPU path returns what the CPU path returns
(dtype, shape, and for lossless codestreams the same pixels), its encodes
are read back by the CPU decoder (exactly, when lossless), ``out=`` works
with numpy, pinned and CuPy arrays, and anything it cannot do raises
instead of quietly running on the CPU. Foreign decoders read its encodes
in test_foreign_readers.py.
"""
from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc
from opencodecs.backends import available
from opencodecs.core.codec import has_codec
from opencodecs.core.errors import OpenCodecsError

pytestmark = pytest.mark.skipif(
    not available("nvimgcodec"),
    reason="nvImageCodec, CuPy or a CUDA device is missing")

GPU = {"backend": "nvimgcodec"}


def _codec(name):
    if not has_codec(name, op="encode"):
        pytest.skip(f"{name} not built")
    return oc.get_codec(name)


def _image(shape, dtype, seed=0):
    rs = np.random.RandomState(seed)
    info = np.iinfo(dtype)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    base = (np.sin(yy / 9.0) + np.cos(xx / 13.0) + 2) / 4
    if len(shape) == 3:
        base = np.stack([np.roll(base, 7 * i, 1) for i in range(shape[2])], -1)
    span = min(float(info.max) - float(info.min), 60000.0)
    a = info.min + base * span + rs.normal(0, span / 200, base.shape)
    return np.clip(a, info.min, info.max).astype(dtype)


def _psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return np.inf if mse == 0 else 10 * np.log10(peak ** 2 / mse)


LOSSLESS = [
    ((97, 131), np.uint8), ((97, 131), np.uint16), ((97, 131), np.int16),
    ((97, 131, 3), np.uint8), ((97, 131, 3), np.uint16),
    ((97, 131, 4), np.uint8),
]


@pytest.mark.parametrize("codec", ["jpeg2k", "htj2k"])
@pytest.mark.parametrize("shape, dtype", LOSSLESS,
                         ids=[f"{s}-{np.dtype(d).name}" for s, d in LOSSLESS])
def test_lossless_decode_equals_cpu(codec, shape, dtype):
    c = _codec(codec)
    x = _image(shape, dtype)
    blob = c.encode(x)
    cpu = c.decode(blob)
    gpu = c.decode(blob, **GPU)
    assert gpu.dtype == cpu.dtype and gpu.shape == cpu.shape
    np.testing.assert_array_equal(gpu, cpu)


@pytest.mark.parametrize("codec", ["jpeg2k", "htj2k"])
@pytest.mark.parametrize("shape, dtype", LOSSLESS,
                         ids=[f"{s}-{np.dtype(d).name}" for s, d in LOSSLESS])
def test_lossless_encode_round_trips_on_cpu(codec, shape, dtype):
    c = _codec(codec)
    x = _image(shape, dtype)
    blob = c.encode(x, **GPU)
    np.testing.assert_array_equal(c.decode(blob), x)


def test_jpeg2k_options_on_gpu():
    c = _codec("jpeg2k")
    x = _image((64, 80, 3), np.uint8)
    raw = c.encode(x, codecformat="j2k", **GPU)
    assert raw[:4] == b"\xff\x4f\xff\x51"
    assert c.encode(x, **GPU)[:12] == b"\x00\x00\x00\x0cjP  \r\n\x87\n"
    np.testing.assert_array_equal(c.decode(raw), x)
    # Twelve-bit samples in uint16 come back as stored.
    x12 = (_image((64, 80), np.uint16) >> 4).astype(np.uint16)
    np.testing.assert_array_equal(
        c.decode(c.encode(x12, bitspersample=12), **GPU), x12)
    planar = c.decode(c.encode(x), planar=True, **GPU)
    np.testing.assert_array_equal(planar, np.moveaxis(x, -1, 0))


def test_lossy_jpeg2k_meets_its_psnr_target():
    c = _codec("jpeg2k")
    x = _image((128, 160, 3), np.uint8)
    blob = c.encode(x, level=40, **GPU)
    # nvJPEG2000 aims at the target on one quality layer; it lands
    # within about a dB of it.
    assert _psnr(c.decode(blob), x) >= 38
    assert len(blob) < len(c.encode(x, **GPU))
    # nvJPEG2000 has no rate target, so a ratio is refused, not ignored.
    with pytest.raises(ValueError, match="ratio"):
        c.encode(x, ratio=20, **GPU)
    with pytest.raises(ValueError, match="ratio"):
        c.encode(x, lossless=False, **GPU)


def test_lossy_htj2k_quality():
    c = _codec("htj2k")
    x = _image((128, 160, 3), np.uint8)
    blob = c.encode(x, level=90, **GPU)
    assert _psnr(c.decode(blob), x) >= 35


def test_lossy_jpeg2k_decode_close_to_cpu():
    c = _codec("jpeg2k")
    x = _image((128, 160, 3), np.uint8)
    blob = c.encode(x, level=40)
    diff = np.abs(c.decode(blob, **GPU).astype(int) - c.decode(blob))
    assert diff.max() <= 2


@pytest.mark.parametrize("shape", [(120, 168, 3), (120, 168)])
def test_jpeg_decode_close_to_cpu(shape):
    c = _codec("jpeg")
    x = _image(shape, np.uint8)
    blob = c.encode(x, level=90)
    cpu = c.decode(blob)
    gpu = c.decode(blob, **GPU)
    assert gpu.shape == cpu.shape and gpu.dtype == cpu.dtype
    # IDCT and chroma-upsampling rounding: 3 levels at most, measured.
    assert np.abs(gpu.astype(int) - cpu).max() <= 4


@pytest.mark.parametrize("shape", [(120, 168, 3), (120, 168)])
def test_jpeg_encode_matches_cpu_quality(shape):
    c = _codec("jpeg")
    x = _image(shape, np.uint8)
    gpu = c.encode(x, level=90, **GPU)
    cpu = c.encode(x, level=90)
    assert abs(_psnr(c.decode(gpu), x) - _psnr(c.decode(cpu), x)) < 0.5
    assert 0.8 < len(gpu) / len(cpu) < 1.2
    assert c.encode(x, level=90, subsampling="444", optimize=True, **GPU)


def test_out_numpy_pinned_and_cupy():
    import cupy as cp
    from opencodecs.backends import pinned_empty
    c = _codec("htj2k")
    x = _image((64, 80, 3), np.uint16)
    blob = c.encode(x)
    out = np.empty_like(x)
    assert c.decode(blob, out=out, **GPU) is out
    np.testing.assert_array_equal(out, x)
    pinned = pinned_empty(x.shape, x.dtype)
    assert c.decode(blob, out=pinned, **GPU) is pinned
    np.testing.assert_array_equal(pinned, x)
    device = cp.empty(x.shape, x.dtype)
    assert c.decode(blob, out=device, **GPU) is device
    np.testing.assert_array_equal(device.get(), x)
    with pytest.raises(ValueError, match="shape"):
        c.decode(blob, out=np.empty((64, 80), np.uint16), **GPU)
    with pytest.raises(ValueError, match="dtype"):
        c.decode(blob, out=np.empty(x.shape, np.uint8), **GPU)


def test_encode_takes_a_device_array():
    import cupy as cp
    c = _codec("jpeg2k")
    x = _image((64, 80), np.uint16)
    np.testing.assert_array_equal(c.decode(c.encode(cp.asarray(x), **GPU)), x)


def test_engine_is_made_once_per_device():
    from opencodecs.backends import _nvimgcodec
    c = _codec("jpeg2k")
    blob = c.encode(_image((32, 32), np.uint8))
    c.decode(blob, **GPU)
    engine = _nvimgcodec._engine()
    decoder = engine.decoder("jpeg2k")
    c.decode(blob, **GPU)
    assert _nvimgcodec._engine() is engine
    assert engine.decoder("jpeg2k") is decoder


def test_concurrent_calls_from_threads():
    from concurrent.futures import ThreadPoolExecutor
    j2k, jpg = _codec("jpeg2k"), _codec("jpeg")
    xs = [_image((48 + i, 64), np.uint16, seed=i) for i in range(16)]
    blobs = [j2k.encode(x) for x in xs]
    with ThreadPoolExecutor(8) as pool:
        got = list(pool.map(lambda b: j2k.decode(b, **GPU), blobs))
        enc = list(pool.map(lambda x: j2k.encode(x, **GPU), xs))
        jp = list(pool.map(lambda x: jpg.encode(x, **GPU),
                           [(x >> 8).astype(np.uint8) for x in xs]))
    for x, g, e, j in zip(xs, got, enc, jp):
        np.testing.assert_array_equal(g, x)
        np.testing.assert_array_equal(j2k.decode(e), x)
        assert jpg.decode(j).shape == x.shape


def test_unsupported_input_raises_instead_of_using_the_cpu():
    j2k, ht, jpg = _codec("jpeg2k"), _codec("htj2k"), _codec("jpeg")
    x = _image((32, 48), np.uint8)
    with pytest.raises(ValueError, match="reduce"):
        j2k.decode(j2k.encode(x), reduce=1, **GPU)
    with pytest.raises(ValueError, match="reduce"):
        ht.decode(ht.encode(x), reduce=1, **GPU)
    with pytest.raises(ValueError, match="scale"):
        jpg.decode(jpg.encode(x), scale=0.5, **GPU)
    with pytest.raises(OpenCodecsError):
        jpg.decode(jpg.encode(x, lossless=True), **GPU)
    with pytest.raises(OpenCodecsError, match="12 bits"):
        jpg.decode(jpg.encode(x.astype(np.uint16) * 16), **GPU)
    with pytest.raises(OpenCodecsError):
        ht.decode(ht.encode(np.ones((16, 16), np.float32)), **GPU)
    with pytest.raises(ValueError):
        jpg.encode(x, lossless=True, **GPU)
    with pytest.raises(ValueError):
        ht.encode(x, level=0.01, **GPU)
    with pytest.raises(ValueError):
        j2k.encode(x.astype(np.int8), **GPU)
    with pytest.raises(ValueError):
        j2k.encode(x, colorspace="gray", **GPU)
