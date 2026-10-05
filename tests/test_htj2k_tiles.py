"""HTJ2K codestreams of several tiles decode one tile per thread, identically.

A multi-tile codestream is cut into single-tile codestreams that decode in
parallel. Every case here compares that against the same codestream
decoded whole on one thread (``numthreads=1``), and against the source
pixels where the encode is lossless.
"""
from __future__ import annotations

import numpy as np
import pytest

_openjph = pytest.importorskip("opencodecs.codecs._openjph")


def _image(shape, dtype, seed=0):
    rs = np.random.RandomState(seed)
    if np.dtype(dtype).kind == "f":
        return rs.standard_normal(shape).astype(dtype)
    info = np.iinfo(dtype)
    base = np.linspace(info.min // 4, info.max // 4, int(np.prod(shape))).reshape(shape)
    return (base + rs.randint(0, 50, shape)).astype(dtype)


@pytest.mark.parametrize("shape,tile", [
    ((700, 900), (256, 256)),        # edge tiles partial on both axes
    ((1024, 512), (512, 128)),       # non-square tiles
    ((300, 1100), (1024, 256)),      # tiles taller than the image
])
@pytest.mark.parametrize("dtype", [np.uint8, np.uint16, np.int16, np.int32])
def test_tiled_decode_matches_whole_decode(shape, tile, dtype):
    img = _image(shape, dtype)
    cs = _openjph.encode(img, tile=tile)
    assert _openjph._tile_plan(cs) is not None
    whole = _openjph.decode(cs, numthreads=1)
    np.testing.assert_array_equal(whole, img)
    # The tiles really decode on their own (no fallback to the whole decode).
    tiled = _openjph._decode_tiled(cs, 3, None, None)
    assert tiled is not None
    np.testing.assert_array_equal(tiled, whole)
    for numthreads in (None, 2, 7):
        got = _openjph.decode(cs, numthreads=numthreads)
        assert got.dtype == whole.dtype
        np.testing.assert_array_equal(got, whole)


@pytest.mark.parametrize("planar", [None, True, False])
@pytest.mark.parametrize("components,rgb", [(3, None), (3, False), (4, None), (2, None)])
def test_tiled_multicomponent_layouts(planar, components, rgb):
    img = _image((600, 520, components), np.uint16, seed=components)
    cs = _openjph.encode(img, tile=(256, 192), rgb=rgb)
    whole = _openjph.decode(cs, numthreads=1, planar=planar)
    got = _openjph._decode_tiled(cs, 4, planar, None)
    assert got is not None and got.shape == whole.shape
    np.testing.assert_array_equal(got, whole)
    np.testing.assert_array_equal(_openjph.decode(cs, numthreads=4, planar=planar), whole)


def test_tiled_lossy_and_float():
    img = _image((520, 610), np.uint16, seed=3)
    cs = _openjph.encode(img, 0.02, tile=(128, 256))
    np.testing.assert_array_equal(_openjph._decode_tiled(cs, 5, None, None),
                                  _openjph.decode(cs, numthreads=1))
    flt = _image((300, 400), np.float32, seed=4)
    cs = _openjph.encode(flt, tile=(128, 128))
    got = _openjph._decode_tiled(cs, 3, None, None)
    assert got.dtype == np.float32
    np.testing.assert_array_equal(got, flt)


def test_tiled_decode_into_out():
    img = _image((512, 768), np.uint16, seed=5)
    cs = _openjph.encode(img, tile=(256, 256), tlm=True)
    out = np.zeros_like(img)
    assert _openjph.decode(cs, out=out, numthreads=4) is out
    np.testing.assert_array_equal(out, img)
    with pytest.raises(ValueError):
        _openjph.decode(cs, out=np.zeros((10, 10), np.uint16), numthreads=4)


def test_tile_parts_and_reduce_keep_working():
    img = _image((640, 640), np.uint16, seed=6)
    cs = _openjph.encode(img, tile=(256, 256), tilepart=3)
    np.testing.assert_array_equal(_openjph.decode(cs, numthreads=4), img)
    np.testing.assert_array_equal(_openjph.decode(cs, reduce=1, numthreads=4),
                                  _openjph.decode(cs, reduce=1, numthreads=1))


def _outcome(cs, numthreads):
    try:
        return "pixels", _openjph.decode(cs, numthreads=numthreads)
    except Exception as exc:  # noqa: BLE001 - compared below
        return "error", (type(exc), str(exc))


def test_untiled_and_damaged_codestreams_decode_as_before():
    img = _image((300, 300), np.uint16, seed=7)
    assert _openjph._tile_plan(_openjph.encode(img)) is None
    cs = _openjph.encode(img, tile=(128, 128))
    plan = _openjph._tile_plan(cs)
    second = plan[2][1][5][0][0]                   # where tile 1's tile-part starts
    damaged = {
        "truncated": cs[:len(cs) // 2],
        "garbage in a tile": cs[:second + 40] + b"\x55" * 64 + cs[second + 104:],
        "bad tile index": cs[:second + 4] + b"\x7f\x7f" + cs[second + 6:],
        "bad tile-part length": cs[:second + 6] + b"\x00\x00\x00\x05" + cs[second + 10:],
    }
    for name, data in damaged.items():
        whole = _outcome(data, 1)
        tiled = _outcome(data, 4)
        assert whole[0] == tiled[0], name
        if whole[0] == "error":
            assert whole[1] == tiled[1], name
        else:
            np.testing.assert_array_equal(whole[1], tiled[1], err_msg=name)
