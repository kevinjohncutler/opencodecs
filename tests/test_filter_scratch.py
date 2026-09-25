"""Predictor destinations and reusable scratch across layout boundaries."""
import numpy as np
import pytest
from opencodecs._predictor_codec import DeltaCodec, XorCodec, FloatpredCodec
from opencodecs.core.scratch import ScratchBuffer


@pytest.mark.parametrize('codec', [DeltaCodec(), XorCodec()])
@pytest.mark.parametrize('axis', [0, 1, -1])
@pytest.mark.parametrize('distance', [1, 3])
@pytest.mark.parametrize('dtype', ['u1', '<u2', '>u2', 'i4'])
def test_predictor_destination(codec, axis, distance, dtype):
    a = np.arange(7 * 11 * 13).astype(dtype).reshape(7, 11, 13)
    encoded = codec.encode(a, axis=axis, dist=distance)
    storage = np.empty((7, 11, 26), dtype=dtype)
    for out in (np.empty_like(a), storage[..., ::2]):
        got = codec.decode(encoded, dtype=a.dtype, shape=a.shape,
                           axis=axis, dist=distance, out=out)
        assert got is out
        np.testing.assert_array_equal(got, a)


@pytest.mark.parametrize('axis', [0, 1, -1])
@pytest.mark.parametrize('distance', [1, 3])
@pytest.mark.parametrize('dtype', ['f2', 'f4', '>f8'])
def test_float_predictor_scratch(axis, distance, dtype):
    a = np.linspace(-10, 10, 7 * 11 * 13).astype(dtype).reshape(7, 11, 13)
    codec = FloatpredCodec()
    encoded = codec.encode(a, axis=axis, dist=distance)
    scratch = ScratchBuffer()
    out = np.empty_like(a)
    for _ in range(3):
        got = codec.decode(encoded, dtype=a.dtype, shape=a.shape, axis=axis,
                           dist=distance, out=out, scratch=scratch)
        assert got is out
        np.testing.assert_array_equal(got.view('u1'), a.view('u1'))
    assert scratch.capacity == a.nbytes
    with pytest.raises(ValueError, match='overlap'):
        codec.decode(encoded, dtype=a.dtype, shape=a.shape, out=out, scratch=out)


def test_scratch_growth_preserves_exported_lifetime():
    scratch = ScratchBuffer()
    first = scratch.array((3,), 'u2')
    first[:] = [1, 2, 3]
    larger = scratch.array((40,), 'u2')
    larger[:] = 7
    np.testing.assert_array_equal(first, [1, 2, 3])
    scratch.clear()
    assert scratch.capacity == 0
    assert np.all(larger == 7)
