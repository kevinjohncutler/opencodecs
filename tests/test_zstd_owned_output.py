"""Owned encoded views retain bytes while avoiding the final payload copy."""
import gc
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from opencodecs.codecs import _zstd


@pytest.mark.parametrize("threads", [None, 0, 1, 2])
def test_zstandard_owned_output_matches_public_bytes(threads):
    rng = np.random.default_rng(928)
    results = []
    for count in (0, 1, 4096, 256 << 10, 1 << 20):
        source = rng.integers(0, 256, count, dtype="u1")
        expected = _zstd.encode(source, numthreads=threads)
        actual = _zstd.encode_buffer(source, numthreads=threads)
        assert isinstance(expected, bytes)
        assert isinstance(actual, memoryview)
        assert actual.readonly and actual.contiguous
        assert bytes(actual) == expected
        assert isinstance(actual.obj, bytes)
        assert actual.obj[len(actual):] == bytes(len(actual.obj) - len(actual))
        results.append((actual, source.tobytes()))
        source.fill(0)
    gc.collect()
    for actual, expected in results:
        assert _zstd.decode(actual) == expected
        with pytest.raises(TypeError):
            actual[0] = 0


def test_zstandard_owned_output_is_independent_between_workers():
    sources = [bytes([i]) * (256 << 10) for i in range(12)]
    with ThreadPoolExecutor(max_workers=4) as executor:
        views = list(executor.map(_zstd.encode_buffer, sources))
    assert [_zstd.decode(view) for view in views] == sources
