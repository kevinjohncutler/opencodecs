"""blosc2 from many threads at once.

Encode used to select its compressor through blosc1_set_compressor, which
is process-global, and then compress on blosc2's global context; decode and
partial decode used the global context too. Concurrent calls could pick up
another thread's compressor, and all of them queued on one mutex. Each call
now has a context of its own, so concurrent results must equal serial ones.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

blosc2 = pytest.importorskip("opencodecs.codecs._blosc2")

COMPRESSORS = ("zstd", "lz4", "blosclz", "zlib", "lz4hc")


def _inputs():
    rs = np.random.RandomState(0)
    return [rs.randint(0, 4000, 50_000 + 997 * i).astype("u2").tobytes() for i in range(8)]


def test_concurrent_encodes_match_serial_ones():
    data = _inputs()
    jobs = [(d, c, lvl) for d in data for c in COMPRESSORS for lvl in (1, 5)]
    serial = [bytes(blosc2.encode(d, compressor=c, level=lvl)) for d, c, lvl in jobs]
    with ThreadPoolExecutor(8) as pool:
        for _ in range(3):
            parallel = list(pool.map(
                lambda job: bytes(blosc2.encode(job[0], compressor=job[1], level=job[2])), jobs))
            assert parallel == serial


def test_concurrent_decodes_and_partial_decodes():
    data = _inputs()
    chunks = [bytes(blosc2.encode(d, compressor=c, typesize=2)) for d in data for c in COMPRESSORS]
    expected = [d for d in data for _ in COMPRESSORS]
    with ThreadPoolExecutor(8) as pool:
        assert list(pool.map(lambda c: bytes(blosc2.decode(c)), chunks)) == expected
        parts = list(pool.map(lambda c: bytes(blosc2.decode_partial(c, 100, 500)), chunks))
    assert parts == [d[200:1200] for d in expected]


@pytest.mark.parametrize("typesize", [1, 2, 8])
def test_shuffled_chunks_round_trip_at_every_item_size(typesize):
    data = np.arange(30_000, dtype="u2").tobytes()
    for compressor in COMPRESSORS:
        chunk = blosc2.encode(data, compressor=compressor, typesize=typesize, shuffle=True)
        assert bytes(blosc2.decode(chunk)) == data
