"""EER indexing is O(1), and accumulation uses more than one core.

EER frames are independent event bitstreams at their own IFD offsets,
so both properties are free from the format and neither was being
taken. is_chunked was already True while random access went through
Reader's default linear walk: measured 1200 ms to reach frame 720 of
721, against 1.62 ms for the same frame through frame(). A flag saying
"indexing is offered" is not the same as one saying "indexing is
cheap".
"""

from __future__ import annotations

import pathlib
import time

import numpy as np
import pytest

import opencodecs as oc

EER = (pathlib.Path(__file__).resolve().parent.parent
       / ".test_data" / "eer" / "empiar10568_falcon4.eer")
needs_corpus = pytest.mark.skipif(
    not EER.is_file(), reason="fetch the EMPIAR-10568 EER corpus entry first")


@pytest.fixture(scope="module")
def reader():
    with oc.get_codec("eer").open(str(EER)) as r:
        yield r


@needs_corpus
def test_indexing_does_not_depend_on_the_index(reader):
    """The last frame must not cost more than the first.

    A ratio rather than an absolute time, so this says something on any
    machine. The linear walk this replaced was 700x, so the bar can sit
    far from both.
    """
    n = reader.n_frames
    reader[0], reader[n - 1]                        # warm

    def timed(i):
        best = min(_elapsed(lambda: reader[i]) for _ in range(3))
        return best

    first, last = timed(0), timed(n - 1)
    assert last < first * 10, (
        f"frame {n - 1} took {last * 1e3:.1f} ms against {first * 1e3:.1f} ms "
        f"for frame 0; random access is walking the file")


def _elapsed(fn):
    t = time.perf_counter()
    fn()
    return time.perf_counter() - t


@needs_corpus
def test_indexing_agrees_with_iteration(reader):
    """O(1) access must return what the walk returned."""
    for i in (0, 1, 7, reader.n_frames - 1):
        assert np.array_equal(reader[i], reader.frame(i))
    for i, frame in enumerate(reader.iter_frames()):
        assert np.array_equal(frame, reader[i])
        if i >= 3:
            break


@needs_corpus
def test_negative_and_out_of_range_indices(reader):
    n = reader.n_frames
    assert np.array_equal(reader[-1], reader.frame(n - 1))
    assert np.array_equal(reader[-n], reader.frame(0))
    for bad in (n, n + 5, -n - 1):
        with pytest.raises(IndexError):
            reader[bad]


@needs_corpus
@pytest.mark.parametrize("numthreads", [None, 1, 2, 4, 8])
def test_threaded_sum_is_bit_identical(reader, numthreads):
    """Per-worker accumulators, so the reduction order changes.

    Integer addition is associative, so it must still be exact. This is
    the test that would catch someone "optimizing" the accumulator to
    float.
    """
    ref = reader.sum(0, 24, dtype=np.uint32, numthreads=1)
    got = reader.sum(0, 24, dtype=np.uint32, numthreads=numthreads)
    assert np.array_equal(got, ref)
    assert got.dtype == ref.dtype


@needs_corpus
def test_weighted_sum_still_works(reader):
    """The weighted path stays serial; it must not have been broken."""
    w = np.linspace(0.5, 1.5, 8)
    out = reader.sum(0, 8, weights=w, dtype=np.float32)
    manual = sum(reader.frame(i) * w[k] for k, i in enumerate(range(8)))
    assert np.allclose(out, manual)


@needs_corpus
def test_sum_range_validation_is_unchanged(reader):
    for bad in ((5, 3), (-1, 4), (0, reader.n_frames + 1)):
        with pytest.raises(ValueError):
            reader.sum(*bad)


@needs_corpus
@pytest.mark.perf
def test_threaded_sum_is_faster(reader):
    reader.sum(0, 60, dtype=np.uint32, numthreads=1)
    serial = min(_elapsed(
        lambda: reader.sum(0, 60, dtype=np.uint32, numthreads=1))
        for _ in range(2))
    threaded = min(_elapsed(
        lambda: reader.sum(0, 60, dtype=np.uint32, numthreads=8))
        for _ in range(2))
    assert serial / threaded > 1.5, f"{serial / threaded:.2f}x on 8 threads"
