"""DataSource ABC + read_many + range coalescing tests.

Pure unit tests on the core/io.py primitives. Integration with the
real HTTPDataSource / FileDataSource is covered by test_tiff_http.py
and test_hdf5_http.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from opencodecs.core.io import DataSource, coalesce_ranges


# ---------------------------------------------------------------------------
# coalesce_ranges
# ---------------------------------------------------------------------------


def test_coalesce_merges_adjacent():
    merged, splits = coalesce_ranges(
        [(0, 100), (100, 100), (200, 100)],
        max_gap=0, max_combined=1_000_000,
    )
    assert merged == [(0, 300)]
    # Each input is one slice into the single merged blob.
    assert splits == [
        [(0, 0, 100)],
        [(0, 100, 200)],
        [(0, 200, 300)],
    ]


def test_coalesce_respects_gap_limit():
    merged, splits = coalesce_ranges(
        [(0, 100), (200, 100)],  # gap of 100
        max_gap=50,
        max_combined=1_000_000,
    )
    assert merged == [(0, 100), (200, 100)]
    assert splits == [
        [(0, 0, 100)],
        [(1, 0, 100)],
    ]


def test_coalesce_bridges_small_gap():
    merged, splits = coalesce_ranges(
        [(0, 100), (200, 100)],
        max_gap=100,  # gap is exactly 100
        max_combined=1_000_000,
    )
    assert merged == [(0, 300)]


def test_coalesce_respects_max_combined():
    merged, _ = coalesce_ranges(
        [(0, 600), (600, 600)],
        max_gap=0,
        max_combined=1000,  # combined would be 1200 > 1000
    )
    assert merged == [(0, 600), (600, 600)]


def test_coalesce_handles_unsorted_input():
    merged, splits = coalesce_ranges(
        [(200, 100), (0, 100), (100, 100)],
        max_gap=0,
        max_combined=1_000_000,
    )
    assert merged == [(0, 300)]
    # splits MUST be returned in original input order.
    assert splits[0][0] == (0, 200, 300)  # (200, 100)
    assert splits[1][0] == (0, 0, 100)    # (0, 100)
    assert splits[2][0] == (0, 100, 200)  # (100, 100)


def test_coalesce_empty_input():
    merged, splits = coalesce_ranges([])
    assert merged == []
    assert splits == []


# ---------------------------------------------------------------------------
# DataSource ABC (default read_many loops read_at)
# ---------------------------------------------------------------------------


class _ByteDataSource(DataSource):
    """Trivial in-memory DataSource for testing the ABC contract."""

    def __init__(self, data: bytes):
        self._data = data
        self.size = len(data)
        self.read_calls = 0

    def read_at(self, offset: int, n: int) -> bytes:
        self.read_calls += 1
        return self._data[offset:offset + n]


def test_data_source_callable_alias_routes_to_read_at():
    ds = _ByteDataSource(b"0123456789")
    assert ds(2, 4) == b"2345"
    assert ds.read_calls == 1


def test_data_source_default_read_many_loops():
    ds = _ByteDataSource(b"0123456789")
    out = ds.read_many([(0, 3), (5, 2), (8, 2)])
    assert out == [b"012", b"56", b"89"]
    assert ds.read_calls == 3


def test_data_source_read_many_empty():
    ds = _ByteDataSource(b"abc")
    assert ds.read_many([]) == []
    assert ds.read_calls == 0


def test_data_source_context_manager_closes():
    closed = []

    class _S(_ByteDataSource):
        def close(self):
            closed.append(True)

    with _S(b"x") as ds:
        ds.read_at(0, 1)
    assert closed == [True]
