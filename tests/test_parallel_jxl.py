"""Tests for opencodecs.parallel — multi-file JXL parallel decode.

Skipped on platforms without the libjxl Cython extension built (Windows
without manual libjxl install). On Mac and Linux with libjxl this
exercises the previously-uncovered 18% module.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc
from opencodecs import parallel


pytestmark = pytest.mark.skipif(
    not parallel._HAVE_BACKEND,
    reason="libjxl backend not available (parallel module is a stub)",
)


def _encode_some_jxl_files(tmp_path, n: int = 4) -> list[str]:
    """Write `n` small JXL files and return their paths."""
    paths = []
    rng = np.random.default_rng(0)
    for i in range(n):
        arr = rng.integers(0, 256, (32, 48, 3), dtype=np.uint8)
        p = tmp_path / f"f{i:02d}.jxl"
        oc.write(str(p), arr, format="jxl", lossless=True)
        paths.append(str(p))
    return paths


# ---------------------------------------------------------------------------
# frame_count
# ---------------------------------------------------------------------------


def test_frame_count_single_frame(tmp_path):
    paths = _encode_some_jxl_files(tmp_path, 1)
    assert parallel.frame_count(paths[0]) == 1


def _write_stack(path, stack):
    """Write an animated multi-frame JXL stack with is_last properly set."""
    with oc.JxlWriter(str(path), animation=True, lossless=True) as w:
        for i, frame in enumerate(stack):
            w.write_frame(frame, is_last=(i == len(stack) - 1))


def test_frame_count_multi_frame(tmp_path):
    """Multi-frame JXL: write a stack via JxlWriter, verify frame_count."""
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 256, (5, 32, 48, 3), dtype=np.uint8)
    p = str(tmp_path / "stack.jxl")
    _write_stack(p, stack)
    assert parallel.frame_count(p) == 5


def test_frame_count_accepts_bytes(tmp_path):
    paths = _encode_some_jxl_files(tmp_path, 1)
    data = Path(paths[0]).read_bytes()
    assert parallel.frame_count(data) == 1


def test_frame_count_accepts_path_object(tmp_path):
    paths = _encode_some_jxl_files(tmp_path, 1)
    assert parallel.frame_count(Path(paths[0])) == 1


# ---------------------------------------------------------------------------
# read_files: ordered parallel decode of N JXL files
# ---------------------------------------------------------------------------


def test_read_files_returns_per_file_arrays(tmp_path):
    """parallel.read_files reads N files concurrently and returns a
    list of decoded arrays in the input order."""
    paths = _encode_some_jxl_files(tmp_path, 4)
    arrays = parallel.read_files(paths, n_workers=4)
    assert len(arrays) == 4
    for arr in arrays:
        assert arr.shape == (32, 48, 3)
        assert arr.dtype == np.uint8


def test_read_files_results_match_serial(tmp_path):
    """Parallel-decoded arrays should match serial decode byte-for-byte
    (lossless JXL → fully deterministic)."""
    paths = _encode_some_jxl_files(tmp_path, 4)
    parallel_results = parallel.read_files(paths, n_workers=4)
    serial_results = [oc.read(p) for p in paths]
    for p_arr, s_arr in zip(parallel_results, serial_results):
        np.testing.assert_array_equal(p_arr, s_arr)


def test_read_files_preserves_input_order(tmp_path):
    """Even though decode completion order is non-deterministic with a
    thread pool, read_files must align results to the input list order."""
    paths = _encode_some_jxl_files(tmp_path, 6)
    parallel_results = parallel.read_files(paths, n_workers=4)
    for i, (p, arr) in enumerate(zip(paths, parallel_results)):
        np.testing.assert_array_equal(arr, oc.read(p))


def test_read_files_with_n_workers_one_works(tmp_path):
    """n_workers=1 should fall back to serial without breaking."""
    paths = _encode_some_jxl_files(tmp_path, 2)
    arrays = parallel.read_files(paths, n_workers=1)
    assert len(arrays) == 2


def test_read_files_empty_input(tmp_path):
    """Decoding an empty list returns an empty list, not an error."""
    arrays = parallel.read_files([], n_workers=4)
    assert arrays == []


def test_read_files_default_n_workers(tmp_path):
    """n_workers=None uses CPU-count default and still returns correctly."""
    paths = _encode_some_jxl_files(tmp_path, 3)
    arrays = parallel.read_files(paths)
    assert len(arrays) == 3


# ---------------------------------------------------------------------------
# iter_files: streaming decode (completion order)
# ---------------------------------------------------------------------------


def test_iter_files_yields_path_array_pairs(tmp_path):
    paths = _encode_some_jxl_files(tmp_path, 3)
    seen = []
    for p, arr in parallel.iter_files(paths, n_workers=2):
        seen.append(p)
        assert arr.shape == (32, 48, 3)
    assert sorted(seen) == sorted(Path(p) for p in paths)


def test_iter_files_empty_returns_empty_iterator(tmp_path):
    assert list(parallel.iter_files([], n_workers=2)) == []


def test_iter_files_serial_path(tmp_path):
    """n_workers=1 takes the simple sequential branch — must still work."""
    paths = _encode_some_jxl_files(tmp_path, 2)
    out = list(parallel.iter_files(paths, n_workers=1))
    assert len(out) == 2


# ---------------------------------------------------------------------------
# reduce_files: max/sum projection over many JXL files
# ---------------------------------------------------------------------------


def test_reduce_files_max_projection(tmp_path):
    """reduce(np.maximum) over N frames should match a simple serial
    np.maximum.reduce — proves the lock-protected fold is correct."""
    paths = _encode_some_jxl_files(tmp_path, 4)
    proj = parallel.reduce_files(paths, np.maximum, n_workers=4)
    expected = np.maximum.reduce([oc.read(p) for p in paths])
    np.testing.assert_array_equal(proj, expected)


def test_reduce_files_empty_with_init():
    """No paths + explicit init → returns init."""
    init = np.zeros((4, 4), dtype=np.uint8)
    out = parallel.reduce_files([], np.maximum, init=init)
    assert out is init


def test_reduce_files_empty_no_init():
    """No paths, no init → returns None."""
    assert parallel.reduce_files([], np.maximum) is None


# ---------------------------------------------------------------------------
# decode_frames_parallel: parallel decode of multi-frame container
# ---------------------------------------------------------------------------


def test_decode_frames_parallel_all_frames(tmp_path):
    """All frames decoded in parallel match the serial round-trip."""
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 256, (4, 32, 48, 3), dtype=np.uint8)
    p = str(tmp_path / "stack.jxl")
    _write_stack(p, stack)

    frames = parallel.decode_frames_parallel(p, n_workers=2)
    assert len(frames) == 4
    for got, want in zip(frames, stack):
        np.testing.assert_array_equal(got, want)


def test_decode_frames_parallel_subset_indices(tmp_path):
    """Specifying indices=[2, 0] returns those frames in that order."""
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 256, (4, 32, 48, 3), dtype=np.uint8)
    p = str(tmp_path / "stack.jxl")
    _write_stack(p, stack)

    frames = parallel.decode_frames_parallel(p, indices=[2, 0], n_workers=2)
    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0], stack[2])
    np.testing.assert_array_equal(frames[1], stack[0])


def test_decode_frames_parallel_serial_path(tmp_path):
    """n_workers=1 takes the JxlReader fast-path; verify results still match."""
    rng = np.random.default_rng(0)
    stack = rng.integers(0, 256, (3, 32, 48, 3), dtype=np.uint8)
    p = str(tmp_path / "stack.jxl")
    _write_stack(p, stack)
    frames = parallel.decode_frames_parallel(p, n_workers=1)
    assert len(frames) == 3
    for got, want in zip(frames, stack):
        np.testing.assert_array_equal(got, want)


def test_decode_frames_parallel_empty_indices(tmp_path):
    paths = _encode_some_jxl_files(tmp_path, 1)
    assert parallel.decode_frames_parallel(paths[0], indices=[]) == []


def test_decode_frames_parallel_accepts_bytes(tmp_path):
    paths = _encode_some_jxl_files(tmp_path, 1)
    data = Path(paths[0]).read_bytes()
    frames = parallel.decode_frames_parallel(data, n_workers=1)
    assert len(frames) == 1


# ---------------------------------------------------------------------------
# read_bytes / open_uncached
# ---------------------------------------------------------------------------


def test_read_bytes_cached(tmp_path):
    p = tmp_path / "f.bin"
    payload = os.urandom(8192)
    p.write_bytes(payload)
    assert parallel.read_bytes(p) == payload


def test_read_bytes_uncached(tmp_path):
    """uncached=True takes the F_NOCACHE / POSIX_FADV_DONTNEED branch.
    Result must be byte-identical to a normal read."""
    p = tmp_path / "f.bin"
    payload = os.urandom(8192)
    p.write_bytes(payload)
    assert parallel.read_bytes(p, uncached=True) == payload


def test_open_uncached_returns_readable_fd(tmp_path):
    p = tmp_path / "f.bin"
    payload = b"hello"
    p.write_bytes(payload)
    fd = parallel.open_uncached(p)
    try:
        assert os.read(fd, len(payload)) == payload
    finally:
        os.close(fd)
