"""Parallel / network-aware I/O helpers for JPEG XL.

A single JXL codestream must be fed to libjxl sequentially — there's no
way to decode different parts of one file in parallel. The wins from
parallel I/O are:

  * Reading **many small JXL files** in parallel (decode N files at once
    on a thread pool — releases the GIL inside JxlDecoderProcessInput).
  * **Cold-cache reads** from NAS — opening with ``F_NOCACHE`` (macOS) /
    ``POSIX_FADV_DONTNEED`` (Linux) bypasses the kernel page cache so
    benchmarks measure the real network read, not a warm-cache replay.

Patterns ported from ``hiprpy.io.czi`` / ``hiprpy.io.npy``.
"""

from __future__ import annotations

import os
import sys

# fcntl is POSIX-only; on Windows the F_NOCACHE optimisation below is
# a no-op anyway, so we silently fall back when the module is absent.
try:
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - Windows-only branch
    _fcntl = None
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

# Backend optional — see opencodecs.jxl for the full pattern. If libjxl
# isn't built we still allow `import opencodecs.parallel` and only fail
# at call time, with a clear message.
try:
    from .codecs._jxl import (
        decode as _jxl_decode,
        frame_count as _jxl_frame_count,
    )
    _HAVE_BACKEND = True
except ImportError as _exc:  # pragma: no cover - libjxl-missing stub; tested via import_or_stubs
    _HAVE_BACKEND = False
    _IMPORT_ERROR = _exc

    def _missing(*_a, **_kw):
        raise ImportError(
            "opencodecs.parallel needs the libjxl Cython extension; "
            f"it didn't build on this platform: {_IMPORT_ERROR}. See INSTALL.md."
        )

    _jxl_decode = _jxl_frame_count = _missing  # type: ignore[assignment]

# F_NOCACHE on Darwin is fcntl command 48 (not exposed in Python's fcntl).
_F_NOCACHE_DARWIN = 48


def _default_n_workers() -> int:
    return min(32, (os.cpu_count() or 4))


# ---------------------------------------------------------------------------
# Cache-bypass open
# ---------------------------------------------------------------------------


def open_uncached(path: str | Path) -> int:
    """Open ``path`` for reading, asking the kernel to bypass its page cache.

    Returns a raw fd (caller must os.close). Useful for cold-cache benchmarks
    of NAS reads — without this, a second read of the same file hits the
    kernel buffer cache and returns in microseconds regardless of the
    underlying network.
    """
    fd = os.open(str(path), os.O_RDONLY | getattr(os, "O_BINARY", 0))
    if sys.platform == "darwin":
        try:
            _fcntl.fcntl(fd, _F_NOCACHE_DARWIN, 1)
        except OSError:  # pragma: no cover - rare kernel reject of F_NOCACHE
            pass
    elif sys.platform.startswith("linux"):  # pragma: no cover - Linux-only branch
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        except (AttributeError, OSError):
            pass
    return fd


def read_bytes(path: str | Path, *, uncached: bool = False) -> bytes:
    """Read entire file as bytes, optionally bypassing the page cache."""
    if not uncached:
        return Path(path).read_bytes()
    fd = open_uncached(path)
    try:
        size = os.fstat(fd).st_size
        return os.read(fd, size) if size <= 1 << 30 else _read_full(fd, size)
    finally:
        os.close(fd)


def _read_full(fd: int, size: int) -> bytes:  # pragma: no cover - >1GB-file path
    """Read exactly `size` bytes from fd, looping over short reads.

    Only invoked when ``read_bytes(uncached=True)`` is called on a file
    larger than 1 GB; not exercised in the standard test corpus.
    """
    chunks: list[bytes] = []
    remaining = size
    while remaining > 0:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


# ---------------------------------------------------------------------------
# Parallel decode of many .jxl files
# ---------------------------------------------------------------------------


def iter_files(
    paths: Iterable[str | Path],
    *,
    n_workers: int | None = None,
    uncached: bool = False,
    decode_threads: int = 1,
) -> Iterator[tuple[Path, np.ndarray]]:
    """Decode N JXL files concurrently; yield ``(path, array)`` as each completes.

    Outer-level parallelism is one thread per file; ``decode_threads`` controls
    the parallelism *inside* each decode (libjxl's own thread pool). For many
    small files, ``decode_threads=1`` and a wider outer pool is usually best.
    For a few large files, the opposite.

    Order is **completion order, not input order** — wrap with ``sorted(...)``
    or use ``read_files`` if you need ordered output.
    """
    paths = [Path(p) for p in paths]
    if not paths:
        return iter(())
    workers = n_workers if n_workers is not None else _default_n_workers()
    workers = max(1, min(workers, len(paths)))

    def _one(p: Path) -> tuple[Path, np.ndarray]:
        data = read_bytes(p, uncached=uncached)
        # parse_color=False matches the fast decode path (no ICC fetch).
        arr = _jxl_decode(data, numthreads=decode_threads, parse_color=False)
        return p, arr

    if workers == 1:
        for p in paths:
            yield _one(p)
        return

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, p): p for p in paths}
        for fut in as_completed(futures):
            yield fut.result()


def read_files(
    paths: Iterable[str | Path],
    *,
    n_workers: int | None = None,
    uncached: bool = False,
    decode_threads: int = 1,
) -> list[np.ndarray]:
    """Decode N JXL files concurrently; return a list aligned to ``paths``."""
    paths = [Path(p) for p in paths]
    results: list[np.ndarray | None] = [None] * len(paths)
    by_path = {p: i for i, p in enumerate(paths)}
    for p, arr in iter_files(
        paths,
        n_workers=n_workers,
        uncached=uncached,
        decode_threads=decode_threads,
    ):
        results[by_path[p]] = arr
    return [r for r in results if r is not None]  # type: ignore[misc]


def reduce_files(
    paths: Iterable[str | Path],
    reducer: Callable[[np.ndarray | None, np.ndarray], np.ndarray],
    *,
    init: np.ndarray | None = None,
    n_workers: int | None = None,
    uncached: bool = False,
    decode_threads: int = 1,
) -> np.ndarray | None:
    """Decode files in parallel and fold them through ``reducer``.

    ``reducer(accumulator, frame) -> new_accumulator``. The reduction is
    serialized under a lock — the parallelism is in the I/O+decode of the
    next batch of frames while the previous reduction completes. Useful for
    max/mean/sum projections over a stack of independent JXL frames.

    Example: max-projection over a stack of grayscale JXLs::

        proj = reduce_files(paths, np.maximum)
    """
    accumulator: np.ndarray | None = init
    lock = threading.Lock()

    for _, arr in iter_files(
        paths,
        n_workers=n_workers,
        uncached=uncached,
        decode_threads=decode_threads,
    ):
        with lock:
            if accumulator is None:
                accumulator = arr
            else:
                accumulator = reducer(accumulator, arr)

    return accumulator


# ---------------------------------------------------------------------------
# Parallel decode of a multi-frame JXL into per-frame ndarrays
# ---------------------------------------------------------------------------


def decode_frames_parallel(
    src,
    *,
    indices: Iterable[int] | None = None,
    n_workers: int | None = None,
    decode_threads: int | None = None,
    uncached: bool = False,
) -> list[np.ndarray]:
    """Decode N frames of a multi-frame JPEG XL in parallel.

    Reads the encoded bytes once (off NAS / cache as ``uncached``), then
    fans ``index=i`` decodes across a thread pool. libjxl's
    ``JxlDecoderSkipFrames`` lets each worker fast-forward to its target
    frame at bitstream-parse cost (no pixel decode for skipped frames),
    so the per-worker overhead is small.

    Returns a list aligned to ``indices`` (or ``[0..N-1]`` if None). For
    a 16-frame stack this is the parallel-multi-frame analog of
    ``read_files`` for the multi-file case — except all frames live in
    one container with the ``jxli`` frame-index box (set by JxlWriter
    when ``animation=True``).

    Parameters
    ----------
    src : str | os.PathLike | bytes | bytes-like | file-like
        The JXL container.
    indices : iterable of int, optional
        Frame indices to decode. Default: all frames in order.
    n_workers : int, optional
        Thread pool size. Default: min(len(indices), CPU count, 32).
    decode_threads : int, default 1
        libjxl threads per decode. With many workers, keep this at 1 to
        avoid oversubscription.
    uncached : bool, default False
        Force a cold-cache read of ``src`` (F_NOCACHE / POSIX_FADV_DONTNEED).
    """
    # Materialize the bytes once. All workers share the same buffer.
    if isinstance(src, (str, Path)):
        data = read_bytes(src, uncached=uncached)
    elif hasattr(src, "read"):
        data = src.read()
    else:
        data = bytes(src)

    if indices is None:
        n = _jxl_frame_count(data)
        indices = range(n)
    indices = list(indices)
    if not indices:
        return []

    workers = (n_workers if n_workers is not None
               else min(_default_n_workers(), len(indices)))
    workers = max(1, min(workers, len(indices)))

    # Auto-budget libjxl threads per worker so total parallelism ~= CPU.
    # User can pin explicitly with decode_threads=N.
    if decode_threads is None:
        cpu = _default_n_workers()
        decode_threads = max(1, cpu // workers)

    def _one(i: int) -> np.ndarray:
        return _jxl_decode(data, index=i, numthreads=decode_threads)

    if workers == 1:
        # Each per-frame index= call walks the bitstream from frame 0,
        # so naively running them serially is O(N^2). Fast-path: just
        # sequentially iterate once and pick out the requested frames.
        from .codecs._jxl import JxlReader
        wanted = set(indices)
        order = {i: pos for pos, i in enumerate(indices)}
        results: list = [None] * len(indices)
        with JxlReader(data, numthreads=decode_threads) as r:
            for fi, frame in enumerate(r.iter_frames()):
                if fi in wanted:
                    results[order[fi]] = frame
                if len(wanted) > 0 and fi >= max(wanted):
                    break
        return results

    results: list = [None] * len(indices)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {ex.submit(_one, i): pos for pos, i in enumerate(indices)}
        for fut in as_completed(futures):
            results[futures[fut]] = fut.result()
    return results


def frame_count(src) -> int:
    """Return the number of frames in a JPEG XL container.

    Pure bitstream-parse — no pixel decode. ~5-10 ms on a 16-frame stack.
    """
    if isinstance(src, (str, Path)):
        data = src
    elif hasattr(src, "read") and not isinstance(src, (bytes, bytearray, memoryview)):
        data = src.read()
    else:
        data = src
    return _jxl_frame_count(data)


__all__ = [
    "open_uncached",
    "read_bytes",
    "iter_files",
    "read_files",
    "reduce_files",
    "decode_frames_parallel",
    "frame_count",
]
