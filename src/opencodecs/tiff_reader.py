"""Parallel-pread TIFF tile reader on top of tifffile.

tifffile's built-in ``asarray()`` reads tiles sequentially through a single
``BufferedReader`` (with a lock around every seek+read), then dispatches
decode to a ThreadPoolExecutor. On NAS or other high-latency storage that
serializes I/O behind the lock and starves the decoder.

This module skips ``read_segments`` entirely and uses ``os.pread`` from
worker threads. ``os.pread`` releases the GIL, takes its own offset, and
doesn't seek the shared file descriptor, so reads issue in parallel at the
OS level. We pipeline read+decode+place in one worker pool.

On Windows ``os.pread`` is unavailable; we fall back to per-thread file
handles (each worker opens its own fd, seeks, reads). That gives the same
correctness with slightly more overhead from the open/close churn — still
a win over tifffile's serialized-fd approach.

Usage::

    from opencodecs.tiff_reader import imread

    arr = imread('big.tif')                 # full eager read
    arr = imread('big.tif', n_workers=16)   # tune concurrency
"""

from __future__ import annotations

import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import tifffile


_HAS_PREAD = hasattr(os, "pread")


def _read_at(fd_or_path, offset: int, nbytes: int,
             tls: "threading.local | None" = None) -> bytes:
    """Read ``nbytes`` from ``offset`` without disturbing a shared fd.

    On POSIX, ``fd_or_path`` is a shared int fd and we use ``os.pread``.
    On Windows, ``fd_or_path`` is the file path; each thread keeps its
    own fd in ``tls.fd`` for the life of the worker.
    """
    if _HAS_PREAD:
        return os.pread(fd_or_path, nbytes, offset)
    # Windows fallback: per-thread fd, seek+read
    fd = getattr(tls, "fd", None)
    if fd is None:
        fd = os.open(str(fd_or_path), os.O_RDONLY | getattr(os, "O_BINARY", 0))
        tls.fd = fd
    os.lseek(fd, offset, 0)
    return os.read(fd, nbytes)


def _read_one_page_parallel(
    page: "tifffile.TiffPage",
    fd_or_path,
    n_workers: int,
    tls_factory=None,
) -> np.ndarray:
    """Decode one TiffPage using parallel pread + decode.

    Mirrors the placement logic from tifffile's ``TiffPage.asarray``:
    segments are returned as ``(segment, (s, d, h, w, _), shape)`` where
    the (s, d, h, w) tuple gives sample/depth/height/width offsets into
    the canonical 5D ``keyframe.shaped`` array. We allocate an array of
    that shape, write decoded segments in, then reshape to ``page.shape``.
    """
    page.init_decode()
    decode = page.decode
    offsets = page.dataoffsets
    bytecounts = page.databytecounts
    n_segments = len(offsets)
    if n_segments == 0:  # pragma: no cover - n_segments==0 is empty-page edge
        return page.asarray()

    out = np.empty(page.shaped, dtype=page.dtype)
    nodata = page.nodata
    image_depth = page.imagedepth
    image_height = page.imagelength
    image_width = page.imagewidth

    tls = tls_factory() if tls_factory is not None else None

    def _read_decode_place(idx: int) -> None:
        bc = int(bytecounts[idx])
        off = int(offsets[idx])
        if bc == 0 or off == 0:  # pragma: no cover - empty-tile sparse-TIFF edge
            data = None
        else:
            data = _read_at(fd_or_path, off, bc, tls)
        seg, segidx, shape = decode(data, idx)
        s, d, h, w, _ = segidx
        if seg is None:  # pragma: no cover - empty-tile sparse-TIFF edge
            out[s,
                d:d + shape[0],
                h:h + shape[1],
                w:w + shape[2]] = nodata
        else:
            out[s,
                d:d + shape[0],
                h:h + shape[1],
                w:w + shape[2]] = seg[
                    :image_depth - d,
                    :image_height - h,
                    :image_width - w,
                ]

    # Track per-thread tls so we can close the worker-owned fds (Windows path).
    _all_tls: list = []
    if tls_factory is not None:
        # When a fresh tls_factory was passed, we need worker-fd cleanup.
        # Re-wrap to capture each thread's tls in a list.
        original_tls = tls

        def _read_decode_place_with_tls(idx: int) -> None:  # noqa: D401
            tls_local = getattr(threading.current_thread(), "_tiff_tls", None)
            if tls_local is None:
                tls_local = tls_factory()
                threading.current_thread()._tiff_tls = tls_local
                _all_tls.append(tls_local)
            bc = int(bytecounts[idx])
            off = int(offsets[idx])
            if bc == 0 or off == 0:  # pragma: no cover
                data = None
            else:
                data = _read_at(fd_or_path, off, bc, tls_local)
            seg, segidx, shape = decode(data, idx)
            s, d, h, w, _ = segidx
            if seg is None:  # pragma: no cover
                out[s, d:d + shape[0], h:h + shape[1], w:w + shape[2]] = nodata
            else:
                out[s, d:d + shape[0], h:h + shape[1], w:w + shape[2]] = seg[
                    :image_depth - d, :image_height - h, :image_width - w]

        worker = _read_decode_place_with_tls
    else:
        worker = _read_decode_place

    try:
        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            # Drain the iterator so exceptions propagate.
            for _ in ex.map(worker, range(n_segments)):
                pass
    finally:
        # Close any per-thread fds the workers opened.
        for t in _all_tls:
            fd_owned = getattr(t, "fd", None)
            if fd_owned is not None:
                try:
                    os.close(fd_owned)
                except OSError:
                    pass

    return out.reshape(page.shape)


def imread(
    path: str | Path,
    *,
    page: int = 0,
    n_workers: int | None = None,
) -> np.ndarray:
    """Read a TIFF page using parallel pread + decode.

    ``n_workers`` defaults to ``os.cpu_count()``.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    with tifffile.TiffFile(str(path)) as tf:
        tpage = tf.pages[page]
        if _HAS_PREAD:
            # POSIX: open one shared fd for all workers; pread releases the
            # GIL and takes its own offset.
            fd = os.open(str(path), os.O_RDONLY)
            try:
                return _read_one_page_parallel(tpage, fd, n_workers)
            finally:
                os.close(fd)
        else:
            # Windows: each worker opens its own fd via tls_factory.
            return _read_one_page_parallel(
                tpage, str(path), n_workers, tls_factory=threading.local,
            )


def imread_stack(
    path: str | Path,
    *,
    pages: range | list[int] | None = None,
    n_workers: int | None = None,
) -> np.ndarray:
    """Read multiple TIFF pages in parallel and stack them along axis 0.

    Unlike tifffile, decodes all pages concurrently via ``os.pread`` —
    a real win for multi-frame stacks where each page is independent.
    """
    if n_workers is None:
        n_workers = os.cpu_count() or 4

    with tifffile.TiffFile(str(path)) as tf:
        if pages is None:
            pages = list(range(len(tf.pages)))
        else:
            pages = list(pages)
        # Eagerly resolve all pages (forces tifffile to walk the IFD chain).
        tpages = [tf.pages[i] for i in pages]
        for p in tpages:
            p.init_decode()

    # POSIX: shared fd + os.pread; Windows: per-thread fd via threading.local.
    fd = os.open(str(path), os.O_RDONLY) if _HAS_PREAD else None
    use_path = None if _HAS_PREAD else str(path)
    tls = threading.local() if not _HAS_PREAD else None
    _per_thread_tls: list = []

    try:
        # Allocate stacked output.
        first = tpages[0]
        stack_shape = (len(tpages), *first.shape)
        stack = np.empty(stack_shape, dtype=first.dtype)

        # Flatten (page, tile) pairs so the thread pool sees fine-grained
        # work and saturates with at most ``n_workers`` outstanding preads.
        tasks = []
        per_page_views: list[np.ndarray] = []
        for pi, page in enumerate(tpages):
            view = stack[pi].reshape(page.shaped)
            per_page_views.append(view)
            for ti in range(len(page.dataoffsets)):
                tasks.append((pi, ti))

        def _worker(task) -> None:
            pi, ti = task
            page = tpages[pi]
            decode = page.decode
            view = per_page_views[pi]
            bc = int(page.databytecounts[ti])
            off = int(page.dataoffsets[ti])
            if bc == 0 or off == 0:  # pragma: no cover - empty-tile sparse-TIFF edge
                data = None
            elif _HAS_PREAD:
                data = os.pread(fd, bc, off)
            else:
                # Windows: per-thread fd kept on threading.current_thread()
                t = threading.current_thread()
                t_tls = getattr(t, "_tiff_stack_tls", None)
                if t_tls is None:
                    t_tls = threading.local()
                    t._tiff_stack_tls = t_tls
                    _per_thread_tls.append(t_tls)
                data = _read_at(use_path, off, bc, t_tls)
            seg, segidx, shape = decode(data, ti)
            s, d, h, w, _ = segidx
            if seg is None:  # pragma: no cover - empty-tile sparse-TIFF edge
                view[s, d:d + shape[0], h:h + shape[1], w:w + shape[2]] = page.nodata
            else:
                view[s, d:d + shape[0], h:h + shape[1], w:w + shape[2]] = seg[
                    :page.imagedepth - d,
                    :page.imagelength - h,
                    :page.imagewidth - w,
                ]

        with ThreadPoolExecutor(max_workers=n_workers) as ex:
            list(ex.map(_worker, tasks))
        return stack
    finally:
        if fd is not None:
            os.close(fd)
        for t in _per_thread_tls:
            t_fd = getattr(t, "fd", None)
            if t_fd is not None:
                try:
                    os.close(t_fd)
                except OSError:
                    pass


__all__ = ["imread", "imread_stack"]
