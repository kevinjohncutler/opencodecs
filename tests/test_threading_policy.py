"""How opencodecs spends threads: one shared pool, and a fair share each.

The batch helpers used to build a thread pool per call (0.2 ms for 2
workers, 1.3 ms for 16) and every automatic call sized itself as if it
had the machine to itself. These pin down the replacement: one pool for
the life of the process, calls that never return while their work is
still running, automatic counts divided between overlapping calls,
explicit counts honored, ``numthreads=1`` meaning no threads at all, and
pools that still work in a forked child.
"""
from __future__ import annotations

import concurrent.futures
import os
import subprocess
import sys
import textwrap
import threading
import time

import numpy as np
import pytest

from opencodecs.core import parallel
from opencodecs.core.parallel import (auto_threads, fair_share, map_batches,
                                      parallel_call, resolve_workers,
                                      run_batched)
from opencodecs.core.pipeline import WorkerBudget, map_bounded


@pytest.fixture
def pools_created(monkeypatch):
    made = []
    init = concurrent.futures.ThreadPoolExecutor.__init__

    def counting(self, *a, **k):
        made.append(self)
        return init(self, *a, **k)

    monkeypatch.setattr(concurrent.futures.ThreadPoolExecutor, "__init__", counting)
    return made


@pytest.fixture
def threads_started(monkeypatch):
    started = []
    start = threading.Thread.start

    def counting(self, *a, **k):
        started.append(self)
        return start(self, *a, **k)

    monkeypatch.setattr(threading.Thread, "start", counting)
    return started


def test_batch_helpers_share_one_pool_across_calls(pools_created):
    out = np.zeros(64)

    def fill(i):
        out[i] = i

    for _ in range(10):
        run_batched(fill, list(range(64)), 4)
        assert [len(b) for b in map_batches(lambda i: i, range(40), 4, batch_size=8)]
        assert list(map_bounded(lambda i: i * 2, range(20), 4)) == [i * 2 for i in range(20)]
    assert out.tolist() == list(range(64))
    # At most the one lazily created pool, however many calls.
    assert len(pools_created) <= 1


def test_run_batched_does_not_return_while_a_batch_still_runs():
    finished = []

    def fn(i):
        if i == 0:
            raise ValueError("first item fails")
        time.sleep(0.02)
        finished.append(i)

    with pytest.raises(ValueError, match="first item fails"):
        run_batched(fn, list(range(16)), 4)
    snapshot = len(finished)
    time.sleep(0.2)
    # Nothing may still be writing after the caller has its error.
    assert len(finished) == snapshot


def test_closing_map_batches_waits_for_its_running_batches():
    finished = []

    def fn(i):
        time.sleep(0.02)
        finished.append(i)
        return i

    it = map_batches(fn, range(64), 4, batch_size=2)
    next(it)
    it.close()
    snapshot = len(finished)
    time.sleep(0.2)
    assert len(finished) == snapshot


def test_automatic_counts_take_a_fair_share_and_explicit_ones_do_not(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 16)
    big = 1 << 40
    assert resolve_workers(None, 256, output_bytes=big) == 16  # alone: full width
    with parallel_call(), parallel_call(), parallel_call():
        assert fair_share() == 4
        assert resolve_workers(None, 256, output_bytes=big) == 4
        assert resolve_workers(12, 256) == 12
        assert auto_threads(None) == 4
        assert auto_threads(7) == 7
    assert resolve_workers(None, 256, output_bytes=big) == 16


def test_auto_threads_sizes_by_work_and_is_serial_inside_a_worker(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 64)
    assert auto_threads(None, work=512 * 512, per_thread=1 << 16) == 4
    assert auto_threads(None, work=4096 * 4096, per_thread=1 << 16, max_threads=16) == 16
    assert auto_threads(None, work=0, max_threads=8) == 8  # unknown work: the cap
    assert WorkerBudget(2).run(lambda _: auto_threads(None, max_threads=8), None) == 1


def test_numthreads_one_starts_no_threads_for_tiff(tmp_path, threads_started):
    tifffile = pytest.importorskip("tifffile")
    import opencodecs as oc
    img = (np.arange(768 * 1024, dtype=np.uint16) % 4093).reshape(768, 1024)
    path = tmp_path / "tiles.tif"
    tifffile.imwrite(path, img, tile=(128, 128), compression="zlib", predictor=2)
    del threads_started[:]
    np.testing.assert_array_equal(oc.read(str(path), numthreads=1), img)
    assert threads_started == []
    np.testing.assert_array_equal(oc.read(str(path)), img)


def _write_jpegls_dicom(path, frames):
    imagecodecs = pytest.importorskip("imagecodecs")
    pytest.importorskip("pydicom")
    from pydicom.dataset import Dataset, FileMetaDataset
    from pydicom.encaps import encapsulate
    from pydicom.uid import JPEGLSLossless, generate_uid
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = "1.2.840.10008.5.1.4.1.1.7"
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = JPEGLSLossless
    ds = Dataset()
    ds.file_meta = meta
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.Modality = "OT"
    ds.Rows, ds.Columns = frames.shape[1:]
    ds.NumberOfFrames = len(frames)
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated, ds.BitsStored, ds.HighBit = 16, 12, 11
    ds.PixelRepresentation = 0
    ds.PixelData = encapsulate([imagecodecs.jpegls_encode(f) for f in frames])
    ds["PixelData"].is_undefined_length = True
    ds.save_as(path, enforce_file_format=True)


def test_dicom_honors_numthreads(tmp_path, threads_started):
    """The codec used to drop numthreads, so numthreads=1 still threaded."""
    import opencodecs as oc
    pytest.importorskip("opencodecs.codecs._charls")
    # Large enough (3 MB out) that the automatic count is more than one
    # worker, so a dropped numthreads shows up as threads started.
    frames = (np.arange(8 * 512 * 384, dtype=np.uint16) % 4000).reshape(8, 512, 384)
    path = tmp_path / "frames.dcm"
    _write_jpegls_dicom(path, frames)
    del threads_started[:]
    np.testing.assert_array_equal(oc.read(str(path), numthreads=1), frames)
    assert threads_started == []
    np.testing.assert_array_equal(oc.read(str(path), numthreads=4), frames)


@pytest.mark.skipif(not hasattr(os, "fork"), reason="needs fork")
def test_pools_work_in_a_forked_child():
    """A child inherits pool objects without their threads; work must still run."""
    script = textwrap.dedent("""
        import os, sys
        from opencodecs.core.parallel import run_batched, map_batches
        from opencodecs.core.io import get_reader_pool
        out = [0] * 32
        def fill(i): out[i] = i
        run_batched(fill, list(range(32)), 4)          # parent creates the pool
        get_reader_pool("zarr").submit(abs, -1).result()
        pid = os.fork()
        if pid == 0:
            run_batched(fill, list(range(32)), 4)
            ok = sum(len(b) for b in map_batches(abs, range(20), 4, batch_size=4)) == 20
            ok = ok and get_reader_pool("zarr").submit(abs, -2).result() == 2
            os._exit(0 if ok and out == list(range(32)) else 1)
        _, status = os.waitpid(pid, 0)
        sys.exit(os.waitstatus_to_exitcode(status))
    """)
    env = dict(os.environ)
    proc = subprocess.run([sys.executable, "-c", script], env=env, timeout=120)
    assert proc.returncode == 0


def test_jxl_default_decode_matches_serial():
    imagecodecs = pytest.importorskip("imagecodecs")
    _jxl = pytest.importorskip("opencodecs.codecs._jxl")
    rgb = (np.arange(300 * 420 * 3) % 251).astype(np.uint8).reshape(300, 420, 3)
    blob = imagecodecs.jpegxl_encode(rgb, level=90)
    np.testing.assert_array_equal(_jxl.decode(blob), _jxl.decode(blob, numthreads=1))


def test_avif_decode_thread_defaults():
    _avif = pytest.importorskip("opencodecs.codecs._avif")
    assert 1 <= _avif._decode_threads(None) <= 8
    assert _avif._decode_threads(3) == 3
    assert _avif._decode_threads(0) == (os.cpu_count() or 4)  # explicit: every core


def test_abandoned_iterators_give_their_share_back():
    """A half-consumed iterator counts as in flight only while it exists."""
    import gc
    before = parallel._INFLIGHT
    it = map_batches(lambda i: i, range(100), 4, batch_size=2)
    next(it)
    assert parallel._INFLIGHT == before + 1
    del it
    gc.collect()
    assert parallel._INFLIGHT == before
    it = map_bounded(lambda i: i, range(100), 4)
    next(it)
    del it
    gc.collect()
    assert parallel._INFLIGHT == before
