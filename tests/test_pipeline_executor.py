"""A bounded iterator borrows its executor without owning other jobs."""
from concurrent.futures import ThreadPoolExecutor
import threading

import pytest

from opencodecs.core.pipeline import map_bounded


def test_shared_executor_close_waits_only_for_own_jobs():
    unrelated_release = threading.Event()
    own_started = threading.Event()
    own_release = threading.Event()
    closed = threading.Event()
    with ThreadPoolExecutor(max_workers=4) as pool:
        unrelated = pool.submit(unrelated_release.wait)
        def work(value):
            if value:
                own_started.set()
                own_release.wait()
            return value
        stream = map_bounded(work, range(3), workers=3, executor=pool)
        try:
            assert next(stream) == 0
            assert own_started.wait(2)
            def close():
                stream.close()
                closed.set()
            closer = threading.Thread(target=close)
            closer.start()
            assert not closed.wait(0.02)
            own_release.set()
            assert closed.wait(2)
            closer.join()
            assert not unrelated.done()
            assert pool.submit(lambda: 42).result(timeout=2) == 42
        finally:
            own_release.set()
            unrelated_release.set()
            stream.close()


def test_shared_executor_survives_decoder_error():
    with ThreadPoolExecutor(max_workers=2) as pool:
        def decode(value):
            if value == 1:
                raise ValueError("broken piece")
            return value
        with pytest.raises(ValueError, match="broken piece"):
            list(map_bounded(decode, range(5), workers=2, executor=pool))
        assert pool.submit(lambda: "still usable").result(timeout=2) == "still usable"


def test_shared_executor_nested_pipeline_runs_inline():
    with ThreadPoolExecutor(max_workers=2) as pool:
        def outer(value):
            return list(map_bounded(lambda n: n + value, range(3), workers=2,
                                    executor=pool))
        assert list(map_bounded(outer, range(2), workers=2, executor=pool)) == [[0, 1, 2], [1, 2, 3]]


@pytest.mark.parametrize('adapter', ['shared', 'ndtiff'])
def test_interrupted_wait_joins_current_borrowed_pool_job(adapter, monkeypatch):
    started, release, finished, returned = (threading.Event() for _ in range(4))
    errors = []
    def work(value):
        started.set()
        try:
            assert release.wait(3)
            return value
        finally:
            finished.set()
    class InterruptedFuture:
        def __init__(self, future):
            self.future = future
        def result(self):
            assert started.wait(3)
            raise KeyboardInterrupt('fixture interrupted wait')
        def cancel(self):
            return self.future.cancel()
        def exception(self):
            return self.future.exception()
    with ThreadPoolExecutor(max_workers=2) as pool:
        class Borrowed:
            def submit(self, fn, *args):
                return InterruptedFuture(pool.submit(fn, *args))
        if adapter == 'shared':
            stream = map_bounded(work, [1], workers=2, max_pending=1, executor=Borrowed())
        else:
            from opencodecs._ndtiff import NDTiffDataset
            reader = object.__new__(NDTiffDataset)
            reader.entries = [1]
            reader._read_entry = work
            monkeypatch.setattr('opencodecs._ndtiff._get_pool', lambda: Borrowed())
            stream = reader.iter_frames_parallel(prefetch=1)
        def caller():
            try:
                next(stream)
            except BaseException as exc:
                errors.append(exc)
            finally:
                returned.set()
        thread = threading.Thread(target=caller)
        thread.start()
        try:
            assert started.wait(3)
            assert not returned.wait(0.05)
            release.set()
            assert returned.wait(3)
            assert finished.is_set()
            assert len(errors) == 1 and isinstance(errors[0], KeyboardInterrupt)
            assert pool.submit(lambda: 'usable').result(timeout=2) == 'usable'
        finally:
            release.set()
            thread.join(timeout=3)
            stream.close()
