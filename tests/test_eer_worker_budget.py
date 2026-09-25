"""Threaded event accumulation propagates failures and shares worker slots."""
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pytest
from opencodecs._eer_reader import EerReader
from opencodecs.core.pipeline import WorkerBudget, in_worker


class SyntheticReader(EerReader):
    shape = (4, 7)


def test_accumulator_propagates_decode_error():
    reader = object.__new__(SyntheticReader)
    def frame(i):
        if i == 3:
            raise ValueError('bad event frame')
        return np.full(reader.shape, i, 'u1')
    reader.frame = frame
    with pytest.raises(ValueError, match='bad event frame'):
        reader._sum_threaded(0, 8, np.uint16, 3)


def test_accumulators_share_explicit_budget():
    reader = object.__new__(SyntheticReader)
    active = peak = 0
    lock = threading.Lock()
    def frame(i):
        nonlocal active, peak
        assert in_worker()
        with lock:
            active += 1
            peak = max(peak, active)
        try:
            time.sleep(.002)
            return np.full(reader.shape, i, 'u1')
        finally:
            with lock:
                active -= 1
    reader.frame = frame
    budget = WorkerBudget(2)
    with ThreadPoolExecutor(2) as callers:
        calls = [callers.submit(reader._sum_threaded, 0, 8, np.uint16, 3, budget)
                 for _ in range(2)]
        for call in calls:
            np.testing.assert_array_equal(call.result(), np.full(reader.shape, 28))
    assert peak == 2
    assert active == 0
