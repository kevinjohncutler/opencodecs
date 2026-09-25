"""Bounded, ordered batches shared by streaming codec consumers."""
from contextlib import closing
import threading
import pytest
from opencodecs.core.parallel import map_batches


@pytest.mark.parametrize('workers', [1, 3])
def test_batches_are_ordered_and_input_is_bounded(workers):
    consumed = []
    def source():
        for value in range(100):
            consumed.append(value)
            yield value
    with closing(map_batches(lambda x: x * 2, source(), workers,
                             batch_size=4, max_pending=2)) as batches:
        assert next(batches) == (0, 2, 4, 6)
        assert len(consumed) <= 8
        result = [0, 2, 4, 6]
        for batch in batches:
            result.extend(batch)
    assert result == list(range(0, 200, 2))


def test_early_close_stops_consuming_and_joins_workers():
    consumed = []
    def source():
        for value in range(100):
            consumed.append(value)
            yield value
    with closing(map_batches(lambda x: x, source(), 2, batch_size=3,
                             max_pending=2, name='early-close-test')) as batches:
        assert next(batches) == (0, 1, 2)
    assert len(consumed) == 6
    assert not any(t.name.startswith('opencodecs-early-close-test')
                   for t in threading.enumerate())


def test_worker_errors_propagate_and_cleanup():
    def fail(value):
        if value == 2:
            raise ValueError('bad segment')
        return value
    with pytest.raises(ValueError, match='bad segment'):
        list(map_batches(fail, range(100), 2, batch_size=2,
                         max_pending=2, name='worker-error-test'))
    assert not any(t.name.startswith('opencodecs-worker-error-test')
                   for t in threading.enumerate())


@pytest.mark.parametrize('option', [{'batch_size': 0}, {'max_pending': 0}])
def test_invalid_limits_are_rejected(option):
    with pytest.raises(ValueError):
        list(map_batches(lambda x: x, [], 2, **option))
