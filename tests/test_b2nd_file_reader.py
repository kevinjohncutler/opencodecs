"""Persistent B2nd storage reads use the native index and owned context."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._b2nd_codec import B2ndCodec


@pytest.fixture
def stored_array(tmp_path):
    pytest.importorskip("opencodecs.codecs._b2nd")
    blosc2 = pytest.importorskip("blosc2")
    data = np.arange(47 * 61 * 19, dtype=np.uint16).reshape(47, 61, 19)
    path = tmp_path / "independent.b2nd"
    stored = blosc2.asarray(data, chunks=(8, 13, 7), blocks=(4, 5, 3),
                            urlpath=str(path), mode="w", contiguous=True)
    del stored
    return path, data


def test_repeated_boxes_and_output(stored_array, monkeypatch):
    path, data = stored_array
    import opencodecs._b2nd_codec as adapter
    def reject_full_read(*args):
        raise AssertionError("file reader must not read a complete cframe")
    monkeypatch.setattr(adapter, "_read_src", reject_full_read)
    with oc.open(path, format="b2nd", numthreads=2) as reader:
        assert reader.shape == data.shape and reader.dtype == data.dtype
        for start, stop in [((1, 2, 3), (9, 17, 13)), ((39, 50, 11), (47, 61, 19)),
                            ((0, 0, 0), (1, 1, 1))]:
            expected = data[tuple(slice(a, b) for a, b in zip(start, stop))]
            out = np.empty_like(expected)
            assert reader.read_slice(start, stop, out=out) is out
            np.testing.assert_array_equal(out, expected)
        np.testing.assert_array_equal(reader.read(), data)
        assert reader.read_slice((0, 0, 0), (0, 2, 3)).shape == (0, 2, 3)
    np.testing.assert_array_equal(out, expected)
    with pytest.raises(ValueError, match="closed"):
        reader.read()
    np.testing.assert_array_equal(B2ndCodec().decode_slice(path, (2, 3, 4), (9, 12, 11)),
                                  data[2:9, 3:12, 4:11])


def test_owned_context_parallel_callers(stored_array):
    path, data = stored_array
    with oc.open(path, format="b2nd") as reader, ThreadPoolExecutor(4) as pool:
        results = list(pool.map(lambda i: reader.read_slice((i, 1, 2), (i+1, 8, 9)), range(15)))
    for i, result in enumerate(results):
        np.testing.assert_array_equal(result, data[i:i+1, 1:8, 2:9])


def test_file_reader_validation(stored_array):
    path, data = stored_array
    with oc.open(path, format="b2nd") as reader:
        with pytest.raises(ValueError):
            reader.read_slice((-1, 0, 0), (1, 1, 1))
        with pytest.raises(ValueError):
            reader.read_slice((0,), (1,))
        with pytest.raises(TypeError):
            reader.read_slice((0.5, 0, 0), (1, 1, 1))
        out = np.zeros((1, 1, 1), dtype=data.dtype)
        out.flags.writeable = False
        with pytest.raises(ValueError, match="writable"):
            reader.read_slice((0, 0, 0), (1, 1, 1), out=out)
        with pytest.raises(ValueError, match="dtype"):
            reader.read_slice((0, 0, 0), (1, 1, 1), out=np.empty((1, 1, 1), dtype="f4"))
    reader.close()


def test_file_reader_corrupt_header(tmp_path):
    pytest.importorskip("opencodecs.codecs._b2nd")
    path = tmp_path / "invalid.b2nd"
    path.write_bytes(b"not a Blosc2 frame" * 16)
    with pytest.raises(RuntimeError):
        oc.open(path, format="b2nd")
