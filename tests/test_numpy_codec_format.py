"""The numpy codec writes what ``numpy.save`` writes and reads ``.npz``.

``numpy.save`` is the reference implementation of the ``.npy`` format
(NEP 1, ``numpy.lib.format``), so its bytes are the expected output, and
imagecodecs ``numpy_encode`` / ``numpy_decode`` are the API reference.
"""

from __future__ import annotations

import io
import zipfile

import numpy as np
import pytest

import opencodecs as oc
from _ic_reference import skip_if_old_imagecodecs  # noqa: E402

pytestmark = skip_if_old_imagecodecs


def _np_save(arr) -> bytes:
    buf = io.BytesIO()
    np.save(buf, arr, allow_pickle=False)
    return buf.getvalue()


def _codec():
    return oc.get_codec("numpy")


CASES = {
    "datetime64[s]": np.arange(4).astype("datetime64[s]"),
    "timedelta64[ms]": np.arange(6).astype("timedelta64[ms]").reshape(2, 3),
    "big-endian datetime": np.arange(3).astype(">M8[D]"),
    "structured with datetime": np.zeros(3, [("t", "M8[us]"), ("v", "<f4")]),
    "fortran 2-D": np.asfortranarray(np.arange(12, dtype="<i4").reshape(3, 4)),
    "fortran 3-D": np.asfortranarray(
        np.arange(60, dtype=">u2").reshape(3, 4, 5)),
    "fortran datetime": np.asfortranarray(
        np.arange(6).astype("M8[s]").reshape(2, 3)),
    "c 2-D": np.arange(12, dtype="<f8").reshape(3, 4),
    "strided": np.arange(40, dtype="<i2").reshape(5, 8)[::2, ::3],
    "1x1 (both orders)": np.ones((1, 1), "<u1"),
}


@pytest.mark.parametrize("name", list(CASES))
def test_encode_bytes_equal_numpy_save(name):
    arr = CASES[name]
    blob = _codec().encode(arr)
    assert blob == _np_save(arr)
    back = _codec().decode(blob)
    assert back.dtype == arr.dtype
    np.testing.assert_array_equal(back, arr)


@pytest.mark.parametrize("name", ["datetime64[s]", "fortran 2-D"])
def test_encode_matches_imagecodecs(name):
    imagecodecs = pytest.importorskip("imagecodecs")
    arr = CASES[name]
    assert _codec().encode(arr) == imagecodecs.numpy_encode(arr)


def test_encode_to_destination_streams_the_same_bytes():
    arr = CASES["fortran 3-D"]
    sink = io.BytesIO()
    assert _codec().encode(arr, dest=sink) is None
    assert sink.getvalue() == _np_save(arr)


def _npz(*arrays, compressed=True, **named):
    buf = io.BytesIO()
    (np.savez_compressed if compressed else np.savez)(buf, *arrays, **named)
    return buf.getvalue()


def test_decode_npz_returns_member_array():
    a = np.arange(100, dtype="<u2")
    b = np.linspace(0, 1, 7)
    blob = _npz(a, b)
    np.testing.assert_array_equal(_codec().decode(blob), a)
    np.testing.assert_array_equal(_codec().decode(blob, index=1), b)
    np.testing.assert_array_equal(_codec().decode(blob, index=-1), b)
    np.testing.assert_array_equal(_codec().decode(blob, index="arr_1"), b)
    for bad in (2, -3, "nope"):
        with pytest.raises(KeyError):
            _codec().decode(blob, index=bad)
    out = np.empty_like(a)
    assert _codec().decode(blob, out=out) is out
    np.testing.assert_array_equal(out, a)


def test_decode_reads_imagecodecs_npz():
    imagecodecs = pytest.importorskip("imagecodecs")
    a = np.arange(100, dtype="<u2")
    blob = imagecodecs.numpy_encode(a, level=5)
    assert blob[:4] == b"PK\x03\x04"
    got = _codec().decode(blob)
    assert isinstance(got, np.ndarray)
    np.testing.assert_array_equal(got, imagecodecs.numpy_decode(blob))


@pytest.mark.parametrize("level", [1, 5, 9, True, -1])
def test_level_writes_compressed_npz(level):
    a = np.zeros((64, 64), "<f4")
    blob = _codec().encode(a, level=level)
    assert blob[:4] == b"PK\x03\x04"
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        (info,) = zf.infolist()
        assert info.filename == "arr_0.npy"
        assert info.compress_type == zipfile.ZIP_DEFLATED
        assert zf.read(info) == _np_save(a)
    assert len(blob) < a.nbytes // 10
    with np.load(io.BytesIO(blob)) as npz:
        np.testing.assert_array_equal(npz["arr_0"], a)
    # Deterministic: a fixed entry timestamp.
    assert _codec().encode(a, level=level) == blob
    try:
        import imagecodecs
    except ImportError:
        return
    np.testing.assert_array_equal(imagecodecs.numpy_decode(blob), a)


def test_level_controls_compression():
    a = np.random.default_rng(0).integers(0, 4, 200_000).astype("<u1")
    fast = _codec().encode(a, level=1)
    best = _codec().encode(a, level=9)
    assert len(best) < len(fast)


def test_falsy_level_writes_npy():
    a = np.arange(5)
    assert _codec().encode(a, level=0) == _np_save(a)
    assert _codec().encode(a, level=None) == _np_save(a)


def test_out_of_range_npz_index_raises_as_imagecodecs_does():
    imagecodecs = pytest.importorskip("imagecodecs")
    blob = _npz(np.arange(3), np.arange(4))
    for bad in (2, -3, "nope"):
        with pytest.raises(KeyError):
            imagecodecs.numpy_decode(blob, index=bad)
        with pytest.raises(KeyError):
            _codec().decode(blob, index=bad)


def test_decode_passes_options_to_numpy_load():
    """imagecodecs ``numpy_decode`` hands extra keywords to ``numpy.load``;
    so does the codec, so none is dropped silently."""
    obj = np.array([{"a": 1}, None], dtype=object)
    buf = io.BytesIO()
    np.save(buf, obj, allow_pickle=True)
    blob = buf.getvalue()
    with pytest.raises(ValueError):
        _codec().decode(blob)  # numpy.load's own default refuses pickles
    got = _codec().decode(blob, allow_pickle=True)
    assert got.dtype == object and got[0] == {"a": 1} and got[1] is None
    with pytest.raises(ValueError):
        _codec().decode(_np_save(np.arange(3)), max_header_size=1)
    with pytest.raises(TypeError):
        _codec().decode(_np_save(np.arange(3)), not_a_numpy_load_option=1)
    try:
        import imagecodecs
    except ImportError:
        return
    assert imagecodecs.numpy_decode(blob, allow_pickle=True)[0] == {"a": 1}


def test_encode_level_by_position_and_unknown_options_raise():
    a = np.arange(1000, dtype="<u2")
    assert _codec().encode(a, 6) == _codec().encode(a, level=6)
    assert _codec().encode(a, 0) == _np_save(a)
    with pytest.raises(TypeError):
        _codec().encode(a, allow_pickle=True)
    try:
        import imagecodecs
    except ImportError:
        return
    with pytest.raises(TypeError):
        imagecodecs.numpy_encode(a, allow_pickle=True)


def test_object_arrays_are_refused():
    """The one exception to "writes what numpy.save writes": numpy.save
    and imagecodecs pickle an object array, which this codec refuses
    with a clear error rather than write a pickle."""
    arr = np.array([1, "a", None], dtype=object)
    with pytest.raises(ValueError, match="Object arrays"):
        oc.get_codec("numpy").encode(arr)
    buf = io.BytesIO()
    np.save(buf, arr)                     # numpy.save does write it
    assert buf.getvalue()[:6] == b"\x93NUMPY"
