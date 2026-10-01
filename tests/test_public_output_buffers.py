"""Public decode destinations must mutate caller storage and retain ownership."""

import importlib
import io

import numpy as np
import pytest

import opencodecs as oc
from opencodecs.core.buffers import array_output, byte_output


BYTE_CASES = [
    ("deflate", {}), ("zstd", {}), ("lz4", {}), ("brotli", {}),
    ("snappy", {}), ("blosc2", {}), ("aec", {"bits_per_sample": 8}),
    ("bitshuffle", {"itemsize": 1}), ("byteshuffle", {"itemsize": 1}), ("none", {}),
]
# Decoders that return imagecodecs' ndarray prefix for an ndarray out=.
NDARRAY_PREFIX = {"deflate", "zstd", "lz4", "brotli", "snappy", "blosc2"}
ARRAY_CASES = [
    ("jpeg", np.uint8), ("png", np.uint16), ("jpegls", np.uint16),
    ("jpeg2k", np.uint16), ("mozjpeg", np.uint8), ("webp", np.uint8),
    ("lerc", np.float32), ("pcodec", np.float32), ("sz3", np.float32),
    ("avif", np.uint8), ("heif", np.uint8), ("rgbe", np.float32),
    ("b2nd", np.float32), ("qoi", np.uint8),
]


def available(name):
    module = importlib.import_module(f"opencodecs._{name}_codec")
    if not getattr(module, "_HAVE_BACKEND", True):
        pytest.skip(f"{name} native backend unavailable")


@pytest.mark.parametrize("name,options", BYTE_CASES)
@pytest.mark.parametrize("kind", ["bytearray", "memoryview", "array"])
def test_public_byte_destinations(name, options, kind):
    available(name)
    data = bytes(range(256)) * 16
    encoded = oc.write(None, data, format=name, **options)
    storage = bytearray(b"\xa5" * (len(data) + 13))
    destination = storage
    if kind == "memoryview":
        destination = memoryview(storage)
    elif kind == "array":
        destination = np.frombuffer(storage, dtype=np.uint8)
    result = oc.read(encoded, format=name, out=destination, **options)
    if kind == "array" and name in NDARRAY_PREFIX:
        # imagecodecs returns an ndarray slice of an ndarray out=.
        assert isinstance(result, np.ndarray) and result.dtype == np.uint8
        assert np.shares_memory(result, destination)
    else:
        assert isinstance(result, memoryview)
    assert bytes(result) == data
    assert storage[:len(data)] == data
    assert storage[len(data):] == b"\xa5" * 13
    result[0] = 37
    assert storage[0] == 37
    if isinstance(result, memoryview):
        result.release()
    if kind == "memoryview":
        assert destination[0] == 37


@pytest.mark.parametrize("name,options", BYTE_CASES)
def test_byte_destinations_reject_invalid_buffers(name, options):
    available(name)
    data = bytes(range(256)) * 4
    encoded = oc.write(None, data, format=name, **options)
    for destination in (bytes(len(data)), memoryview(bytearray(len(data) * 2))[::2]):
        with pytest.raises((TypeError, ValueError)):
            oc.read(encoded, format=name, out=destination, **options)
    with pytest.raises((ValueError, RuntimeError)):
        oc.read(encoded, format=name, out=bytearray(1), **options)


# SZ3's stream does not reliably record its data type (as for
# imagecodecs.sz3_decode, the caller passes it).
DECODE_OPTIONS = {"sz3": {"dtype": np.float32}}


@pytest.mark.parametrize("name,dtype", ARRAY_CASES)
def test_public_array_destinations(name, dtype):
    available(name)
    data = (np.arange(64 * 80 * 3).reshape(64, 80, 3) % 127).astype(dtype)
    encoded = oc.write(None, data, format=name)
    expected = oc.read(encoded, format=name, **DECODE_OPTIONS.get(name, {}))
    destination = np.empty_like(expected)
    actual = oc.read(encoded, format=name, out=destination)
    assert actual is destination
    np.testing.assert_array_equal(actual, expected)
    destination.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        oc.read(encoded, format=name, out=destination)
    with pytest.raises((ValueError, RuntimeError)):
        oc.read(encoded, format=name, out=np.empty((1,), dtype=expected.dtype))
    wide = np.empty(expected.shape[:-1] + (expected.shape[-1] * 2,), dtype=expected.dtype)
    with pytest.raises(ValueError, match="contiguous"):
        oc.read(encoded, format=name, out=wide[..., ::2])


def test_destination_helpers_do_not_release_caller_view():
    caller = memoryview(bytearray(12))
    view = byte_output(caller)
    view.release()
    caller[0] = 7
    assert caller[0] == 7
    with pytest.raises(ValueError):
        byte_output(-1)
    with pytest.raises(TypeError):
        array_output(bytearray(12))


@pytest.mark.parametrize("name", ["jpeg", "mozjpeg", "jpeg2k", "htj2k"])
def test_public_reduced_decode_matches_native(name):
    available(name)
    data = (np.arange(128 * 160 * 3).reshape(128, 160, 3) % 251).astype(np.uint8)
    encoded = oc.write(None, data, format=name)
    module = "_openjph" if name == "htj2k" else "_" + name
    native = importlib.import_module("opencodecs.codecs." + module)
    options = {"scale": (1, 2)} if name in ("jpeg", "mozjpeg") else {"reduce": 1}
    expected = native.decode(encoded, **options)
    actual = oc.read(encoded, format=name, **options)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape[0] == 64


@pytest.mark.parametrize("name,options", BYTE_CASES)
def test_public_byte_size_hint(name, options):
    available(name)
    raw = bytes(range(128)) * 17
    encoded = oc.write(None, raw, format=name, **options)
    assert oc.read(encoded, format=name, out=len(raw), **options) == raw


def test_webp_animation_destination_is_explicitly_unsupported():
    available("webp")
    imagecodecs = pytest.importorskip("imagecodecs")
    frames = np.zeros((2, 24, 32, 3), dtype=np.uint8)
    frames[1, :, :, 1] = 191
    try:
        encoded = imagecodecs.webp_encode(frames, lossless=True)
    except ValueError:
        # Older imagecodecs releases cannot encode a frame stack.
        pytest.skip("this imagecodecs cannot write an animated WebP")
    with pytest.raises(ValueError, match="animation"):
        oc.read(encoded, format="webp", out=np.empty_like(frames))


@pytest.mark.parametrize("name", ["webp", "avif", "heif"])
def test_common_still_writer_rejects_sequence_at_submission(name):
    from opencodecs.core.codec import get_codec
    available(name)
    writer = get_codec(name).writer()
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    writer.write_frame(frame)
    with pytest.raises(ValueError, match="multiple.frame|sequence"):
        writer.write_frame(frame)
    assert writer.close()


def test_heif_destination_callback_short_writes_and_errors():
    import io
    available("heif")
    image = np.arange(32 * 48 * 3, dtype=np.uint8).reshape(32, 48, 3)
    expected = oc.read(oc.write(None, image, format="heif"), format="heif")

    class ShortSink(io.BytesIO):
        def write(self, data):
            return super().write(memoryview(data)[:13])

    sink = ShortSink()
    assert oc.write(sink, image, format="heif") is None
    assert not sink.closed
    np.testing.assert_array_equal(oc.read(sink.getvalue(), format="heif"), expected)

    class Broken:
        def write(self, data):
            raise OSError("destination stopped")
    with pytest.raises(OSError, match="destination stopped"):
        oc.write(Broken(), image, format="heif")


def test_b2nd_destination_file_uses_native_storage(tmp_path):
    available("b2nd")
    array = np.arange(19 * 31 * 7, dtype=np.uint16).reshape(19, 31, 7)
    path = tmp_path / "array.b2nd"
    path.write_bytes(b"old content")
    assert oc.write(path, array, format="b2nd", storage_output=True) is None
    np.testing.assert_array_equal(oc.read(path, format="b2nd"), array)
    blosc2 = pytest.importorskip("blosc2")
    if not hasattr(blosc2, "open"):
        pytest.skip("independent Blosc2 lacks persistent-array support")
    reference = blosc2.open(str(path))
    np.testing.assert_array_equal(reference[:], array)


def test_b2nd_destination_default_preserves_buffer_path(tmp_path, monkeypatch):
    available("b2nd")
    import opencodecs._b2nd_codec as adapter
    native_encode = adapter._b2nd_encode
    calls = []

    def capture(data, **kwargs):
        calls.append(kwargs)
        return native_encode(data, **kwargs)

    monkeypatch.setattr(adapter, "_b2nd_encode", capture)
    array = np.arange(32, dtype=np.uint16)
    path = tmp_path / "default.b2nd"
    oc.write(path, array, format="b2nd")
    assert "path" not in calls[0]
    np.testing.assert_array_equal(oc.read(path, format="b2nd"), array)
    for destination in (None, io.BytesIO()):
        with pytest.raises(ValueError, match="filesystem destination"):
            oc.write(destination, array, format="b2nd", storage_output=True)
