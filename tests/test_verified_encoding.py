"""Compression verification preserves representations, not approximate values."""

from concurrent.futures import ThreadPoolExecutor
import gc
import threading
import tracemalloc
import weakref
import zlib

import numpy as np
import pytest

from opencodecs.core import segment_compression as segments
from opencodecs.core.pipeline import WorkerBudget, map_bounded
from opencodecs.core.verification import (
    LosslessVerificationError, assert_bit_exact, verification_reservation,
    verify_segment, verify_with_decoder, DeferredVerification,
)


@pytest.mark.parametrize("codec", ["none", "deflate", "zstd"])
def test_actual_byte_roundtrip_and_independent_decode(codec):
    if codec != "none":
        pytest.importorskip(f"opencodecs.codecs._{codec}")
    raw = np.arange(12345, dtype=">u2").tobytes()
    encoded = segments.encode_segment(raw, codec)
    assert verify_segment(raw, encoded, codec) is None
    if codec == "deflate":
        assert zlib.decompress(encoded) == raw
    elif codec == "zstd":
        reference = pytest.importorskip("zstandard")
        assert reference.ZstdDecompressor().decompress(encoded) == raw
    else:
        assert encoded == raw


def test_corrupt_or_wrong_payload():
    pytest.importorskip("opencodecs.codecs._deflate")
    raw = b"correct payload" * 100
    with pytest.raises(LosslessVerificationError, match="decode failed") as failure:
        verify_segment(raw, b"not compressed", "deflate")
    assert failure.value.__cause__ is not None
    wrong = segments.encode_segment(b"incorrect payload" * 100, "deflate")
    with pytest.raises(LosslessVerificationError, match="byte length differs"):
        verify_segment(raw, wrong, "deflate")
    changed = bytearray(raw)
    changed[4] ^= 1
    with pytest.raises(LosslessVerificationError, match="bytes differ"):
        verify_segment(raw, changed, "none")


@pytest.mark.parametrize("dtype,patterns", [
    ("u4", [0, 0x80000000, 0x7FC12345, 0x7FC12346, 0x3F800000]),
    ("u8", [0, 0x8000000000000000, 0x7FF8000000000123,
            0x7FF8000000000124, 0x3FF0000000000000]),
])
def test_float_bits_byteorder_nan_payload_signed_zero(dtype, patterns):
    words = np.array(patterns, dtype=dtype)
    values = words.view("f" + dtype[-1])
    swapped = values.byteswap().view(values.dtype.newbyteorder())
    assert_bit_exact(values, swapped)
    np.testing.assert_array_equal(values.view(dtype), words)
    changed = words.copy()
    changed[0] = patterns[1]
    with pytest.raises(LosslessVerificationError, match="pixel bits differ"):
        assert_bit_exact(values, changed.view(values.dtype))
    changed = words.copy()
    changed[2] = patterns[3]
    with pytest.raises(LosslessVerificationError, match="pixel bits differ"):
        assert_bit_exact(values, changed.view(values.dtype))


def test_complex_strided_and_empty_arrays():
    words = np.array([0x7FC12345, 0x80000000, 0, 0x7FC54321], dtype="u4")
    values = np.tile(words.view("c8"), (8, 9)).T[::2]
    other = values.byteswap().view(values.dtype.newbyteorder())
    assert_bit_exact(values, other)
    assert_bit_exact(np.empty((0, 3), "u2"), np.empty((0, 3), ">u2"))


@pytest.mark.parametrize("other,reason", [
    (np.zeros((2, 3), "u2"), "shape differs"),
    (np.zeros((3, 2), "i2"), "dtype differs"),
    (np.zeros((3, 2), "u4"), "dtype differs"),
    (b"not an image", "not an array"),
])
def test_image_metadata_mismatch(other, reason):
    with pytest.raises(LosslessVerificationError, match=reason):
        assert_bit_exact(np.zeros((3, 2), "u2"), other, codec="fixture")


def test_actual_lossy_image_is_rejected():
    pytest.importorskip("opencodecs.codecs._jpeg")
    original = np.random.default_rng(83).integers(0, 256, (48, 61, 3), dtype="u1")
    encoded = segments.encode_segment(original, "jpeg", level=60)
    with pytest.raises(LosslessVerificationError, match="pixel bits differ"):
        verify_segment(original, encoded, "jpeg")


def test_actual_lossless_image_roundtrip():
    pytest.importorskip("opencodecs.codecs._jpeg2k")
    original = np.arange(71 * 53, dtype="u2").reshape(71, 53)
    encoded = segments.encode_segment(original, "jpeg2000", lossless=True, numthreads=1)
    verify_segment(original, encoded, "jpeg2000", {"numthreads": 1})


def test_shared_decode_resolution_and_options(monkeypatch):
    calls = []
    def decode(data, codec, **kwargs):
        calls.append((data, codec, kwargs))
        return b"abc"
    monkeypatch.setattr(segments, "decode_segment", decode)
    options = {"numthreads": 3}
    verify_segment(b"abc", b"encoded", "none", options)
    assert calls == [(b"encoded", "none", options)]
    assert options == {"numthreads": 3}
    with pytest.raises(ValueError, match="do not supply out"):
        verify_segment(b"abc", b"encoded", "none", {"out": bytearray(3)})


def test_shared_decoder_obeys_outer_worker_budget(monkeypatch):
    original = np.arange(6, dtype="u2").reshape(2, 3)
    calls = []
    def decode(encoded, **kwargs):
        calls.append(kwargs["numthreads"])
        return original.copy()
    monkeypatch.setattr(segments, "_lookup_fn", lambda code, side: decode)
    with DeferredVerification() as verifier:
        verifier.submit(verify_segment, original, b"encoded", "jpeg2000",
                        {"numthreads": 8})
        verifier.finish()
    assert calls == [1]


def test_byte_codec_array_representation_is_exact():
    data = np.arange(123, dtype=">u2")
    verify_segment(data, data.tobytes(), "none")
    with pytest.raises(LosslessVerificationError, match="bytes differ"):
        verify_segment(data, data.astype("<u2").tobytes(), "none")


def test_output_lifetimes_and_bounded_blocks(monkeypatch):
    original = np.arange(100000, dtype="u8").reshape(100, 1000).T
    encoded_storage = bytearray(b"owned encoded bytes")
    encoded = memoryview(encoded_storage).toreadonly()
    references = []
    def decode(payload):
        assert payload is encoded
        result = original.copy()
        references.append(weakref.ref(result))
        return result
    verify_with_decoder(original, encoded, decode, codec="fixture")
    gc.collect()
    assert references[0]() is None
    assert bytes(encoded) == bytes(encoded_storage)
    assert verification_reservation(original) == original.nbytes + 4 * 65536
    assert verification_reservation(3) == 15
    with pytest.raises(ValueError):
        verification_reservation(-1)


def test_concurrent_verification_failure_joins_workers():
    lock = threading.Lock()
    active = 0
    def work(number):
        nonlocal active
        with lock:
            active += 1
        try:
            raw = bytes([number]) * 10000
            verify_with_decoder(raw, raw, lambda value: value if number != 2 else b"bad")
            return number
        finally:
            with lock:
                active -= 1
    with ThreadPoolExecutor(3) as executor:
        with pytest.raises(LosslessVerificationError):
            list(map_bounded(work, range(12), workers=3, max_pending=4,
                             budget=WorkerBudget(3), executor=executor))
        assert active == 0
        assert executor.submit(lambda: 9).result() == 9


def test_custom_decoder_error_and_unsupported_objects():
    def broken(_):
        raise OSError("decoder stopped")
    with pytest.raises(LosslessVerificationError, match="decoder stopped") as error:
        verify_with_decoder(b"raw", b"encoded", broken, codec="gzip")
    assert isinstance(error.value.__cause__, OSError)
    with pytest.raises(LosslessVerificationError, match="unsupported"):
        assert_bit_exact(np.array([object()]), np.array([object()]))


def test_deferred_verification_overlaps_and_bounds_pending():
    from opencodecs.core.pipeline import native_workers
    started, release = threading.Event(), threading.Event()
    def check(value):
        started.set()
        assert release.wait(5)
        assert native_workers(7) == 1
        return value
    with DeferredVerification() as verifier:
        verifier.submit(check, 17)
        try:
            assert started.wait(5)
            with pytest.raises(RuntimeError, match="pending"):
                verifier.submit(check, 18)
            # The producer can encode its next payload while verification waits.
            assert zlib.decompress(zlib.compress(b"next payload")) == b"next payload"
        finally:
            release.set()
        assert verifier.finish() == 17
        assert verifier.finish() is None
    with pytest.raises(RuntimeError, match="closed"):
        verifier.submit(check, 19)


def test_deferred_failure_propagates_and_does_not_mask_producer_error():
    def fail():
        raise LosslessVerificationError("mismatch")
    with pytest.raises(LosslessVerificationError, match="mismatch"):
        with DeferredVerification() as verifier:
            verifier.submit(fail)
    verifier.close()
    with pytest.raises(OSError, match="producer stopped"):
        with DeferredVerification() as verifier:
            verifier.submit(fail)
            raise OSError("producer stopped")


def test_last_comparison_block_is_verified():
    original = np.zeros(100003, dtype="u8")
    changed = original.copy()
    changed[-1] = 1
    with pytest.raises(LosslessVerificationError, match="pixel bits differ"):
        assert_bit_exact(original, changed)
    with pytest.raises(LosslessVerificationError, match="bytes differ"):
        assert_bit_exact(original.tobytes(), changed.tobytes())


def test_comparison_scratch_is_bounded_for_large_strided_images():
    original = np.arange(1024 * 1024, dtype="u8").reshape(1024, 1024).T
    decoded = original.byteswap().view(original.dtype.newbyteorder())
    tracemalloc.start()
    try:
        assert_bit_exact(original, decoded)
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    # Both full images exist before tracing. Comparison must not flatten-copy
    # either complete image or allocate an image-sized equality matrix.
    assert peak < 512 * 1024
