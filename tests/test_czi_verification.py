"""CZI segment verification uses the real decoder before destination emission."""

import threading

import numpy as np
import pytest

import opencodecs._czi_writer as implementation
from opencodecs._czi_reader import CziReader, CziPyramidReader
from opencodecs._czi_writer import CziWriter, CziPyramidWriter, CziWriterError
from opencodecs.core.verification import (
    DeferredVerification, LosslessVerificationError, assert_bit_exact,
)


def _native(compression, hilo=True):
    if compression != "none":
        pytest.importorskip("opencodecs.codecs._zstd")
    if compression == "zstdhdr" and hilo:
        pytest.importorskip("opencodecs.codecs._bytetools")


def _image(dtype):
    if dtype == "f4":
        bits = np.array([0, 0x80000000, 0x7FC12345, 0xFFC43210,
                         0x3F800000, 0x7F800000], dtype="u4")
        return np.tile(bits.view("f4"), (31, 11))
    return np.arange(31 * 66, dtype="u2").astype(dtype).reshape(31, 66)


@pytest.mark.parametrize("compression,hilo", [
    ("none", True), ("zstd", True), ("zstdhdr", True), ("zstdhdr", False),
])
@pytest.mark.parametrize("dtype", ["u1", "u2", "f4"])
def test_verified_file_matches_unverified_and_actual_reader(tmp_path, compression, hilo, dtype):
    _native(compression, hilo)
    image = _image(dtype)
    verified, plain = tmp_path / "verified.czi", tmp_path / "plain.czi"
    for path, verify in ((verified, True), (plain, False)):
        writer = CziWriter(path, compression=compression, hilo=hilo, verify=verify)
        writer.write_frame(image)
        writer.close()
        writer.close()
    assert verified.read_bytes() == plain.read_bytes()
    with CziReader(verified) as reader:
        assert len(reader.entries) == 1
        assert_bit_exact(image, reader[0])


def test_raw_verified_file_has_independent_reader_parity(tmp_path):
    reference = pytest.importorskip("czifile")
    image = _image("u2")
    path = tmp_path / "reference.czi"
    with CziWriter(path, verify=True) as writer:
        writer.write(image)
    with reference.CziFile(path) as reader:
        assert_bit_exact(image, np.squeeze(reader.asarray()))


@pytest.mark.parametrize("compression", ["none", "zstd", "zstdhdr"])
def test_corrupt_first_payload_never_reaches_destination(tmp_path, monkeypatch, compression):
    _native(compression)
    build = implementation._build_subblock
    emitted = []
    def corrupt(*args, **kwargs):
        segment, entry = build(*args, **kwargs)
        payload = bytearray(segment.payload)
        payload[3 if compression == "zstdhdr" else 0] ^= 1
        return segment.with_payload(bytes(payload)), entry
    monkeypatch.setattr(implementation, "_build_subblock", corrupt)
    monkeypatch.setattr(CziWriter, "_emit_subblock", lambda *args: emitted.append(args))
    path = tmp_path / "existing.czi"
    path.write_bytes(b"previous destination")
    writer = CziWriter(path, compression=compression, verify=True)
    writer.write(_image("u2"))
    reason = "pixel bits differ" if compression == "none" else "decode failed"
    with pytest.raises(LosslessVerificationError, match=reason):
        writer.close()
    assert not emitted
    assert path.read_bytes() == b"previous destination"
    assert writer._pending_verified is None
    assert writer._file is None
    with pytest.raises(CziWriterError, match="closed"):
        writer.write(_image("u2"))
    writer.close()


def test_corrupt_later_payload_is_not_emitted(tmp_path, monkeypatch):
    build, emit = implementation._build_subblock, CziWriter._emit_subblock
    built, emitted = 0, []
    def corrupt_second(*args, **kwargs):
        nonlocal built
        segment, entry = build(*args, **kwargs)
        built += 1
        if built == 2:
            payload = bytearray(segment.payload)
            payload[0] ^= 1
            segment = segment.with_payload(bytes(payload))
        return segment, entry
    def record(self, segment, entry):
        emitted.append(entry.copy())
        emit(self, segment, entry)
    monkeypatch.setattr(implementation, "_build_subblock", corrupt_second)
    monkeypatch.setattr(CziWriter, "_emit_subblock", record)
    writer = CziWriter(tmp_path / "partial.czi", verify=True)
    writer.write(_image("u2"))
    writer.write(_image("u2"))
    assert len(emitted) == 1
    with pytest.raises(LosslessVerificationError):
        writer.close()
    assert len(emitted) == 1
    assert writer._file is None and writer._pending_verified is None


def test_reused_array_is_snapshotted_before_async_verification(tmp_path, monkeypatch):
    verify = CziWriter._verify_subblock
    started, release = threading.Event(), threading.Event()
    count = 0
    def delayed(original, segment, entry):
        nonlocal count
        count += 1
        if count == 1:
            started.set()
            assert release.wait(5)
        verify(original, segment, entry)
    monkeypatch.setattr(CziWriter, "_verify_subblock", staticmethod(delayed))
    array = np.full((21, 37), 9, dtype="u2")
    expected = [array.copy()]
    path = tmp_path / "reused.czi"
    with CziWriter(path, verify=True) as writer:
        writer.write(array)
        try:
            assert started.wait(5)
            array.fill(17)
        finally:
            release.set()
        for value in (17, 31, 48):
            array.fill(value)
            expected.append(array.copy())
            writer.write(array)
        array.fill(65535)
    with CziReader(path) as reader:
        assert len(reader.entries) == len(expected)
        for index, pixels in enumerate(expected):
            assert_bit_exact(pixels, reader[index])


def test_next_segment_encoding_overlaps_previous_verification(tmp_path, monkeypatch):
    build, verify = implementation._build_subblock, CziWriter._verify_subblock
    started, next_encoded = threading.Event(), threading.Event()
    built, checked = 0, 0
    def check(original, segment, entry):
        nonlocal checked
        checked += 1
        if checked == 1:
            started.set()
            assert next_encoded.wait(5), "writer waited before encoding its next segment"
        verify(original, segment, entry)
    def encode(*args, **kwargs):
        nonlocal built
        result = build(*args, **kwargs)
        built += 1
        if built == 2:
            assert started.wait(5)
            next_encoded.set()
        return result
    monkeypatch.setattr(CziWriter, "_verify_subblock", staticmethod(check))
    monkeypatch.setattr(implementation, "_build_subblock", encode)
    path = tmp_path / "overlap.czi"
    with CziWriter(path, verify=True) as writer:
        writer.write(_image("u1"))
        writer.write(_image("u1"))
    assert checked == built == 2
    with CziReader(path) as reader:
        assert len(reader.entries) == 2


def test_encode_failure_joins_verification_and_preserves_original_error(tmp_path, monkeypatch):
    build = implementation._build_subblock
    started, release, finished = (threading.Event() for _ in range(3))
    built = 0
    def check(*args):
        started.set()
        try:
            assert release.wait(5)
            raise LosslessVerificationError("earlier verification failed")
        finally:
            finished.set()
    def encode(*args, **kwargs):
        nonlocal built
        built += 1
        if built == 2:
            release.set()
            raise OSError("encoder stopped")
        return build(*args, **kwargs)
    monkeypatch.setattr(CziWriter, "_verify_subblock", staticmethod(check))
    monkeypatch.setattr(implementation, "_build_subblock", encode)
    path = tmp_path / "failed.czi"
    writer = CziWriter(path, verify=True)
    writer.write(_image("u2"))
    assert started.wait(5)
    with pytest.raises(OSError, match="encoder stopped"):
        writer.write(_image("u2"))
    assert finished.is_set()
    assert writer._pending_verified is None and writer._file is None
    assert not path.exists()
    writer.close()


def test_destination_failure_closes_verification_worker(tmp_path, monkeypatch):
    def fail(*args):
        raise OSError("destination stopped")
    monkeypatch.setattr(CziWriter, "_emit_subblock", fail)
    writer = CziWriter(tmp_path / "sink.czi", verify=True)
    writer.write(_image("u1"))
    with pytest.raises(OSError, match="destination stopped"):
        writer.close()
    with pytest.raises(RuntimeError, match="closed"):
        writer._verification.submit(lambda: None)
    assert writer._pending_verified is None


def test_interrupted_verifier_shutdown_closes_emitted_file(tmp_path, monkeypatch):
    writer = CziWriter(tmp_path / "interrupted.czi", verify=True)
    writer.write(_image("u2"))
    close = writer._verification.close
    calls, handles = [], []
    def interrupted_once():
        calls.append(True)
        if len(calls) == 1:
            handles.append(writer._file)
            assert handles[0] is not None and not handles[0].closed
            raise KeyboardInterrupt
        close()
    monkeypatch.setattr(writer._verification, "close", interrupted_once)
    with pytest.raises(KeyboardInterrupt):
        writer.close()
    assert len(calls) == 2
    assert handles[0].closed
    assert writer._file is None and writer._pending_verified is None
    with pytest.raises(RuntimeError, match="closed"):
        writer._verification.submit(lambda: None)
    writer.close()


def test_final_close_drains_last_segment_and_empty_close_stops_worker(tmp_path):
    path = tmp_path / "last.czi"
    writer = CziWriter(path, verify=True)
    writer.write(_image("u2"))
    assert not path.exists()
    writer.close()
    with CziReader(path) as reader:
        assert_bit_exact(_image("u2"), reader[0])
    empty = CziWriter(tmp_path / "empty.czi", verify=True)
    with pytest.raises(CziWriterError, match="no frames"):
        empty.close()
    with pytest.raises(RuntimeError, match="closed"):
        empty._verification.submit(lambda: None)


def test_pyramid_verification_preserves_level_layout(tmp_path):
    base = _image("u2")
    levels = [base, base[::2, ::2], base[::4, ::4]]
    path = tmp_path / "pyramid.czi"
    with CziPyramidWriter(path, verify=True) as writer:
        writer.write_pyramid(levels)
    with CziPyramidReader(CziReader(path)) as reader:
        assert reader.n_levels == len(levels)
        for index, expected in enumerate(levels):
            actual = reader.read_region(index, y=(0, expected.shape[0]),
                                        x=(0, expected.shape[1]))
            assert_bit_exact(expected, actual)


def test_deferred_interrupt_keeps_running_task_reserved():
    started, release = threading.Event(), threading.Event()
    def check():
        started.set()
        assert release.wait(5)
        return 29
    with DeferredVerification() as verifier:
        verifier.submit(check)
        assert started.wait(5)
        future = verifier._pending
        class InterruptOnce:
            interrupted = False
            def result(self):
                if not self.interrupted:
                    self.interrupted = True
                    raise KeyboardInterrupt
                return future.result()
            def done(self):
                return future.done()
        pending = verifier._pending = InterruptOnce()
        try:
            with pytest.raises(KeyboardInterrupt):
                verifier.finish()
            assert verifier._pending is pending
            with pytest.raises(RuntimeError, match="pending"):
                verifier.submit(lambda: None)
        finally:
            release.set()
        assert verifier.finish() == 29
        assert verifier._pending is None


@pytest.mark.parametrize("byteorder", ["=", "<", ">"])
def test_contiguous_comparison_avoids_element_iteration(byteorder):
    class NoFlatIteration(np.ndarray):
        @property
        def flat(self):
            raise AssertionError("contiguous comparison must use bounded byte blocks")
    original = _image("f4")
    if byteorder == ">" and np.little_endian:
        original = original.byteswap().view(original.dtype.newbyteorder(">"))
    elif byteorder == "<" and not np.little_endian:
        original = original.byteswap().view(original.dtype.newbyteorder("<"))
    original = original.view(NoFlatIteration)
    decoded = original.copy()
    assert_bit_exact(original, decoded)
    decoded.view("u1").reshape(-1)[-1] ^= 1
    with pytest.raises(LosslessVerificationError, match="pixel bits differ"):
        assert_bit_exact(original, decoded)
