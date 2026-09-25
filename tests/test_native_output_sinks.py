"""Destination draining, native buffer bounds, and callback error propagation."""

import io

import numpy as np
import pytest

from opencodecs.core.codec import get_codec


class ShortSink(io.BytesIO):
    def write(self, data):
        return super().write(data[:17])


class BrokenSink(io.BytesIO):
    def write(self, data):
        raise OSError("injected destination failure")


@pytest.mark.parametrize("name", ["gif", "jxl"])
def test_animation_drains_before_close_with_independent_parity(name):
    imagecodecs = pytest.importorskip("imagecodecs")
    codec = get_codec(name)
    options = {"animation": True, "lossless": True, "numthreads": 1} if name == "jxl" else {}
    frames = np.random.default_rng(51).integers(0, 256, (40, 48, 64), dtype="u1")
    sink = ShortSink()
    writer = codec.writer(sink, **options)
    capacities = []
    for frame in frames:
        writer.write_frame(frame)
        capacities.append(writer._inner.output_buffer_capacity)
    assert len(sink.getvalue()) > 0
    assert writer.close() is None
    assert writer.close() is None
    assert not sink.closed
    assert max(capacities) <= 4 * frames[0].nbytes + 65536
    with codec.writer(**options) as memory_writer:
        for frame in frames:
            memory_writer.write_frame(frame)
    blob = memory_writer.close()
    assert sink.getvalue() == blob
    decoded = getattr(imagecodecs, "gif_decode" if name == "gif" else "jpegxl_decode")(blob)
    if name == "gif":
        np.testing.assert_array_equal(decoded[..., :3], np.repeat(frames[..., None], 3, axis=-1))
    else:
        np.testing.assert_array_equal(decoded, frames)
    assert codec.streaming_output and codec.writer_buffering == "frame"


@pytest.mark.parametrize("name", ["gif", "jxl"])
def test_animation_destination_failure_propagates_and_preserves_borrowed_stream(name):
    codec = get_codec(name)
    options = {"animation": True, "lossless": True, "numthreads": 1} if name == "jxl" else {}
    sink = BrokenSink()
    writer = codec.writer(sink, **options)
    arr = np.arange(1024, dtype="u1").reshape(32, 32)
    with pytest.raises(OSError, match="injected destination"):
        with writer:
            writer.write_frame(arr)
            writer.write_frame(arr)
    assert not sink.closed


def test_brunsli_direct_transcode_roundtrip_and_short_writes(tmp_path):
    from opencodecs.codecs._brunsli import encode_jpeg, decode_jpeg
    jpeg_codec = get_codec("jpeg")
    arr = np.random.default_rng(7).integers(0, 256, (64, 96, 3), dtype="u1")
    jpeg = jpeg_codec.encode(arr)
    expected = encode_jpeg(jpeg)
    sink = ShortSink()
    assert encode_jpeg(jpeg, dest=sink) is None
    assert sink.getvalue() == expected
    assert not sink.closed
    recovered = ShortSink()
    assert decode_jpeg(expected, dest=recovered) is None
    assert recovered.getvalue() == jpeg
    assert not recovered.closed
    path = tmp_path / "out.brn"
    assert get_codec("brunsli").encode(jpeg, dest=path) is None
    assert path.read_bytes() == expected
    path = tmp_path / "restored.jpg"
    assert get_codec("brunsli").decode(expected, asjpeg=True, dest=path) is None
    assert path.read_bytes() == jpeg


def test_brunsli_callback_error_propagates():
    from opencodecs.codecs._brunsli import encode_jpeg, decode_jpeg
    jpeg = get_codec("jpeg").encode(np.arange(4096, dtype="u1").reshape(64, 64))
    for fn, source in ((encode_jpeg, jpeg), (decode_jpeg, encode_jpeg(jpeg))):
        sink = BrokenSink()
        with pytest.raises(OSError, match="injected destination"):
            fn(source, dest=sink)
        assert not sink.closed


@pytest.mark.parametrize("name", ["gif", "jxl"])
def test_owned_animation_file_is_visible_before_close(tmp_path, name):
    codec = get_codec(name)
    options = {"animation": True, "lossless": True, "numthreads": 1} if name == "jxl" else {}
    path = tmp_path / f"visible.{name}"
    with codec.writer(path, **options) as writer:
        frame = np.arange(1024, dtype="u1").reshape(32, 32)
        writer.write_frame(frame)
        writer.write_frame(frame)
        assert path.stat().st_size > 0


def test_jxl_final_frame_failure_releases_pending_frame():
    writer = get_codec("jxl").writer(BrokenSink(), animation=True, lossless=True, numthreads=1)
    writer.write_frame(np.zeros((16, 16), dtype="u1"))
    with pytest.raises(OSError, match="injected destination"):
        writer.close()
    assert writer._pending is None
    assert writer.close() is None
