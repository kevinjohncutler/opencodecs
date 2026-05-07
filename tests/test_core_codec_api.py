"""Tests for opencodecs.core.codec — Reader/Writer/Codec ABCs + registry.

Most coverage of these classes comes indirectly through real codecs, but
the ABC default methods (``Reader.read``, ``Reader.__getitem__`` fallback,
``Codec.__repr__``, registry lookup helpers) are easier to verify with
small targeted tests against minimal subclasses.
"""

from __future__ import annotations

import io

import numpy as np
import pytest

from opencodecs.core.codec import (
    Codec,
    Reader,
    Writer,
    _SingleFrameReader,
    _resolve_codec,
    codec_for_bytes,
    codec_for_path,
    get_codec,
    has_codec,
    list_codecs,
    register_codec,
)


# ---------------------------------------------------------------------------
# Reader ABC default methods
# ---------------------------------------------------------------------------


class _ListReader(Reader):
    """Minimal Reader for testing default method behavior."""

    is_chunked = True

    def __init__(self, frames: list[np.ndarray]):
        self._frames = frames
        self.n_frames = len(frames)
        self.shape = frames[0].shape if frames else ()
        self.dtype = frames[0].dtype if frames else np.dtype("uint8")

    def iter_frames(self):
        yield from self._frames


def test_reader_iter_protocol():
    """`for frame in reader` should work via __iter__."""
    frames = [np.zeros((4, 4), dtype=np.uint8), np.ones((4, 4), dtype=np.uint8)]
    r = _ListReader(frames)
    out = list(r)
    assert len(out) == 2
    np.testing.assert_array_equal(out[1], np.ones((4, 4), dtype=np.uint8))


def test_reader_getitem_fallback():
    """Default __getitem__ iterates to the requested index."""
    frames = [np.full((2, 2), i, dtype=np.uint8) for i in range(3)]
    r = _ListReader(frames)
    np.testing.assert_array_equal(r[1], np.full((2, 2), 1, dtype=np.uint8))


def test_reader_getitem_out_of_range_raises():
    frames = [np.zeros((2, 2), dtype=np.uint8)]
    r = _ListReader(frames)
    with pytest.raises(IndexError):
        _ = r[42]


def test_reader_getitem_unsupported_when_not_chunked():
    """A non-chunked reader should refuse [idx] access."""
    class NotChunked(Reader):
        is_chunked = False
        n_frames = 1

        def iter_frames(self):
            yield np.zeros((1,), dtype=np.uint8)

    with pytest.raises(TypeError, match="random access"):
        _ = NotChunked()[0]


def test_reader_read_default_single_frame():
    """Reader.read() on a 1-frame stream returns that frame, not stacked."""
    arr = np.full((4, 4), 5, dtype=np.uint8)
    r = _ListReader([arr])
    np.testing.assert_array_equal(r.read(), arr)


def test_reader_read_default_multi_frame_stacks():
    """Reader.read() on N>1 frames returns a stack along axis 0."""
    frames = [np.full((4, 4), i, dtype=np.uint8) for i in range(3)]
    r = _ListReader(frames)
    out = r.read()
    assert out.shape == (3, 4, 4)


def test_reader_read_empty_raises():
    r = _ListReader([])
    with pytest.raises(ValueError, match="empty"):
        r.read()


def test_reader_context_manager_calls_close():
    closed = []

    class TrackClose(Reader):
        is_chunked = False
        n_frames = 0

        def iter_frames(self):
            return iter(())

        def close(self):
            closed.append(True)

    with TrackClose():
        pass
    assert closed == [True]


def test_single_frame_reader_round_trip():
    arr = np.arange(12).reshape(3, 4).astype(np.uint8)
    r = _SingleFrameReader(arr)
    assert r.shape == (3, 4)
    assert r.dtype == np.uint8
    assert r.n_frames == 1
    np.testing.assert_array_equal(r.read(), arr)
    np.testing.assert_array_equal(list(r.iter_frames())[0], arr)


# ---------------------------------------------------------------------------
# Writer ABC default behavior
# ---------------------------------------------------------------------------


def test_writer_context_manager_swallows_close_on_exception():
    """If body raises, Writer.__exit__ does NOT call close (per impl)
    and propagates the exception."""

    closed = []

    class W(Writer):
        def write_frame(self, arr, **opts):
            pass

        def close(self):
            closed.append(True)
            return None

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with W():
            raise Boom()
    # Per implementation, close is only called on clean exit.
    assert closed == []


def test_writer_context_manager_calls_close_on_clean_exit():
    closed = []

    class W(Writer):
        def write_frame(self, arr, **opts):
            pass

        def close(self):
            closed.append(True)
            return None

    with W():
        pass
    assert closed == [True]


# ---------------------------------------------------------------------------
# Codec ABC default behavior
# ---------------------------------------------------------------------------


class _StubCodec(Codec):
    name = "_test_stub"
    file_extensions = (".tstub",)
    can_encode = False
    can_decode = False


def test_codec_encode_unsupported_raises():
    c = _StubCodec()
    with pytest.raises(NotImplementedError, match="encode not supported"):
        c.encode(np.zeros((1,)))


def test_codec_decode_unsupported_raises():
    c = _StubCodec()
    with pytest.raises(NotImplementedError, match="decode not supported"):
        c.decode(b"")


def test_codec_signature_default_returns_false():
    """Default signature() always returns False (no magic by default)."""
    assert _StubCodec().signature(b"\x00" * 32) is False


def test_codec_repr_includes_capability_flags():
    """__repr__ should mark native/delegate/stub + rw/ro/wo + multi/chunked/parallel."""

    class Cap(Codec):
        name = "_caps"
        has_native = True
        can_encode = True
        can_decode = True
        multi_frame = True
        chunked = True
        parallel_decode = True

    r = repr(Cap())
    assert "native" in r and "rw" in r and "multi" in r and "chunked" in r and "parallel" in r


def test_codec_repr_decode_only():
    class RO(Codec):
        name = "_ro"
        has_delegate = True
        can_decode = True

    r = repr(RO())
    assert "delegate" in r and "ro" in r


def test_codec_repr_encode_only():
    class WO(Codec):
        name = "_wo"
        can_encode = True

    r = repr(WO())
    assert "stub" in r and "wo" in r


def test_codec_open_default_wraps_decode_in_single_frame_reader():
    """A codec without a streaming open() should fall back to a
    full-decode-then-wrap path."""

    class FakeJpeg(Codec):
        name = "_fake_jpeg"
        can_decode = True

        def decode(self, src, **opts):
            return np.full((4, 4), 7, dtype=np.uint8)

    r = FakeJpeg().open(b"unused")
    assert isinstance(r, _SingleFrameReader)
    np.testing.assert_array_equal(r.read(), np.full((4, 4), 7, dtype=np.uint8))


# ---------------------------------------------------------------------------
# Registry: register / lookup / list / has
# ---------------------------------------------------------------------------


def test_register_codec_requires_name():
    class Anon(Codec):
        name = ""

    with pytest.raises(ValueError, match="no .name"):
        register_codec(Anon())


def test_register_and_lookup_by_name_and_alias():
    class Mine(Codec):
        name = "_test_mine"
        aliases = ("_mine_alt",)

    inst = Mine()
    register_codec(inst)
    try:
        assert get_codec("_test_mine") is inst
        assert get_codec("_mine_alt") is inst
        # Case-insensitive
        assert get_codec("_TEST_MINE") is inst
    finally:
        # Clean up registry to avoid leaking state into other tests.
        from opencodecs.core.codec import _REGISTRY
        _REGISTRY.pop("_test_mine", None)
        _REGISTRY.pop("_mine_alt", None)


def test_get_codec_unknown_raises_keyerror():
    with pytest.raises(KeyError, match="no codec named"):
        get_codec("_definitely_not_a_codec_xyz")


def test_has_codec_true_for_existing():
    """A real registered codec exists."""
    assert has_codec("zstd") is True


def test_has_codec_false_for_missing():
    assert has_codec("_not_a_real_codec") is False


def test_has_codec_op_filter():
    """has_codec(name, op='encode'/'decode') checks capability flag."""
    # zstd encodes and decodes
    assert has_codec("zstd", op="encode") is True
    assert has_codec("zstd", op="decode") is True


def test_has_codec_op_on_missing_returns_false():
    assert has_codec("_missing_codec", op="encode") is False


def test_list_codecs_returns_descriptors():
    descriptors = list_codecs()
    assert all(isinstance(d, dict) for d in descriptors)
    names = {d["name"] for d in descriptors}
    # A few that we know are always registered
    assert {"zstd", "lz4", "brotli"} <= names
    # Schema sanity
    sample = descriptors[0]
    for key in ("name", "extensions", "native", "delegate",
                "encode", "decode", "multi_frame", "chunked"):
        assert key in sample


# ---------------------------------------------------------------------------
# codec_for_path / codec_for_bytes
# ---------------------------------------------------------------------------


def test_codec_for_path_resolves_known_extensions():
    c = codec_for_path("foo.png")
    assert c.name == "png"


def test_codec_for_path_unknown_extension_raises():
    with pytest.raises(KeyError):
        codec_for_path("foo.unknownextension")


def test_codec_for_path_no_extension_raises():
    with pytest.raises(KeyError, match="no extension"):
        codec_for_path("noext")


def test_codec_for_bytes_recognizes_png_signature():
    """PNG magic: 0x89 P N G ..."""
    head = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
    c = codec_for_bytes(head)
    assert c.name == "png"


def test_codec_for_bytes_unrecognized_raises():
    with pytest.raises(KeyError):
        codec_for_bytes(b"\x00\x01\x02\x03\x04")


# ---------------------------------------------------------------------------
# _resolve_codec: dispatcher used by top-level read/open
# ---------------------------------------------------------------------------


def test_resolve_codec_explicit_format_wins():
    c = _resolve_codec(b"\x00\x00\x00", format="zstd")
    assert c.name == "zstd"


def test_resolve_codec_path_by_extension(tmp_path):
    p = tmp_path / "x.png"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)
    assert _resolve_codec(p).name == "png"


def test_resolve_codec_path_falls_back_to_magic_bytes(tmp_path):
    """Unknown extension falls back to peeking magic bytes."""
    p = tmp_path / "mystery.bin"
    p.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 24)
    assert _resolve_codec(p).name == "png"


def test_resolve_codec_bytes_input_uses_signatures():
    head = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
    assert _resolve_codec(head).name == "png"


def test_resolve_codec_file_like_seekable_input():
    head = b"\x89PNG\r\n\x1a\n" + b"\x00" * 24
    bio = io.BytesIO(head)
    c = _resolve_codec(bio)
    assert c.name == "png"
    # Position must be reset so the caller can read the bytes again.
    assert bio.tell() == 0


def test_resolve_codec_unsupported_type_raises():
    with pytest.raises(KeyError):
        _resolve_codec(42)


def test_resolve_codec_unrecognized_path_raises(tmp_path):
    """Path with no extension and unrecognized magic — must raise."""
    p = tmp_path / "noext"
    p.write_bytes(b"\x00\x01\x02\x03")
    with pytest.raises(KeyError):
        _resolve_codec(p)
