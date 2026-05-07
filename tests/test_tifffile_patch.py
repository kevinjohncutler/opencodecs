"""Tests for opencodecs.tifffile_patch.

The patch module repoints tifffile's compression dispatch at opencodecs's
native codecs. Coverage was 0% before this file. Tests cover:

  * Each adapter function (``zstd_decode`` / ``deflate_encode`` / etc.)
    round-trips bytes through opencodecs's Cython codecs with the exact
    signature tifffile uses.
  * ``install()`` / ``uninstall()`` are idempotent and don't leak state.
  * The ``patched()`` context manager installs and reverts cleanly even
    when the body raises.
  * After install, ``tifffile.tifffile.imagecodecs`` resolves to a shim
    that forwards original attributes but overrides ours.
"""

from __future__ import annotations

import pytest

import opencodecs as oc
from opencodecs import tifffile_patch as patch


# ---------------------------------------------------------------------------
# Direct adapter functions: each must round-trip the bytes
# ---------------------------------------------------------------------------


def _need(codec_name: str) -> None:
    if not oc.has_codec(codec_name):
        pytest.skip(f"codec {codec_name!r} not available")


def test_zstd_roundtrip():
    _need("zstd")
    payload = b"opencodecs tifffile patch round-trip" * 100
    enc = patch.zstd_encode(payload, level=3)
    assert patch.zstd_decode(enc) == payload


def test_zstd_decode_with_out_size_hint():
    """tifffile passes ``out=`` as a bytes-buffer hint; the adapter
    coerces the result to that length when ``out`` is bytes-like."""
    _need("zstd")
    payload = b"x" * 1024
    enc = patch.zstd_encode(payload)
    out_buf = bytearray(1024)
    decoded = patch.zstd_decode(enc, out=out_buf)
    # Bytes-like ``out`` → length-clamped return.
    assert len(decoded) == 1024
    assert decoded == payload


def test_zstd_decode_with_int_out():
    """``out`` as an int is a size hint we currently ignore — verify
    we still return correctly-sized bytes."""
    _need("zstd")
    payload = b"y" * 256
    enc = patch.zstd_encode(payload)
    decoded = patch.zstd_decode(enc, out=256)
    assert decoded == payload


def test_deflate_roundtrip():
    _need("deflate")
    payload = b"deflate test" * 50
    enc = patch.deflate_encode(payload, level=6)
    assert patch.deflate_decode(enc) == payload


def test_zlib_aliases_to_deflate():
    """tifffile sometimes calls zlib_*; they must alias to deflate_*."""
    _need("deflate")
    payload = b"zlib alias" * 100
    enc = patch.zlib_encode(payload)
    assert patch.zlib_decode(enc) == payload


def test_lz4_roundtrip():
    _need("lz4")
    payload = b"lz4 frame" * 200
    enc = patch.lz4_encode(payload)
    assert patch.lz4_decode(enc) == payload


def test_png_roundtrip():
    _need("png")
    import numpy as np
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (16, 24, 3), dtype=np.uint8)
    enc = patch.png_encode(arr)
    decoded = patch.png_decode(enc)
    np.testing.assert_array_equal(np.squeeze(decoded), arr)


def test_jpeg_roundtrip_lossy():
    _need("jpeg")
    import numpy as np
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    enc = patch.jpeg_encode(arr, level=90)
    # JPEG is lossy — just verify decode succeeds & shape matches.
    decoded = patch.jpeg_decode(enc)
    assert np.squeeze(decoded).shape == arr.shape


def test_webp_roundtrip_lossless():
    _need("webp")
    import numpy as np
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    enc = patch.webp_encode(arr, lossless=True)
    decoded = patch.webp_decode(enc)
    np.testing.assert_array_equal(np.squeeze(decoded), arr)


def test_jpeg2k_roundtrip_lossless():
    _need("jpeg2k")
    import numpy as np
    rng = np.random.default_rng(0)
    # 64x64 — large enough for libopenjp2's default 6-level wavelet
    # decomposition. Smaller (e.g. 16x16) fails opj_start_compress.
    arr = rng.integers(0, 256, (64, 64, 3), dtype=np.uint8)
    enc = patch.jpeg2k_encode(arr, lossless=True)
    decoded = patch.jpeg2k_decode(enc)
    np.testing.assert_array_equal(np.squeeze(decoded), arr)


def test_jpegxl_roundtrip_lossless():
    if not oc.has_codec("jxl"):
        pytest.skip("jxl backend not built")
    import numpy as np
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    enc = patch.jpegxl_encode(arr, lossless=True)
    decoded = patch.jpegxl_decode(enc)
    np.testing.assert_array_equal(np.squeeze(decoded), arr)


# ---------------------------------------------------------------------------
# Buffer-protocol input: tifffile passes memoryview / bytearray over the wire
# ---------------------------------------------------------------------------


def test_adapters_accept_memoryview():
    _need("zstd")
    payload = b"memoryview test" * 100
    enc = patch.zstd_encode(memoryview(payload))
    assert patch.zstd_decode(memoryview(enc)) == payload


def test_adapters_accept_bytearray():
    _need("zstd")
    payload = b"bytearray test" * 100
    enc = patch.zstd_encode(bytearray(payload))
    assert patch.zstd_decode(bytearray(enc)) == payload


# ---------------------------------------------------------------------------
# install() / uninstall() / patched() lifecycle
# ---------------------------------------------------------------------------


@pytest.fixture
def reset_patch_state():
    """Make sure each test starts and ends with patch uninstalled."""
    patch.uninstall()
    yield
    patch.uninstall()


def test_install_is_idempotent(reset_patch_state):
    """Calling install() twice in a row should not double-wrap."""
    patch.install()
    patch.install()  # second call returns silently
    assert patch._installed is True


def test_uninstall_when_not_installed_is_safe(reset_patch_state):
    """uninstall() before install() is a no-op."""
    patch.uninstall()  # before any install
    assert patch._installed is False


def test_install_replaces_tifffile_imagecodecs(reset_patch_state):
    """After install, tifffile.tifffile.imagecodecs must point at our shim.
    The shim's zstd_decode is our function, not the original."""
    pytest.importorskip("tifffile")
    import tifffile.tifffile as tt
    patch.install()
    assert tt.imagecodecs.zstd_decode is patch.zstd_decode
    # Forwarded attribute: imagecodecs has many more attrs than we override;
    # the shim should still expose them.
    assert hasattr(tt.imagecodecs, "TIFF") or hasattr(tt.imagecodecs, "version")


def test_uninstall_restores_original(reset_patch_state):
    """After install + uninstall, tifffile.imagecodecs is back to normal."""
    pytest.importorskip("tifffile")
    import tifffile.tifffile as tt
    original = tt.imagecodecs
    patch.install()
    assert tt.imagecodecs is not original  # shim swapped in
    patch.uninstall()
    assert tt.imagecodecs is original  # back to the original


def test_patched_context_manager_installs_and_reverts(reset_patch_state):
    pytest.importorskip("tifffile")
    import tifffile.tifffile as tt
    original = tt.imagecodecs

    assert tt.imagecodecs is original
    with patch.patched():
        assert tt.imagecodecs is not original
    assert tt.imagecodecs is original


def test_patched_reverts_on_exception(reset_patch_state):
    """If the body of patched() raises, uninstall must still run."""
    pytest.importorskip("tifffile")
    import tifffile.tifffile as tt
    original = tt.imagecodecs

    class Boom(Exception):
        pass

    with pytest.raises(Boom):
        with patch.patched():
            assert tt.imagecodecs is not original
            raise Boom()
    assert tt.imagecodecs is original
    assert patch._installed is False
