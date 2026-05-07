"""Tests for the optional-backend pattern that lets opencodecs import
cleanly on platforms where some Cython extensions didn't build.

On Mac and Linux, every codec backend builds, so the failure path of
``import_or_stubs`` is never exercised by the rest of the test suite.
This module forces that path with a fake non-existent module name and
verifies the stubs behave as documented:

  * Importing the codec adapter succeeds even when its backend is
    missing.
  * ``_HAVE_BACKEND`` reports the truth.
  * Calling a stubbed function raises ``ImportError`` (not segfault,
    not silent None return) with a message that points at INSTALL.md.

These guarantees are what makes the package usable on Windows-without-
libjxl, Linux-without-TJ-v3, etc. — the tests pass on every platform
because they don't depend on any specific extension being built.
"""

from __future__ import annotations

import pytest

from opencodecs.core._optional_backend import import_or_stubs


def test_real_module_import_succeeds():
    """Sanity: importing real symbols from a real module returns the symbols."""
    encode_fn, decode_fn, *_, have = import_or_stubs(
        "opencodecs.core._optional_backend",
        "import_or_stubs", "_stub_factory",
    )
    assert have is True
    # Real callables, not stubs.
    assert callable(encode_fn)
    assert callable(decode_fn)


def test_missing_module_returns_stubs():
    """Importing from a non-existent module yields stubs + have=False."""
    *attrs, have = import_or_stubs(
        "opencodecs.codecs._does_not_exist",
        "encode", "decode", "check_signature",
    )
    assert have is False
    assert len(attrs) == 3
    for stub in attrs:
        assert callable(stub)


def test_stub_call_raises_importerror():
    """Calling a stub raises ImportError — predictable type for catch blocks."""
    *attrs, have = import_or_stubs(
        "opencodecs.codecs._does_not_exist",
        "encode", "decode",
    )
    encode_stub, decode_stub = attrs
    with pytest.raises(ImportError):
        encode_stub(b"whatever")
    with pytest.raises(ImportError):
        decode_stub(b"whatever")


def test_stub_error_message_points_at_install_docs():
    """The ImportError message should mention INSTALL.md so users
    know how to fix it."""
    stub, _have = import_or_stubs(
        "opencodecs.codecs._does_not_exist", "encode",
    )
    with pytest.raises(ImportError, match="INSTALL.md"):
        stub(b"whatever")


def test_stub_error_message_includes_codec_name():
    """The error should name the codec so the user knows which one
    failed (especially when multiple imports failed at once)."""
    *attrs, have = import_or_stubs(
        "opencodecs.codecs._fake_zstd",
        "encode",
    )
    stub = attrs[0]
    with pytest.raises(ImportError, match="fake_zstd"):
        stub(b"x")


def test_stub_preserves_function_name():
    """For diagnostics: the stub's __name__ should be the attribute name
    so tracebacks make sense."""
    encode_stub, decode_stub, _ = import_or_stubs(
        "opencodecs.codecs._does_not_exist",
        "encode", "decode",
    )
    assert encode_stub.__name__ == "encode"
    assert decode_stub.__name__ == "decode"


def test_partial_failure_returns_all_stubs():
    """If the import fails, we get stubs for ALL requested names — not a
    mix of real + stub, which would be hard to reason about."""
    *attrs, have = import_or_stubs(
        "opencodecs.codecs._missing_codec",
        "a", "b", "c", "d", "e",
    )
    assert have is False
    assert len(attrs) == 5
    assert all(callable(s) for s in attrs)
    # Every stub should fail on call (none silently return).
    for stub in attrs:
        with pytest.raises(ImportError):
            stub()


def test_have_backend_flag_is_truthy_only_on_success():
    _, have_real = import_or_stubs(
        "opencodecs.core._optional_backend", "import_or_stubs",
    )
    _, have_fake = import_or_stubs(
        "opencodecs.codecs._missing", "encode",
    )
    assert have_real is True
    assert have_fake is False


# ---------------------------------------------------------------------------
# Verify each codec adapter exposes the contract
# ---------------------------------------------------------------------------


_ADAPTERS = [
    "_zstd_codec", "_lz4_codec", "_brotli_codec", "_blosc2_codec",
    "_deflate_codec", "_qoi_codec", "_png_codec", "_jpeg_codec",
    "_webp_codec", "_jpeg2k_codec", "_avif_codec", "_heif_codec",
    "_jxl_codec",
]


@pytest.mark.parametrize("modname", _ADAPTERS)
def test_adapter_has_have_backend_flag(modname):
    """Every codec adapter exposes a _HAVE_BACKEND flag for callers
    that want to check availability without try/except."""
    import importlib
    mod = importlib.import_module(f"opencodecs.{modname}")
    assert hasattr(mod, "_HAVE_BACKEND"), (
        f"{modname} should expose _HAVE_BACKEND")
    assert isinstance(mod._HAVE_BACKEND, bool)


@pytest.mark.parametrize("modname", ["jxl", "parallel"])
def test_surface_module_has_have_backend(modname):
    """Direct-API surface modules also expose _HAVE_BACKEND."""
    import importlib
    mod = importlib.import_module(f"opencodecs.{modname}")
    assert hasattr(mod, "_HAVE_BACKEND")
    assert isinstance(mod._HAVE_BACKEND, bool)
