"""The opt-in hardware backends, as seen by a user who has not opted in.

These run on every machine. The backends themselves are tested in
test_backend_nvimgcodec.py and test_backend_imageio.py, which skip where
the hardware is absent. What is checked here is the promise that matters
everywhere else: importing opencodecs loads none of the optional
packages, ``backend=None`` is the CPU path, an unknown backend name is an
error, and asking for a backend that cannot run raises rather than
falling back to the CPU.
"""
from __future__ import annotations

import subprocess
import sys
import textwrap

import numpy as np
import pytest

import opencodecs as oc
from opencodecs.backends import BackendUnavailable, BACKENDS, select
from opencodecs.core.codec import has_codec


OPTIONAL = ("cupy", "nvidia.nvimgcodec", "objc", "Quartz", "Foundation",
            "AppKit")


def test_import_loads_no_optional_package():
    """Not on import, and not on an ordinary decode or encode either."""
    code = textwrap.dedent(f"""
        import sys
        import numpy as np
        import opencodecs as oc
        import opencodecs.backends
        if oc.has_codec("jpeg2k", op="encode"):
            c = oc.get_codec("jpeg2k")
            c.decode(c.encode(np.zeros((8, 8), np.uint8)))
        loaded = [m for m in {OPTIONAL!r} if m in sys.modules]
        print(",".join(loaded))
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         text=True, check=True).stdout.strip()
    assert out == "", f"importing opencodecs loaded {out}"


def test_backend_names_are_validated():
    assert select(None, "jpeg2k", "decode") is None
    assert select("native", "jpeg2k", "decode") is None
    assert select("NATIVE", "heif", "encode") is None
    with pytest.raises(ValueError, match="nvimgcodec"):
        select("cuda", "jpeg2k", "decode")
    # A real backend, but not one this codec offers.
    with pytest.raises(ValueError, match="imageio"):
        select("nvimgcodec", "heif", "decode")
    with pytest.raises(ValueError):
        select("imageio", "jpeg", "encode")
    with pytest.raises(ValueError):
        select("nvimgcodec", "png", "decode")


def test_every_listed_codec_takes_backend():
    for name, codecs in BACKENDS.items():
        for codec in codecs:
            if not has_codec(codec):
                continue
            with pytest.raises(ValueError, match="unknown backend"):
                oc.get_codec(codec).decode(b"\0" * 32, backend="no-such")


@pytest.mark.parametrize("codec, x", [
    ("jpeg2k", np.arange(64, dtype=np.uint16).reshape(8, 8)),
    ("htj2k", np.arange(64, dtype=np.uint16).reshape(8, 8)),
    ("jpeg", np.full((16, 16, 3), 77, np.uint8)),
])
def test_native_backend_is_the_default_path(codec, x):
    if not has_codec(codec, op="encode"):
        pytest.skip(f"{codec} not built")
    c = oc.get_codec(codec)
    blob = c.encode(x)
    assert c.encode(x, backend="native") == blob
    assert c.encode(x, backend=None) == blob
    np.testing.assert_array_equal(c.decode(blob, backend="native"),
                                  c.decode(blob))


def test_missing_cuda_packages_raise(monkeypatch):
    """No CuPy: a clear BackendUnavailable, which is also an ImportError."""
    from opencodecs.backends import _nvimgcodec
    monkeypatch.setattr(_nvimgcodec, "_modules", None)
    monkeypatch.setitem(sys.modules, "cupy", None)
    with pytest.raises(BackendUnavailable, match="CuPy"):
        _nvimgcodec.load()
    with pytest.raises(ImportError):
        _nvimgcodec.load()
    if has_codec("jpeg2k", op="encode"):
        c = oc.get_codec("jpeg2k")
        blob = c.encode(np.zeros((8, 8), np.uint8))
        with pytest.raises(BackendUnavailable):
            c.decode(blob, backend="nvimgcodec")
        with pytest.raises(BackendUnavailable):
            c.encode(np.zeros((8, 8), np.uint8), backend="nvimgcodec")
    from opencodecs.backends import available
    assert available("nvimgcodec") is False


def test_imageio_off_macos_raises(monkeypatch):
    from opencodecs.backends import _imageio
    monkeypatch.setattr(_imageio, "_api", None)
    # Where the frameworks are absent, as anywhere but macOS. The names must
    # not exist either: for a missing ".../X.framework/X" path, macOS's
    # loader falls back to the system's X.framework and loads it anyway.
    monkeypatch.setattr(_imageio, "_FRAMEWORK",
                        "/nonexistent/{0}Absent.framework/{0}Absent")
    with pytest.raises(BackendUnavailable, match="macOS"):
        _imageio.load()
    if has_codec("heif"):
        with pytest.raises(BackendUnavailable):
            oc.get_codec("heif").decode(b"\0" * 64, backend="imageio")


# ------------------------------------------------- HEIF routing (any OS)
# Which files backend="imageio" hands to Apple's decoder is decided from
# the container's metadata alone, in Python, so it is tested everywhere.

def _heif(x, **kw):
    if not has_codec("heif", op="encode"):
        pytest.skip("heif encoder not built")
    try:
        return oc.get_codec("heif").encode(x, **kw)
    except Exception as e:  # noqa: BLE001
        if "encoder" in str(e).lower():
            pytest.skip(f"heif encoder not available: {e}")
        raise


def _rgb(h, w):
    yy, xx = np.mgrid[0:h, 0:w]
    base = (np.sin(yy / 7.0) + np.cos(xx / 11.0) + 2) * 60
    return np.stack([np.roll(base, 5 * i, 1) for i in range(3)],
                    -1).astype(np.uint8)


def test_route_single_picture_goes_to_imageio():
    from opencodecs.backends._imageio import route
    blob = _heif(_rgb(64, 96), level=90)
    assert route(blob) == ("imageio", "")
    # Options only libheif implements keep the file on libheif.
    assert route(blob, index=0)[0] == "native"
    assert route(blob, photometric="rgb")[0] == "native"


def test_route_keeps_other_files_on_libheif():
    from opencodecs.backends._imageio import route
    gray = _heif(np.full((64, 64), 100, np.uint8), level=90)
    assert route(gray) == ("native", "monochrome")
    alpha = _heif(np.dstack([_rgb(64, 64), np.full((64, 64), 9, np.uint8)]),
                  level=90)
    assert route(alpha)[0] == "native"
    deep = _heif((_rgb(64, 64).astype(np.uint16) * 4), level=90)
    assert route(deep)[0] == "native"
    odd = _heif(_rgb(63, 95), level=90)      # cropped by a clap box
    assert route(odd)[0] == "native"
    assert route(b"not a heif at all")[0] == "native"
