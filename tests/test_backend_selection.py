"""The opt-in hardware backends, as seen by a user who has not opted in.

These run on every machine. The backends themselves are tested in
test_backend_nvimgcodec.py, test_backend_nvvideocodec.py and
test_backend_imageio.py, which skip where
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


OPTIONAL = ("cupy", "nvidia.nvimgcodec", "PyNvVideoCodec", "objc", "Quartz",
            "Foundation", "AppKit")


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


# ------------------------------------- backend="nvvideocodec", no GPU
# The container parsing, routing and color decisions of the NVDEC path
# are Python over the file's metadata, so they are tested everywhere.

def test_missing_pynvvideocodec_raises(monkeypatch):
    from opencodecs.backends import _nvvideocodec
    monkeypatch.setattr(_nvvideocodec, "_modules", None)
    monkeypatch.setitem(sys.modules, "PyNvVideoCodec", None)
    with pytest.raises(BackendUnavailable):
        _nvvideocodec.load()
    if has_codec("heif"):
        with pytest.raises(BackendUnavailable):
            oc.get_codec("heif").decode(b"\0" * 64, backend="nvvideocodec")
    if has_codec("avif", op="encode"):
        with pytest.raises(BackendUnavailable):
            oc.get_codec("avif").encode(_rgb(64, 64), level=50,
                                        backend="nvvideocodec")
    from opencodecs.backends import available
    assert available("nvvideocodec") is False


def test_nvvideocodec_plan_keeps_other_files_on_libheif():
    from opencodecs.backends._nvvideocodec import _plan
    color = _heif(_rgb(64, 96), level=90)
    plan, reason = _plan(color)
    assert reason == "" and (plan.width, plan.height) == (96, 64)
    assert (plan.chroma, plan.bits, len(plan.tiles)) == (3, 8, 1)
    assert plan.tiles[0][:4] == b"\0\0\0\1"
    assert _plan(color, index=0)[0] is None
    assert _plan(color, photometric="rgb")[0] is None
    gray = _heif(np.full((64, 64), 100, np.uint8), level=90)
    assert _plan(gray) == (None, "monochrome")
    alpha = _heif(np.dstack([_rgb(64, 64), np.full((64, 64), 9, np.uint8)]),
                  level=90)
    assert "alpha" in _plan(alpha)[1]
    assert "clap" in _plan(_heif(_rgb(63, 95), level=90))[1]
    assert _plan(b"not a heif at all")[0] is None
    # Deep color is the GPU's to decode (if its NVDEC takes the format).
    deep, _ = _plan(_heif(_rgb(64, 64).astype(np.uint16) * 4, level=90))
    assert deep is not None and deep.bits == 10


def test_nvvideocodec_color_follows_libheif():
    """The conversion libheif would pick: the container's nclx (this
    encoder writes one: identity matrix for lossless), and the fixed-point
    path only for 8-bit 4:2:0 at full range."""
    from opencodecs.backends import _nvvideocodec as nv
    lossless, _ = nv._plan(_heif(_rgb(64, 96)))
    assert lossless.nclx[2] == 0
    assert int(nv._convert_args(lossless)[0]) == 0
    lossy, _ = nv._plan(_heif(_rgb(64, 96), level=50))
    lossy.nclx = (1, 13, 6, True)
    assert int(nv._convert_args(lossy)[0]) == 2          # 4:4:4: float path
    lossy.chroma = 1
    assert int(nv._convert_args(lossy)[0]) == 1          # 4:2:0 full range
    lossy.nclx = (1, 13, 6, False)
    assert int(nv._convert_args(lossy)[0]) == 2          # limited range
    # libheif's BT.601 defaults for an unspecified matrix, and its float32
    # arithmetic for BT.709.
    np.testing.assert_allclose(nv._ycbcr_to_rgb_coefficients(2, 2),
                               (1.402, -0.344136, -0.714136, 1.772), rtol=1e-6)
    r_cr, g_cb, g_cr, b_cb = nv._ycbcr_to_rgb_coefficients(1, 1)
    assert r_cr == np.float32(2) * (np.float32(1) - np.float32(0.2126))
    assert abs(float(g_cb) + 0.1873) < 1e-3 and abs(float(b_cb) - 1.8556) < 1e-4


def test_hevc_sps_and_hvcc_from_libheif_stream():
    from opencodecs.backends import _bitstreams as bs, _heifbox as hb
    blob = _heif(_rgb(64, 96), level=50)
    items = hb.parse(blob)
    s, e = items.properties(items.primary)[b"hvcC"]
    config = hb.hvcc(items.data, s, e)
    nals = {bs.nal_type(n): n for n in config["nals"]}
    sps = bs.hevc_sps(nals[33])
    assert (sps["chroma"], sps["luma_bits"]) == (3, 8)
    assert sps["width"] >= 96 and sps["height"] >= 64
    rebuilt = bs.hvcc_record(nals[32], nals[33], nals[34])
    # Profile, compatibility flags, level and the format fields agree
    # with libheif's own record. (libheif writes zeros for the constraint
    # flags, bytes 6 to 11; ours are the SPS's, as the format specifies.)
    ours, theirs = rebuilt[8:], bytes(blob[s:e])
    assert ours[:6] == theirs[:6] and ours[12] == theirs[12]
    assert ours[6:12] == bs.rbsp(nals[33])[6:12]
    assert ours[16:19] == theirs[16:19]
    assert hb.hvcc(rebuilt, 8, len(rebuilt))["nals"] == config["nals"]
    # Rewriting the VUI color keeps every other field.
    patched = bs.hevc_sps_rewrite(nals[33], signal=(9, 16, 9, False),
                                  drop_hrd=True)
    again = bs.hevc_sps(patched)
    assert again["vui"] == (9, 16, 9, False)
    for key in ("chroma", "width", "height", "luma_bits", "dpb"):
        assert again[key] == sps[key], key


def test_av1_sequence_header_color_rewrite():
    if not has_codec("avif", op="encode"):
        pytest.skip("avif encoder not built")
    from opencodecs.backends import _bitstreams as bs, _heifbox as hb
    blob = oc.get_codec("avif").encode(_rgb(64, 96), level=60,
                                       yuv_format="420")
    items = hb.parse(blob)
    obus = bs.split_obus(items.item_data(items.primary))
    kind, seq, at = next(o for o in obus if o[0] == bs.OBU_SEQUENCE_HEADER)
    before = bs.av1_sequence_header(seq[at:])
    assert (before["width"], before["height"]) == (96, 64)
    new = bs.av1_set_color(seq, at, (9, 16, 9, False))
    after = bs.av1_sequence_header(new[at:])
    assert after["color"] == (9, 16, 9, False)
    for key in ("profile", "level", "tier", "bits", "subsampling", "width",
                "height", "separate_uv_delta_q", "film_grain"):
        assert after[key] == before[key], key
    record = bs.av1c_record(after, new)
    assert record[4:8] == b"av1C" and record[8] == 0x81
