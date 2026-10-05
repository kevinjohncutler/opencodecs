"""backend="nvvideocodec": HEIF decode on NVDEC, AVIF and HEIF encode on NVENC.

Skipped unless PyNvVideoCodec, CuPy and an NVIDIA GPU are all present.
Checked: a file NVDEC decodes comes back exactly as libheif returns it
(same dtype, shape and pixels: the YCbCr to RGB conversion is libheif's
own arithmetic), grids included; files the GPU should not take (gray,
alpha, cropped, an image other than the primary) are decoded by libheif
with the same result as backend=None; ``out=`` takes numpy, pinned and
CuPy arrays; the NVENC encodes decode on the CPU path within a quality
bound, and anything the encoders cannot write raises. Foreign decoders
read the encodes in test_foreign_readers.py; the container parsing and
routing that need no GPU are tested in test_backend_selection.py.
"""
from __future__ import annotations

import struct

import numpy as np
import pytest

import opencodecs as oc
from opencodecs.backends import available, pinned_empty
from opencodecs.core.codec import has_codec

pytestmark = pytest.mark.skipif(
    not available("nvvideocodec"),
    reason="PyNvVideoCodec, CuPy or an NVIDIA GPU is missing")

GPU = {"backend": "nvvideocodec"}


def _codec(name):
    if not has_codec(name, op="encode"):
        pytest.skip(f"{name} encoder not built")
    return oc.get_codec(name)


def _rgb(h, w, top=255, dtype=np.uint8, seed=0):
    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = (np.sin(yy / 17.0) + np.cos(xx / 23.0) + 2) / 4
    base = np.stack([np.roll(base, 9 * i, 1) for i in range(3)], -1)
    a = base * top * 0.9 + top * 0.05 + rs.normal(0, top / 120, base.shape)
    return np.clip(np.rint(a), 0, top).astype(dtype)


def _psnr(a, b, peak=255.0):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return np.inf if mse == 0 else 10 * np.log10(peak ** 2 / mse)


def _route(blob):
    from opencodecs.backends._nvvideocodec import route
    return route(blob)


def _grid(tiles, rows, cols, width, height):
    """A HEIF whose primary image is a grid of the given single-image
    HEIFs (row-major), each tile's hvcC and coded data copied over."""
    from opencodecs.backends import _heifbox as hb
    parts = []
    for blob in tiles:
        items = hb.parse(blob)
        s, e = items.properties(items.primary)[b"hvcC"]
        parts.append((bytes(blob[s - 8:e]), items.size(items.primary),
                      items.item_data(items.primary)))

    def box(kind, *payload):
        body = b"".join(payload)
        return struct.pack(">I4s", 8 + len(body), kind) + body

    def full(kind, version, *payload):
        return box(kind, struct.pack(">I", version << 24), *payload)

    n = len(tiles)
    ids = list(range(2, n + 2))
    infe = [full(b"infe", 2, struct.pack(">HH", 1, 0), b"grid", b"\0")]
    infe += [full(b"infe", 2, struct.pack(">HH", i, 0), b"hvc1", b"\0")
             for i in ids]
    iinf = full(b"iinf", 0, struct.pack(">H", n + 1), *infe)
    dimg = box(b"dimg", struct.pack(">HH", 1, n),
               *(struct.pack(">H", i) for i in ids))
    iref = full(b"iref", 0, dimg)
    desc = bytes([0, 0, rows - 1, cols - 1]) + struct.pack(">HH", width, height)
    props = [full(b"ispe", 0, struct.pack(">II", width, height))]
    assoc = [(1, [(1, False)])]
    for k, (hvcc, (tw, th), _) in enumerate(parts):
        props.append(hvcc)
        props.append(full(b"ispe", 0, struct.pack(">II", tw, th)))
        assoc.append((ids[k], [(len(props) - 1, True), (len(props), False)]))
    ipma = full(b"ipma", 0, struct.pack(">I", len(assoc)), *(
        struct.pack(">HB", item, len(p)) + bytes((0x80 if e else 0) | i
                                                 for i, e in p)
        for item, p in assoc))
    iprp = box(b"iprp", box(b"ipco", *props), ipma)
    hdlr = full(b"hdlr", 0, struct.pack(">I", 0), b"pict", b"\0" * 13)
    pitm = full(b"pitm", 0, struct.pack(">H", 1))
    idat = box(b"idat", desc)
    ftyp = box(b"ftyp", b"heic", struct.pack(">I", 0), b"mif1", b"heic")

    def iloc(base):
        # version 1: construction method 1 (idat) for the grid descriptor
        entries = [struct.pack(">HHHHII", 1, 1, 0, 1, 0, len(desc))]
        at = base
        for k, (_, _, data) in enumerate(parts):
            entries.append(struct.pack(">HHHHII", ids[k], 0, 0, 1, at,
                                       len(data)))
            at += len(data)
        return full(b"iloc", 1, bytes([0x44, 0x00]),
                    struct.pack(">H", n + 1), *entries)

    def meta(base):
        return full(b"meta", 0, hdlr, pitm, iloc(base), iinf, iref, iprp, idat)

    base = len(ftyp) + len(meta(0)) + 8
    return ftyp + meta(base) + box(b"mdat", *(d for _, _, d in parts))


# ------------------------------------------------------------- decode

def test_lossless_file_decodes_exactly():
    heif = _codec("heif")
    x = _rgb(160, 192)
    blob = heif.encode(x)
    assert _route(blob) == ("nvvideocodec", "")
    got = heif.decode(blob, **GPU)
    assert got.shape == x.shape and got.dtype == np.uint8
    np.testing.assert_array_equal(got, x)


@pytest.mark.parametrize("level", [30, 90])
@pytest.mark.parametrize("deep", [False, True], ids=["8bit", "10bit"])
def test_lossy_444_equals_libheif(level, deep):
    heif = _codec("heif")
    x = _rgb(200, 264, 1023, np.uint16) if deep else _rgb(200, 264)
    blob = heif.encode(x, level=level)
    assert _route(blob)[0] == "nvvideocodec"
    cpu = heif.decode(blob)
    got = heif.decode(blob, **GPU)
    assert got.dtype == cpu.dtype and got.shape == cpu.shape
    np.testing.assert_array_equal(got, cpu)


@pytest.mark.parametrize("deep", [False, True], ids=["8bit", "10bit"])
def test_nvenc_420_file_equals_libheif(deep):
    """4:2:0 (NVENC's own output): libheif's fixed-point 8-bit and its
    floating-point 10-bit conversion, both reproduced exactly."""
    heif = _codec("heif")
    x = _rgb(256, 320, 1023, np.uint16) if deep else _rgb(256, 320)
    blob = heif.encode(x, level=70, **GPU)
    assert _route(blob)[0] == "nvvideocodec"
    cpu = heif.decode(blob)
    np.testing.assert_array_equal(heif.decode(blob, **GPU), cpu)


def test_grid_equals_libheif():
    heif = _codec("heif")
    x = _rgb(512, 768)
    tiles = [heif.encode(np.ascontiguousarray(x[r:r + 256, c:c + 256]),
                         level=60)
             for r in (0, 256) for c in (0, 256, 512)]
    # Output smaller than the tiles cover: the last row and column clip.
    blob = _grid(tiles, 2, 3, 700, 500)
    assert _route(blob)[0] == "nvvideocodec"
    cpu = heif.decode(blob)
    assert cpu.shape == (500, 700, 3)
    np.testing.assert_array_equal(heif.decode(blob, **GPU), cpu)


def test_sizes_and_formats_interleaved():
    """One decoder session per chroma format and depth, reconfigured for
    each size: decoding small after large and back stays exact."""
    heif = _codec("heif")
    blobs = [heif.encode(_rgb(h, w, seed=h), level=50)
             for h, w in [(512, 640), (160, 200), (512, 640), (144, 176)]]
    blobs.append(heif.encode(_rgb(160, 200, 1023, np.uint16), level=50))
    for blob in blobs + blobs[::-1]:
        np.testing.assert_array_equal(heif.decode(blob, **GPU),
                                      heif.decode(blob))


@pytest.mark.parametrize("make, reason", [
    (lambda h: h.encode(np.full((64, 64), 100, np.uint8), level=90),
     "monochrome"),
    (lambda h: h.encode(np.dstack([_rgb(64, 64), np.full((64, 64), 9,
                                                          np.uint8)]),
                        level=90), "alpha"),
    (lambda h: h.encode(_rgb(63, 95), level=90), "clap"),
    (lambda h: h.encode(_rgb(64, 96), level=90), "smaller than"),
])
def test_other_files_go_to_libheif(make, reason):
    heif = _codec("heif")
    blob = make(heif)
    route = _route(blob)
    assert route[0] == "native" and reason in route[1]
    np.testing.assert_array_equal(heif.decode(blob, **GPU), heif.decode(blob))


def test_index_and_photometric_go_to_libheif():
    heif = _codec("heif")
    gray = heif.encode(np.full((64, 64), 100, np.uint8), level=90)
    np.testing.assert_array_equal(
        heif.decode(gray, photometric="monochrome", **GPU),
        heif.decode(gray, photometric="monochrome"))
    color = heif.encode(_rgb(64, 96), level=90)
    np.testing.assert_array_equal(heif.decode(color, index=0, **GPU),
                                  heif.decode(color, index=0))


def test_out_numpy_pinned_and_cupy():
    import cupy as cp
    heif = _codec("heif")
    x = _rgb(160, 224)
    blob = heif.encode(x, level=70)
    cpu = heif.decode(blob)
    out = np.empty_like(cpu)
    assert heif.decode(blob, out=out, **GPU) is out
    np.testing.assert_array_equal(out, cpu)
    pinned = pinned_empty(cpu.shape, cpu.dtype)
    assert heif.decode(blob, out=pinned, **GPU) is pinned
    np.testing.assert_array_equal(pinned, cpu)
    dev = cp.empty(cpu.shape, cpu.dtype)
    assert heif.decode(blob, out=dev, **GPU) is dev
    np.testing.assert_array_equal(dev.get(), cpu)
    # A file routed to libheif still fills a device out=.
    gray = heif.encode(np.full((64, 64), 100, np.uint8), level=90)
    dev = cp.empty((64, 64, 3), np.uint8)
    assert heif.decode(gray, out=dev, **GPU) is dev
    np.testing.assert_array_equal(dev.get(), heif.decode(gray))
    with pytest.raises(ValueError, match="out="):
        heif.decode(blob, out=np.empty((10, 10, 3), np.uint8), **GPU)


def test_close_and_reopen():
    from opencodecs.backends import _nvvideocodec
    heif = _codec("heif")
    blob = heif.encode(_rgb(160, 200), level=60)
    first = heif.decode(blob, **GPU)
    _nvvideocodec.close()
    np.testing.assert_array_equal(heif.decode(blob, **GPU), first)


# ------------------------------------------------------------- encode

@pytest.mark.parametrize("deep", [False, True], ids=["8bit", "10bit"])
def test_avif_encode_reads_back(deep):
    avif = _codec("avif")
    x = _rgb(256, 320, 1023, np.uint16) if deep else _rgb(256, 320)
    blob = avif.encode(x, level=80, **GPU)
    assert blob[4:12] == b"ftypavif"
    got = avif.decode(blob)
    assert got.shape == x.shape and got.dtype == x.dtype
    assert _psnr(got, x, 1023.0 if deep else 255.0) >= 36


def test_avif_level_trades_size_for_quality():
    avif = _codec("avif")
    x = _rgb(256, 320)
    blobs = [avif.encode(x, level=lv, **GPU) for lv in (20, 60, 95)]
    sizes = [len(b) for b in blobs]
    assert sizes[0] < sizes[1] < sizes[2]
    q = [_psnr(avif.decode(b), x) for b in blobs]
    assert q[0] < q[1] < q[2]


def test_avif_cupy_input_and_icc():
    import cupy as cp
    avif = _codec("avif")
    x = _rgb(256, 320)
    icc = b"\0" * 128
    a = avif.encode(cp.asarray(x), level=70, iccprofile=icc, **GPU)
    b = avif.encode(x, level=70, iccprofile=icc, **GPU)
    assert a == b
    assert avif.read_icc_profile(a) == icc


def test_heif_encode_reads_back():
    heif = _codec("heif")
    x = _rgb(256, 320)
    blob = heif.encode(x, level=80, **GPU)
    assert blob[4:12] == b"ftypheic"
    assert _psnr(heif.decode(blob), x) >= 36
    # Odd sizes: padded to even and cropped back by a clap box, as libheif
    # writes them; libheif returns the requested size.
    odd = _rgb(255, 321)
    got = heif.decode(heif.encode(odd, level=80, **GPU))
    assert got.shape == odd.shape and _psnr(got, odd) >= 36


def test_heif_encode_above_level5_size_reads_on_cpu():
    """Above 8,912,896 samples NVENC's HRD parameters held values that
    libde265 (libheif's HEVC decoder) refuses; the SPS is rewritten
    without them, and with the color the pixels were converted with."""
    from opencodecs.backends import _bitstreams as bs, _heifbox as hb
    heif = _codec("heif")
    x = _rgb(3008, 3072)
    blob = heif.encode(x, level=60, **GPU)
    items = hb.parse(blob)
    config = hb.hvcc(items.data, *items.properties(items.primary)[b"hvcC"])
    sps = next(n for n in config["nals"] if bs.nal_type(n) == 33)
    info = bs.hevc_sps(sps)
    assert info["vui"] == (1, 13, 6, True) and "hrd_end" in info["at"]
    got = heif.decode(blob)
    assert got.shape == x.shape and _psnr(got, x) >= 36
    np.testing.assert_array_equal(heif.decode(blob, **GPU), got)


@pytest.mark.parametrize("codec, x, kw, match", [
    ("avif", None, {}, "level"),
    ("avif", None, {"level": 100}, "level"),
    ("avif", None, {"level": 60, "lossless": True}, "lossy"),
    ("avif", None, {"level": 60, "yuv_format": "444"}, "4:2:0"),
    ("avif", None, {"level": 60, "tile_cols_log2": 1}, "tile_cols_log2"),
    ("avif", None, {"level": 60, "codec": "aom"}, "codec"),
    ("avif", "gray", {"level": 60}, "RGB"),
    ("avif", "rgba", {"level": 60}, "RGB"),
    ("avif", "deep", {"level": 60}, "10 bits"),
    ("avif", "tiny", {"level": 60}, "pixels"),
    ("avif", "odd", {"level": 60}, "even"),
    ("heif", None, {}, "level"),
    ("heif", None, {"level": 60, "lossless": True}, "lossy"),
    ("heif", None, {"level": 60, "color": "srgb"}, "color"),
])
def test_encode_refuses_what_it_cannot_write(codec, x, kw, match):
    c = _codec(codec)
    data = {None: _rgb(256, 320), "gray": _rgb(256, 320)[..., 0],
            "rgba": np.dstack([_rgb(256, 320), _rgb(256, 320)[..., :1]]),
            "deep": _rgb(256, 320, 4095, np.uint16),
            "tiny": _rgb(16, 16), "odd": _rgb(255, 320)}[x]
    with pytest.raises(ValueError, match=match):
        c.encode(data, **GPU, **kw)
