"""Everything opencodecs writes must read back in a decoder that is not ours.

WHY THIS FILE EXISTS
--------------------
test_czi_foreign_readers.py exists because a CZI this package wrote, and
read back without complaint, was refused by libCZI, czicompress and ZEN.
Its own reader was lenient in exactly the way its writer was wrong, so
every round-trip test passed. That trap is not specific to CZI: any
encoder tested only against our own decoder can drift from the format
while the suite stays green.

So every case here encodes with opencodecs and decodes with something
else: the reference library's own binding where one exists (zstandard,
lz4, brotli, python-blosc2, cramjam, lerc, zfpy, pcodec, the standard
library), imagecodecs, and format-level readers (tifffile, mrcfile,
nibabel, pydicom, numcodecs, numpy). Nothing here calls an opencodecs
decoder.

Every codec that can encode has an entry in CASES, and
test_every_encoder_has_a_case fails when a new one does not, so the
coverage cannot fall behind silently. A reader that is not installed is a
skip, never a pass. Lossless settings must come back exactly; lossy ones
within the bound the setting promises.

WHAT TO DO WHEN ONE OF THESE FAILS
----------------------------------
Assume our encoder is wrong, not the reader. These readers open files
from every other producer of the format; we are the newcomer.
"""
from __future__ import annotations

import bz2
import gzip
import io
import lzma
import zlib
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pytest

from opencodecs.backends import BackendUnavailable
from opencodecs.core.codec import get_codec, has_codec, list_codecs

from _ic_reference import IMAGECODECS_REFERENCE  # noqa: E402


# ---------------------------------------------------------------- readers

def _imagecodecs():
    """imagecodecs, or a skip when it is missing or older than the release
    the compatibility tests pin (older ones crash on some inputs)."""
    ic = pytest.importorskip("imagecodecs", reason="imagecodecs missing")
    from packaging.version import Version
    if Version(ic.__version__) < Version(IMAGECODECS_REFERENCE):
        pytest.skip(f"imagecodecs {ic.__version__} < {IMAGECODECS_REFERENCE}")
    return ic


def _mod(name: str):
    return pytest.importorskip(name, reason=f"{name} missing")


def _ic_func(func: str):
    """imagecodecs.<func>, or a skip when this imagecodecs build lacks it
    (wheels for some platforms leave out brunsli or HEIF). imagecodecs
    resolves its functions lazily, so a missing one raises ImportError on
    the first call, not on lookup."""
    try:
        f = getattr(_imagecodecs(), func)
    except (ImportError, AttributeError) as e:
        pytest.skip(f"imagecodecs built without {func}: {e}")

    def call(*a, **kw):
        try:
            return f(*a, **kw)
        except ImportError as e:
            pytest.skip(f"imagecodecs built without {func}: {e}")
    return call


def ic(func: str, *args_from_x: Callable, typed: bool = False, **kw):
    """A reader that calls imagecodecs.<func>(blob, *f(x) for f in
    args_from_x, **kw). typed=True hands it the stream as an array of the
    input's dtype and shape, as the filter decoders expect."""
    def read(blob, x):
        data = np.frombuffer(blob, x.dtype).reshape(x.shape) if typed else blob
        return _ic_func(func)(data, *(f(x) for f in args_from_x), **kw)
    read.__name__ = f"imagecodecs.{func}"
    return read


def named(label: str, fn: Callable) -> Callable:
    fn.__name__ = label
    return fn


SHAPE = lambda x: x.shape          # noqa: E731
DTYPE = lambda x: x.dtype          # noqa: E731


def _zstandard(blob, x):
    return _mod("zstandard").ZstdDecompressor().decompressobj().decompress(blob)


def _lz4_frame(blob, x):
    return _mod("lz4.frame").decompress(blob)


def _brotli(blob, x):
    return _mod("brotli").decompress(blob)


def _blosc2(blob, x):
    return _mod("blosc2").decompress(blob)


def _b2nd(blob, x):
    return _mod("blosc2").ndarray_from_cframe(blob)[...]


def _snappy_raw(blob, x):
    return bytes(_mod("cramjam").snappy.decompress_raw(blob))


def _snappy_framed(blob, x):
    return bytes(_mod("cramjam").snappy.decompress(blob))


def _numcodecs_shuffle(blob, x):
    return bytes(_mod("numcodecs").Shuffle(elementsize=x.itemsize).decode(blob))


def _lerc_pkg(blob, x):
    result, data, *_ = _mod("lerc").decode(blob)
    assert result == 0, f"lerc.decode returned {result}"
    return data


def _zfpy(blob, x):
    return _mod("zfpy").decompress_numpy(bytes(blob))


def _pcodec_pkg(blob, x):
    # pcodec codes a flat sequence of numbers; the shape is the caller's.
    return _mod("pcodec.standalone").simple_decompress(bytes(blob)).reshape(x.shape)


def _qoi_pkg(blob, x):
    return _mod("qoi").decode(bytes(blob))


def _np_load(blob, x):
    return np.load(io.BytesIO(blob), allow_pickle=False)


def _mrcfile(blob, x):
    """mrcfile opens paths only; permissive=False makes any header
    deviation from the MRC2014 standard an error, not a warning."""
    import os
    import tempfile
    mrcfile = _mod("mrcfile")
    fd, path = tempfile.mkstemp(suffix=".mrc")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(blob)
        with mrcfile.open(path, permissive=False) as m:
            return np.array(m.data)
    finally:
        os.unlink(path)


def _nibabel(blob, x):
    nib = _mod("nibabel")
    img = nib.Nifti1Image.from_bytes(bytes(blob))
    return np.asarray(img.dataobj)


def _pillow_heif(blob, x):
    """libheif through pillow-heif's own decode, not through Pillow."""
    ph = _mod("pillow_heif")
    return np.asarray(ph.open_heif(bytes(blob), convert_hdr_to_8bit=False))


def _tifffile_via_imagecodecs():
    """tifffile, for a TIFF whose tiles it hands to imagecodecs to
    decompress: then imagecodecs must be the reference release too.
    imagecodecs 2025.3.30, all Python 3.10 can install, aborted the
    process decoding one of these files."""
    tifffile = _mod("tifffile")
    _imagecodecs()
    return tifffile


def _tifffile(blob, x):
    return _mod("tifffile").imread(io.BytesIO(blob))


def _pydicom_rle(blob, x):
    _mod("pydicom")
    try:
        from pydicom.pixels.decoders.rle import _rle_decode_frame
    except ImportError:
        pytest.skip("pydicom has no RLE frame decoder")
    rows, cols = x.shape[:2]
    spp = x.shape[2] if x.ndim == 3 else 1
    frame = _rle_decode_frame(bytes(blob), rows, cols, spp, x.itemsize * 8,
                              segment_order=">")
    # Planar by sample; with segment_order=">" pydicom reassembles each
    # sample little-endian.
    a = np.frombuffer(frame, x.dtype.newbyteorder("<"))
    a = a.reshape((spp, rows, cols)) if spp > 1 else a.reshape(rows, cols)
    return np.moveaxis(a, 0, -1) if spp > 1 else a


def _gray_from_rgb(a):
    """GIF stores palette indices; a decoder hands back RGB. A grayscale
    image must come back with three equal channels."""
    a = np.asarray(a)
    if a.ndim == 3:
        assert (a[..., 0] == a[..., 1]).all() and (a[..., 1] == a[..., 2]).all(), \
            "gray GIF decoded with unequal channels"
        a = a[..., 0]
    return a


def _cumsum(blob, x):
    """TIFF predictor 2 undone with numpy: a running sum along the last
    axis, modulo 2**bits, on the unsigned bit pattern."""
    u = np.dtype(f"u{x.itemsize}")
    d = np.frombuffer(blob, u).reshape(x.shape)
    return np.cumsum(d, axis=-1, dtype=u).view(x.dtype)


def _xor_accumulate(blob, x):
    u = np.dtype(f"u{x.itemsize}")
    d = np.frombuffer(blob, u).reshape(x.shape)
    return np.bitwise_xor.accumulate(d, axis=-1).view(x.dtype)


# ----------------------------------------------------------------- inputs

def _smooth(shape, top, dtype, seed=3):
    rs = np.random.RandomState(seed)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    base = (np.sin(yy / 7.0) + np.cos(xx / 11.0) + 2) * (top / 4)
    if len(shape) == 3:
        base = np.stack([np.roll(base, 5 * i, 1) for i in range(shape[2])], -1)
    a = np.clip(base + rs.normal(0, top / 64, base.shape), 0, top)
    return a.astype(dtype)


INPUTS = {
    "u16": lambda: _smooth((48, 64), 4095, np.uint16),
    "u8": lambda: _smooth((48, 64), 255, np.uint8),
    "rgb": lambda: _smooth((48, 64, 3), 255, np.uint8),
    "f4": lambda: _smooth((32, 48), 100.0, np.float32),
    "f4_3d": lambda: _smooth((32, 48), 100.0, np.float32)[None].repeat(4, 0),
    "rgb_f4": lambda: (_smooth((32, 48, 3), 1000.0, np.float32) + 1.0),
}


# ------------------------------------------------------------------ cases

@dataclass
class Case:
    codec: str
    data: str
    readers: list
    kw: dict = field(default_factory=dict)
    # None = exact; ("abs", e) / ("rel", e) / ("psnr", db) for lossy.
    tol: tuple | None = None
    id: str = ""

    def __post_init__(self):
        self.id = self.id or self.codec


CASES = [
    # General-purpose byte streams: the library's own binding, the stdlib,
    # and imagecodecs.
    Case("zstd", "u16", [_zstandard, ic("zstd_decode")]),
    Case("lz4", "u16", [_lz4_frame, ic("lz4f_decode")]),
    Case("brotli", "u16", [_brotli, ic("brotli_decode")]),
    Case("blosc2", "u16", [_blosc2, ic("blosc2_decode")]),
    Case("bz2", "u16", [named("bz2", lambda b, x: bz2.decompress(b)),
                        ic("bz2_decode")]),
    Case("lzma", "u16", [named("lzma", lambda b, x: lzma.decompress(b)),
                         ic("lzma_decode")]),
    Case("gzip", "u16", [named("gzip", lambda b, x: gzip.decompress(b)),
                         ic("gzip_decode")]),
    Case("deflate", "u16", [named("zlib", lambda b, x: zlib.decompress(b)),
                            ic("zlib_decode")]),
    Case("deflate", "u16", [named("zlib raw", lambda b, x: zlib.decompress(b, -15)),
                            named("imagecodecs.deflate_decode", lambda b, x:
                                  _ic_func("deflate_decode")(b, raw=True,
                                                             out=x.nbytes))],
         kw={"raw": True}, id="deflate-raw"),
    Case("snappy", "u16", [_snappy_raw, ic("snappy_decode")]),
    Case("snappy_framed", "u16", [_snappy_framed]),
    Case("none", "u16", [named("identity", lambda b, x: bytes(b))]),

    # Filters and bit-level transforms: an independent implementation of
    # the inverse, and imagecodecs.
    Case("byteshuffle", "u16", [_numcodecs_shuffle]),
    Case("bitshuffle", "u16", [ic("bitshuffle_decode", itemsize=2)]),
    Case("delta", "u16", [_cumsum, ic("delta_decode", typed=True)]),
    Case("xor", "u16", [_xor_accumulate, ic("xor_decode", typed=True)]),
    Case("floatpred", "f4", [ic("floatpred_decode", typed=True)]),
    Case("packints", "u16", [named("imagecodecs.packints_decode", lambda b, x:
                                   _ic_func("packints_decode")(b, x.dtype, 12)
                                   .reshape(x.shape))],
         kw={"bitspersample": 12}),

    # Scientific compressors.
    Case("aec", "u16", [named("imagecodecs.aec_decode", lambda b, x:
                              _imagecodecs().aec_decode(b, out=np.empty_like(x)))]),
    Case("rcomp", "u16", [ic("rcomp_decode", SHAPE, DTYPE)]),
    Case("pcodec", "u16", [_pcodec_pkg, ic("pcodec_decode", SHAPE, DTYPE)]),
    Case("lerc", "u16", [_lerc_pkg, ic("lerc_decode")]),
    Case("zfp", "f4", [_zfpy, ic("zfp_decode")], kw={"mode": "reversible"}),
    Case("sz3", "f4", [ic("sz3_decode", SHAPE, DTYPE)],
         kw={"mode": "abs", "abs": 1e-3}, tol=("abs", 1e-3)),
    Case("sperr", "f4_3d", [ic("sperr_decode")],
         kw={"mode": "pwe", "pwe": 1e-3}, tol=("abs", 1e-3)),
    Case("b2nd", "u16", [_b2nd]),
    Case("numpy", "u16", [_np_load, ic("numpy_decode")]),

    # Image formats, lossless settings: exact.
    Case("png", "rgb", [ic("png_decode")]),
    Case("png", "u16", [ic("png_decode")], id="png-u16"),
    Case("qoi", "rgb", [_qoi_pkg, ic("qoi_decode")]),
    Case("bmp", "rgb", [ic("bmp_decode")]),
    Case("gif", "u8", [named("imagecodecs.gif_decode", lambda b, x:
                             _gray_from_rgb(_ic_func("gif_decode")(b)))]),
    Case("webp", "rgb", [ic("webp_decode")], kw={"lossless": True}),
    Case("jxl", "rgb", [ic("jpegxl_decode")], kw={"lossless": True}),
    Case("jxl", "u16", [ic("jpegxl_decode")], kw={"lossless": True}, id="jxl-u16"),
    Case("jpeg2k", "u16", [ic("jpeg2k_decode")]),
    Case("jpeg2k", "rgb", [ic("jpeg2k_decode")], id="jpeg2k-rgb"),
    Case("htj2k", "u16", [ic("htj2k_decode"), ic("jpeg2k_decode")]),
    Case("jpegls", "u16", [ic("jpegls_decode")]),
    Case("jpeg", "u8", [ic("jpeg8_decode")], kw={"lossless": True},
         id="jpeg-lossless"),
    Case("dicomrle", "u16", [_pydicom_rle, ic("dicomrle_decode", DTYPE)]),

    # Image formats, lossy settings: within a quality floor.
    Case("jpeg", "rgb", [ic("jpeg8_decode")], kw={"level": 90}, tol=("psnr", 35)),
    Case("mozjpeg", "rgb", [ic("jpeg8_decode")], kw={"level": 90}, tol=("psnr", 35)),
    Case("brunsli", "rgb", [ic("brunsli_decode")], tol=("psnr", 35)),
    Case("webp", "rgb", [ic("webp_decode")], kw={"level": 90}, tol=("psnr", 30),
         id="webp-lossy"),
    Case("avif", "rgb", [ic("avif_decode")], kw={"level": 90}, tol=("psnr", 30)),
    Case("heif", "rgb", [ic("heif_decode"), _pillow_heif], kw={"level": 90},
         tol=("psnr", 30)),
    Case("jxl", "rgb", [ic("jpegxl_decode")], kw={"level": 90}, tol=("psnr", 30),
         id="jxl-level"),
    Case("rgbe", "rgb_f4", [ic("rgbe_decode")], tol=("rgbe", 1 / 128)),

    # The opt-in hardware encoders (opencodecs.backends). Their output
    # goes through the same foreign readers; a host without the GPU or
    # without macOS skips them.
    Case("jpeg2k", "u16", [ic("jpeg2k_decode")], kw={"backend": "nvimgcodec"},
         id="jpeg2k-nvimgcodec"),
    Case("jpeg2k", "rgb", [ic("jpeg2k_decode")], kw={"backend": "nvimgcodec"},
         id="jpeg2k-rgb-nvimgcodec"),
    Case("htj2k", "u16", [ic("htj2k_decode"), ic("jpeg2k_decode")],
         kw={"backend": "nvimgcodec"}, id="htj2k-nvimgcodec"),
    Case("htj2k", "rgb", [ic("htj2k_decode"), ic("jpeg2k_decode")],
         kw={"backend": "nvimgcodec"}, id="htj2k-rgb-nvimgcodec"),
    Case("jpeg", "rgb", [ic("jpeg8_decode")],
         kw={"level": 90, "backend": "nvimgcodec"}, tol=("psnr", 35),
         id="jpeg-nvimgcodec"),
    Case("jpeg", "u8", [ic("jpeg8_decode")],
         kw={"level": 90, "backend": "nvimgcodec"}, tol=("psnr", 35),
         id="jpeg-gray-nvimgcodec"),
    Case("heif", "rgb", [ic("heif_decode"), _pillow_heif],
         kw={"level": 90, "backend": "imageio"}, tol=("psnr", 30),
         id="heif-imageio"),

    # Whole-file formats: the format's own reader.
    Case("tiff", "u16", [_tifffile]),
    Case("tiff", "rgb", [_tifffile], id="tiff-rgb"),
    Case("mrc", "u16", [_mrcfile]),
    Case("nifti", "u16", [_nibabel]),
]

# Codecs that write no stream a foreign decoder could read, and why.
NOT_A_STREAM = {
    "quantize": "rounds the array in place; its output is an array, not an "
                "encoded stream (its values are pinned against imagecodecs "
                "in test_filter_compat.py)",
}

# Encoders a build may lack: the Linux wheels decode AVIF and HEIF only.
_NO_ENCODER = ("No codec available", "Unsupported file-type",
               "no encoder", "encoder not available")


def _encode(case: Case, x: np.ndarray) -> bytes:
    # A build without the library (the CI test build has no SZ3, SPERR,
    # pcodec, MozJPEG or brunsli; the wheels do) has nothing to test.
    if not has_codec(case.codec, op="encode"):
        pytest.skip(f"{case.codec} encoder not built")
    codec = get_codec(case.codec)
    try:
        blob = codec.encode(x, **case.kw)
    except BackendUnavailable as e:
        pytest.skip(f"backend={case.kw.get('backend')!r} unavailable: {e}")
    except Exception as e:  # noqa: BLE001 -- narrowed below
        if case.codec in ("avif", "heif") and any(s in str(e) for s in _NO_ENCODER):
            pytest.skip(f"{case.codec} encoder not built: {e}")
        raise
    if isinstance(blob, np.ndarray):
        blob = blob.tobytes()
    return bytes(blob)


def _as_array(got, x: np.ndarray) -> np.ndarray:
    if isinstance(got, (bytes, bytearray, memoryview)):
        return np.frombuffer(got, x.dtype).reshape(x.shape)
    got = np.asarray(got)
    if got.shape != x.shape:
        got = np.squeeze(got)
        assert got.shape == np.squeeze(x).shape, f"shape {got.shape} != {x.shape}"
        x = np.squeeze(x)
    return got


def _check(got, x, tol, label):
    a = _as_array(got, x)
    ref = x if a.shape == x.shape else np.squeeze(x)
    if tol is None:
        assert a.dtype == ref.dtype, f"{label}: dtype {a.dtype} != {ref.dtype}"
        assert np.array_equal(a, ref), f"{label}: not the pixels we encoded"
        return
    kind, bound = tol
    diff = np.abs(a.astype(np.float64) - ref.astype(np.float64))
    if kind == "abs":
        assert diff.max() <= bound * (1 + 1e-6), f"{label}: max error {diff.max()}"
    elif kind == "rel":
        rel = diff / np.maximum(np.abs(ref.astype(np.float64)), 1e-30)
        assert rel.max() <= bound, f"{label}: max relative error {rel.max()}"
    elif kind == "rgbe":
        # One exponent per pixel: each channel is exact to 1/128 of the
        # pixel's brightest channel (8-bit mantissas), not of itself.
        peak = np.abs(ref.astype(np.float64)).max(axis=-1, keepdims=True)
        rel = diff / peak
        assert rel.max() <= bound, f"{label}: error {rel.max()} of the pixel peak"
    elif kind == "psnr":
        peak = 255.0 if ref.dtype == np.uint8 else float(np.iinfo(ref.dtype).max)
        mse = float(np.mean(diff ** 2))
        psnr = np.inf if mse == 0 else 10 * np.log10(peak ** 2 / mse)
        assert psnr >= bound, f"{label}: PSNR {psnr:.1f} dB < {bound}"


PAIRS = [pytest.param(c, r, id=f"{c.id}-{r.__name__}")
         for c in CASES for r in c.readers]


@pytest.mark.parametrize("case, reader", PAIRS)
def test_foreign_reader_decodes_what_we_encode(case, reader):
    x = INPUTS[case.data]()
    blob = _encode(case, x)
    _check(reader(blob, x), x, case.tol, f"{case.id} via {reader.__name__}")


def test_every_encoder_has_a_case():
    """A codec that gains an encoder must gain a foreign reader here too."""
    encoders = {r["name"] for r in list_codecs() if r.get("encode")}
    covered = {c.codec for c in CASES} | set(NOT_A_STREAM)
    missing = sorted(encoders - covered)
    assert not missing, f"encoders with no foreign-reader case: {missing}"


# ================================================================ writers
# Files rather than streams: each writer's output opened by the reader the
# rest of the world uses for that format. The CZI writer has its own file,
# test_czi_foreign_readers.py; its pyramid writer is here.

TIFF_CASES = [
    # (compression, input, writer kwargs, tolerance)
    ("none", "u16", {}, None),
    ("none", "u16", {"tile": (32, 32)}, None),
    ("deflate", "u16", {"predictor": 2}, None),
    ("deflate", "rgb", {"tile": (32, 32), "predictor": 2}, None),
    ("zstd", "u16", {"tile": (32, 32), "predictor": 2}, None),
    ("lzw", "u16", {"predictor": 2}, None),
    ("deflate", "f4", {"predictor": 3}, None),
    ("zstd", "f4", {"tile": (16, 16), "predictor": 3}, None),
    ("jpeg2000", "u16", {"tile": (32, 32)}, None),
    ("jxl", "rgb", {"tile": (32, 32)}, None),
    ("jxl", "rgb", {"tile": (32, 32), "compression_level": 90}, ("psnr", 30)),
    ("webp", "rgb", {"tile": (32, 32)}, ("psnr", 30)),
    ("lerc", "f4", {"tile": (16, 16)}, None),
    ("jpeg", "rgb", {"tile": (32, 32)}, ("psnr", 30)),
    ("jpeg", "u8", {}, ("psnr", 30)),
]


@pytest.mark.parametrize("compression, data, kw, tol", [
    pytest.param(*c, id=f"{c[0]}-{c[1]}-" + "-".join(f"{k}{v}" for k, v in c[2].items()))
    for c in TIFF_CASES])
def test_tifffile_reads_our_tiff(tmp_path, compression, data, kw, tol):
    tifffile = _mod("tifffile") if compression == "none" else _tifffile_via_imagecodecs()
    from opencodecs import TiffWriter
    x = INPUTS[data]()
    path = tmp_path / "w.tif"
    with TiffWriter(str(path)) as w:
        w.write_page(x, compression=compression, **kw)
    with tifffile.TiffFile(str(path)) as t:
        page = t.pages[0]
        assert page.shape[:2] == x.shape[:2]
        got = page.asarray()
    _check(got, x, tol, f"TIFF {compression} via tifffile")


def test_tifffile_reads_our_pyramid_and_bigtiff(tmp_path):
    tifffile = _tifffile_via_imagecodecs()          # zstd tiles
    from opencodecs import TiffWriter
    x = INPUTS["u16"]()
    levels = [x, x[::2, ::2].copy(), x[::4, ::4].copy()]
    path = tmp_path / "pyr.tif"
    with TiffWriter(str(path), bigtiff=True) as w:
        w.write_pyramid(levels, tile=(16, 16), compression="zstd")
    with tifffile.TiffFile(str(path)) as t:
        assert t.is_bigtiff
        series = t.series[0]
        assert len(series.levels) == 3, "tifffile does not see a pyramid"
        for lev, ref in zip(series.levels, levels):
            assert np.array_equal(lev.asarray(), ref)


def test_tifffile_reads_our_multipage_stack(tmp_path):
    tifffile = _tifffile_via_imagecodecs()          # deflate pages
    from opencodecs import TiffWriter
    stack = np.stack([INPUTS["u16"]() + i for i in range(4)])
    path = tmp_path / "stack.tif"
    with TiffWriter(str(path)) as w:
        for frame in stack:
            w.write_page(frame, compression="deflate")
    got = tifffile.imread(str(path), key=slice(None))
    assert np.array_equal(got, stack)


@pytest.mark.parametrize("compression", ["none", "deflate", "zstd"])
def test_ndtiff_dataset_reads_our_ndtiff(tmp_path, compression):
    """pycromanager's ndtiff reads the index and the stack files; a
    compressed stack is also checked page by page in tifffile."""
    from opencodecs._ndtiff_writer import NDTiffWriter
    frames = np.stack([INPUTS["u16"]() + i for i in range(3)])
    out = tmp_path / "acq"
    with NDTiffWriter(out, base_name="Acq", compression=compression) as w:
        for z, f in enumerate(frames):
            w.write_frame(f, {"z": z})
    files = sorted(p for p in out.iterdir() if p.suffix == ".tif")
    assert files, "no stack file written"
    tifffile = _mod("tifffile") if compression == "none" else _tifffile_via_imagecodecs()
    pages = [p for f in files for p in tifffile.TiffFile(str(f)).pages]
    assert len(pages) == 3
    for page, ref in zip(pages, frames):
        assert np.array_equal(page.asarray(), ref)
    if compression == "none":
        ndtiff = _mod("ndtiff")
        ds = ndtiff.Dataset(str(out))
        for z, ref in enumerate(frames):
            assert np.array_equal(np.asarray(ds.read_image(z=z)), ref)


ZARR_CASES = [
    pytest.param(fmt, comp, id=f"{comp}-v{fmt}")
    for fmt in (2, 3) for comp in ("none", "zstd", "gzip", "blosc2")
] + [pytest.param(2, "blosc", id="blosc-v2")]   # zarr v3 has no Blosc1 spec


def _register_ocf_blosc2(zarr_format):
    """No zarr specification defines Blosc2. ocf-blosc2 (Open Climate Fix)
    is the numcodecs plugin for it, and the names our writer uses are its
    names. For v2 its entry point registers "blosc2" with numcodecs on
    install; for v3 zarr has to be told about its wrapper by hand."""
    _mod("ocf_blosc2")
    if zarr_format == 3:
        import zarr.registry
        try:
            from ocf_blosc2.ocf_blosc2_v3 import Blosc2
        except ImportError:
            # ocf-blosc2's own v3 module imports a numcodecs helper that
            # numcodecs 0.16 removed when the v3 wrappers moved into zarr.
            # Wrap it the way zarr wraps its own numcodecs codecs; the
            # decoding is still ocf-blosc2's, found in numcodecs' registry.
            from zarr.codecs.numcodecs import _NumcodecsBytesBytesCodec

            class Blosc2(_NumcodecsBytesBytesCodec, codec_name="blosc2"):
                pass
        zarr.registry.register_codec("numcodecs.blosc2", Blosc2)


@pytest.mark.parametrize("zarr_format, compressor", ZARR_CASES)
def test_zarr_reads_our_omezarr(tmp_path, zarr_format, compressor):
    zarr = _mod("zarr")
    if zarr_format == 3 and int(zarr.__version__.split(".")[0]) < 3:
        pytest.skip(f"zarr {zarr.__version__} reads v2 stores only")
    if compressor == "blosc2":
        _register_ocf_blosc2(zarr_format)
    from opencodecs import write_omezarr_pyramid
    x = INPUTS["u16"]()
    levels = [x, x[::2, ::2].copy()]
    path = tmp_path / "img.zarr"
    write_omezarr_pyramid(str(path), levels, chunks=(16, 16),
                          compressor=compressor, zarr_format=zarr_format)
    group = zarr.open_group(str(path), mode="r")
    attrs = dict(group.attrs)
    ms = attrs.get("multiscales") or attrs.get("ome", {}).get("multiscales")
    assert ms, "no OME-NGFF multiscales metadata"
    paths = [d["path"] for d in ms[0]["datasets"]]
    assert len(paths) == 2
    for p, ref in zip(paths, levels):
        assert np.array_equal(np.asarray(group[p][...]), ref)


def test_mrcfile_reads_our_mrc_file(tmp_path):
    mrcfile = _mod("mrcfile")
    from opencodecs import write_mrc
    x = INPUTS["f4"]()[None].repeat(3, 0)
    path = tmp_path / "v.mrc"
    write_mrc(str(path), x)
    with mrcfile.open(str(path), permissive=False) as m:
        assert np.array_equal(np.asarray(m.data), x)
    assert mrcfile.validate(str(path), print_file=io.StringIO())


@pytest.mark.parametrize("compress", [False, True])
def test_nibabel_reads_our_nifti_file(tmp_path, compress):
    nib = _mod("nibabel")
    from opencodecs import write_nifti
    x = INPUTS["u16"]()[..., None].repeat(3, -1)
    path = tmp_path / ("v.nii.gz" if compress else "v.nii")
    write_nifti(str(path), x, compress=compress)
    img = nib.load(str(path))
    assert np.array_equal(np.asarray(img.dataobj), x)


def test_imagecodecs_reads_our_rgbe_file(tmp_path):
    from opencodecs import rgbe_imwrite
    x = INPUTS["rgb_f4"]()
    path = tmp_path / "x.hdr"
    rgbe_imwrite(str(path), x)
    got = _ic_func("rgbe_decode")(path.read_bytes())
    _check(got, x, ("rgbe", 1 / 128), "RGBE file via imagecodecs")


def test_imagecodecs_reads_our_jxl_animation():
    from opencodecs import JxlWriter
    frames = np.stack([INPUTS["rgb"]() // (i + 1) for i in range(3)])
    w = JxlWriter(None, lossless=True, animation=True)
    for i, f in enumerate(frames):
        w.write_frame(f, is_last=(i == len(frames) - 1))
    data = w.close()
    dec = _ic_func("jpegxl_decode")
    for i, ref in enumerate(frames):
        assert np.array_equal(np.squeeze(dec(data, index=i)), ref), f"frame {i}"


@pytest.mark.parametrize("compression", ["none", "zstdhdr"])
def test_libczi_reads_our_czi_pyramid(tmp_path, compression):
    czirw = _mod("pylibCZIrw.czi")
    from opencodecs import CziPyramidWriter
    x = INPUTS["u16"]()
    levels = [x, x[::2, ::2].copy(), x[::4, ::4].copy()]
    path = tmp_path / "pyr.czi"
    with CziPyramidWriter(str(path), compression=compression) as w:
        w.write_pyramid(levels)
    with czirw.open_czi(str(path)) as r:
        got = np.squeeze(r.read())
    assert got.shape == x.shape
    assert np.array_equal(got, x), "libCZI sees different full-resolution pixels"
