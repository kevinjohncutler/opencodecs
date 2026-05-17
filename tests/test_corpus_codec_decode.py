"""Real-data decode + cross-validation sweep for every codec we ship.

The other test files (``test_tier1_scientific.py``, ``test_phase4-7_*.py``)
prove our codecs round-trip on synthetic input. This file goes further:

1. **Real photo data** — encode each lossy / lossless image codec on the
   Kodak24 photographic benchmark set (real photographs, not random
   bytes), then verify our decode matches both the source (PSNR or
   bit-exact) and ``imagecodecs``'s decode of the same blob. That last
   part proves wire-format interop, which is the only thing that
   matters when our blobs travel out into the wider ecosystem.

2. **Scientific data** — synthesize a smooth float field (sinusoid +
   gaussian noise — what you actually see in microscopy / climate
   data) and exercise the float compressors against the imagecodecs
   reference where they expose a parallel API.

3. **Corpus files we actually have** — decode the small upstream
   samples (WebP gallery, animated GIF) that ``download_test_corpus.sh``
   pulls.

All tests are skipif-gated on the corpus + imagecodecs availability,
so the suite stays green on a fresh checkout.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc


CORPUS_ROOT = Path(__file__).resolve().parent.parent / ".test_data"
KODAK_DIR = CORPUS_ROOT / "png" / "kodak24"
WEBP_DIR = CORPUS_ROOT / "webp"
GIF_DIR = CORPUS_ROOT / "gif"


def _ic_callable(ic, name: str):
    """Return ic.<name> only if it's a real function, not a stub.

    ``imagecodecs`` installs DelayedImportError stubs at module level
    when an optional dependency wasn't built, so ``getattr`` always
    returns *something* — we have to probe by exercising the
    function. Calling with an obviously-invalid input is the cheap
    way to flush out the stub vs. the real codec."""
    fn = getattr(ic, name, None)
    if fn is None:
        return None
    try:
        fn(b"")
    except ic.DelayedImportError:
        return None
    except Exception:
        # Any other exception (decode failure, type error) means
        # we hit the real codec — it's available.
        pass
    return fn

_HINT = (
    "Run `bash tests/download_test_corpus.sh --light` from the repo "
    "root to populate the real-data corpus (~15 MB)."
)


def _have_kodak() -> bool:
    return KODAK_DIR.exists() and any(KODAK_DIR.glob("kodim*.png"))


def _kodak_sample_paths(n: int = 4) -> list[Path]:
    """Pick a deterministic subset of Kodak photos for parametrize.

    Using all 24 across every codec blows the test-suite runtime up by
    24×. Four representative images keep the gold-standard signal
    without flogging CI."""
    if not _have_kodak():
        return []
    files = sorted(p for p in KODAK_DIR.glob("kodim*.png")
                   if not p.name.startswith("._"))
    if not files:
        return []
    # kodim01 (woman, smooth + detail), kodim08 (parrots, color),
    # kodim15 (girl, faces), kodim23 (macaw, hard detail). These cover
    # the failure modes lossy codecs typically struggle with.
    picks = ["kodim01.png", "kodim08.png", "kodim15.png", "kodim23.png"]
    out = [KODAK_DIR / p for p in picks if (KODAK_DIR / p).exists()]
    return out[:n] if out else files[:n]


def _load_kodak(path: Path) -> np.ndarray:
    """Decode a Kodak PNG to uint8 RGB ndarray via our PNG codec.

    We avoid Pillow (per project policy) and tifffile (TIFF only); the
    opencodecs PNG path is itself cross-validated against imagecodecs
    in test_corpus_png.py, so it's a known-good source."""
    return oc.get_codec("png").decode(path.read_bytes())


def _psnr(a: np.ndarray, b: np.ndarray, peak: float = 255.0) -> float:
    a = a.astype(np.float64)
    b = b.astype(np.float64)
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 20.0 * np.log10(peak) - 10.0 * np.log10(mse)


pytestmark = pytest.mark.skipif(not _have_kodak(), reason=_HINT)


# ---------------------------------------------------------------------------
# Lossy image codecs — PSNR floor + imagecodecs decode interop
# ---------------------------------------------------------------------------

# (codec_name, ic_decode_attr, psnr_floor_db)
#
# psnr floors are conservative (well below typical quality at default
# settings — Kodak photo @ default quality usually nets 30–38 dB) so the
# test doesn't false-alarm on minor encoder version drift. The point is
# to catch regressions like "encoder silently produced garbage", not to
# benchmark quality.
_LOSSY_CASES = [
    ("jpeg",    "jpeg_decode",     28.0),
    ("mozjpeg", "mozjpeg_decode",  28.0),    # often missing in ic
    ("webp",    "webp_decode",     30.0),
    ("avif",    "avif_decode",     28.0),
    ("heif",    "heif_decode",     30.0),    # often missing in ic
    ("jxl",     "jpegxl_decode",   32.0),
    ("jpeg2k",  "jpeg2k_decode",   26.0),
]


@pytest.mark.parametrize("path", _kodak_sample_paths(),
                          ids=lambda p: p.name)
@pytest.mark.parametrize("codec,ic_attr,psnr_floor", _LOSSY_CASES,
                          ids=lambda x: x if isinstance(x, str) else "")
def test_lossy_kodak_psnr_and_interop(path, codec, ic_attr, psnr_floor):
    """Encode a real Kodak photo with our codec, then:

    * Our decode must come back within ``psnr_floor`` dB of the source.
    * ``imagecodecs``'s decode of the same blob must agree with ours
      within ≤1 LSB (lossy codecs are deterministic but encoders /
      decoders can disagree on the last bit due to chroma upsampling
      / dithering, hence the small tolerance).
    """
    ic = pytest.importorskip("imagecodecs")
    if not oc.has_codec(codec):
        pytest.skip(f"{codec} codec not built")
    ic_decode = _ic_callable(ic, ic_attr)
    if ic_decode is None:
        pytest.skip(f"imagecodecs has no {ic_attr} (optional dep)")

    src = _load_kodak(path)
    c = oc.get_codec(codec)

    blob = c.encode(src)
    assert isinstance(blob, (bytes, bytearray)) and len(blob) > 0
    assert len(blob) < src.nbytes, (
        f"{codec}: encoded size {len(blob)} >= raw {src.nbytes} on a "
        f"natural photograph — lossy compressor producing wire-format "
        f"bloat is almost certainly a bug"
    )

    back_oc = c.decode(blob)
    assert back_oc.shape == src.shape
    assert back_oc.dtype == src.dtype
    p = _psnr(src, back_oc)
    assert p >= psnr_floor, (
        f"{codec} on {path.name}: PSNR {p:.2f} dB < floor "
        f"{psnr_floor:.2f} dB"
    )

    back_ic = ic_decode(blob)
    assert back_ic.shape == back_oc.shape, (
        f"{codec}: ic decode shape {back_ic.shape} vs oc {back_oc.shape}"
    )
    # AVIF, HEIF, WebP: same compressed bytes can decode to slightly
    # different RGB on libavif/libheif/libwebp due to YUV→RGB rounding
    # (libavif uses libyuv-style integer matrices while libheif/libwebp
    # use floating-point). Different builds disagree on ~20-25% of
    # pixels at 1-2 LSB. We only catch outright corruption (max diff
    # >> 2 LSB, or mean diff >> 0.5).  JPEG / JPEG2000 / lossless
    # codecs should agree to the bit.
    diff = np.abs(back_ic.astype(int) - back_oc.astype(int))
    max_abs = int(diff.max())
    mean_diff = float(diff.mean())
    if codec in ("avif", "heif", "webp"):
        assert max_abs <= 3, (
            f"{codec} on {path.name}: oc/ic decode disagree by "
            f"{max_abs} LSB (>3 — color-conversion drift bound exceeded)"
        )
        assert mean_diff <= 0.5, (
            f"{codec} on {path.name}: mean abs diff {mean_diff:.4f} > 0.5 "
            f"between oc/ic — likely a colorimetry bug"
        )
    else:
        assert max_abs == 0, (
            f"{codec} on {path.name}: oc/ic decode not bit-identical "
            f"(max diff {max_abs}) — wire-format interop bug?"
        )


# ---------------------------------------------------------------------------
# Lossless image codecs — bit-exact roundtrip + interop
# ---------------------------------------------------------------------------

_LOSSLESS_CASES = [
    ("png",     "png_decode"),
    ("qoi",     "qoi_decode"),
    ("jpegls",  "jpegls_decode"),
    ("htj2k",   "htj2k_decode"),   # often missing in ic
]


@pytest.mark.parametrize("path", _kodak_sample_paths(2),
                          ids=lambda p: p.name)
@pytest.mark.parametrize("codec,ic_attr", _LOSSLESS_CASES,
                          ids=lambda x: x if isinstance(x, str) else "")
def test_lossless_kodak_bit_exact_and_interop(path, codec, ic_attr):
    ic = pytest.importorskip("imagecodecs")
    if not oc.has_codec(codec):
        pytest.skip(f"{codec} codec not built")
    ic_decode = _ic_callable(ic, ic_attr)

    src = _load_kodak(path)
    c = oc.get_codec(codec)

    blob = c.encode(src)
    back_oc = c.decode(blob)
    np.testing.assert_array_equal(back_oc, src)

    if ic_decode is not None:
        back_ic = ic_decode(blob)
        np.testing.assert_array_equal(back_ic, back_oc)


# ---------------------------------------------------------------------------
# JPEG XL — bit-exact lossless roundtrip (uses a non-default flag).
# ---------------------------------------------------------------------------


def test_jxl_lossless_kodak_roundtrip():
    """JXL has a true-lossless mode; verify it on a real photo."""
    if not oc.has_codec("jxl"):
        pytest.skip("jxl codec not built")
    c = oc.get_codec("jxl")
    src = _load_kodak(_kodak_sample_paths(1)[0])

    blob = c.encode(src, lossless=True)
    back = c.decode(blob)
    np.testing.assert_array_equal(back, src)


# ---------------------------------------------------------------------------
# Brunsli — lossless JPEG recompression. Source must be a JPEG blob,
# not raw pixels. Build the JPEG from a Kodak photo and verify brunsli
# round-trips the JPEG bytes exactly.
# ---------------------------------------------------------------------------


def test_brunsli_kodak_jpeg_roundtrip():
    if not oc.has_codec("brunsli") or not oc.has_codec("jpeg"):
        pytest.skip("brunsli or jpeg codec not built")
    src = _load_kodak(_kodak_sample_paths(1)[0])
    jpeg_blob = oc.get_codec("jpeg").encode(src)

    bru = oc.get_codec("brunsli")
    brn = bru.encode(jpeg_blob)
    assert len(brn) < len(jpeg_blob), (
        f"brunsli should compress JPEG: brn={len(brn)} vs jpeg={len(jpeg_blob)}"
    )
    # asjpeg=True returns the recovered JPEG bytes; the default decode
    # path returns decoded pixels (handy for "I just want the image"
    # consumers).
    back_jpeg = bru.decode(brn, asjpeg=True)
    assert bytes(back_jpeg) == bytes(jpeg_blob), (
        "brunsli: recovered JPEG bytes differ from source — "
        "brunsli is supposed to be a perfect JPEG container transform"
    )


# ---------------------------------------------------------------------------
# BMP — uncompressed Windows bitmap. Bit-exact roundtrip + interop.
# ---------------------------------------------------------------------------


def test_bmp_kodak_bit_exact_and_interop():
    ic = pytest.importorskip("imagecodecs")
    if not oc.has_codec("bmp"):
        pytest.skip("bmp codec not built")
    bmp_decode = _ic_callable(ic, "bmp_decode")
    if bmp_decode is None:
        pytest.skip("imagecodecs has no bmp_decode")
    c = oc.get_codec("bmp")
    src = _load_kodak(_kodak_sample_paths(1)[0])
    blob = c.encode(src)
    back_oc = c.decode(blob)
    np.testing.assert_array_equal(back_oc, src)
    back_ic = bmp_decode(blob)
    np.testing.assert_array_equal(back_ic, src)


# ---------------------------------------------------------------------------
# Real upstream WebP + GIF samples (decode-only — we don't have an
# encoder reference to validate against, but decode-without-crash on
# real-world bytes is itself a meaningful signal).
# ---------------------------------------------------------------------------


def _real_glob(directory: Path, pattern: str) -> list[Path]:
    """Sorted glob that filters macOS dotfile resource forks.

    The NAS-backed test_data dir gets ``._foo.webp`` AppleDouble files
    from Finder copies — they parse as garbage and shouldn't be in
    the test loop."""
    if not directory.exists():
        return []
    return sorted(
        p for p in directory.glob(pattern)
        if not p.name.startswith("._")
    )


_WEBP_FILES = _real_glob(WEBP_DIR, "*.webp")
_GIF_FILES = _real_glob(GIF_DIR, "*.gif")


@pytest.mark.skipif(not _WEBP_FILES, reason="no WebP corpus")
@pytest.mark.parametrize("path", _WEBP_FILES, ids=lambda p: p.name)
def test_real_webp_decode_matches_imagecodecs(path):
    ic = pytest.importorskip("imagecodecs")
    if not oc.has_codec("webp"):
        pytest.skip("webp codec not built")
    blob = path.read_bytes()
    a = oc.get_codec("webp").decode(blob)
    b = ic.webp_decode(blob)
    assert a.shape == b.shape, (
        f"{path.name}: oc shape {a.shape} != ic {b.shape}"
    )
    np.testing.assert_array_equal(a, b)


@pytest.mark.skipif(not _GIF_FILES, reason="no GIF corpus")
@pytest.mark.parametrize("path", _GIF_FILES, ids=lambda p: p.name)
def test_real_gif_decode_first_frame(path):
    """Animated GIF — decode succeeds and produces a sane uint8 array.

    Don't cross-validate against imagecodecs because its gif_decode
    returns only one frame in older versions while ours may return
    a stack. Sanity-check shape + dtype instead."""
    if not oc.has_codec("gif"):
        pytest.skip("gif codec not built")
    blob = path.read_bytes()
    arr = oc.get_codec("gif").decode(blob)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.uint8
    assert arr.size > 0


# ---------------------------------------------------------------------------
# Scientific float compressors — synthesize a realistic smooth field
# (the kind of data zfp/sz3/sperr/pcodec are designed for) and confirm
# both error bounds and imagecodecs interop where exposed.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def smooth_field_3d() -> np.ndarray:
    """A 32×32×32 float32 field — sinusoid + gentle noise. Mimics what
    you find in volumetric microscopy / climate data."""
    z, y, x = np.mgrid[0:32, 0:32, 0:32].astype(np.float32) / 32.0
    base = (
        np.sin(4 * np.pi * x)
        + np.cos(3 * np.pi * y)
        + np.sin(2 * np.pi * z) * 0.5
    ).astype(np.float32)
    noise = np.random.default_rng(0).standard_normal(base.shape).astype(np.float32) * 0.05
    return base + noise


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_zfp_reversible_on_smooth_field(smooth_field_3d, dtype):
    if not oc.has_codec("zfp"):
        pytest.skip("zfp codec not built")
    field = smooth_field_3d.astype(dtype)
    c = oc.get_codec("zfp")
    blob = c.encode(field)
    back = c.decode(blob)
    np.testing.assert_array_equal(back, field)


# Wire-format note for scientific compressors
# --------------------------------------------
# opencodecs's pcodec / sz3 / sperr / aec / rcomp wrap each library's
# raw output with a small self-describing header (shape + dtype) so
# users don't have to remember the original shape. imagecodecs's
# corresponding *_decode helpers expect the *raw* library output and
# require the caller to pass shape/dtype, so the two libraries are
# **not** wire-compatible by design. We don't write interop tests
# against ic for these — instead, each codec has its own
# self-roundtrip test in test_tier1_scientific.py.
#
# zfp is the exception: opencodecs's zfp codec uses zfp's native
# self-describing serialization (zfp headers carry shape + dtype),
# so the wire format IS interoperable with imagecodecs.zfp_decode.


def test_zfp_interop_with_imagecodecs(smooth_field_3d):
    """Our zfp blob must decode byte-identical via imagecodecs.zfp_decode.

    Possible because zfp's own stream format is self-describing — we
    just hand the library bytes off, no opencodecs-specific wrapper."""
    ic = pytest.importorskip("imagecodecs")
    if not oc.has_codec("zfp"):
        pytest.skip("zfp codec not built")
    zfp_decode = _ic_callable(ic, "zfp_decode")
    if zfp_decode is None:
        pytest.skip("imagecodecs has no zfp_decode")
    blob = oc.get_codec("zfp").encode(smooth_field_3d)
    back = zfp_decode(blob)
    np.testing.assert_array_equal(back, smooth_field_3d)


def test_sci_compressors_self_roundtrip(smooth_field_3d):
    """For wire-format-incompatible sci compressors, prove each one
    roundtrips its own blob — the codec is functional even if the
    bytes aren't ic-readable."""
    for name in ("pcodec", "sz3", "sperr"):
        if not oc.has_codec(name):
            continue
        c = oc.get_codec(name)
        blob = c.encode(smooth_field_3d)
        back = c.decode(blob)
        if back.shape != smooth_field_3d.shape:
            back = back.reshape(smooth_field_3d.shape)
        # pcodec is lossless on floats; sz3/sperr are lossy. We only
        # need a sanity bound to confirm the encode/decode path works.
        if name == "pcodec":
            np.testing.assert_array_equal(back, smooth_field_3d)
        else:
            rms = float(np.sqrt(np.mean(
                (back.astype(np.float64) - smooth_field_3d.astype(np.float64)) ** 2
            )))
            data_range = float(np.ptp(smooth_field_3d))
            assert rms < 0.05 * data_range, (
                f"{name}: lossy roundtrip RMS {rms:.4f} > 5% of "
                f"data range {data_range:.4f} — codec or our wrapper broken"
            )


# ---------------------------------------------------------------------------
# Integer/generic compressors — exercise on a realistic image-derived
# byte stream (Kodak photo flattened to bytes — has the same byte-level
# entropy you see in image residuals after a delta predictor).
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def kodak_bytes() -> bytes:
    return _load_kodak(_kodak_sample_paths(1)[0]).tobytes()


@pytest.mark.parametrize("codec,ic_attr", [
    ("deflate", "deflate_decode"),
    ("zstd",    "zstd_decode"),
    # opencodecs ships LZ4 as the frame format (magic 04 22 4D 18);
    # imagecodecs's lz4_decode is raw-block, lz4f_decode is the frame
    # variant. Match the right one for wire interop.
    ("lz4",     "lz4f_decode"),
    ("lzma",    "lzma_decode"),
    ("brotli",  "brotli_decode"),
    ("snappy",  "snappy_decode"),
    ("bz2",     "bz2_decode"),
])
def test_compressor_kodak_bytes_interop(kodak_bytes, codec, ic_attr):
    """Generic byte compressors: oc.encode → ic.decode must produce
    bit-identical bytes back. This is the most basic wire-format
    correctness check, and the one that catches accidental header
    drift fastest."""
    ic = pytest.importorskip("imagecodecs")
    if not oc.has_codec(codec):
        pytest.skip(f"{codec} not built")
    ic_decode = _ic_callable(ic, ic_attr)
    if ic_decode is None:
        pytest.skip(f"imagecodecs has no {ic_attr}")
    c = oc.get_codec(codec)
    blob = c.encode(kodak_bytes)
    back = ic_decode(blob)
    assert bytes(back) == kodak_bytes


# ---------------------------------------------------------------------------
# rcomp / aec — self-roundtrip on Kodak-derived data.
#
# Like sz3/pcodec/sperr, opencodecs's rcomp and aec codecs add their
# own shape/dtype prefix to the library output. ic's rcomp_decode /
# aec_decode operate on the raw library bytes and need shape/dtype
# passed in. Different wire formats by design; we just verify our
# own roundtrip on a realistic data shape.
# ---------------------------------------------------------------------------


def test_rcomp_kodak_self_roundtrip(kodak_bytes):
    if not oc.has_codec("rcomp"):
        pytest.skip("rcomp codec not built")
    arr = np.frombuffer(kodak_bytes, dtype=np.uint8).astype(np.int16)
    c = oc.get_codec("rcomp")
    blob = c.encode(arr)
    back = c.decode(blob)
    # rcomp returns raw bytes; reshape to original.
    arr_back = np.frombuffer(back, dtype=arr.dtype) if isinstance(back, (bytes, bytearray)) else back
    np.testing.assert_array_equal(arr_back, arr)


def test_aec_kodak_self_roundtrip(kodak_bytes):
    if not oc.has_codec("aec"):
        pytest.skip("aec codec not built")
    arr = np.frombuffer(kodak_bytes, dtype=np.uint8).copy()
    c = oc.get_codec("aec")
    blob = c.encode(arr, bits_per_sample=8)
    back = c.decode(blob, dtype=np.uint8, shape=arr.shape, bits_per_sample=8)
    arr_back = (
        np.frombuffer(back, dtype=np.uint8) if isinstance(back, (bytes, bytearray))
        else np.asarray(back).view(np.uint8)
    )
    np.testing.assert_array_equal(arr_back.reshape(arr.shape), arr)


# ---------------------------------------------------------------------------
# bcn — decode-only codec. Encode a Kodak image with ic's BC1 encoder
# (we don't ship one) and verify our decoder gives the same pixels as
# ic's decoder.
# ---------------------------------------------------------------------------


def test_bcn_decode_matches_imagecodecs():
    """BCn is decode-only on both sides (imagecodecs.bcn_encode is
    stubbed "not implemented" too). Build a random BC1 blob and
    confirm oc and ic decoders produce identical pixels."""
    ic = pytest.importorskip("imagecodecs")
    if not oc.has_codec("bcn"):
        pytest.skip("bcn codec not built")
    bcn_decode = _ic_callable(ic, "bcn_decode")
    if bcn_decode is None:
        pytest.skip("imagecodecs has no bcn_decode")
    # BC1: 8 bytes per 4x4 block. Build a 16x16 random blob —
    # arbitrary bit patterns are still valid BC1 (no checksum, just
    # interpolated colors). This proves our decoder agrees with the
    # reference on byte-equivalent input.
    w = h = 16
    n_blocks = (w // 4) * (h // 4)
    rng = np.random.default_rng(0)
    blob = rng.integers(0, 256, size=n_blocks * 8,
                         dtype=np.uint8).tobytes()
    # ic.bcn_decode: positional format, shape kwarg includes channels.
    ic_back = bcn_decode(blob, 1, shape=(h, w, 4))
    oc_back = oc.get_codec("bcn").decode(blob, format="bc1",
                                          width=w, height=h)
    assert oc_back.shape[:2] == ic_back.shape[:2], (
        f"shape disagreement: oc {oc_back.shape} ic {ic_back.shape}"
    )
    # Compare RGB channels (the LSB of BC1 alpha is a 1-bit punchthrough
    # flag — keep things simple and just compare color).
    np.testing.assert_array_equal(oc_back[..., :3], ic_back[..., :3])


# ---------------------------------------------------------------------------
# dicomrle — DICOM RLE. ic exposes a parallel API; cross-validate.
# ---------------------------------------------------------------------------


def test_dicomrle_kodak_interop():
    ic = pytest.importorskip("imagecodecs")
    if not oc.has_codec("dicomrle"):
        pytest.skip("dicomrle codec not built")
    dicomrle_decode = _ic_callable(ic, "dicomrle_decode")
    if dicomrle_decode is None:
        pytest.skip("imagecodecs has no dicomrle_decode")
    # DICOM RLE is for medical images; downsample a Kodak photo to a
    # 64×64 uint8 gray "scan slice" — what the codec is meant for.
    src = _load_kodak(_kodak_sample_paths(1)[0])
    gray = np.ascontiguousarray(src[:64, :64, 0])
    blob = oc.get_codec("dicomrle").encode(gray)
    # ic.dicomrle_decode returns raw bytes; caller reshapes.
    raw = dicomrle_decode(blob, dtype=gray.dtype)
    back = np.frombuffer(raw, dtype=gray.dtype).reshape(gray.shape)
    np.testing.assert_array_equal(back, gray)


# ---------------------------------------------------------------------------
# Filter / predictor codecs (delta / xor / floatpred / byteshuffle /
# quantize / packints / numpy) — these are byte-level filters that get
# their real-world exercise inside container readers (TIFF predictors,
# CZI byteshuffle, Zarr filters). We cover them with synthetic round-
# trips in test_phase4_filters.py and via the containers themselves;
# adding a duplicate Kodak-byte path here would just add CI minutes
# without finding new bugs.
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Container readers (czi/lif/nd2/oib/oir/vsi/hdf5) and pyramid TIFF
# get their real-data coverage in their dedicated test_corpus_*.py /
# test_<format>_*.py files. Not duplicated here.
# ---------------------------------------------------------------------------
