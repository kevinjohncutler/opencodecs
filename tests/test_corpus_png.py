"""PNG decoder validation against real-world PNG corpora.

Two public corpora:
  * **PngSuite** (~150 files): canonical PNG spec-coverage suite.
    Covers every bit depth (1/2/4/8/16), color type (gray/RGB/
    palette/gray+a/RGBA), filter, gamma, transparency, interlacing,
    and corruption-detection edge case.
  * **Kodak24** (24 photographic PNGs): the standard photo-codec
    quality bench set. Real images, not synthetic, so they exercise
    every defilter path the libspng patch needs to land on (and let
    us measure compression ratios + decode speed against imagecodecs
    on data users actually have).

Tests are skipif-gated on file presence — they run only after
``bash tests/download_test_corpus.sh`` populates ``.test_data/png/``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc

CORPUS_ROOT = Path(__file__).resolve().parent.parent / ".test_data" / "png"
PNGSUITE_DIR = CORPUS_ROOT / "pngsuite"
KODAK_DIR = CORPUS_ROOT / "kodak24"

_HINT = (
    "Run `bash tests/download_test_corpus.sh --light` from the repo "
    "root to populate the PNG corpus (~12 MB)."
)


# ---------------------------------------------------------------------------
# PngSuite — variant coverage
# ---------------------------------------------------------------------------


def _pngsuite_valid_files() -> list[Path]:
    """All PngSuite files that should decode successfully.

    PngSuite's naming convention:
      * ``bas*`` (basic non-interlaced), ``bgw*`` (basic interlaced w/bKGD),
        ``f*`` (filter test), ``s*`` (size), ``t*`` (transparency),
        ``g*`` (gamma), ``c*`` (chroma), ``z*`` (zlib), ``oi*`` (odd
        IDAT layout): all valid PNGs.
      * ``x*`` (corruption-detection): malformed PNGs that should be
        REJECTED. Skipped here (covered separately).
    """
    if not PNGSUITE_DIR.exists():
        return []
    return sorted(
        p for p in PNGSUITE_DIR.glob("*.png")
        if not p.name.startswith(("x", "._"))
        and p.name != "PngSuite.png"
    )


def _kodak_files() -> list[Path]:
    if not KODAK_DIR.exists():
        return []
    return sorted(
        p for p in KODAK_DIR.glob("kodim*.png")
        if not p.name.startswith("._")
    )


@pytest.mark.skipif(not PNGSUITE_DIR.exists(), reason=_HINT)
def test_pngsuite_files_present():
    files = _pngsuite_valid_files()
    # Sanity floor: corpus has ~140 valid files; abort if extraction
    # produced fewer (probably a partial download).
    assert len(files) >= 100, f"only {len(files)} pngsuite files"


@pytest.mark.skipif(not _pngsuite_valid_files(), reason=_HINT)
@pytest.mark.parametrize("path", _pngsuite_valid_files(), ids=lambda p: p.name)
def test_pngsuite_decodes(path):
    """Every valid PngSuite file must decode without raising. We don't
    cross-check pixel values against a reference (the corpus has no
    golden ndarray); we just verify decode succeeds and produces a
    plausible ndarray."""
    data = path.read_bytes()
    arr = oc.read(data, format="png")
    assert isinstance(arr, np.ndarray)
    assert arr.size > 0
    # PngSuite filenames encode the expected geometry: basn0g01 -> bit
    # depth 1, basn0g08 -> bit depth 8, basn6a16 -> RGBA 16-bit, etc.
    # Spot-check: 16-bit files should come back as uint16; 8-bit and
    # below should be uint8.
    if "16" in path.stem[-3:]:
        assert arr.dtype == np.uint16, (
            f"{path.name}: expected uint16, got {arr.dtype}"
        )
    elif any(d in path.stem[-3:] for d in ("01", "02", "04", "08")):
        assert arr.dtype == np.uint8, (
            f"{path.name}: expected uint8, got {arr.dtype}"
        )


@pytest.mark.skipif(not _pngsuite_valid_files(), reason=_HINT)
def test_pngsuite_corruption_detection_files_rejected():
    """PngSuite includes ``x*`` files that are deliberately malformed.
    Our decoder should refuse to decode them — at minimum not
    silently return wrong data."""
    if not PNGSUITE_DIR.exists():
        pytest.skip(_HINT)
    bad = sorted(
        p for p in PNGSUITE_DIR.glob("x*.png")
        if not p.name.startswith("._")
    )
    if not bad:
        pytest.skip("no corruption-detection files in corpus")
    rejected = 0
    for path in bad:
        try:
            oc.read(path.read_bytes(), format="png")
        except Exception:
            rejected += 1
    # The corruption suite has ~12 files; we expect most to be
    # rejected. libspng is strict; libpng less so. Floor at 50%.
    assert rejected >= len(bad) // 2, (
        f"only {rejected}/{len(bad)} corrupt files were rejected"
    )


# ---------------------------------------------------------------------------
# Kodak24 — cross-check pixel values against imagecodecs's libpng
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _kodak_files(), reason=_HINT)
@pytest.mark.parametrize("path", _kodak_files(), ids=lambda p: p.name)
def test_kodak_oc_matches_imagecodecs(path):
    """Every Kodak image must decode byte-identical between our
    libspng path and imagecodecs's libpng. Real photographic data
    is the most relevant correctness check after our libspng
    grayscale-defilter patch (commit 3a9feaf)."""
    ic = pytest.importorskip("imagecodecs")
    data = path.read_bytes()
    a = oc.read(data, format="png")
    b = ic.png_decode(data)
    assert a.shape == b.shape, f"{path.name}: shape mismatch"
    assert a.dtype == b.dtype, f"{path.name}: dtype mismatch"
    assert np.array_equal(a, b), f"{path.name}: pixel mismatch"


@pytest.mark.skipif(not _kodak_files(), reason=_HINT)
def test_kodak_decode_is_competitive_with_imagecodecs():
    """Aggregate decode time across all 24 Kodak photos: oc should be
    no slower than 1.05x imagecodecs. Sentinel against a regression
    in the libspng defilter patch."""
    import time
    ic = pytest.importorskip("imagecodecs")
    files = _kodak_files()
    payloads = [p.read_bytes() for p in files]
    # warmup
    for d in payloads:
        oc.read(d, format="png"); ic.png_decode(d)

    t0 = time.perf_counter_ns()
    for d in payloads: oc.read(d, format="png")
    t_oc = (time.perf_counter_ns() - t0) / 1e6
    t0 = time.perf_counter_ns()
    for d in payloads: ic.png_decode(d)
    t_ic = (time.perf_counter_ns() - t0) / 1e6
    # Allow up to 5% slowdown — we expect to be at parity or faster
    # on RGB u8 (NEON path) and at parity on gray (libspng patch).
    assert t_oc < t_ic * 1.05, (
        f"oc Kodak24 decode {t_oc:.0f} ms > 1.05 * ic {t_ic:.0f} ms"
    )
