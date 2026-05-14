"""Real-world corpus smoke tests for FITS, HEIF, LERC, DICOM.

Each test verifies that the corresponding opencodecs reader/decoder
can open a public real-world sample. These complement the synthetic
encode/decode round-trip tests, which never see "real" bytes
emitted by a different toolchain.

All tests are skipif-gated on file presence — run
``bash tests/download_test_corpus.sh --light`` from the repo root
to populate the corpus (~12 MB).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

CORPUS = Path(__file__).resolve().parent.parent / ".test_data"
_HINT = (
    "Run `bash tests/download_test_corpus.sh --light` from the repo "
    "root to populate the corpus."
)


# ---------------------------------------------------------------------------
# FITS — Hubble HorseHead nebula
# ---------------------------------------------------------------------------

FITS_SAMPLE = CORPUS / "fits" / "HorseHead.fits"


@pytest.mark.skipif(not FITS_SAMPLE.exists(), reason=_HINT)
def test_fits_horsehead_opens():
    """Astropy tutorials sample. Real Hubble exposure as int16.
    Verifies _fits parses the primary HDU header + reads the data."""
    from opencodecs._fits import FitsStream
    with FitsStream(str(FITS_SAMPLE)) as fs:
        hdu = fs.hdu(0)
        # Expected geometry per the astropy tutorial: 893 x 891 int16
        arr = hdu.asarray()
        assert arr.ndim == 2
        assert arr.shape == (893, 891)
        assert arr.dtype.kind == "i"
        assert arr.dtype.itemsize == 2
        # Real exposure has non-trivial dynamic range
        assert arr.max() > arr.min()


@pytest.mark.skipif(not FITS_SAMPLE.exists(), reason=_HINT)
def test_fits_horsehead_matches_astropy():
    astropy = pytest.importorskip("astropy.io.fits")
    from opencodecs._fits import FitsStream

    with astropy.open(str(FITS_SAMPLE)) as hdul:
        ref = hdul[0].data
    with FitsStream(str(FITS_SAMPLE)) as fs:
        ours = fs.hdu(0).asarray()
    assert np.array_equal(ours, ref)


# ---------------------------------------------------------------------------
# HEIF — Nokia conformance C001 (single still HEIC)
# ---------------------------------------------------------------------------

HEIF_SAMPLE = CORPUS / "heif" / "C001.heic"


@pytest.mark.skipif(not HEIF_SAMPLE.exists(), reason=_HINT)
def test_heif_conformance_c001_decodes():
    import opencodecs as oc
    data = HEIF_SAMPLE.read_bytes()
    codec = oc.get_codec("heif")
    arr = codec.decode(data)
    # C001 is a 1280x720 RGB still per the conformance suite docs.
    assert arr.ndim == 3
    assert arr.shape[2] in (3, 4)
    assert arr.dtype == np.uint8
    assert arr.shape[0] > 0 and arr.shape[1] > 0


# ---------------------------------------------------------------------------
# LERC — ESRI reference test files
# ---------------------------------------------------------------------------

LERC_FLOAT = CORPUS / "lerc" / "california_float.lerc2"
LERC_BYTE = CORPUS / "lerc" / "bluemarble_byte.lerc2"


@pytest.mark.skipif(not LERC_FLOAT.exists(), reason=_HINT)
def test_lerc_california_float_decodes():
    """ESRI's float32 elevation raster. Tests our _lerc against bytes
    written by the canonical encoder, on a different shape than our
    synthetic tests usually generate (400x400 single-band float32)."""
    import opencodecs as oc
    data = LERC_FLOAT.read_bytes()
    arr = oc.get_codec("lerc").decode(data)
    assert arr.dtype == np.float32
    # The file name encodes the shape: 400x400x1
    assert arr.shape == (400, 400) or arr.shape == (400, 400, 1)


@pytest.mark.skipif(not LERC_BYTE.exists(), reason=_HINT)
def test_lerc_bluemarble_byte_decodes():
    """ESRI's uint8 RGB raster (256x256, 3 bands). Validates the
    multi-band path. LERC returns multi-band data as (n_bands, H, W)
    — bands-first, not channels-last."""
    import opencodecs as oc
    data = LERC_BYTE.read_bytes()
    arr = oc.get_codec("lerc").decode(data)
    assert arr.dtype == np.uint8
    assert arr.shape == (3, 256, 256)


# NOTE: no oc-vs-imagecodecs LERC cross-check here. Both libraries
# vendor their own libLerc.4.dylib and export the same global symbols;
# dyld resolves all calls to whichever was loaded second, which
# corrupts internal state and SIGABRTs the process. See
# [[beating-imagecodecs-perf-patterns]] § "libLerc symbol clash". The
# decode tests above + ESRI reference bytes are enough to validate
# our decoder against the canonical encoder. For a head-to-head bench
# use bench_h2h_lerc which spawns each library in its own subprocess.


# ---------------------------------------------------------------------------
# DICOM — pydicom-data emri_small (functional MRI volume)
# ---------------------------------------------------------------------------

DICOM_SAMPLE = CORPUS / "dicom" / "emri_small.dcm"


@pytest.mark.skipif(not DICOM_SAMPLE.exists(), reason=_HINT)
def test_dicom_emri_small_pixel_data_via_pydicom():
    """pydicom is the reference DICOM toolkit. We don't yet have a
    standalone DICOM decoder in opencodecs (only DICOMweb HTTP), so
    this just locks down that the corpus file is structurally valid
    — anchor for a future native DICOM reader."""
    pydicom = pytest.importorskip("pydicom")
    ds = pydicom.dcmread(str(DICOM_SAMPLE))
    arr = ds.pixel_array
    # emri_small is a 64x64 multi-slice MRI int16 volume
    assert arr.ndim >= 2
    assert arr.dtype in (np.int16, np.uint16)
