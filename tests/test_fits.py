"""FITS reader tests.

Uses astropy.io.fits to generate ground-truth FITS files, then
verifies our native FitsStream decodes them pixel-equal. Covers:

* All standard BITPIX values: 8 / 16 / 32 / 64 / -32 / -64
* Unsigned-int convention via BZERO/BSCALE (BITPIX=16, BZERO=32768 ->
  uint16; same for 32/64).
* Multi-extension files (primary + IMAGE_HDU extensions).
* Streaming over a callable read_at (HTTP-range simulation).
"""

from __future__ import annotations

import io

import numpy as np
import pytest

astropy_fits = pytest.importorskip("astropy.io.fits")
from opencodecs._fits import FitsStream, FitsHDU, imread


@pytest.mark.parametrize("dtype", [
    np.uint8,
    np.int16, np.uint16,        # signed16; uint16 via BZERO=32768
    np.int32, np.uint32,
    np.int64, np.uint64,
    np.float32, np.float64,
])
def test_primary_hdu_dtype_round_trip(tmp_path, dtype):
    """Every supported BITPIX value reads back pixel-equal to astropy."""
    rng = np.random.default_rng(0)
    if np.issubdtype(dtype, np.floating):
        arr = rng.standard_normal((48, 64)).astype(dtype)
    elif dtype in (np.uint64, np.int64):
        arr = rng.integers(0, 1 << 30, size=(48, 64), dtype=dtype)
    elif dtype in (np.uint32, np.int32):
        arr = rng.integers(0, 1 << 28, size=(48, 64), dtype=dtype)
    else:
        info = np.iinfo(dtype)
        arr = rng.integers(info.min, info.max + 1,
                           size=(48, 64), dtype=dtype)
    p = tmp_path / f"{np.dtype(dtype).name}.fits"
    astropy_fits.writeto(str(p), arr, overwrite=True)
    back = imread(p)
    np.testing.assert_array_equal(back, arr)


def test_3d_image(tmp_path):
    """3-D arrays: (NAXIS3, NAXIS2, NAXIS1) per FITS, returned in
    numpy's slowest-first order."""
    arr = np.arange(2 * 8 * 16, dtype=np.int16).reshape(2, 8, 16)
    p = tmp_path / "cube.fits"
    astropy_fits.writeto(str(p), arr, overwrite=True)
    with FitsStream(p) as f:
        hdu = f.hdu(0)
        assert hdu.shape == arr.shape
        np.testing.assert_array_equal(hdu.asarray(), arr)


def test_multi_extension(tmp_path):
    """Primary + N IMAGE_HDU extensions — each one is a separate HDU
    in our stream and pixel-equal to what astropy wrote."""
    primary = np.arange(8 * 16, dtype=np.float32).reshape(8, 16)
    ext1 = np.arange(4 * 8, dtype=np.int16).reshape(4, 8)
    ext2 = np.arange(2 * 4, dtype=np.uint16).reshape(2, 4)

    hdu_list = astropy_fits.HDUList([
        astropy_fits.PrimaryHDU(primary),
        astropy_fits.ImageHDU(ext1, name="EXT1"),
        astropy_fits.ImageHDU(ext2, name="EXT2"),
    ])
    p = tmp_path / "multi.fits"
    hdu_list.writeto(str(p), overwrite=True)

    with FitsStream(p) as f:
        assert f.n_hdus == 3
        np.testing.assert_array_equal(f.hdu(0).asarray(), primary)
        np.testing.assert_array_equal(f.hdu(1).asarray(), ext1)
        np.testing.assert_array_equal(f.hdu(2).asarray(), ext2)


def test_bzero_bscale_unsigned_int_convention(tmp_path):
    """BITPIX=16 with BZERO=32768 and BSCALE=1 means 'this is uint16
    stored as int16 with offset' — astropy writes uint16 arrays that
    way, our reader must reverse the transform."""
    arr = np.array([[0, 1, 32767, 32768, 65535]], dtype=np.uint16)
    p = tmp_path / "u16.fits"
    astropy_fits.writeto(str(p), arr, overwrite=True)
    with FitsStream(p) as f:
        hdu = f.hdu(0)
        assert hdu.dtype == np.dtype("u2")
    back = imread(p)
    np.testing.assert_array_equal(back, arr)


def test_read_via_bytes_buffer(tmp_path):
    """FitsStream accepts a bytes buffer directly (no fd open)."""
    arr = np.arange(4 * 8, dtype=np.int16).reshape(4, 8)
    p = tmp_path / "buf.fits"
    astropy_fits.writeto(str(p), arr, overwrite=True)
    data = p.read_bytes()
    with FitsStream(data) as f:
        np.testing.assert_array_equal(f.hdu(0).asarray(), arr)


def test_read_via_callable_read_at(tmp_path):
    """FitsStream accepts a callable read_at(offset, n) -> bytes —
    the same interface our HTTPDataSource exposes."""
    arr = np.arange(16 * 16, dtype=np.uint16).reshape(16, 16) * 17
    p = tmp_path / "callable.fits"
    astropy_fits.writeto(str(p), arr, overwrite=True)
    data = p.read_bytes()
    # Use a counted closure to confirm callable was used.
    calls = []
    def _read_at(offset, n):
        calls.append((offset, n))
        return data[offset:offset + n]
    with FitsStream(_read_at) as f:
        np.testing.assert_array_equal(f.hdu(0).asarray(), arr)
    assert len(calls) > 0, "callable read_at wasn't invoked"


def test_invalid_file_raises(tmp_path):
    p = tmp_path / "bogus.fits"
    p.write_bytes(b"not a fits file at all" + b"\x00" * 3000)
    with FitsStream(p) as f:
        # No HDUs should parse.
        assert f.n_hdus == 0


# ---------------------------------------------------------------------------
# Compressed-image HDUs (BINTABLE + ZIMAGE=T)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ctype", ["RICE_1", "GZIP_1", "GZIP_2", "NOCOMPRESS"])
def test_compressed_fits_int16(tmp_path, ctype):
    """All four supported compression types round-trip int16 tiles."""
    img = np.arange(64 * 128, dtype=np.int16).reshape(64, 128) * 7 - 5000
    p = tmp_path / f"int16_{ctype}.fits"
    hdul = astropy_fits.HDUList([
        astropy_fits.PrimaryHDU(),
        astropy_fits.CompImageHDU(img, compression_type=ctype),
    ])
    hdul.writeto(p)
    arr = FitsStream(p).hdu(1).asarray()
    np.testing.assert_array_equal(arr, img.astype(arr.dtype, copy=False))


@pytest.mark.parametrize("ctype", ["RICE_1", "GZIP_1", "GZIP_2"])
def test_compressed_fits_int32(tmp_path, ctype):
    img = np.arange(32 * 64, dtype=np.int32).reshape(32, 64) * 1000
    p = tmp_path / f"int32_{ctype}.fits"
    astropy_fits.HDUList([
        astropy_fits.PrimaryHDU(),
        astropy_fits.CompImageHDU(img, compression_type=ctype),
    ]).writeto(p)
    arr = FitsStream(p).hdu(1).asarray()
    np.testing.assert_array_equal(arr, img.astype(arr.dtype, copy=False))


@pytest.mark.parametrize("ctype", ["RICE_1", "GZIP_1", "GZIP_2"])
def test_compressed_fits_float32_matches_astropy(tmp_path, ctype):
    """Float compression in FITS is lossy (per-tile ZSCALE/ZZERO
    quantization). Our decode should produce the same bytes astropy's
    own decoder produces — pixel-equal not original-equal."""
    img = (np.arange(32 * 64, dtype=np.float32).reshape(32, 64) / 100.0)
    p = tmp_path / f"float32_{ctype}.fits"
    astropy_fits.HDUList([
        astropy_fits.PrimaryHDU(),
        astropy_fits.CompImageHDU(img, compression_type=ctype),
    ]).writeto(p)
    ours = FitsStream(p).hdu(1).asarray()
    theirs = astropy_fits.open(p)[1].data
    np.testing.assert_array_equal(ours, theirs)


def test_compressed_fits_gzip_fallback_for_incompressible_tile(tmp_path):
    """Real-world compressed FITS uses GZIP_COMPRESSED_DATA as a fallback
    when a tile would compress poorly. Our parser must follow that
    fallback path (count=0 in primary descriptor) and treat the fallback
    tile as raw-bytes (no ZSCALE/ZZERO rescale)."""
    rng = np.random.default_rng(0)
    img = rng.normal(scale=1000.0, size=(48, 96)).astype(np.float32)
    p = tmp_path / "noisy_float.fits"
    astropy_fits.HDUList([
        astropy_fits.PrimaryHDU(),
        astropy_fits.CompImageHDU(img, compression_type="RICE_1"),
    ]).writeto(p)
    ours = FitsStream(p).hdu(1).asarray()
    theirs = astropy_fits.open(p)[1].data
    np.testing.assert_array_equal(ours, theirs)


def test_compressed_fits_hcompress_int16(tmp_path):
    """HCOMPRESS_1 decode for an int16 image — exercises the vendored
    cfitsio fits_hdecompress through opencodecs.codecs._hcomp."""
    img = np.arange(64 * 64, dtype=np.int16).reshape(64, 64) - 1000
    p = tmp_path / "hcompress.fits"
    astropy_fits.HDUList([
        astropy_fits.PrimaryHDU(),
        astropy_fits.CompImageHDU(img, compression_type="HCOMPRESS_1"),
    ]).writeto(p)
    ours = FitsStream(p).hdu(1).asarray()
    theirs = astropy_fits.open(p)[1].data
    np.testing.assert_array_equal(ours, theirs)


def test_compressed_fits_plio_1_roundtrip(tmp_path):
    """PLIO_1 (IRAF mask coding) — astropy writes a PLIO_1-compressed
    int16 mask, opencodecs decodes it back to the original."""
    # Mask-like int16 with small run-lengths — exercises the PLIO
    # run-length encoder's typical workload (segmentation labels).
    rng = np.random.default_rng(0)
    img = np.zeros((48, 48), dtype=np.int16)
    img[10:20, 10:20] = 3
    img[20:35, 5:30] = 7
    img[35:, :] = rng.integers(0, 4, size=(13, 48), dtype=np.int16)
    p = tmp_path / "plio.fits"
    astropy_fits.HDUList([
        astropy_fits.PrimaryHDU(),
        astropy_fits.CompImageHDU(img, compression_type="PLIO_1"),
    ]).writeto(p)
    ours = FitsStream(p).hdu(1).asarray()
    theirs = astropy_fits.open(p)[1].data
    np.testing.assert_array_equal(ours, theirs)
    np.testing.assert_array_equal(ours, img)


def test_plio_rejects_malformed_payloads_without_crashing():
    """PLIO line lists must be bounds-checked against the payload length.

    cfitsio before 4.7.0 decoded PLIO without validating that a line-list
    opcode stays inside the source buffer, so a crafted or truncated
    payload walks off the end. Upstream added the ``srclen`` argument and
    the two range checks in ``pl_l2pi`` on 2026-07-21 ("Added checks for
    group key overflow and for plio compression"); we vendored the fix
    and pass the length through.

    Against the pre-fix decoder this corpus killed the interpreter with
    SIGBUS, so a plain "it returned" assertion is a real regression
    guard. Reachable from any FITS file: _fits_compressed.py hands heap
    bytes straight to the decoder.
    """
    plio = pytest.importorskip("opencodecs.codecs._plio")
    rng = np.random.default_rng(0)
    decoded = rejected = 0
    for _ in range(500):
        n = int(rng.integers(1, 40))
        payload = rng.integers(0, 65536, n, dtype=np.uint16).astype(">u2").tobytes()
        try:
            plio.decode_raw(payload, nelements=int(rng.integers(1, 4000)))
            decoded += 1
        except plio.PlioError:
            rejected += 1
    assert decoded + rejected == 500
    assert rejected > 0, "expected some malformed payloads to be rejected"


@pytest.mark.parametrize("dtype,bytes_per_pixel", [(np.int32, 4), (np.int16, 2),
                                                   (np.int8, 1)])
def test_rice_decode_rejects_truncated_payload(dtype, bytes_per_pixel):
    """The first pixel of a Rice stream is stored unencoded, so a payload
    shorter than one pixel must be rejected rather than read past.

    Upstream cfitsio added the 4-byte form of this guard to fits_rdecomp
    on 2025-03-03 ("Added a buffer size check to fits_rdecomp"); our
    vendored copy predated it. The short and byte decoders read 2 and 1
    bytes the same way, so they are guarded too.

    Not reachable through opencodecs.rcomp_decode, whose 12-byte framing
    header rejects short blobs first, but decode_raw is public and the
    FITS tile path calls into the same C.
    """
    rcomp = pytest.importorskip("opencodecs.codecs._rcomp")
    arr = np.arange(64, dtype=dtype)
    blob = rcomp.encode(arr)
    assert np.array_equal(rcomp.decode(blob).astype(dtype), arr)
    payload = blob[12:]
    for clen in range(bytes_per_pixel):
        with pytest.raises(rcomp.RcompError):
            rcomp.decode_raw(payload[:clen], nelements=64, blocksize=32,
                             bytes_per_pixel=bytes_per_pixel)
