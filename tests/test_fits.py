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
