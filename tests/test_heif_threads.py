"""HEIF decode threads change the speed, never the pixels.

With libheif 1.21 or newer the HEVC decoder is handed the thread count,
and libde265 decodes the wavefront rows x265 writes in parallel. Every
thread count must give the image a single thread gives.
"""
from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(not oc.has_codec("heif"), reason="libheif not built here")

_heif = pytest.importorskip("opencodecs.codecs._heif")


def _image(h, w, seed=0):
    rs = np.random.RandomState(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = 120 + 80 * np.sin(x / 37.0) * np.cos(y / 23.0)
    rgb = np.stack([base, base[::-1], base[:, ::-1]], axis=-1)
    return np.clip(rgb + rs.normal(0, 4, rgb.shape), 0, 255).astype(np.uint8)


@pytest.mark.parametrize("shape", [(64, 80), (700, 520)])
def test_thread_counts_decode_identically(shape):
    try:
        data = oc.get_codec("heif").encode(_image(*shape), level=70)
    except Exception as exc:  # noqa: BLE001 - encoder plugins vary by build
        pytest.skip(f"no HEVC encoder here: {exc}")
    one = _heif.decode(data, numthreads=1)
    for numthreads in (None, 2, 8):
        np.testing.assert_array_equal(_heif.decode(data, numthreads=numthreads), one)


def test_zero_threads_means_one():
    try:
        data = oc.get_codec("heif").encode(_image(96, 128, seed=1), level=50)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"no HEVC encoder here: {exc}")
    np.testing.assert_array_equal(_heif.decode(data, numthreads=0),
                                  _heif.decode(data, numthreads=1))
