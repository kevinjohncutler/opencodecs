"""PNG decode: the one-shot inflate path and the carried defilter loops.

A non-interlaced 8- or 16-bit PNG decoded to its own layout is inflated
in one libdeflate call (oc_spng_decode_oneshot in the vendored spng.c)
instead of one zlib call per row, and the sub / average / Paeth
defilters at 1, 2, 6 and 8 bytes per pixel keep the previous pixel in
locals. Neither may change a pixel, and anything the one-shot path does
not expect must fall back to libspng, which then decodes or raises
exactly as before.

Every filter type is written explicitly (filter=), at sizes whose rows
are not a multiple of any SIMD width, so each defilter loop runs on
every pixel size the encoder produces.
"""

from __future__ import annotations

import zlib

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(
    not oc.has_codec("png"), reason="PNG not built here")

FILTERS = {"none": 8, "sub": 16, "up": 32, "avg": 64, "paeth": 128,
           "all": 248}


def _image(h, w, channels, dtype, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = (np.sin(x / 9.0) + np.cos(y / 7.0)) * 0.25 + 0.5
    a = np.clip(base[..., None] + rng.normal(0, 0.1, (h, w, channels)), 0, 1)
    a = np.round(a * np.iinfo(dtype).max).astype(dtype)
    return a[..., 0] if channels == 1 else a


def _png():
    return oc.get_codec("png")


def _libdeflate_build():
    from opencodecs.codecs import _deflate
    return _deflate.backend() == "libdeflate"


@pytest.mark.parametrize("filt", list(FILTERS))
@pytest.mark.parametrize("channels", [1, 2, 3, 4])
@pytest.mark.parametrize("dtype", ["u1", "u2"])
@pytest.mark.parametrize("hw", [(1, 1), (5, 3), (61, 37)],
                         ids=lambda hw: f"{hw[0]}x{hw[1]}")
def test_round_trip_every_filter(filt, channels, dtype, hw):
    a = _image(*hw, channels, np.dtype(dtype), seed=channels)
    blob = _png().encode(a, filter=FILTERS[filt])
    got = _png().decode(blob)
    assert got.dtype == a.dtype and np.array_equal(got, a)


def test_one_shot_path_is_taken():
    """With libdeflate linked, the common layouts take the one-shot path.

    The path declines silently, so without this a change that made it
    decline everything would leave every other test here green.
    """
    if not _libdeflate_build():
        pytest.skip("libspng is built without libdeflate here")
    from opencodecs.codecs import _png as native
    for channels in (1, 2, 3, 4):
        for dtype in (np.uint8, np.uint16):
            blob = _png().encode(_image(40, 23, channels, dtype),
                                 filter=FILTERS["all"])
            assert native._oneshot_decodes(blob), (channels, dtype)


def _chunks(blob):
    pos, out = 8, []
    while pos < len(blob):
        n = int.from_bytes(blob[pos:pos + 4], "big")
        out.append((pos, n, blob[pos + 4:pos + 8]))
        pos += 12 + n
    return out


def _with_idat(blob, payload):
    """``blob`` with its IDAT run replaced by one chunk holding ``payload``."""
    chunks = _chunks(blob)
    idats = [c for c in chunks if c[2] == b"IDAT"]
    start = idats[0][0]
    end = idats[-1][0] + 12 + idats[-1][1]
    chunk = (len(payload).to_bytes(4, "big") + b"IDAT" + payload
             + zlib.crc32(b"IDAT" + payload).to_bytes(4, "big"))
    return blob[:start] + chunk + blob[end:]


@pytest.fixture(scope="module")
def sample():
    a = _image(120, 257, 1, np.uint16, seed=3)
    blob = _png().encode(a, filter=FILTERS["paeth"])
    raw = zlib.decompress(b"".join(
        blob[p + 8:p + 8 + n] for p, n, t in _chunks(blob) if t == b"IDAT"))
    return a, blob, raw


def test_decodable_oddities_still_decode(sample):
    """What libspng accepts, the one-shot path accepts or hands back."""
    a, blob, raw = sample
    stride = 257 * 2 + 1
    cases = {
        "one IDAT": _with_idat(blob, zlib.compress(raw)),
        "data after the image": _with_idat(blob, zlib.compress(raw + raw[:stride * 3])),
        "bytes after the zlib stream": _with_idat(blob, zlib.compress(raw) + b"junk"),
    }
    last = [c for c in _chunks(blob) if c[2] == b"IDAT"][-1]
    crc_at = last[0] + 8 + last[1]
    cases["bad CRC on the last IDAT"] = (
        blob[:crc_at] + bytes([blob[crc_at] ^ 0xFF]) + blob[crc_at + 1:])
    for name, data in cases.items():
        assert np.array_equal(_png().decode(data), a), name


def test_malformed_input_still_raises(sample):
    a, blob, raw = sample
    stride = 257 * 2 + 1
    bad_filter = bytearray(raw)
    bad_filter[stride * 60] = 9
    first = [c for c in _chunks(blob) if c[2] == b"IDAT"][0]
    cases = {
        "short image data": _with_idat(blob, zlib.compress(raw[:-stride])),
        "filter type 9": _with_idat(blob, zlib.compress(bytes(bad_filter))),
        "corrupt zlib stream": _with_idat(
            blob, zlib.compress(raw)[:200] + bytes(64) + zlib.compress(raw)[264:]),
        "bad CRC on the first IDAT": (
            blob[:first[0] + 8] + bytes([blob[first[0] + 8] ^ 1])
            + blob[first[0] + 9:]),
        "truncated file": blob[:len(blob) // 2],
    }
    for name, data in cases.items():
        with pytest.raises(Exception, match="spng_decode_image"):
            _png().decode(data)
        from opencodecs.codecs import _png as native
        if _libdeflate_build():
            assert not native._oneshot_decodes(data), name


def test_out_buffer_gets_the_same_pixels(sample):
    a, blob, _ = sample
    out = np.empty_like(a)
    assert _png().decode(blob, out=out) is out
    assert np.array_equal(out, a)


def test_matches_imagecodecs_reference():
    """An independent decoder (imagecodecs, libpng) reads the same pixels."""
    imagecodecs = pytest.importorskip("imagecodecs")
    for channels in (1, 2, 3, 4):
        for dtype in (np.uint8, np.uint16):
            a = _image(33, 71, channels, dtype, seed=7)
            for filt in ("sub", "avg", "paeth"):
                blob = _png().encode(a, filter=FILTERS[filt])
                assert np.array_equal(imagecodecs.png_decode(blob),
                                      _png().decode(blob))
