"""CharLS / JPEG-LS tests.

JPEG-LS is the predictive lossless / near-lossless JPEG variant used
heavily in DICOM medical imaging. Verified by round-trip + by cross-
decode against imagecodecs.jpegls_decode when present.
"""

from __future__ import annotations

import numpy as np
import pytest
from _ic_reference import skip_if_old_imagecodecs  # noqa: E402

pytestmark = skip_if_old_imagecodecs

mod = pytest.importorskip("opencodecs.codecs._charls")
encode = mod.encode
decode = mod.decode


@pytest.mark.parametrize("dtype", [np.uint8, np.uint16])
def test_charls_lossless_grayscale(dtype):
    rng = np.random.default_rng(0)
    if dtype is np.uint8:
        arr = rng.integers(0, 256, size=(64, 96), dtype=dtype)
    else:
        arr = rng.integers(0, 4000, size=(64, 96), dtype=dtype)
    enc = encode(arr)
    back = decode(enc)
    assert back.dtype == arr.dtype
    assert back.shape == arr.shape
    np.testing.assert_array_equal(back, arr)


def test_charls_lossless_rgb_u8():
    rng = np.random.default_rng(1)
    arr = rng.integers(0, 256, size=(48, 64, 3), dtype=np.uint8)
    enc = encode(arr)
    back = decode(enc)
    np.testing.assert_array_equal(back, arr)


def test_charls_lossless_rgba_u8():
    rng = np.random.default_rng(2)
    arr = rng.integers(0, 256, size=(32, 48, 4), dtype=np.uint8)
    enc = encode(arr)
    back = decode(enc)
    np.testing.assert_array_equal(back, arr)


@pytest.mark.parametrize("near_lossless", [1, 3, 5])
def test_charls_near_lossless_error_bound(near_lossless):
    """Near-lossless mode guarantees per-sample error <= near_lossless."""
    rng = np.random.default_rng(3)
    arr = rng.integers(0, 4000, size=(64, 96), dtype=np.uint16)
    enc = encode(arr, near_lossless=near_lossless)
    back = decode(enc)
    diff = np.abs(back.astype(int) - arr.astype(int)).max()
    assert diff <= near_lossless, (
        f"near_lossless={near_lossless} but max diff was {diff}"
    )


def test_charls_near_lossless_shrinks_file():
    """Bounded-error mode produces smaller files than lossless."""
    rng = np.random.default_rng(4)
    arr = rng.integers(0, 4000, size=(96, 128), dtype=np.uint16)
    lossless = len(encode(arr, near_lossless=0))
    nl3 = len(encode(arr, near_lossless=3))
    assert nl3 < lossless, f"nl3={nl3} should be < lossless={lossless}"


def test_charls_imagecodecs_cross_decode():
    """Output decodable by imagecodecs.jpegls_decode (the reference
    JPEG-LS decoder)."""
    imagecodecs = pytest.importorskip("imagecodecs")
    if not hasattr(imagecodecs, "jpegls_decode"):
        pytest.skip("imagecodecs has no jpegls_decode")
    arr = np.arange(48 * 64, dtype=np.uint16).reshape(48, 64) * 17 % 4000
    arr = arr.astype(np.uint16)
    enc = encode(arr)
    via_ic = imagecodecs.jpegls_decode(enc)
    np.testing.assert_array_equal(via_ic.reshape(arr.shape), arr)


def test_charls_rejects_unsupported_dtype():
    arr = np.zeros((16, 16), dtype=np.float32)
    with pytest.raises(Exception):
        encode(arr)


def test_charls_rejects_5_channel():
    arr = np.zeros((16, 16, 5), dtype=np.uint8)
    with pytest.raises(Exception):
        encode(arr)


# ---------------------------------------------------------------------------
# Interleave mode NONE (ILV=0): one scan per component.
#
# ITU-T T.87 | ISO/IEC 14495-1 Annex C.2.3 and Table C.3 allow a
# multi-component frame to be coded as one scan per component (ILV=0),
# which DICOM archives and other encoders write. We used to hand CharLS
# the interleaved row stride for such a frame and fail with "output
# buffer too small". Our encoder always writes sample interleave, so the
# fixture is assembled by hand from the spec: SOI, an SOF55 with Nf=C,
# then C scans with Ns=1, Cs=c+1 and ILV=0, each carrying the entropy
# coded data of a single-component stream, then EOI.
# ---------------------------------------------------------------------------


def _jls_segments(stream):
    """Split a single-component JPEG-LS stream into its parts.

    Returns (sof_payload, [lse segments], sos_payload, scan_data). Any
    APPn segment (imagecodecs writes a SPIFF header) is dropped.
    """
    assert stream[:2] == b"\xff\xd8" and stream[-2:] == b"\xff\xd9"
    pos, sof, lse = 2, None, []
    while True:
        marker = stream[pos + 1]
        length = int.from_bytes(stream[pos + 2:pos + 4], "big")
        payload = stream[pos + 4:pos + 2 + length]
        if marker == 0xF7:
            sof = payload
        elif marker == 0xF8:
            lse.append(stream[pos:pos + 2 + length])
        elif marker == 0xDA:
            return sof, lse, payload, stream[pos + 2 + length:-2]
        pos += 2 + length


def _segment(marker, payload):
    return bytes([0xFF, marker]) + (len(payload) + 2).to_bytes(2, "big") + payload


def _ilv0_stream(single_component_streams):
    """Join C one-component streams into one ILV=0 frame of C components."""
    parts = [_jls_segments(s) for s in single_component_streams]
    sof0, lse0, _, _ = parts[0]
    ncomp = len(parts)
    sof = bytearray(sof0[:5]) + bytes([ncomp])
    for c in range(ncomp):
        sof += bytes([c + 1, 0x11, 0])
    out = b"\xff\xd8" + _segment(0xF7, bytes(sof)) + b"".join(lse0)
    for c, (_, _, sos, scan) in enumerate(parts):
        near = sos[3]
        out += _segment(0xDA, bytes([1, c + 1, 0, near, 0, 0])) + scan
    return out + b"\xff\xd9"


def _scan_ilv(stream):
    _, _, sos, _ = _jls_segments(stream)
    return sos[1 + 2 * sos[0] + 1]


@pytest.mark.parametrize("dtype,top", [(np.uint8, 255), (np.uint16, 4095)])
@pytest.mark.parametrize("ncomp", [3, 4])
@pytest.mark.parametrize("near", [0, 2])
def test_charls_decodes_interleave_none(dtype, top, ncomp, near):
    rng = np.random.default_rng(10 + ncomp)
    arr = rng.integers(0, top + 1, size=(23, 41, ncomp)).astype(dtype)
    arr[5:10] = arr[5, 0]  # a flat band, so run mode is exercised
    planes = [encode(np.ascontiguousarray(arr[..., c]), near_lossless=near)
              for c in range(ncomp)]
    stream = _ilv0_stream(planes)
    assert _scan_ilv(stream) == 0
    back = decode(stream)
    assert back.shape == arr.shape and back.dtype == arr.dtype
    assert back.flags["C_CONTIGUOUS"]
    err = np.abs(back.astype(np.int64) - arr.astype(np.int64)).max()
    assert err <= near
    # out= receives the same (H, W, C) layout.
    out = np.zeros_like(arr)
    assert decode(stream, out=out) is out
    np.testing.assert_array_equal(out, back)


def test_charls_interleave_none_matches_imagecodecs():
    """An ILV=0 frame built from imagecodecs' own streams decodes the
    same here as in imagecodecs, the decoder this was reported against."""
    imagecodecs = pytest.importorskip("imagecodecs")
    if not hasattr(imagecodecs, "jpegls_encode"):
        pytest.skip("imagecodecs has no JPEG-LS")
    rng = np.random.default_rng(20)
    arr = rng.integers(0, 4096, size=(33, 47, 3)).astype(np.uint16)
    stream = _ilv0_stream([imagecodecs.jpegls_encode(
        np.ascontiguousarray(arr[..., c])) for c in range(3)])
    reference = imagecodecs.jpegls_decode(stream)
    np.testing.assert_array_equal(reference, arr)
    np.testing.assert_array_equal(decode(stream), reference)


def test_charls_decodes_imagecodecs_line_interleave():
    """imagecodecs writes RGBA with line interleave (ILV=1)."""
    imagecodecs = pytest.importorskip("imagecodecs")
    if not hasattr(imagecodecs, "jpegls_encode"):
        pytest.skip("imagecodecs has no JPEG-LS")
    rng = np.random.default_rng(21)
    arr = rng.integers(0, 256, size=(31, 45, 4), dtype=np.uint8)
    stream = imagecodecs.jpegls_encode(arr)
    assert _scan_ilv(stream) == 1
    np.testing.assert_array_equal(decode(stream), arr)


# ---------------------------------------------------------------------------
# ``level`` is imagecodecs' name for NEAR; it used to be swallowed.
# ---------------------------------------------------------------------------


def _strip_spiff(stream):
    """Drop APP8 (SPIFF) segments, which imagecodecs adds and we do not."""
    out, pos = bytearray(stream[:2]), 2
    while stream[pos + 1] != 0xDA:
        length = int.from_bytes(stream[pos + 2:pos + 4], "big")
        if stream[pos + 1] != 0xE8:
            out += stream[pos:pos + 2 + length]
        pos += 2 + length
    return bytes(out + stream[pos:])


@pytest.mark.parametrize("shape", [(40, 56), (40, 56, 3)])
@pytest.mark.parametrize("level", [1, 3])
def test_charls_level_is_near_lossless_like_imagecodecs(shape, level):
    imagecodecs = pytest.importorskip("imagecodecs")
    if not hasattr(imagecodecs, "jpegls_encode"):
        pytest.skip("imagecodecs has no JPEG-LS")
    rng = np.random.default_rng(30)
    arr = rng.integers(0, 256, size=shape, dtype=np.uint8)
    ours = encode(arr, level=level)
    assert ours == encode(arr, near_lossless=level)
    assert ours != encode(arr)
    # Without the SPIFF header imagecodecs writes the same codestream.
    assert ours == _strip_spiff(imagecodecs.jpegls_encode(arr, level=level))
    import opencodecs as oc
    assert oc.get_codec("jpegls").encode(arr, level=level) == ours


def test_charls_level_and_near_lossless_must_agree():
    arr = np.zeros((16, 16), dtype=np.uint8)
    assert encode(arr, level=2, near_lossless=2) == encode(arr, level=2)
    with pytest.raises(ValueError):
        encode(arr, level=2, near_lossless=0)
    with pytest.raises(ValueError):
        encode(arr, near_lossless=-1)


def test_jpegls_codec_rejects_unknown_encode_options():
    import opencodecs as oc
    with pytest.raises(TypeError):
        oc.get_codec("jpegls").encode(np.zeros((8, 8), np.uint8), bogus=1)


def test_jpegls_codec_rejects_unknown_decode_options():
    """imagecodecs' jpegls_decode takes only out; decode used to drop
    anything else."""
    import opencodecs as oc
    codec = oc.get_codec("jpegls")
    arr = np.arange(64, dtype=np.uint8).reshape(8, 8)
    blob = codec.encode(arr)
    with pytest.raises(TypeError):
        codec.decode(blob, bogus=1)
    with pytest.raises(TypeError):
        codec.decode(blob, index=0)
    out = np.empty_like(arr)
    np.testing.assert_array_equal(codec.decode(blob, out=out), arr)
