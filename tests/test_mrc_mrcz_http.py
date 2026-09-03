"""MRCZ decoding, and MRC read over HTTP range requests.

Two things the plain reader could not do. MRCZ keeps the MRC header and
blosc-compresses the voxels, so the ordinary size arithmetic would read
compressed bytes as samples. And MRC's layout is a fixed header plus
contiguous planes, which is exactly the shape that makes range requests
worth having: the interesting assertion is not that a URL opens but that
almost none of the file crosses the wire.
"""

from __future__ import annotations

import numpy as np
import pytest

from opencodecs._mrc import MrcError, MrcStream
from opencodecs._mrc_writer import encode_mrc, mrc_header

from _range_http_server import range_http_server


def build_mrcz(arr, *, codec=None):
    """An MRCZ file: an MRC header with EXTTYP set, then blosc frames."""
    blosc2 = pytest.importorskip("blosc2")
    h = bytearray(mrc_header(arr.shape, arr.dtype))
    h[104:108] = b"MRCZ"
    kwargs = {"codec": codec} if codec is not None else {}
    return bytes(h) + blosc2.compress2(arr.tobytes(), **kwargs)


# --------------------------------------------------------------------
# MRCZ
# --------------------------------------------------------------------

def test_mrcz_is_detected_and_decoded():
    a = (np.arange(2 * 8 * 8, dtype="f4") % 17).reshape(2, 8, 8)
    with MrcStream(build_mrcz(a)) as r:
        assert r.is_mrcz
        assert np.array_equal(r.asarray(), a)


@pytest.mark.parametrize("dtype", ["i2", "u2", "f4"])
def test_mrcz_dtypes(dtype):
    a = (np.arange(3 * 4 * 5) % 29).astype(dtype).reshape(3, 4, 5)
    with MrcStream(build_mrcz(a)) as r:
        assert np.array_equal(np.asarray(r.asarray(), dtype=dtype), a)


def test_mrcz_plane_access():
    a = (np.arange(3 * 4 * 5, dtype="f4")).reshape(3, 4, 5)
    with MrcStream(build_mrcz(a)) as r:
        for i in range(3):
            assert np.array_equal(r.plane(i), a[i])


def test_mrcz_codec_choice_does_not_matter():
    """blosc records its own codec, so the reader need not be told."""
    blosc2 = pytest.importorskip("blosc2")
    a = (np.arange(64, dtype="f4")).reshape(4, 16)
    for codec in (blosc2.Codec.ZSTD, blosc2.Codec.LZ4, blosc2.Codec.BLOSCLZ):
        with MrcStream(build_mrcz(a, codec=codec)) as r:
            assert np.array_equal(r.asarray(), a), codec


def test_mrcz_truncated_payload_raises():
    """Cut the payload in half, not by a fixed number of bytes.

    Lopping off a few bytes is not reliably a corruption: blosc frames
    carry padding, so a small truncation can leave the decompressible
    content intact and there is then nothing to raise about. Removing
    half the payload always loses data, which is the case worth pinning.
    """
    from opencodecs._mrc import HEADER_SIZE
    a = np.arange(4096, dtype="f4").reshape(16, 256)
    blob = build_mrcz(a)
    payload = len(blob) - HEADER_SIZE
    with pytest.raises(MrcError):
        MrcStream(blob[:HEADER_SIZE + payload // 2]).asarray()


def test_plain_mrc_is_not_treated_as_mrcz():
    a = np.arange(24, dtype="f4").reshape(2, 3, 4)
    with MrcStream(encode_mrc(a)) as r:
        assert not r.is_mrcz
        assert np.array_equal(r.asarray(), a)


# --------------------------------------------------------------------
# HTTP
# --------------------------------------------------------------------

def test_reading_a_header_over_http_transfers_almost_nothing(tmp_path):
    """Opening a remote map must not download it.

    The claim worth testing is byte savings, so this counts what the
    server actually sent rather than checking that a URL works.
    """
    a = np.zeros((64, 128, 128), dtype="f4")           # 4 MB of voxels
    blob = encode_mrc(a)
    (tmp_path / "vol.mrc").write_bytes(blob)

    with range_http_server(tmp_path) as (url, tracker):
        with MrcStream(f"{url}/vol.mrc") as r:
            assert r.shape == (64, 128, 128)
            assert r.dtype == np.dtype("<f4")
        served_for_header = tracker.bytes_served
    # One 64 KiB prefetch block, which is HTTPDataSource's unit: the
    # 1024-byte header cannot be fetched more finely than that, and the
    # point stands anyway at 64 KiB against 4 MB.
    assert served_for_header <= 64 * 1024, (
        f"opening the header pulled {served_for_header} bytes of a "
        f"{len(blob)}-byte file")
    assert served_for_header < len(blob) // 10


def test_reading_one_plane_over_http_skips_the_rest(tmp_path):
    a = np.arange(16 * 64 * 64, dtype="f4").reshape(16, 64, 64)
    blob = encode_mrc(a)
    (tmp_path / "vol.mrc").write_bytes(blob)
    plane_bytes = 64 * 64 * 4

    with range_http_server(tmp_path) as (url, tracker):
        with MrcStream(f"{url}/vol.mrc") as r:
            got = r.plane(9)
        served = tracker.bytes_served
    assert np.array_equal(got, a[9])
    # Header plus one plane, with room for read-ahead, but nowhere near
    # the whole volume.
    assert served < len(blob) // 2, (
        f"reading one {plane_bytes}-byte plane moved {served} bytes of "
        f"{len(blob)}")


def test_http_and_local_reads_agree(tmp_path):
    a = (np.arange(4 * 32 * 32, dtype="f4") % 91).reshape(4, 32, 32)
    (tmp_path / "vol.mrc").write_bytes(encode_mrc(a))
    with range_http_server(tmp_path) as (url, _):
        with MrcStream(f"{url}/vol.mrc") as remote:
            over_http = remote.asarray()
    with MrcStream(str(tmp_path / "vol.mrc")) as local:
        on_disk = local.asarray()
    assert np.array_equal(over_http, on_disk)
    assert np.array_equal(over_http, a)
