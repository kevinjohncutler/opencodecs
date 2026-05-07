"""CZI tests using a synthetic fixture (no lab-CZI mount required).

The fixture in ``_czi_fixture.py`` builds tiny but valid Zeiss ZISRAW
files in-memory by hand. That makes these tests:

  * **CI-friendly** — no external data, runs on any platform with the
    czi codec registered (i.e. the native zstd extension built).
  * **Round-trip accurate** — we encode known pixel buffers and decode
    them back through the production CziReader, so any divergence
    between writer and reader breaks loudly.
  * **Compression-coverage complete** — the lab CZIs we have are all
    ZSTDHDR; the fixture also exercises uncompressed (type 0) and raw
    ZSTD0 (type 5) decode branches that real corpus tests can't reach.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc

pytestmark = pytest.mark.skipif(
    not oc.has_codec("czi"),
    reason="czi codec requires native zstd + bytetools extensions",
)

from _czi_fixture import czi_bytes, write_czi  # noqa: E402


# ---------------------------------------------------------------------------
# Single-frame round-trips: every supported compression branch
# ---------------------------------------------------------------------------


def test_czi_fixture_uncompressed_uint8():
    arr = np.random.RandomState(1).randint(0, 256, (16, 24), dtype=np.uint8)
    data = czi_bytes(arr, compression=0)
    with oc.get_codec("czi").open(data) as r:
        np.testing.assert_array_equal(np.squeeze(r.read()), arr)


def test_czi_fixture_zstd0_uint8():
    """Compression type 5 (raw ZSTD) — exercises the _decode_one branch
    that's NEVER hit by the lab corpus (which is all ZSTDHDR)."""
    arr = np.random.RandomState(2).randint(0, 256, (32, 32), dtype=np.uint8)
    data = czi_bytes(arr, compression=5)
    with oc.get_codec("czi").open(data) as r:
        np.testing.assert_array_equal(np.squeeze(r.read()), arr)


def test_czi_fixture_zstdhdr_uint8_no_shuffle():
    """ZSTDHDR with hilo=False — exercises the no-byteshuffle branch
    that the lab corpus also doesn't reach (Zen always shuffles)."""
    arr = np.random.RandomState(3).randint(0, 256, (16, 16), dtype=np.uint8)
    data = czi_bytes(arr, compression=6, hilo=False)
    with oc.get_codec("czi").open(data) as r:
        np.testing.assert_array_equal(np.squeeze(r.read()), arr)


def test_czi_fixture_zstdhdr_uint16_hilo():
    """ZSTDHDR with byte-plane shuffle — the standard lab Zen format."""
    arr = np.random.RandomState(4).randint(
        0, 65535, (24, 32), dtype=np.uint16,
    )
    data = czi_bytes(arr, compression=6, hilo=True)
    with oc.get_codec("czi").open(data) as r:
        np.testing.assert_array_equal(np.squeeze(r.read()), arr)


def test_czi_fixture_zstdhdr_float32_hilo():
    """Float32 ZSTDHDR — 4-byte-plane shuffle path (general k=4 case)."""
    arr = np.random.RandomState(5).rand(16, 16).astype(np.float32)
    data = czi_bytes(arr, compression=6, hilo=True)
    with oc.get_codec("czi").open(data) as r:
        np.testing.assert_array_equal(np.squeeze(r.read()), arr)


# ---------------------------------------------------------------------------
# Multi-frame: stacked sub-blocks → CziReader.n_frames + iter_frames
# ---------------------------------------------------------------------------


def test_czi_fixture_multi_frame_iter():
    stack = np.random.RandomState(6).randint(
        0, 65535, (4, 16, 16), dtype=np.uint16,
    )
    data = czi_bytes(stack, compression=6, hilo=True)
    with oc.get_codec("czi").open(data) as r:
        assert r.n_frames == 4
        for i, frame in enumerate(r.iter_frames()):
            np.testing.assert_array_equal(np.squeeze(frame), stack[i])


def test_czi_fixture_multi_frame_random_access():
    stack = np.random.RandomState(7).randint(
        0, 256, (5, 16, 16), dtype=np.uint8,
    )
    data = czi_bytes(stack, compression=5)
    with oc.get_codec("czi").open(data) as r:
        # Random-access + slice
        np.testing.assert_array_equal(np.squeeze(r[2]), stack[2])
        np.testing.assert_array_equal(np.squeeze(r[-1]), stack[-1])
        all_via_slice = np.squeeze(r[:])
        np.testing.assert_array_equal(all_via_slice, stack)


def test_czi_fixture_read_squeeze_false():
    """squeeze=False keeps singleton tile axes."""
    stack = np.random.RandomState(8).randint(
        0, 256, (3, 8, 8), dtype=np.uint8,
    )
    data = czi_bytes(stack, compression=0)
    with oc.get_codec("czi").open(data) as r:
        out = r.read(squeeze=False)
        # n_frames + (Y, X, S) including singleton sample axis
        assert out.shape[0] == 3


# ---------------------------------------------------------------------------
# Disk path: write to file, read back via codec.open(path)
# ---------------------------------------------------------------------------


def test_czi_fixture_write_and_open_path(tmp_path):
    """The most common CI smoke test: synthesize a CZI on disk, then
    open it via the public ``oc.get_codec("czi").open`` surface."""
    arr = np.random.RandomState(9).randint(0, 256, (24, 24), dtype=np.uint8)
    p = tmp_path / "synthetic.czi"
    write_czi(p, arr, compression=6, hilo=True)
    assert p.is_file()
    assert p.read_bytes()[:10] == b"ZISRAWFILE"
    with oc.get_codec("czi").open(p) as r:
        out = np.squeeze(r.read())
        np.testing.assert_array_equal(out, arr)


def test_czi_fixture_signature_accepts_synth(tmp_path):
    """opencodecs.read auto-detects CZI from path extension; signature
    check should also accept synthetic bytes."""
    arr = np.zeros((4, 4), dtype=np.uint8)
    data = czi_bytes(arr, compression=0)
    assert oc.get_codec("czi").signature(data[:32]) is True


# ---------------------------------------------------------------------------
# Metadata path: CziReader.metadata_bytes returns the embedded XML
# ---------------------------------------------------------------------------


def test_czi_fixture_metadata_bytes():
    arr = np.zeros((4, 4), dtype=np.uint8)
    data = czi_bytes(
        arr, compression=0,
        metadata_xml=b"<Metadata><Custom>fixture</Custom></Metadata>",
    )
    with oc.get_codec("czi").open(data) as r:
        meta = r.metadata_bytes
        assert b"fixture" in meta
        # Repeated access returns the same object (cache).
        assert r.metadata_bytes is meta


def test_czi_fixture_metadata_xml_decoded():
    arr = np.zeros((4, 4), dtype=np.uint8)
    data = czi_bytes(
        arr, compression=0, metadata_xml=b"<Metadata>hello</Metadata>",
    )
    with oc.get_codec("czi").open(data) as r:
        assert "hello" in r.metadata_xml
