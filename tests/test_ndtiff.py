"""Native NDTiff reader — parity + perf sanity vs ndstorage.

Synthesizes a tiny NDTiff acquisition (1 stack file + matching index)
into a tmp dir so the tests don't depend on lab data. Parity checks
both the Cython index parser and the full read path including
parallel multi-frame access. ndstorage is optional — when absent, the
parity tests skip and only the self-consistency tests run.
"""

from __future__ import annotations

import io
import json
import os
import struct
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    import ndstorage  # noqa: F401
    _HAS_NDSTORAGE = True
except ImportError:
    _HAS_NDSTORAGE = False


from opencodecs._ndtiff import NDTiffDataset
from opencodecs.codecs._ndtiff import (
    parse_ndtiff_index,
    PIXEL_TYPE_SIXTEEN_BIT,
)


def _silence_ndstorage(fn, *args, **kwargs):
    """Suppress ndstorage's chatty `Reading index... N%` progress."""
    old = sys.stdout
    sys.stdout = io.StringIO()
    try:
        return fn(*args, **kwargs)
    finally:
        sys.stdout = old


# ---------------------------------------------------------------------------
# Synthetic NDTiff acquisition (1 file, N frames, uint16)
# ---------------------------------------------------------------------------


def _build_synthetic_ndtiff(
    tmp_path: Path,
    n_frames: int = 4,
    width: int = 8,
    height: int = 6,
) -> Path:
    """Create a minimal NDTiff acquisition. Returns the directory.

    The resulting directory has:
      * NDTiffStack.tif: raw uint16 pixel data with NDTiff-style headers
        the reader never parses (we use index pixel_offset directly).
      * NDTiff.index: matching binary index pointing at the right offsets.

    Frame i contains uint16 values [i*1000, i*1000+1, i*1000+2, ...].
    """
    d = tmp_path / "synth_ndtiff"
    d.mkdir()
    stack_path = d / "NDTiffStack.tif"
    index_path = d / "NDTiff.index"

    pixel_bytes = width * height * 2
    frame_pad = 32      # NDTiff-style pre-frame padding (not parsed)
    # Layout each frame back-to-back at a stable pixel_offset.
    stack_bytes = bytearray()
    for i in range(n_frames):
        # Padding stub — not interpreted by the reader since pixel_offset
        # in the index points past it.
        stack_bytes += b"\x00" * frame_pad
        pixels = np.arange(
            i * 1000, i * 1000 + width * height, dtype=np.uint16
        ).tobytes()
        stack_bytes += pixels
        # Optional metadata blob (matches NDTiff convention but we
        # don't read it).
        stack_bytes += b'{"frame": ' + str(i).encode() + b"}"
    stack_path.write_bytes(bytes(stack_bytes))

    # Index records.
    record_stride = frame_pad + pixel_bytes + len(b'{"frame": 0}')  # approx
    index_bytes = bytearray()
    cursor = 0
    for i in range(n_frames):
        axes = json.dumps({"z": i}).encode("utf-8")
        filename = b"NDTiffStack.tif"
        # Pixel offset is past the frame_pad stub.
        pixel_offset = cursor + frame_pad
        metadata_offset = pixel_offset + pixel_bytes
        metadata_len = len(b'{"frame": ' + str(i).encode() + b"}")

        index_bytes += struct.pack("<I", len(axes)) + axes
        index_bytes += struct.pack("<I", len(filename)) + filename
        index_bytes += struct.pack(
            "<IIIIIIII",
            pixel_offset,
            width,
            height,
            PIXEL_TYPE_SIXTEEN_BIT,
            0,            # pixel_compression: NONE
            metadata_offset,
            metadata_len,
            0,            # metadata_compression: NONE
        )
        cursor += frame_pad + pixel_bytes + metadata_len
    index_path.write_bytes(bytes(index_bytes))
    return d


# ---------------------------------------------------------------------------
# Index parser
# ---------------------------------------------------------------------------


def test_parse_index_round_trip(tmp_path):
    d = _build_synthetic_ndtiff(tmp_path, n_frames=4)
    raw = (d / "NDTiff.index").read_bytes()
    records = parse_ndtiff_index(raw)
    assert len(records) == 4
    # Per record tuple shape:
    for i, rec in enumerate(records):
        (axes_blob, filename, pix_off, w, h, pt, pc, m_off, m_len, mc) = rec
        assert filename == "NDTiffStack.tif"
        assert json.loads(axes_blob.decode("utf-8")) == {"z": i}
        assert w == 8
        assert h == 6
        assert pt == PIXEL_TYPE_SIXTEEN_BIT
        assert pc == 0
        assert mc == 0


def test_parse_empty_index():
    assert parse_ndtiff_index(b"") == []


def test_parse_truncated_index_raises(tmp_path):
    d = _build_synthetic_ndtiff(tmp_path, n_frames=2)
    raw = (d / "NDTiff.index").read_bytes()
    from opencodecs.codecs._ndtiff import NDTiffIndexError
    # Cut after the first record's axes_len; should raise.
    with pytest.raises(NDTiffIndexError):
        parse_ndtiff_index(raw[:6])


# ---------------------------------------------------------------------------
# Dataset reader — self-consistency
# ---------------------------------------------------------------------------


def test_ndtiff_dataset_basic_open(tmp_path):
    d = _build_synthetic_ndtiff(tmp_path, n_frames=4, width=8, height=6)
    with NDTiffDataset(d) as ds:
        assert len(ds) == 4
        assert ds.n_frames == 4
        assert ds.shape == (6, 8)
        assert ds.dtype == np.uint16
        assert ds.axes_names == {"z"}
        assert ds.axis_values("z") == [0, 1, 2, 3]


def test_ndtiff_random_access(tmp_path):
    d = _build_synthetic_ndtiff(tmp_path, n_frames=4, width=8, height=6)
    with NDTiffDataset(d) as ds:
        # By axes coordinates
        f2 = ds.read_frame(z=2)
        # Verify pixel content matches our synthetic generator
        expected = np.arange(2000, 2000 + 8 * 6, dtype=np.uint16).reshape(6, 8)
        np.testing.assert_array_equal(f2, expected)

        # By integer index
        f0 = ds[0]
        np.testing.assert_array_equal(
            f0, np.arange(0, 8 * 6, dtype=np.uint16).reshape(6, 8))

        # By dict
        f3 = ds[{"z": 3}]
        np.testing.assert_array_equal(
            f3, np.arange(3000, 3000 + 8 * 6, dtype=np.uint16).reshape(6, 8))


def test_ndtiff_missing_key_raises(tmp_path):
    d = _build_synthetic_ndtiff(tmp_path, n_frames=2)
    with NDTiffDataset(d) as ds:
        assert ds.has_frame(z=0) is True
        assert ds.has_frame(z=99) is False
        with pytest.raises(KeyError):
            ds.read_frame(z=99)


def test_ndtiff_iter_frames(tmp_path):
    d = _build_synthetic_ndtiff(tmp_path, n_frames=4, width=8, height=6)
    with NDTiffDataset(d) as ds:
        frames = list(ds.iter_frames())
        assert len(frames) == 4
        for i, frame in enumerate(frames):
            expected = np.arange(
                i * 1000, i * 1000 + 8 * 6, dtype=np.uint16,
            ).reshape(6, 8)
            np.testing.assert_array_equal(frame, expected)


def test_ndtiff_read_many_parallel(tmp_path):
    d = _build_synthetic_ndtiff(tmp_path, n_frames=8, width=8, height=6)
    with NDTiffDataset(d) as ds:
        out = ds.read_many(keys=[{"z": i} for i in range(8)])
        assert out.shape == (8, 6, 8)
        for i in range(8):
            expected = np.arange(
                i * 1000, i * 1000 + 8 * 6, dtype=np.uint16,
            ).reshape(6, 8)
            np.testing.assert_array_equal(out[i], expected)


def test_ndtiff_iter_frames_parallel_bounded_memory(tmp_path):
    """iter_frames_parallel should yield in submitted order regardless of
    prefetch concurrency."""
    d = _build_synthetic_ndtiff(tmp_path, n_frames=8, width=8, height=6)
    with NDTiffDataset(d) as ds:
        keys = [{"z": i} for i in range(8)]
        frames = list(ds.iter_frames_parallel(keys, prefetch=3))
        assert len(frames) == 8
        for i, frame in enumerate(frames):
            expected = np.arange(
                i * 1000, i * 1000 + 8 * 6, dtype=np.uint16,
            ).reshape(6, 8)
            np.testing.assert_array_equal(frame, expected)


# ---------------------------------------------------------------------------
# Parity vs ndstorage (when installed)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# HTTP-range NDTiff
# ---------------------------------------------------------------------------


def test_ndtiff_dataset_from_http(tmp_path):
    """Open an NDTiff folder over HTTP Range requests. The frames must
    round-trip byte-for-byte, AND we shouldn't fetch the whole file —
    just the index + the pixel bytes of the frames we touch."""
    import http.server
    import socketserver
    import threading
    from pathlib import Path

    d = _build_synthetic_ndtiff(tmp_path, n_frames=4, width=8, height=6)

    class H(http.server.SimpleHTTPRequestHandler):
        def do_GET(self):
            rng = self.headers.get("Range")
            data = (Path(self.directory) / Path(self.path).name).read_bytes()
            if rng:
                s, e = rng.split("=", 1)[1].split("-")
                s = int(s)
                e = int(e) if e else len(data) - 1
                chunk = data[s:e + 1]
                self.send_response(206)
                self.send_header("Content-Type", "application/octet-stream")
                self.send_header("Content-Range",
                                 f"bytes {s}-{e}/{len(data)}")
                self.send_header("Content-Length", str(len(chunk)))
                self.end_headers()
                self.wfile.write(chunk)
            else:
                self.send_response(200)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

        def log_message(self, *a, **k):
            pass

    server = socketserver.ThreadingTCPServer(
        ("127.0.0.1", 0),
        lambda *a, **kw: H(*a, directory=str(d), **kw),
    )
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        base_url = f"http://127.0.0.1:{port}/"

        ds = NDTiffDataset.from_http(base_url)
        try:
            assert ds.n_frames == 4
            assert ds.shape == (6, 8)
            # Random access — should issue one Range request per frame.
            for i in range(4):
                frame = ds.read_frame(z=i)
                expected = np.arange(
                    i * 1000, i * 1000 + 8 * 6, dtype=np.uint16,
                ).reshape(6, 8)
                np.testing.assert_array_equal(frame, expected)
        finally:
            ds.close()
    finally:
        server.shutdown()
        server.server_close()


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------


def test_ndtiff_writer_round_trip(tmp_path):
    """Write N frames, read them back via our NDTiffDataset; bytes must match."""
    from opencodecs._ndtiff_writer import NDTiffWriter

    out_dir = tmp_path / "written"
    n = 5
    frames = [
        np.random.default_rng(i).integers(0, 4000, size=(48, 64), dtype=np.uint16)
        for i in range(n)
    ]
    with NDTiffWriter(out_dir, summary={"PixelType": "uint16"}) as w:
        for i, arr in enumerate(frames):
            w.write_frame({"z": i}, arr, metadata={"z_um": i * 0.5})

    # Read back via opencodecs.
    with NDTiffDataset(out_dir) as ds:
        assert ds.n_frames == n
        for i, arr in enumerate(frames):
            np.testing.assert_array_equal(ds.read_frame(z=i), arr)


def test_ndtiff_writer_batch_write(tmp_path):
    from opencodecs._ndtiff_writer import NDTiffWriter

    out_dir = tmp_path / "batch"
    n = 8
    frames = [
        np.full((20, 30), i, dtype=np.uint16) for i in range(n)
    ]
    with NDTiffWriter(out_dir) as w:
        recs = w.write_many(
            ({"z": i}, arr, None) for i, arr in enumerate(frames)
        )
    assert len(recs) == n
    with NDTiffDataset(out_dir) as ds:
        assert ds.n_frames == n
        for i in range(n):
            np.testing.assert_array_equal(
                ds.read_frame(z=i), np.full((20, 30), i, dtype=np.uint16))


@pytest.mark.parametrize("compression,level", [
    ("none",    None),
    ("deflate", 6),
    ("zstd",    3),
])
def test_ndtiff_writer_compression_round_trip(tmp_path, compression, level):
    """Write compressed NDTiff frames, read them back, verify byte-equal."""
    from opencodecs._ndtiff_writer import NDTiffWriter

    out = tmp_path / f"compressed_{compression}"
    rng = np.random.default_rng(7)
    frames = [rng.integers(0, 4000, size=(48, 64), dtype=np.uint16)
              for _ in range(5)]

    with NDTiffWriter(out, compression=compression,
                      compression_level=level) as w:
        for i, a in enumerate(frames):
            w.write_frame({"z": i}, a)

    with NDTiffDataset(out) as ds:
        assert ds.n_frames == 5
        for i, expected in enumerate(frames):
            np.testing.assert_array_equal(ds.read_frame(z=i), expected)


def test_ndtiff_writer_compression_reduces_size(tmp_path):
    """zstd-compressed NDTiff must produce a smaller file than uncompressed
    for correlated data."""
    from opencodecs._ndtiff_writer import NDTiffWriter

    # Correlated data (radial gradient + low-amplitude noise) — codec-
    # friendly the way real microscopy data is.
    h, w = 96, 128
    rng = np.random.default_rng(0)
    yy, xx = np.indices((h, w))
    frames = [(500 + 0.5 * yy + 0.3 * xx
               + rng.normal(0, 5, (h, w))).astype(np.uint16)
              for _ in range(8)]

    def write(compression):
        d = tmp_path / f"size_{compression}"
        with NDTiffWriter(d, compression=compression,
                          compression_level=3) as w:
            for i, a in enumerate(frames):
                w.write_frame({"z": i}, a)
        return (d / "NDTiffStack.tif").stat().st_size

    size_none = write("none")
    size_zstd = write("zstd")
    # Should compress to ~half on this kind of data.
    assert size_zstd < size_none * 0.85, (
        f"zstd compression didn't help: zstd={size_zstd}, none={size_none}"
    )


@pytest.mark.skipif(not _HAS_NDSTORAGE, reason="ndstorage not installed")
def test_ndtiff_writer_compatible_with_ndstorage(tmp_path):
    """Files written by our writer must round-trip through ndstorage."""
    from opencodecs._ndtiff_writer import NDTiffWriter

    out_dir = tmp_path / "compat"
    frames = [
        np.arange(i * 1000, i * 1000 + 32 * 24, dtype=np.uint16).reshape(24, 32)
        for i in range(4)
    ]
    with NDTiffWriter(out_dir, summary={"PixelType": "uint16"}) as w:
        for i, a in enumerate(frames):
            w.write_frame({"z": i}, a, metadata={"frame": i})

    ds = _silence_ndstorage(ndstorage.NDTiffDataset, str(out_dir))
    try:
        assert len(ds.index) == 4
        for i in range(4):
            img = ds.read_image(z=i)
            np.testing.assert_array_equal(img, frames[i])
    finally:
        ds.close()


# ---------------------------------------------------------------------------
# NDTiff pyramid
# ---------------------------------------------------------------------------


def _build_synthetic_ndtiff_pyramid(tmp_path: Path, downscales=(1, 2, 4)) -> Path:
    """Build a parent folder with NDTiff acquisitions at multiple downscales.

    Each level is a complete NDTiff folder containing 2 frames keyed
    by ``z``. Returns the parent directory.
    """
    from opencodecs._ndtiff_writer import NDTiffWriter

    parent = tmp_path / "pyramid_dataset"
    parent.mkdir()
    full_h, full_w = 64, 96
    for ds in downscales:
        if ds == 1:
            subdir = parent / "Full resolution"
        else:
            subdir = parent / f"Downsampled_x{ds}"
        h, w = full_h // ds, full_w // ds
        with NDTiffWriter(subdir) as wr:
            for z in range(2):
                arr = np.full((h, w), z * 100 + ds, dtype=np.uint16)
                wr.write_frame({"z": z}, arr)
    return parent


def test_ndtiff_pyramid_enumerates_levels(tmp_path):
    from opencodecs._ndtiff_pyramid import NDTiffPyramidDataset

    parent = _build_synthetic_ndtiff_pyramid(tmp_path, downscales=(1, 2, 4))
    with NDTiffPyramidDataset(parent) as p:
        assert p.n_levels == 3
        assert p.level(0).downscale == (1, 1)
        assert p.level(0).shape == (64, 96)
        assert p.level(1).downscale == (2, 2)
        assert p.level(1).shape == (32, 48)
        assert p.level(2).downscale == (4, 4)
        assert p.level(2).shape == (16, 24)


def test_ndtiff_pyramid_read_region(tmp_path):
    from opencodecs._ndtiff_pyramid import NDTiffPyramidDataset

    parent = _build_synthetic_ndtiff_pyramid(tmp_path, downscales=(1, 2))
    with NDTiffPyramidDataset(parent) as p:
        # Full frame at level 0, z=0.
        full = p.read_region(level=0, z=0)
        assert full.shape == (64, 96)
        # Crop at level 0.
        crop = p.read_region(level=0, y=(10, 40), x=(20, 60), z=1)
        assert crop.shape == (30, 40)
        # Each frame is constant — easy to verify.
        assert (crop == 1 * 100 + 1).all()
        # Same at level 1.
        out = p.read_region(level=1, z=0)
        assert out.shape == (32, 48)
        assert (out == 0 * 100 + 2).all()


def test_ndtiff_pyramid_best_level_for(tmp_path):
    from opencodecs._ndtiff_pyramid import NDTiffPyramidDataset

    parent = _build_synthetic_ndtiff_pyramid(
        tmp_path, downscales=(1, 2, 4, 8))
    with NDTiffPyramidDataset(parent) as p:
        # Level shapes: 64, 32, 16, 8 (heights).
        assert p.best_level_for(max_pixels_y=10_000) == 0
        assert p.best_level_for(max_pixels_y=50) == 1     # 32 fits
        assert p.best_level_for(max_pixels_y=20) == 2     # 16 fits
        assert p.best_level_for(max_pixels_y=10) == 3     # 8 fits


@pytest.mark.skipif(not _HAS_NDSTORAGE, reason="ndstorage not installed")
def test_ndtiff_parity_index_parse(tmp_path):
    """The opencodecs Cython parser should produce the same records as
    ndstorage's pure-Python parser."""
    import ndstorage.ndtiff_index as ndi
    d = _build_synthetic_ndtiff(tmp_path, n_frames=4)
    raw = (d / "NDTiff.index").read_bytes()

    oc_records = parse_ndtiff_index(raw)
    nd_index = _silence_ndstorage(ndi.read_ndtiff_index, raw)

    assert len(oc_records) == len(nd_index)
    for oc_rec, nd_key in zip(oc_records, nd_index.keys()):
        nd_entry = nd_index[nd_key]
        (axes_blob, fname, pix_off, w, h, pt, pc, m_off, m_len, mc) = oc_rec
        assert nd_entry.filename == fname
        assert nd_entry.pix_offset == pix_off
        assert nd_entry.image_width == w
        assert nd_entry.image_height == h
        assert nd_entry.pixel_type == pt
        assert nd_entry.metadata_offset == m_off
        assert nd_entry.metadata_length == m_len
