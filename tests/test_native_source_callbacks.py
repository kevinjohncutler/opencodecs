"""Native callbacks preserve source ownership and avoid eager encoded copies."""

import io
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import opencodecs as oc
from opencodecs._jpeg2k_codec import Jpeg2kCodec
from opencodecs._heif_codec import HeifCodec

sys.path.insert(0, str(Path(__file__).parent))
from _range_http_server import range_http_server


class CountingSource:
    def __init__(self, data):
        self.data = data
        self.size = len(data)
        self.requests = []
        self.closed = False

    def read_at(self, offset, size):
        assert size <= 65536
        self.requests.append((offset, size))
        return self.data[offset:offset+size]

    def close(self):
        self.closed = True


@pytest.mark.parametrize("name", ["heif", "avif"])
def test_public_metadata_open_uses_physical_http_ranges(name, tmp_path):
    pytest.importorskip(f"opencodecs.codecs._{name}")
    image = np.random.default_rng(49).integers(0, 256, (512, 512, 3), dtype="u1")
    path = tmp_path / f"image.{name}"
    oc.write(path, image, format=name, numthreads=1)
    expected = oc.read(path, format=name, numthreads=1)
    with range_http_server(tmp_path) as (base, tracker):
        with oc.open(f"{base}/{path.name}", format=name, numthreads=1) as reader:
            assert reader.shape == image.shape and reader.n_frames == 1
            metadata_bytes = tracker.bytes_served
            metadata_requests = tracker.requests
            assert tracker.full_requests == 0
            assert 0 < metadata_bytes < path.stat().st_size // 2
            assert 0 < metadata_requests < 16
            np.testing.assert_array_equal(reader.frame(0), expected)
        print(f"{name}: metadata {metadata_bytes}/{path.stat().st_size} bytes, "
              f"{metadata_requests} physical range requests; "
              f"frame total {tracker.bytes_served} bytes in {tracker.requests} requests")


def test_jpeg2k_tiled_source_reads_only_requested_tiles(tmp_path):
    reference = pytest.importorskip("imagecodecs")
    pytest.importorskip("opencodecs.codecs._jpeg2k")
    data = np.random.default_rng(42).integers(0, 65536, (1024, 1024), dtype="u2")
    command = shutil.which("opj_compress", path=os.environ.get("PATH", "") + ":/opt/homebrew/bin")
    if command is None:
        pytest.skip("independent OpenJPEG command-line encoder unavailable")
    image = tmp_path / "input.pgm"
    image.write_bytes(b"P5\n1024 1024\n65535\n" + data.astype(">u2").tobytes())
    output = tmp_path / "tiled.jp2"
    subprocess.run([command, "-i", str(image), "-o", str(output), "-t", "128,128"],
                   check=True, capture_output=True)
    encoded = output.read_bytes()
    np.testing.assert_array_equal(reference.jpeg2k_decode(encoded), data)
    source = CountingSource(encoded)
    actual = Jpeg2kCodec().decode_region(source, 13, 101, 15, 99, numthreads=1)
    np.testing.assert_array_equal(actual, data[13:101, 15:99])
    assert sum(size for _, size in source.requests) < len(encoded) // 3
    assert not source.closed
    source.requests.clear()
    np.testing.assert_array_equal(Jpeg2kCodec().decode_tile(source, 9, numthreads=1), data[128:256, 128:256])
    assert sum(size for _, size in source.requests) < len(encoded) // 3


@pytest.mark.parametrize("name", ["jpeg2k", "heif"])
def test_native_source_short_reads_and_borrowed_position(name):
    pytest.importorskip(f"opencodecs.codecs._{name}")
    data = np.random.default_rng(37).integers(0, 256, (71, 91, 3), dtype="u1")
    encoded = oc.write(None, data, format=name)
    class ShortFile(io.BytesIO):
        def read(self, size=-1):
            assert size >= 0
            return super().read(min(size, 17))
    source = ShortFile(encoded)
    source.seek(3)
    if name == "jpeg2k":
        actual = Jpeg2kCodec().decode_region(source, 2, 43, 5, 67, numthreads=1)
        expected = oc.read(encoded, format=name)[2:43, 5:67]
    else:
        actual = oc.read(source, format=name, range_reads=True)
        expected = oc.read(encoded, format=name)
    np.testing.assert_array_equal(actual, expected)
    assert source.tell() == 3 and not source.closed


@pytest.mark.parametrize("name", ["jpeg2k", "heif"])
def test_native_source_propagates_original_error(name):
    pytest.importorskip(f"opencodecs.codecs._{name}")
    image = np.zeros((67, 71, 3), dtype="u1")
    encoded = oc.write(None, image, format=name)
    class Broken(CountingSource):
        def read_at(self, offset, size):
            if self.requests:
                raise OSError("range failed")
            return super().read_at(offset, size)
    source = Broken(encoded)
    with pytest.raises(OSError, match="range failed"):
        if name == "jpeg2k":
            Jpeg2kCodec().decode_region(source, 1, 11, 2, 17)
        else:
            oc.read(source, format=name)
    assert not source.closed


def test_heif_metadata_does_not_decode_first_image():
    pytest.importorskip("opencodecs.codecs._heif")
    data = np.random.default_rng(32).integers(0, 256, (512, 512, 3), dtype="u1")
    encoded = oc.write(None, data, format="heif")
    source = CountingSource(encoded)
    with HeifCodec().open(source) as reader:
        assert reader.shape == data.shape
        assert sum(size for _, size in source.requests) < len(encoded) // 2
        expected = oc.read(encoded, format="heif")
        np.testing.assert_array_equal(reader.frame(0), expected)
    assert not source.closed
    with pytest.raises(ValueError, match="closed"):
        reader.frame(0)


def test_heif_default_full_read_keeps_single_fetch(tmp_path, monkeypatch):
    pytest.importorskip("opencodecs.codecs._heif")
    import opencodecs._heif_codec as adapter
    image = np.random.default_rng(84).integers(0, 256, (256, 256, 3), dtype="u1")
    path = tmp_path / "image.heif"
    oc.write(path, image, format="heif")
    expected = oc.read(path.read_bytes(), format="heif")
    original_read = adapter._read_src
    calls = []
    def read(src):
        calls.append(src)
        return original_read(src)
    monkeypatch.setattr(adapter, "_read_src", read)
    np.testing.assert_array_equal(oc.read(path, format="heif"), expected)
    assert calls == [path]
    with range_http_server(tmp_path) as (base, tracker):
        np.testing.assert_array_equal(oc.read(f"{base}/{path.name}", format="heif"), expected)
        assert tracker.requests == tracker.full_requests == 1
        assert tracker.range_requests == 0
        assert tracker.bytes_served == path.stat().st_size


def test_avif_custom_io_metadata_and_frame():
    pytest.importorskip("opencodecs.codecs._avif")
    data = np.random.default_rng(9).integers(0, 256, (256, 256, 3), dtype="u1")
    encoded = oc.write(None, data, format="avif")
    source = CountingSource(encoded)
    with oc.open(source, format="avif", numthreads=1) as reader:
        assert reader.shape == data.shape and reader.n_frames == 1
        assert sum(size for _, size in source.requests) < len(encoded) // 2
        expected = oc.read(encoded, format="avif", numthreads=1)
        np.testing.assert_array_equal(reader.frame(0), expected)
    assert not source.closed
    with pytest.raises(ValueError, match="closed"):
        reader.frame(0)


def test_avif_custom_io_exception():
    pytest.importorskip("opencodecs.codecs._avif")
    class Broken:
        size = 200
        def read_at(self, offset, size):
            raise OSError("AVIF range stopped")
    with pytest.raises(OSError, match="AVIF range stopped"):
        oc.open(Broken(), format="avif")
