"""Independent parity and ownership tests for bounded writer pipelines."""

import io
import json
import weakref

import numpy as np
import pytest

from opencodecs.core._write_helpers import iter_array_buffers, write_all


class ShortSink(io.BytesIO):
    def write(self, data):
        return super().write(data[:17])


@pytest.mark.parametrize("name", ["mrc", "nifti", "numpy"])
@pytest.mark.parametrize("layout", ["contiguous", "strided", "bigendian"])
def test_codec_destinations_match_memory_and_independent_reader(tmp_path, name, layout):
    from opencodecs.core.codec import get_codec
    arr = np.arange(120, dtype="i2").reshape(4, 5, 6)
    if layout == "strided":
        arr = arr[:, ::-1, ::2]
    elif layout == "bigendian":
        arr = arr.astype(">i2")
    codec = get_codec(name)
    expected_bytes = codec.encode(arr)
    sink = ShortSink()
    assert codec.encode(arr, dest=sink) is None
    assert not sink.closed
    assert sink.getvalue() == expected_bytes
    path = tmp_path / {"mrc": "out.mrc", "nifti": "out.nii", "numpy": "out.npy"}[name]
    codec.encode(arr, dest=path)
    assert path.read_bytes() == expected_bytes
    if name == "numpy":
        result = np.load(path, allow_pickle=False)
    elif name == "mrc":
        mrcfile = pytest.importorskip("mrcfile")
        with mrcfile.open(path) as reader:
            result = reader.data.copy()
    else:
        nibabel = pytest.importorskip("nibabel")
        result = np.asanyarray(nibabel.load(path).dataobj)
    np.testing.assert_array_equal(result, arr)


@pytest.mark.parametrize("order", ["C", "F"])
def test_array_buffers_are_bounded_with_conversion(order):
    arr = np.arange(11 * 13 * 17, dtype=">i2").reshape(11, 13, 17)[:, ::-1, ::2]
    chunks = [bytes(block) for block in iter_array_buffers(arr, dtype="<i2", order=order,
                                                          buffer_bytes=128)]
    assert max(map(len, chunks)) <= 128
    assert b"".join(chunks) == arr.astype("<i2").tobytes(order=order)


def test_write_all_rejects_no_progress():
    class Sink:
        def write(self, data):
            return 0
    with pytest.raises(OSError, match="byte count"):
        write_all(Sink(), b"pixels")


def test_nifti_gzip_short_destination():
    from opencodecs._nifti_writer import write_nifti
    import gzip
    arr = np.arange(120, dtype="i2").reshape(4, 5, 6)
    plain = io.BytesIO()
    write_nifti(plain, arr, compress=False)
    sink = ShortSink()
    write_nifti(sink, arr, compress=True)
    assert gzip.decompress(sink.getvalue()) == plain.getvalue()
    assert not sink.closed


@pytest.mark.parametrize("compression", ["none", "deflate", "zstd"])
@pytest.mark.parametrize("workers", [1, 3])
@pytest.mark.parametrize("dtype", ["u2", ">u2"])
def test_ndtiff_generator_reuses_pixels_and_metadata(tmp_path, compression, workers, dtype):
    from opencodecs._ndtiff import NDTiffDataset
    from opencodecs._ndtiff_writer import NDTiffWriter
    pixels = np.empty((32, 48), dtype=dtype)
    axes, metadata = {}, {}
    produced = 0
    emitted = []

    class ObservedWriter(NDTiffWriter):
        def _emit_frame(self, *args):
            emitted.append(produced)
            return super()._emit_frame(*args)

    def frames():
        nonlocal produced
        for i in range(96):
            pixels.fill(i)
            axes["z"] = i
            metadata["exposure"] = i
            produced += 1
            yield axes, pixels, metadata

    with ObservedWriter(tmp_path, compression=compression) as writer:
        records = writer.write_many(frames(), n_workers=workers)
    assert emitted[0] < 96
    assert [record["axes"]["z"] for record in records] == list(range(96))
    with NDTiffDataset(tmp_path) as reader:
        for i in range(96):
            np.testing.assert_array_equal(reader.read_frame(z=i), np.full_like(pixels, i))


@pytest.mark.parametrize("compression", ["jxl", "jpeg2000", "lerc"])
@pytest.mark.parametrize("workers", [1, 2])
def test_ndtiff_image_compressor_preserves_geometry(tmp_path, compression, workers):
    from opencodecs._ndtiff import NDTiffDataset
    from opencodecs._ndtiff_writer import NDTiffWriter
    frames = [(np.arange(32 * 48, dtype="u2").reshape(32, 48) + i).astype(">u2")
              for i in range(3)]
    with NDTiffWriter(tmp_path, compression=compression) as writer:
        writer.write_many((({"z": i}, arr, None) for i, arr in enumerate(frames)),
                          n_workers=workers)
    with NDTiffDataset(tmp_path) as reader:
        for i, expected in enumerate(frames):
            np.testing.assert_array_equal(reader.read_frame(z=i), expected)


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize("sharded", [False, True])
def test_zarr_bounded_results_and_independent_parity(tmp_path, monkeypatch, zarr_format, sharded):
    if sharded and zarr_format == 2:
        pytest.skip("sharding requires Zarr v3")
    zarr = pytest.importorskip("zarr")
    import opencodecs._omezarr_writer as module
    arr = np.arange(137 * 149, dtype="u2").reshape(137, 149)
    real_make = module._make_chunk_bytes
    calls = 0
    def make(*args):
        nonlocal calls
        calls += 1
        if not sharded and calls == 20:
            # Some chunks must already have reached storage.
            assert any(p.is_file() and p.name not in ("zarr.json", ".zarray", ".zattrs")
                       for p in tmp_path.rglob("*"))
        return real_make(*args)
    monkeypatch.setattr(module, "_make_chunk_bytes", make)
    module.write_zarr_array(tmp_path, arr, chunks=(16, 16),
                            shards=(64, 64) if sharded else None,
                            zarr_format=zarr_format, compressor="zstd", workers=3,
                            fill_value=13)
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path), mode="r")[:], arr)


def test_shard_failure_removes_partial_file(tmp_path, monkeypatch):
    import opencodecs._omezarr_writer as module
    real = module._encode_chunk
    calls = 0
    def fail(raw, codec, level):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise RuntimeError("injected encoder failure")
        return real(raw, codec, level)
    monkeypatch.setattr(module, "_encode_chunk", fail)
    with pytest.raises(RuntimeError, match="injected"):
        module.write_zarr_array(tmp_path, np.ones((64, 64), dtype="u2"),
                                chunks=(16, 16), shards=(64, 64),
                                zarr_format=3, workers=1)
    assert not [p for p in (tmp_path / "c").rglob("*") if p.is_file()]


def test_pyramid_consumes_and_releases_one_level_at_a_time(tmp_path):
    from opencodecs._omezarr_writer import write_omezarr_pyramid
    references = []
    def levels():
        for i, size in enumerate((128, 64, 32)):
            if i:
                assert (tmp_path / str(i - 1) / ".zarray").exists()
                assert references[-1]() is None
            level = np.full((size, size), i, dtype="u2")
            references.append(weakref.ref(level))
            yield level
            del level
    write_omezarr_pyramid(tmp_path, levels(), chunks=(16, 16), workers=2)
    metadata = json.loads((tmp_path / ".zattrs").read_text())
    assert [d["path"] for d in metadata["multiscales"][0]["datasets"]] == ["0", "1", "2"]


def test_mrc_statistics_ignore_nonfinite_values_in_bounded_blocks(tmp_path):
    from opencodecs._mrc_writer import write_mrc
    import mrcfile
    data = np.arange(400_000, dtype="f4").reshape(10, 200, 200)
    data.flat[::53] = np.nan
    data.flat[::127] = np.inf
    finite = data[np.isfinite(data)].astype("f8")
    path = tmp_path / "finite.mrc"
    write_mrc(path, data)
    with mrcfile.open(path) as reader:
        assert float(reader.header.dmean) == pytest.approx(finite.mean(), rel=1e-6)
        assert float(reader.header.rms) == pytest.approx(finite.std(), rel=1e-6)
        np.testing.assert_array_equal(reader.data, data)


def test_zarr_big_endian_v3_independent_reader(tmp_path):
    import zarr
    from opencodecs._omezarr_writer import write_zarr_array
    arr = np.arange(35, dtype=">u2").reshape(5, 7)
    write_zarr_array(tmp_path, arr, chunks=(3, 4), compressor="zstd", zarr_format=3)
    np.testing.assert_array_equal(zarr.open_array(str(tmp_path), mode="r")[:], arr)


@pytest.mark.parametrize("dtype", [np.dtype([("value_\u03b1", "f4")]),
                                    np.dtype([(f"column_{i}", "u1") for i in range(5000)])])
def test_numpy_extended_headers_use_public_numpy_formats(dtype):
    from opencodecs._numpy_codec import NumpyCodec
    arr = np.zeros((2,), dtype=dtype)
    blob = NumpyCodec().encode(arr)
    np.testing.assert_array_equal(np.load(io.BytesIO(blob), max_header_size=200_000), arr)


@pytest.mark.parametrize("copy_frames", [True, False])
def test_ndtiff_variable_geometry_and_borrowed_buffers(tmp_path, copy_frames):
    from opencodecs._ndtiff_writer import NDTiffWriter
    from opencodecs._ndtiff import NDTiffDataset
    shapes = [(16, 24), (128, 256), (16, 24), (20, 20, 3)] * 4
    arrays = [np.full(shape, i, dtype="u1") for i, shape in enumerate(shapes)]
    with NDTiffWriter(tmp_path, compression="zstd") as writer:
        records = writer.write_many((({"z": i}, arr, {"i": i}) for i, arr in enumerate(arrays)),
                                    n_workers=3, copy_frames=copy_frames)
        assert len(records) == len(arrays)
    with NDTiffDataset(tmp_path) as dataset:
        for i, arr in enumerate(arrays):
            np.testing.assert_array_equal(dataset.read_frame(z=i), arr)


def test_ndtiff_encoder_failure_stops_lazy_producer(tmp_path, monkeypatch):
    from opencodecs._ndtiff_writer import NDTiffWriter
    consumed = []
    def frames():
        frame = np.zeros((32, 32), dtype="u1")
        for i in range(100):
            consumed.append(i)
            yield {"z": i}, frame, None
    def fail(*args, **kwargs):
        raise RuntimeError("injected encoder failure")
    with NDTiffWriter(tmp_path, compression="zstd") as writer:
        monkeypatch.setattr(writer, "_encode_pixels", fail)
        with pytest.raises(RuntimeError, match="injected encoder failure"):
            writer.write_many(frames(), n_workers=3)
    assert len(consumed) < 100


def test_mrc_complex_nan_statistics_preserve_baseline_header_semantics(tmp_path):
    import mrcfile
    from opencodecs._mrc_writer import write_mrc
    data = np.arange(24, dtype="f4").reshape(4, 6).astype("c8")
    data[1, 2] = complex(float("nan"), 1)
    path = tmp_path / "complex-nan.mrc"
    with np.errstate(invalid="ignore"):
        write_mrc(path, data)
    with mrcfile.open(path) as image:
        np.testing.assert_array_equal(image.data, data)
        for field in ("dmin", "dmax", "dmean", "rms"):
            assert np.isnan(image.header[field])
