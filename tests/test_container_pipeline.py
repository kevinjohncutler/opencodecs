"""Independent container fixtures for bounded reconstruction and placement."""
from __future__ import annotations

import io
import weakref

import numpy as np
import pytest

h5py = pytest.importorskip("h5py")

from opencodecs._h5_common import read_h5_dataset
from opencodecs._hdf5_codec import HdfReader


@pytest.mark.parametrize("dtype", ["<u2", ">u2", "<f4"])
@pytest.mark.parametrize("shuffle,checksum", [(False, False), (True, False), (True, True)])
def test_hdf5_filters_ragged_selection(dtype, shuffle, checksum):
    data = (np.arange(7 * 37 * 53).reshape(7, 37, 53) * 17).astype(dtype)
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as file:
        file.create_dataset("image", data=data, chunks=(2, 16, 16),
                            compression="gzip", shuffle=shuffle, fletcher32=checksum)
    buffer.seek(0)
    with HdfReader(buffer) as reader:
        for selection in (None, np.s_[1:7, 3:33, 7:51], np.s_[-1, 3:33, 7:51],
                          np.s_[..., 10:30], np.s_[2:2]):
            expected = data if selection is None else data[selection]
            actual = read_h5_dataset(reader._ds, selection, numthreads=4)
            np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(reader.read_parallel(-1, n_workers=4), data[-1:])


def test_hdf5_sparse_fill_and_skipped_filters():
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as file:
        ds = file.create_dataset("image", shape=(65, 81), dtype="<u2",
                                 chunks=(16, 16), compression="gzip", shuffle=True,
                                 fillvalue=321)
        raw = (np.arange(256, dtype="<u2").reshape(16, 16) * 31)
        # Both optional filters are skipped in this deliberately raw chunk.
        ds.id.write_direct_chunk((16, 32), raw.tobytes(), filter_mask=3)
        expected = ds[...]
    buffer.seek(0)
    with HdfReader(buffer) as reader:
        np.testing.assert_array_equal(reader.read_parallel(n_workers=4), expected)


def test_hdf5_checksum_corruption_is_not_bypassed():
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as file:
        ds = file.create_dataset("image", data=np.arange(64 * 64, dtype="u2").reshape(64, 64),
                                 chunks=(16, 16), compression="gzip", fletcher32=True)
        mask, payload = ds.id.read_direct_chunk((0, 0))
        bad = bytearray(payload)
        bad[-1] ^= 1
        ds.id.write_direct_chunk((0, 0), bad, filter_mask=mask)
    buffer.seek(0)
    with HdfReader(buffer) as reader:
        with pytest.raises(OSError):
            reader.read_parallel(n_workers=4)


def test_emd_uses_checked_shared_filters(tmp_path):
    from opencodecs._emd import EmdFile
    path = tmp_path / "filters.emd"
    expected = np.arange(65 * 81, dtype="u2").reshape(65, 81)
    with h5py.File(path, "w") as file:
        group = file.create_group("image")
        group.attrs["emd_group_type"] = 1
        group.create_dataset("data", data=expected, chunks=(16, 16),
                             compression="gzip", shuffle=True)
    with EmdFile(path) as reader:
        np.testing.assert_array_equal(reader.asarray(numthreads=4), expected)


def test_imaris_uses_checked_shared_filters(tmp_path):
    from opencodecs._imaris import ImarisReader
    path = tmp_path / "filters.ims"
    expected = np.arange(3 * 65 * 81, dtype="u2").reshape(3, 65, 81)
    with h5py.File(path, "w") as file:
        group = file.create_group("DataSet/ResolutionLevel 0/TimePoint 0/Channel 0")
        group.create_dataset("Data", data=expected, chunks=(1, 16, 16),
                             compression="gzip", shuffle=True)
    with ImarisReader(path) as reader:
        np.testing.assert_array_equal(reader.read(numthreads=4), expected)


def test_nd2_releases_previous_frame_before_next_decode():
    from opencodecs._nd2_native import Nd2NativeReader
    class Parser:
        previous = None
        def read_frame(self, index):
            assert self.previous is None or self.previous() is None
            frame = np.full((17, 23, 2), index, dtype="u2")
            self.previous = weakref.ref(frame)
            return frame
    reader = object.__new__(Nd2NativeReader)
    reader._parser = Parser()
    reader.n_frames = 19
    reader.shape = (19, 17, 23, 2)
    reader.dtype = np.dtype("u2")
    actual = reader.read()
    expected = np.broadcast_to(np.arange(19, dtype="u2")[:, None, None, None], reader.shape)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("shape", [(31, 37, 53), (8, 12, 16), (7, 1, 5)])
@pytest.mark.parametrize("dtype", ["f4", "f8", "i4", "i8"])
def test_zfp_destination_matches_independent_decoder(shape, dtype):
    imagecodecs = pytest.importorskip("imagecodecs")
    from opencodecs._zfp_codec import ZfpCodec
    expected_input = np.arange(np.prod(shape), dtype=dtype).reshape(shape)
    codec = ZfpCodec()
    blob = codec.encode(expected_input, mode="rate", rate=16)
    expected = imagecodecs.zfp_decode(blob)
    for numthreads in (1, 4):
        out = np.empty_like(expected)
        actual = codec.decode(blob, out=out, numthreads=numthreads)
        assert actual is out
        np.testing.assert_array_equal(actual, expected)


def test_zfp_readonly_destination_rejected():
    from opencodecs._zfp_codec import ZfpCodec
    codec = ZfpCodec()
    data = np.ones((16, 16, 16), dtype="f4")
    blob = codec.encode(data, mode="rate", rate=8)
    data.flags.writeable = False
    with pytest.raises(ValueError, match="writable"):
        codec.decode(blob, out=data, numthreads=4)


@pytest.mark.parametrize("byte_budget", [1024, 48 << 10])
def test_hdf5_bounded_reservations(byte_budget):
    from opencodecs.core.pipeline import PipelineStats
    data = np.arange(65 * 81, dtype="u2").reshape(65, 81)
    buffer = io.BytesIO()
    with h5py.File(buffer, "w") as file:
        file.create_dataset("image", data=data, chunks=(16, 16),
                            compression="gzip", shuffle=True)
    buffer.seek(0)
    stats = PipelineStats()
    with HdfReader(buffer) as reader:
        actual = reader.read_parallel(n_workers=4, max_pending_bytes=byte_budget,
                                      pipeline_stats=stats)
    np.testing.assert_array_equal(actual, data)
    assert stats.completed == stats.submitted
    if stats.oversized_items:
        assert stats.peak_pending == 1
    else:
        assert stats.peak_reserved_bytes <= byte_budget
