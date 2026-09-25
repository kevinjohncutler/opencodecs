"""Optional payload verification before publication by container writers."""
import threading

import numpy as np
import pytest

from opencodecs.core.verification import LosslessVerificationError


@pytest.mark.parametrize("compression", ["none", "zstd", "deflate", "jxl", "jpeg2000", "lerc"])
def test_ndtiff_verified_reused_frames(tmp_path, compression):
    from opencodecs._ndtiff_writer import NDTiffWriter
    from opencodecs._ndtiff import NDTiffDataset
    pixels = np.empty((64, 64), dtype=">u2")
    def frames():
        for i in range(12):
            pixels.fill(i * 23)
            yield {"z": i}, pixels, {"i": i}
    with NDTiffWriter(tmp_path, compression=compression, verify=True) as writer:
        writer.write_many(frames(), n_workers=3)
    with NDTiffDataset(tmp_path) as reader:
        for i in range(12):
            np.testing.assert_array_equal(reader.read_frame(z=i), np.full_like(pixels, i * 23))


def test_ndtiff_lossy_verification_fails_before_index_publication(tmp_path):
    from opencodecs._ndtiff_writer import NDTiffWriter
    image = np.random.default_rng(31).integers(0, 256, (64, 64, 3), dtype="u1")
    with NDTiffWriter(tmp_path, compression="jpeg", verify=True) as writer:
        with pytest.raises(LosslessVerificationError):
            writer.write_many((({"z": i}, image, None) for i in range(8)), n_workers=3)
        assert writer.frame_count == 0
    assert (tmp_path / "NDTiff.index").read_bytes() == b""


@pytest.mark.parametrize("version,sharded", [(2, False), (3, False), (3, True)])
@pytest.mark.parametrize("compressor", ["none", "zstd", "gzip"])
def test_zarr_verified_float_bits_and_endianness(tmp_path, version, sharded, compressor):
    zarr = pytest.importorskip("zarr")
    from opencodecs._omezarr_writer import write_zarr_array
    bits = (np.arange(11 * 13, dtype="u4") + 0x3F000000).reshape(11, 13)
    bits.flat[:5] = [0, 0x80000000, 0x7FC00001, 0x7FC00002, 0x7F800000]
    values = bits.byteswap().view(">f4")
    write_zarr_array(tmp_path, values, chunks=(4, 5), shards=(8, 10) if sharded else None,
                     zarr_format=version, compressor=compressor, workers=3, verify=True)
    decoded = zarr.open_array(str(tmp_path), mode="r")[:]
    normalized = decoded.astype("<f4", copy=False).view("<u4")
    np.testing.assert_array_equal(normalized, bits)


@pytest.mark.parametrize("compressor", ["blosc", "blosc2"])
def test_zarr_verifies_compressor_internal_transforms(tmp_path, compressor):
    from opencodecs._omezarr_writer import write_zarr_array
    from opencodecs._omezarr import OmeZarrArray
    image = np.arange(32 * 48, dtype="u2").reshape(32, 48)
    write_zarr_array(tmp_path, image, chunks=(16, 16), compressor=compressor,
                     workers=2, verify=True)
    reader = OmeZarrArray(tmp_path)
    np.testing.assert_array_equal(reader.read(), image)


@pytest.mark.parametrize("sharded", [False, True])
def test_zarr_bad_payload_is_never_published(tmp_path, monkeypatch, sharded):
    import opencodecs._omezarr_writer as module
    encode = module._encode_chunk
    def corrupt(raw, codec, level):
        output = encode(raw, codec, level)
        if np.frombuffer(raw, dtype="u2")[0] == 16:
            output = b"invalid compressed payload"
        return output
    monkeypatch.setattr(module, "_encode_chunk", corrupt)
    values = np.arange(32 * 32, dtype="u2").reshape(32, 32)
    with pytest.raises(LosslessVerificationError):
        module.write_zarr_array(tmp_path, values, chunks=(16, 16),
                                 shards=(32, 32) if sharded else None,
                                 zarr_format=3, compressor="zstd", workers=2, verify=True)
    if sharded:
        assert not any(path.is_file() for path in (tmp_path / "c").rglob("*"))
    else:
        assert (tmp_path / "c/0/0").is_file()
        assert not (tmp_path / "c/0/1").exists()


def test_zarr_verification_overlaps_other_chunk_encoding(tmp_path, monkeypatch):
    import opencodecs._omezarr_writer as module
    encode, verify = module._encode_chunk, module._verify_chunk
    verifying, other_encoded = threading.Event(), threading.Event()
    def tracked_encode(raw, codec, level):
        first = np.frombuffer(raw, dtype="u2")[0] == 0
        if not first:
            assert verifying.wait(5), "first verifier did not start"
        result = encode(raw, codec, level)
        if not first:
            other_encoded.set()
        return result
    def tracked_verify(raw, encoded, codec):
        if np.frombuffer(raw, dtype="u2")[0] == 0:
            verifying.set()
            assert other_encoded.wait(5), "other compression could not overlap verification"
        verify(raw, encoded, codec)
    monkeypatch.setattr(module, "_encode_chunk", tracked_encode)
    monkeypatch.setattr(module, "_verify_chunk", tracked_verify)
    module.write_zarr_array(tmp_path, np.arange(16 * 32, dtype="u2").reshape(16, 32),
                             chunks=(16, 16), workers=2, verify=True)
    assert verifying.is_set() and other_encoded.is_set()


def test_verification_reserves_additional_decoded_working_memory(tmp_path, monkeypatch):
    import opencodecs.core.pipeline as pipeline
    from opencodecs._ndtiff_writer import NDTiffWriter
    from opencodecs._omezarr_writer import write_zarr_array
    real_map = pipeline.map_bounded
    seen = []
    def observed(fn, items, *args, **kwargs):
        original = kwargs["size"]
        def size(item):
            value = original(item)
            seen.append(value)
            return value
        kwargs["size"] = size
        return real_map(fn, items, *args, **kwargs)
    monkeypatch.setattr(pipeline, "map_bounded", observed)
    pixels = np.arange(256, dtype="u2").reshape(16, 16)
    for writer_kind in ("ndtiff", "zarr", "shard"):
        reservations = []
        for verify in (False, True):
            seen.clear()
            path = tmp_path / f"{writer_kind}-{verify}"
            if writer_kind == "ndtiff":
                with NDTiffWriter(path, compression="zstd", verify=verify) as writer:
                    writer.write_many((({"z": i}, pixels, None) for i in range(8)), n_workers=2)
            else:
                write_zarr_array(path, pixels, chunks=(8, 8), zarr_format=3,
                                 shards=(16, 16) if writer_kind == "shard" else None,
                                 workers=2, verify=verify)
            reservations.append(min(seen))
        assert reservations[1] > reservations[0]
