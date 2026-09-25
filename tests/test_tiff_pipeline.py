"""Independent fixtures for shared full-page and region reconstruction."""
import io
import numpy as np
import pytest
import opencodecs as oc

tifffile = pytest.importorskip("tifffile")
pytest.importorskip("imagecodecs")
pytestmark = pytest.mark.skipif(not oc.has_codec("tiff"), reason="TIFF backend unavailable")
from opencodecs._tiff_pyramid import TiffPyramidReader
from opencodecs._tiff_writer import TiffWriter


@pytest.mark.parametrize('layout', ['tiles', 'strips'])
@pytest.mark.parametrize('kind', ['uint-predictor', 'float-predictor', 'planar', 'bilevel'])
@pytest.mark.parametrize('workers', [1, 4])
@pytest.mark.parametrize('prefetch', [False, True])
def test_region_matches_independently_written_pixels(layout, kind, workers, prefetch):
    rng = np.random.default_rng(73)
    shape = (137, 197)
    options = dict(compression='deflate')
    if kind == 'float-predictor':
        arr = rng.random(shape, dtype=np.float32)
        options['predictor'] = 3
    elif kind == 'bilevel':
        arr = rng.integers(0, 2, shape, dtype='u1').astype(bool)
    elif kind == 'planar':
        arr = rng.integers(0, 4000, (3, *shape), dtype='u2')
        options.update(photometric='rgb', planarconfig='separate', predictor=2)
    else:
        arr = rng.integers(0, 4000, shape, dtype='u2')
        options.update(predictor=2, byteorder='>')
    options.update({'tile': (32, 48)} if layout == 'tiles' else {'rowsperstrip': 17})
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, **options)
    expected = np.moveaxis(arr, 0, -1) if kind == 'planar' else arr
    from opencodecs._tiff_codec import TiffStream
    with TiffStream(buf.getvalue()) as stream:
        np.testing.assert_array_equal(stream.page(0).asarray(numthreads=workers), expected)
    with TiffPyramidReader(buf.getvalue(), num_decode_workers=workers,
                           prefetch=prefetch, max_buffer_bytes=128 << 10) as reader:
        np.testing.assert_array_equal(reader.read_region(0), expected)
        np.testing.assert_array_equal(reader.read_region(0, y=(9, 132), x=(11, 190)),
                                      expected[9:132, 11:190])


def test_writer_emits_before_consuming_all_tiles(tmp_path):
    class ObservedWriter(TiffWriter):
        generated = 0
        first_pixel_write = None
        def _iter_tile_segments(self, *args):
            for tile in super()._iter_tile_segments(*args):
                self.generated += 1
                yield tile
        def _writev(self, buffers):
            if self.generated and self.first_pixel_write is None:
                self.first_pixel_write = self.generated
            return super()._writev(buffers)
    arr = np.arange(2048 * 4096, dtype='u2').reshape(2048, 4096)
    path = tmp_path / 'bounded.tif'
    with ObservedWriter(path) as writer:
        writer.write_page(arr, tile=(128, 128), compression='deflate', n_workers=2)
        assert 0 < writer.first_pixel_write < writer.generated
    np.testing.assert_array_equal(tifffile.imread(path), arr)


@pytest.mark.parametrize('bigtiff', [False, True])
@pytest.mark.parametrize('streaming', [False, True])
def test_parallel_batching_preserves_exact_file_bytes(tmp_path, bigtiff, streaming):
    arr = np.random.default_rng(17).integers(0, 4000, (259, 389, 3), dtype='u2')
    outputs = []
    for workers in (1, 4):
        path = tmp_path / f'{workers}.tif'
        with TiffWriter(path, bigtiff=bigtiff, streaming=streaming) as writer:
            opts = dict(tile=(32, 48), compression='deflate', predictor=2, n_workers=workers)
            if streaming:
                writer.write_stream([arr], total_pages=1, **opts)
            else:
                writer.write_page(arr, **opts)
        outputs.append(path.read_bytes())
        np.testing.assert_array_equal(tifffile.imread(path), arr)
    assert outputs[0] == outputs[1]


@pytest.mark.parametrize('layout', ['tiles', 'strips'])
@pytest.mark.parametrize('byteorder', ['<', '>'])
@pytest.mark.parametrize('itemsize', [2, 4, 8])
@pytest.mark.parametrize('channels', ['gray', 'rgb', 'planar'])
def test_float_predictor_preserves_bits(layout, byteorder, itemsize, channels):
    from opencodecs._tiff_codec import TiffStream
    shape = ((37, 53) if channels == 'gray' else
             (3, 37, 53) if channels == 'planar' else (37, 53, 3))
    rng = np.random.default_rng(71)
    uint = np.dtype(f'u{itemsize}')
    # Include arbitrary NaN payloads, infinities, and negative zero. Encoded
    # data must stay bytes until prediction has been undone.
    bits = rng.integers(0, np.iinfo(uint).max, shape, dtype=uint)
    bits.flat[:3] = [0, 1 << (8 * itemsize - 1), np.iinfo(uint).max]
    arr = bits.view(f'f{itemsize}')
    options = dict(compression='deflate', predictor=3, byteorder=byteorder)
    if channels != 'gray':
        options.update(photometric='rgb',
                       planarconfig='separate' if channels == 'planar' else 'contig')
    options.update({'tile': (16, 32)} if layout == 'tiles' else {'rowsperstrip': 7})
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, **options)
    expected = np.moveaxis(arr, 0, -1) if channels == 'planar' else arr
    with TiffStream(buf.getvalue()) as reader:
        assert reader.page(0).asarray(numthreads=3).tobytes() == expected.tobytes()
    with TiffPyramidReader(buf.getvalue(), num_decode_workers=3) as reader:
        assert reader.read_region(0, y=(3, 35), x=(7, 51)).tobytes() == expected[3:35, 7:51].tobytes()


@pytest.mark.parametrize('itemsize', [2, 4, 8])
@pytest.mark.parametrize('channels', [1, 3, 4])
def test_native_float_predictor_matches_independent_encoder(itemsize, channels):
    import imagecodecs
    from opencodecs.codecs._tiff import undo_floating_point
    arr = np.arange(7 * 19 * channels, dtype=f'f{itemsize}').reshape(7, 19, channels)
    encoded = imagecodecs.floatpred_encode(arr, axis=-2)
    view = encoded.view('u1').reshape(7, 19, channels * itemsize)
    undo_floating_point(view, itemsize)
    assert encoded.tobytes() == arr.tobytes()


def test_native_float_predictor_validates_byte_layout():
    from opencodecs.codecs._tiff import undo_floating_point, TiffError
    with pytest.raises(TiffError, match='whole samples'):
        undo_floating_point(np.zeros((2, 3, 3), dtype='u1'), 4)
    with pytest.raises((TiffError, ValueError), match='contiguous'):
        undo_floating_point(np.zeros((2, 6, 4), dtype='u1')[:, ::2], 4)
    for shape in [(0, 3, 4), (2, 0, 4)]:
        undo_floating_point(np.zeros(shape, dtype='u1'), 4)


def test_predicted_crop_fetches_only_intersecting_tiles():
    arr = np.random.default_rng(4).integers(0, 4000, (513, 769), dtype='u2')
    buf = io.BytesIO()
    tifffile.imwrite(buf, arr, tile=(32, 48), compression='deflate', predictor=2)
    data = buf.getvalue()
    calls = []
    def read_at(offset, size):
        calls.append((offset, size))
        return data[offset:offset + size]
    with TiffPyramidReader(None, read_at=read_at) as reader:
        page = reader.level(0).reader
        calls.clear()
        patch = reader.read_region(0, y=(5, 27), x=(49, 93))
        np.testing.assert_array_equal(patch, arr[5:27, 49:93])
        assert calls == [(int(page.offsets[1]), int(page.byte_counts[1]))]
