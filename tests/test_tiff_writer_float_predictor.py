"""TIFF floating-point prediction matches independent wire and file decoders."""
from __future__ import annotations
import io
import numpy as np
import pytest

imagecodecs = pytest.importorskip('imagecodecs')
tifffile = pytest.importorskip('tifffile')
from opencodecs._tiff_writer import TiffWriter, TiffWriterError, _apply_float_predictor


def float_bits(dtype, shape):
    dtype = np.dtype(dtype)
    bits = np.random.default_rng(14).integers(0, 256, (int(np.prod(shape)), dtype.itemsize), dtype='u1')
    # Explicit signed zero, infinity, quiet/signaling NaN payloads supplement
    # random bit patterns. Keep all comparisons in unsigned integer space.
    special = {2: [0, 0x8000, 0x7c00, 0xfc00, 0x7e51, 0x7c21],
               4: [0, 0x80000000, 0x7f800000, 0xff800000, 0x7fc01234, 0x7f801234],
               8: [0, 0x8000000000000000, 0x7ff0000000000000, 0xfff0000000000000,
                   0x7ff8000000001234, 0x7ff0000000001234]}
    values = bits.reshape(-1).view(dtype).reshape(shape)
    unsigned = np.dtype(f'{dtype.byteorder}u{dtype.itemsize}')
    values.view(unsigned).flat[:6] = special[dtype.itemsize]
    return values


def assert_bits_equal(actual, expected):
    assert actual.shape == expected.shape
    unsigned = np.dtype(f'u{expected.dtype.itemsize}')
    actual_bits = actual.view(np.dtype(f'{actual.dtype.byteorder}u{actual.dtype.itemsize}')).astype(unsigned)
    expected_bits = expected.view(np.dtype(f'{expected.dtype.byteorder}u{expected.dtype.itemsize}')).astype(unsigned)
    np.testing.assert_array_equal(actual_bits, expected_bits)


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8'])
@pytest.mark.parametrize('byte_order', ['<', '>'])
@pytest.mark.parametrize('channels', [1, 3, 4])
def test_float_predictor_matches_imagecodecs_bytes(dtype, byte_order, channels):
    values = float_bits(byte_order + dtype, (7, 19, channels))
    original = values.tobytes()
    encoded = _apply_float_predictor(values, byte_order)
    independent = imagecodecs.floatpred_encode(values, axis=1)
    assert encoded.tobytes() == independent.tobytes()
    assert values.tobytes() == original


class ForwardOnly:
    def __init__(self):
        self.buffer = bytearray()
    def write(self, data):
        self.buffer.extend(data)
        return len(data)
    def flush(self):
        pass


@pytest.mark.parametrize('dtype', ['f2', 'f4', 'f8'])
@pytest.mark.parametrize('byte_order', ['<', '>'])
@pytest.mark.parametrize('channels', [1, 3])
@pytest.mark.parametrize('tiled', [False, True])
@pytest.mark.parametrize('streaming', [False, True])
def test_float_predictor_writer_matches_tifffile(dtype, byte_order, channels, tiled, streaming):
    pytest.importorskip('opencodecs.codecs._deflate')
    shape = (35, 47) if channels == 1 else (35, 47, channels)
    # Deliberately opposite input and target byte orders.
    source_order = '>' if byte_order == '<' else '<'
    values = float_bits(source_order + dtype, shape)
    original = values.tobytes()
    sink = ForwardOnly() if streaming else io.BytesIO()
    options = dict(predictor=3, compression='deflate', n_workers=4,
                   tile=(16, 32) if tiled else None, rows_per_strip=7)
    with TiffWriter(sink, byte_order=byte_order, streaming=streaming,
                    spool_threshold=128 if streaming else None) as writer:
        if streaming:
            writer.write_stream(iter([values]), total_pages=1, **options)
        else:
            writer.write_page(values, **options)
    encoded = bytes(sink.buffer) if streaming else sink.getvalue()
    with tifffile.TiffFile(io.BytesIO(encoded)) as file:
        assert file.pages[0].predictor == 3
        decoded = file.asarray()
    assert_bits_equal(decoded, values)
    from opencodecs._tiff_codec import TiffStream, _HAVE_BACKEND
    if _HAVE_BACKEND:
        with TiffStream(encoded) as reader:
            assert_bits_equal(reader.read(), values)
    assert values.tobytes() == original


@pytest.mark.parametrize('compression', ['zstd', 'lzw'])
def test_float_predictor_byte_compressors(compression):
    pytest.importorskip('opencodecs.codecs._zstd' if compression == 'zstd' else 'opencodecs.codecs._tiff')
    values = float_bits('<f4', (31, 43, 3))
    sink = io.BytesIO()
    with TiffWriter(sink) as writer:
        writer.write_page(values, compression=compression, predictor=3, rows_per_strip=5)
    assert_bits_equal(tifffile.imread(io.BytesIO(sink.getvalue())), values)


@pytest.mark.parametrize('streaming', [False, True])
@pytest.mark.parametrize('dtype,compression,planar', [('u2', 'deflate', 1),
                                                   ('f4', 'jpeg', 1),
                                                   ('f4', 'none', 1),
                                                   ('f4', 'deflate', 2)])
def test_float_predictor_rejects_incompatible_options(streaming, dtype, compression, planar):
    values = np.ones((7, 11, 3), dtype=dtype)
    sink = ForwardOnly() if streaming else io.BytesIO()
    with TiffWriter(sink, streaming=streaming) as writer:
        options = dict(predictor=3, compression=compression, planar_config=planar)
        with pytest.raises(TiffWriterError, match='predictor 3 requires'):
            if streaming:
                writer.write_stream([values], total_pages=1, **options)
            else:
                writer.write_page(values, **options)
