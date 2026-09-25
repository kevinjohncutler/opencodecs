"""Optional TIFF verification checks reconstructed pixels before emission."""
from __future__ import annotations

import io
import threading
import zlib

import numpy as np
import pytest

from opencodecs._tiff_writer import TiffWriter
from opencodecs.core.verification import LosslessVerificationError, assert_bit_exact


def require_native(compression='deflate'):
    pytest.importorskip('opencodecs.codecs._tiff')
    if compression != 'lzw':
        pytest.importorskip('opencodecs.codecs._' + compression)


def pixels(dtype, shape):
    values = np.random.default_rng(412).integers(0, 256, np.prod(shape) * np.dtype(dtype).itemsize,
                                               dtype='u1').view(dtype).reshape(shape)
    if values.dtype.kind == 'f':
        # Include distinct NaN payloads and both zero signs without float math.
        values.view(np.dtype(dtype.replace('f', 'u'))).flat[:4] = [0, 0x80000000, 0x7fc01234, 0x7f801234]
    return values


@pytest.mark.parametrize('predictor,dtype', [(1, 'u2'), (2, 'u2'), (3, 'f4')])
@pytest.mark.parametrize('byte_order', ['<', '>'])
@pytest.mark.parametrize('channels', [1, 3])
@pytest.mark.parametrize('tiled', [False, True])
@pytest.mark.parametrize('streaming', [False, True])
def test_verify_reconstructs_original_pixels(predictor, dtype, byte_order, channels, tiled, streaming):
    require_native()
    tifffile = pytest.importorskip('tifffile')
    shape = (35, 47) if channels == 1 else (35, 47, channels)
    values = pixels(('>' if byte_order == '<' else '<') + dtype, shape)
    before = values.tobytes()
    sink = io.BytesIO()
    options = dict(compression='deflate', predictor=predictor, verify=True,
                   tile=(16, 32) if tiled else None, rows_per_strip=7, n_workers=4)
    with TiffWriter(sink, byte_order=byte_order, streaming=streaming,
                    spool_threshold=128 if streaming else None) as writer:
        if streaming:
            writer.write_stream([values], total_pages=1, **options)
        else:
            writer.write_page(values, **options)
    assert_bit_exact(values, tifffile.imread(io.BytesIO(sink.getvalue())))
    assert values.tobytes() == before


@pytest.mark.parametrize('compression', ['none', 'deflate', 'zstd', 'lzw', 'jxl'])
def test_verify_compressor_paths(compression):
    pytest.importorskip('opencodecs.codecs._tiff')
    if compression != 'none':
        require_native(compression)
    values = np.random.default_rng(4).integers(0, 256, (39, 47, 3), dtype='u1')
    sink = io.BytesIO()
    with TiffWriter(sink) as writer:
        writer.write_page(values, compression=compression, tile=(16, 32), verify=True)
    tifffile = pytest.importorskip('tifffile')
    assert_bit_exact(values, tifffile.imread(io.BytesIO(sink.getvalue())))


@pytest.mark.parametrize('streaming', [False, True])
def test_verify_rejects_valid_compressed_wrong_pixels_before_emission(monkeypatch, streaming):
    require_native()
    sink = io.BytesIO()
    values = np.arange(31 * 43, dtype='u2').reshape(31, 43)
    def corrupt(self, segment, cmp_code, level):
        altered = bytearray(segment.tobytes())
        altered[0] ^= 1
        return zlib.compress(altered)
    monkeypatch.setattr(TiffWriter, '_encode_segment_bytes', corrupt)
    with TiffWriter(sink, streaming=streaming) as writer:
        with pytest.raises(LosslessVerificationError, match='pixel bits differ'):
            if streaming:
                writer.write_stream([values], total_pages=1, compression='deflate', verify=True)
            else:
                writer.write_page(values, compression='deflate', verify=True)
    assert len(sink.getvalue()) == 8  # Header only; no unverified segment emitted.


@pytest.mark.parametrize('predictor,dtype', [(2, '<u2'), (3, '<f4')])
def test_verify_checks_inverse_prediction(monkeypatch, predictor, dtype):
    require_native()
    from opencodecs._tiff_codec import TiffPage
    values = pixels(dtype, (17, 31))
    monkeypatch.setattr(TiffPage, '_undo_predictor', lambda self, arr: arr)
    with TiffWriter(io.BytesIO()) as writer:
        with pytest.raises(LosslessVerificationError, match='pixel bits differ'):
            writer.write_page(values, compression='deflate', predictor=predictor, verify=True)


def test_verification_overlaps_another_encoding_worker(monkeypatch):
    require_native()
    other_encoding = threading.Event()
    first_verified = threading.Event()
    encode = TiffWriter._encode_segment_bytes
    verify = TiffWriter._verify_segment_pixels
    def wrapped_encode(self, segment, cmp_code, level):
        if segment[0, 0] == 1:
            other_encoding.set()
            assert first_verified.wait(5), 'verification did not overlap the other encoder'
        return encode(self, segment, cmp_code, level)
    def wrapped_verify(self, original, encoded, cmp_code, predictor):
        if original[0, 0] == 0:
            assert other_encoding.wait(5), 'second encoding worker did not start'
            verify(self, original, encoded, cmp_code, predictor)
            first_verified.set()
        else:
            verify(self, original, encoded, cmp_code, predictor)
    monkeypatch.setattr(TiffWriter, '_encode_segment_bytes', wrapped_encode)
    monkeypatch.setattr(TiffWriter, '_verify_segment_pixels', wrapped_verify)
    values = np.zeros((256, 1024), dtype='u2')
    values[128:] = 1
    with TiffWriter(io.BytesIO()) as writer:
        writer.write_page(values, tile=(128, 1024), compression='deflate',
                          verify=True, n_workers=2)
    assert first_verified.is_set()


def test_default_does_not_decode_for_verification(monkeypatch):
    def forbidden(*args):
        raise AssertionError('verification is opt-in')
    monkeypatch.setattr(TiffWriter, '_verify_segment_pixels', forbidden)
    with TiffWriter(io.BytesIO()) as writer:
        writer.write_page(np.zeros((17, 29), dtype='u1'))


@pytest.mark.parametrize('method', ['write_pyramid', 'write_pyramid_auto'])
def test_pyramid_forwards_verification(monkeypatch, method):
    pytest.importorskip('opencodecs.codecs._tiff')
    verified = []
    original = TiffWriter._verify_segment_pixels
    def checked(self, *args):
        verified.append(True)
        return original(self, *args)
    monkeypatch.setattr(TiffWriter, '_verify_segment_pixels', checked)
    values = np.zeros((32, 32), dtype='u1')
    with TiffWriter(io.BytesIO()) as writer:
        if method == 'write_pyramid':
            writer.write_pyramid([values, values[::2, ::2]], tile=(16, 16), verify=True)
        else:
            writer.write_pyramid_auto(values, pyramid_levels=2, tile=(16, 16), verify=True)
    assert verified


@pytest.mark.parametrize('source_order,target_order', [('<', '>'), ('>', '<'), ('>', '>')])
def test_horizontal_prediction_uses_target_endian_values(source_order, target_order):
    tifffile = pytest.importorskip('tifffile')
    pytest.importorskip('imagecodecs')
    class IndependentDeflateWriter(TiffWriter):
        def _encode_segment_bytes(self, segment, cmp_code, level):
            return zlib.compress(segment.tobytes())
    values = (np.arange(17 * 31, dtype='u2').reshape(17, 31) * 257 + 13).astype(source_order + 'u2')
    sink = io.BytesIO()
    with IndependentDeflateWriter(sink, byte_order=target_order) as writer:
        writer.write_page(values, compression='deflate', predictor=2,
                          rows_per_strip=5, n_workers=1)
    assert_bit_exact(values, tifffile.imread(io.BytesIO(sink.getvalue())))


@pytest.mark.parametrize('compression', ['CMP_JXL', 'CMP_JPEG2000'])
def test_tiff_native_decoder_honors_outer_worker_context(monkeypatch, compression):
    from types import SimpleNamespace
    from opencodecs import _tiff_codec
    from opencodecs.core.pipeline import WorkerBudget
    calls = []
    # Constants normally come from the native module. Give them distinct values
    # so this policy test also runs when that optional module is absent.
    monkeypatch.setattr(_tiff_codec, 'CMP_JXL', 50002)
    monkeypatch.setattr(_tiff_codec, 'CMP_JPEG2000', 34712)
    def decoder(raw, **kwargs):
        calls.append(kwargs)
        return np.zeros((1, 1), dtype='u1')
    monkeypatch.setattr(_tiff_codec, '_get_decoder', lambda name: decoder)
    page = SimpleNamespace(compression=getattr(_tiff_codec, compression))
    _tiff_codec.TiffPage._decode_segment(page, b'fixture')
    WorkerBudget(2).run(lambda raw: _tiff_codec.TiffPage._decode_segment(page, raw), b'fixture')
    assert calls == [{}, {'numthreads': 1}]
