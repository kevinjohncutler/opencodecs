"""Whole-stream NRRD iteration decodes once; raw iteration stays selective."""
from __future__ import annotations
import bz2
import gzip
import numpy as np
import pytest
from opencodecs._nrrd import NrrdFile


def make_volume(encoding):
    values = np.arange(5 * 7 * 11, dtype='<u2').reshape(5, 7, 11)
    raw = values.tobytes()
    if encoding == 'gzip':
        payload = gzip.compress(raw)
    elif encoding == 'bzip2':
        payload = bz2.compress(raw)
    elif encoding == 'ascii':
        payload = ' '.join(str(int(value)) for value in values.flat).encode('ascii')
    elif encoding == 'hex':
        payload = raw.hex().encode('ascii')
    else:
        payload = raw
    header = ('NRRD0005\ntype: uint16\ndimension: 3\nsizes: 11 7 5\n'
              f'encoding: {encoding}\nendian: little\n\n').encode('ascii')
    return header + payload, values


@pytest.mark.parametrize('encoding', ['gzip', 'bzip2', 'ascii', 'hex'])
def test_whole_stream_iteration_decodes_once_and_shares_one_volume(encoding):
    encoded, expected = make_volume(encoding)
    with NrrdFile(encoded) as reader:
        calls = []
        original = reader.asarray
        def decode():
            calls.append(True)
            return original()
        reader.asarray = decode
        frames = list(reader.iter_frames())
        np.testing.assert_array_equal(np.stack(frames), expected)
        assert len(calls) == 1
        assert len({id(frame.base) for frame in frames}) == 1


def test_raw_iteration_keeps_exact_frame_reads():
    encoded, expected = make_volume('raw')
    with NrrdFile(encoded) as reader:
        def forbidden():
            raise AssertionError('raw iteration should not materialize the volume')
        reader.asarray = forbidden
        np.testing.assert_array_equal(np.stack(list(reader.iter_frames())), expected)
