"""TIFF decode workers follow the decompression work as well as the output size.

Sizing by output alone held a deflate image with real compressed data (6.8 MB in 31 strips) to 7 workers;
sizing by input alone would serialize a nearly empty mask whose cost is writing 16 MB of output.
"""

from opencodecs import _tiff_codec
from opencodecs.core import parallel


def _workers(**kw):
    return _tiff_codec._resolve_tiff_workers(None, kw.pop("n", 31), **kw)


def test_compressed_input_raises_the_worker_count(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 20)
    by_output = _workers(output_bytes=8_000_000)
    both = _workers(output_bytes=8_000_000, input_bytes=6_800_000)
    assert by_output == 7 and both == _tiff_codec._TIFF_MAX_WORKERS


def test_cheap_input_keeps_the_output_count(monkeypatch):
    monkeypatch.setattr(parallel.os, "cpu_count", lambda: 20)
    assert _workers(n=63, output_bytes=16_000_000, input_bytes=20_000) == _workers(n=63, output_bytes=16_000_000)


def test_explicit_threads_and_uncompressed_are_unchanged():
    assert _tiff_codec._resolve_tiff_workers(5, 31, output_bytes=8_000_000, input_bytes=6_800_000) == 5
    assert _workers(has_decode_work=False, output_bytes=8_000_000, input_bytes=8_000_000) == 1
