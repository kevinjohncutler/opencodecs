"""Explicit JPEG handles retain no stale scale or caller pixel ownership."""

from concurrent.futures import ThreadPoolExecutor
import io

import numpy as np
import pytest

import opencodecs as oc


@pytest.mark.parametrize("name", ["jpeg", "mozjpeg"])
def test_owned_decoder_reconfiguration_and_error_recovery(name):
    pytest.importorskip(f"opencodecs.codecs._{name}")
    rng = np.random.default_rng(121)
    images = [rng.integers(0, 256, shape, dtype="u1")
              for shape in [(65, 71, 3), (79, 63), (97, 81, 3)]]
    codec = oc.get_codec(name)
    with codec.decoder() as decoder:
        for image in images:
            encoded = codec.encode(image)
            for scale in (4, None, 2, None):
                expected = codec.decode(encoded, scale=scale)
                out = np.empty_like(expected)
                assert decoder.decode(encoded, scale=scale, out=out) is out
                np.testing.assert_array_equal(out, expected)
            with pytest.raises(Exception):
                decoder.decode(b"not a JPEG codestream" * 8)
            np.testing.assert_array_equal(decoder.decode(encoded), codec.decode(encoded))
        retained = decoder.decode(encoded)
        with ThreadPoolExecutor(4) as pool:
            results = list(pool.map(lambda _: decoder.decode(encoded), range(16)))
        for result in results:
            np.testing.assert_array_equal(result, retained)
            assert not np.shares_memory(result, retained)
    with pytest.raises(ValueError, match="closed"):
        decoder.decode(encoded)
    decoder.close()


def test_jpeg2k_seekable_sink_offsets_and_short_writes():
    pytest.importorskip("opencodecs.codecs._jpeg2k")
    reference = pytest.importorskip("imagecodecs")
    image = np.random.default_rng(72).integers(0, 65536, (83, 91), dtype="u2")
    class ShortSink(io.BytesIO):
        def write(self, data):
            return super().write(memoryview(data)[:19])
    sink = ShortSink()
    sink.write(b"prefix")
    assert oc.write(sink, image, format="jpeg2k", numthreads=1) is None
    assert not sink.closed
    assert sink.getvalue().startswith(b"prefix")
    assert sink.tell() == len(sink.getvalue())
    np.testing.assert_array_equal(reference.jpeg2k_decode(sink.getvalue()[6:]), image)


def test_jpeg2k_sink_failure_and_forward_only_fallback():
    pytest.importorskip("opencodecs.codecs._jpeg2k")
    image = np.zeros((83, 91), dtype="u2")
    class Broken(io.BytesIO):
        def write(self, data):
            raise OSError("JPEG2000 sink stopped")
    with pytest.raises(OSError, match="JPEG2000 sink stopped"):
        oc.write(Broken(), image, format="jpeg2k", numthreads=1)
    class ForwardOnly:
        def __init__(self):
            self.parts = []
        def write(self, data):
            self.parts.append(bytes(data))
            return len(data)
    sink = ForwardOnly()
    assert oc.write(sink, image, format="jpeg2k", numthreads=1) is None
    np.testing.assert_array_equal(oc.read(b"".join(sink.parts), format="jpeg2k"), image)
