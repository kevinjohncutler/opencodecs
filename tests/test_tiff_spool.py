"""Forward-only page spooling preserves the exact on-disk layout."""
import io
import numpy as np
import pytest

from opencodecs._tiff_writer import TiffWriter


@pytest.mark.parametrize('bigtiff', [False, True])
def test_spooled_pages_identical_and_independently_readable(bigtiff):
    tifffile = pytest.importorskip('tifffile')
    pages = [np.arange(127 * 193, dtype='u2').reshape(127, 193) + i for i in range(3)]
    class Forward(io.BytesIO):
        def seek(self, *_):
            raise AssertionError('destination must remain forward-only')
        def close(self):
            pass
    encoded = []
    for threshold in (None, 128, 1 << 20):
        dest = Forward()
        with TiffWriter(dest, streaming=True, bigtiff=bigtiff,
                        spool_threshold=threshold) as writer:
            writer.write_stream(iter(pages), total_pages=3, tile=(32, 48),
                                compression='none')
        encoded.append(dest.getvalue())
        with tifffile.TiffFile(io.BytesIO(encoded[-1])) as reader:
            for page, expected in zip(reader.pages, pages):
                np.testing.assert_array_equal(page.asarray(), expected)
    assert encoded[0] == encoded[1] == encoded[2]
