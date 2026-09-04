"""dm4 header geometry, which the two real corpus files cannot reach.

Both fixtures in ``test_dm.py`` are .dm3. That is enough to exercise the
tag-tree walk and the type language, and it says nothing at all about
dm4, whose header this reader had wrong: dm4 widened only the root
length, from 4 bytes to 8, so its byte-order flag is at offset 12 and
its root tag directory starts at 16. The reader looked at 16 and 24,
which is what you get by widening the version field too.

Building a file is the only way to test that without a dm4 in the
corpus, and it is worth doing anyway: a synthetic file can be made
minimal and wrong on purpose, which a downloaded one cannot.
"""

from __future__ import annotations

import struct

import numpy as np
import pytest

from opencodecs._dm import DmError, DmFile

T_GROUP, T_ENTRY, T_ARRAY, T_USHORT = 20, 21, 20, 4


def build_dm(version: int, image: np.ndarray, *, little: bool = True) -> bytes:
    """A minimal dm3 or dm4 holding one uint16 image.

    Counts and lengths are big-endian in both versions; only their width
    changes. The samples follow the byte-order flag in the header.
    """
    long_fmt = "q" if version == 4 else "i"

    def count(n):
        return struct.pack(">" + long_fmt, n)

    def entry(kind, label, payload):
        head = bytes([kind]) + struct.pack(">H", len(label)) + label.encode()
        if version == 4:
            head += struct.pack(">q", len(payload))
        return head + payload

    def group(entries):
        return b"\x00\x00" + count(len(entries)) + b"".join(entries)

    def tagdata(typecodes, raw):
        head = b"%%%%" + count(len(typecodes))
        head += b"".join(struct.pack(">" + long_fmt, t) for t in typecodes)
        return head + raw

    order = "<" if little else ">"
    flat = image.astype(order + "u2").tobytes()
    data = entry(T_ENTRY, "Data",
                 tagdata([T_ARRAY, T_USHORT, image.size], flat))
    dims = group([entry(T_ENTRY, "",
                        tagdata([T_USHORT], struct.pack(order + "H", d)))
                  for d in image.shape[::-1]])
    image_data = entry(T_GROUP, "ImageData", group([
        data,
        entry(T_GROUP, "Dimensions", dims),
        entry(T_ENTRY, "DataType",
              tagdata([T_USHORT], struct.pack(order + "H", 10))),
    ]))
    root = group([entry(T_GROUP, "ImageList",
                        group([entry(T_GROUP, "", group([image_data]))]))])

    if version == 3:
        header = (struct.pack(">i", 3) + struct.pack(">I", len(root))
                  + struct.pack(">i", 1 if little else 0))
    else:
        header = (struct.pack(">i", 4) + struct.pack(">Q", len(root))
                  + struct.pack(">i", 1 if little else 0))
    return header + root


IMAGE = (np.arange(4 * 6, dtype="u2") + 3).reshape(4, 6)


@pytest.mark.parametrize("version", [3, 4])
def test_both_versions_reach_the_image(version):
    with DmFile(build_dm(version, IMAGE)) as f:
        assert f.version == version
        assert np.array_equal(f.asarray(), IMAGE)


@pytest.mark.parametrize("version", [3, 4])
def test_header_size_is_where_the_root_directory_starts(version):
    """Pinned directly, because getting it wrong is silent.

    An off-by-eight here does not raise: the tree still parses, produces
    plausible-looking tags, and simply never finds ImageList.
    """
    with DmFile(build_dm(version, IMAGE)) as f:
        assert f._header_size() == (12 if version == 3 else 16)


@pytest.mark.parametrize("version", [3, 4])
def test_byte_order_flag_is_honored(version):
    """The tag tree is big-endian; the samples follow the header flag."""
    with DmFile(build_dm(version, IMAGE, little=False)) as f:
        assert np.array_equal(f.asarray(), IMAGE)


@pytest.mark.parametrize("version", [3, 4])
def test_shape_and_dtype_without_decoding(version):
    with DmFile(build_dm(version, IMAGE)) as f:
        assert f.shape == IMAGE.shape
        assert f.dtype == np.dtype("<u2")


def test_a_dm4_is_not_read_as_a_dm3():
    """The versions differ only in integer width, so confusing them
    produces a plausible tree rather than an error."""
    blob = build_dm(4, IMAGE)
    with DmFile(blob) as f:
        assert f.version == 4
    # Same bytes with the version field lied about: the offsets no
    # longer line up and nothing is found, which is the failure the
    # real bug produced.
    lied = struct.pack(">i", 3) + blob[4:]
    with pytest.raises(DmError):
        DmFile(lied).asarray()
