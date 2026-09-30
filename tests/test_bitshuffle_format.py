"""Bitshuffle output against the format, not only against itself.

A round trip passes for any bijection, so it cannot tell whether the SSE2
path (which MSVC builds now take too) still writes what other bitshuffle
readers expect. The digests below were computed from this library and
matched imagecodecs' independent build of bitshuffle byte for byte; the
test also compares against imagecodecs directly where it is installed.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

GOLDEN = [
    (1, 7, "5f8d6ceeb6b9f6cf"), (1, 64, "68492d48cc975409"),
    (1, 1000, "fa8b67a9c09b554c"), (1, 4099, "1d44dd9deb87c3cb"),
    (2, 7, "1f2c4de08da7564c"), (2, 64, "5c17de9f1fc686b1"),
    (2, 1000, "5f6478d99a002fda"), (2, 4099, "d5b5039f22e318c5"),
    (4, 7, "6d26d9c502af3af3"), (4, 64, "e7461f29c50175e9"),
    (4, 1000, "2fe97d40da0f53a2"), (4, 4099, "060ac17e73c602f4"),
    (8, 7, "9951977c792fa0f4"), (8, 64, "274548ad56b24110"),
    (8, 1000, "d8dfe7ff8434383f"), (8, 4099, "dd12cbb196dd6987"),
]


def _inputs():
    rng = np.random.default_rng(123)
    for itemsize, nelem, digest in GOLDEN:
        raw = rng.integers(0, 256, nelem * itemsize, dtype=np.uint8).tobytes()
        yield itemsize, nelem, digest, raw


def test_encoding_matches_the_reference_digests():
    bs = pytest.importorskip("opencodecs.codecs._bitshuffle")
    for itemsize, nelem, digest, raw in _inputs():
        enc = bytes(bs.encode(raw, itemsize=itemsize))
        assert hashlib.sha256(enc).hexdigest()[:16] == digest, (itemsize, nelem)
        assert bytes(bs.decode(enc, itemsize=itemsize)) == raw


def test_encoding_matches_imagecodecs():
    bs = pytest.importorskip("opencodecs.codecs._bitshuffle")
    imagecodecs = pytest.importorskip("imagecodecs")
    for itemsize, nelem, _, raw in _inputs():
        theirs = imagecodecs.bitshuffle_encode(np.frombuffer(raw, dtype=f"u{itemsize}"))
        theirs = theirs.tobytes() if isinstance(theirs, np.ndarray) else bytes(theirs)
        assert bytes(bs.encode(raw, itemsize=itemsize)) == theirs, (itemsize, nelem)
