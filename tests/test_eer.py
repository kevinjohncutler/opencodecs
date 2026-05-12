"""EER (Electron Event Representation) decoder tests.

The reference test vector ``b'\\x03\\x1b\\xfc\\xb1\\x35\\xfb'`` is taken
straight from the EER format specification and matches imagecodecs'
own test cases. Event positions are pre-computed in the spec, so we
assert against absolute expected values rather than cross-validate.

Additional fuzz cross-validation against imagecodecs.eer_decode (when
present) covers parameter combinations on random bitstreams.
"""

from __future__ import annotations

import numpy as np
import pytest

mod = pytest.importorskip("opencodecs.codecs._eer")
decode = mod.decode
EerError = mod.EerError

# Test vector from the EER specification
SPEC_ENCODED = b"\x03\x1b\xfc\xb1\x35\xfb"


def test_eer_spec_linear():
    """Linear 1x312 frame; expected event positions: 3, 17, 233, 311."""
    im = decode(SPEC_ENCODED, (1, 312), 7, 1, 1)
    hits = np.where(im[0])[0].tolist()
    assert hits == [3, 17, 233, 311]


def test_eer_spec_2d_grid():
    """Same events folded into 20x16."""
    im = decode(SPEC_ENCODED, (20, 16), 7, 1, 1)
    assert im[0, 3]
    assert im[1, 1]
    assert im[14, 9]
    assert im[19, 7]
    assert im.sum() == 4


def test_eer_spec_superres():
    """Super-resolution mode upsamples to 40x32 with sub-pixel hits."""
    im = decode(SPEC_ENCODED, (40, 32), 7, 1, 1, superres=1)
    assert im[0, 7]
    assert im[3, 3]
    assert im[29, 18]
    assert im[39, 14]
    assert im.sum() == 4


def test_eer_uint16_accumulator():
    """Passing a uint16 ``out`` accumulates instead of overwriting."""
    out = np.ones((40, 32), np.uint16)
    decode(SPEC_ENCODED, (40, 32), 7, 1, 1, superres=1, out=out)
    # Each of the four hits adds 1 to the pre-existing 1.
    assert int(out[0, 7]) == 2
    assert int(out[3, 3]) == 2
    assert int(out[29, 18]) == 2
    assert int(out[39, 14]) == 2
    # Background should be the original 1.
    assert int(out.sum()) == (40 * 32) + 4


def test_eer_rejects_shape_too_small():
    """Output shape too small to hold all events -> EerError."""
    with pytest.raises(EerError):
        decode(SPEC_ENCODED, (19, 15), 7, 1, 1)


def test_eer_superres_requires_compatible_shape():
    """In super-resolution mode the output shape must be divisible by
    the super-resolution factor."""
    with pytest.raises(EerError):
        decode(SPEC_ENCODED, (40, 33), 7, 1, 1, superres=1)


def test_eer_rejects_invalid_params():
    with pytest.raises(EerError):
        # skipbits=0 invalid
        decode(SPEC_ENCODED, (16, 16), 0, 1, 1)
    with pytest.raises(EerError):
        # horzbits=0 invalid
        decode(SPEC_ENCODED, (16, 16), 7, 0, 1)


def test_eer_imagecodecs_cross_validate():
    """Random bitstreams must decode identically to imagecodecs.

    EER's "skip" field can advance past the end of small canvases on
    random input — we use a 1024x1024 canvas so most parameter combos
    decode cleanly. When *either* implementation errors we just
    cross-check that *both* implementations error on the same input.
    """
    imagecodecs = pytest.importorskip("imagecodecs")
    if not getattr(imagecodecs, "EER", None) or not imagecodecs.EER.available:
        pytest.skip("imagecodecs EER backend unavailable")

    rng = np.random.default_rng(42)
    data = rng.bytes(4096)
    matched = 0
    for sb in (7, 8, 10):
        for hb in (2, 3):
            for vb in (2, 3):
                if not (8 < sb + hb + vb < 17):
                    continue
                for sr in (0, 1, 2):
                    shape = (1024, 1024)
                    try:
                        ours = decode(data, shape, sb, hb, vb, superres=sr)
                    except EerError:
                        with pytest.raises(Exception):
                            imagecodecs.eer_decode(
                                data, shape, sb, hb, vb, superres=sr
                            )
                        continue
                    theirs = imagecodecs.eer_decode(
                        data, shape, sb, hb, vb, superres=sr
                    )
                    np.testing.assert_array_equal(
                        ours, theirs,
                        err_msg=(
                            f"divergence at sb={sb} hb={hb} vb={vb} sr={sr}"
                        ),
                    )
                    matched += 1
    assert matched > 0, "no parameter combo decoded cleanly"
