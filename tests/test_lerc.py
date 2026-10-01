"""LERC codec: codec version, imagecodecs keywords, masks, outer compression.

The reference is imagecodecs.lerc_encode / lerc_decode. Both packages link
their own libLerc, and two copies of it must not be loaded into one
process, so every imagecodecs call runs in a child interpreter and the
arrays and blobs cross through files.
"""

from __future__ import annotations

import pickle
import subprocess
import sys
import textwrap

import numpy as np
import pytest
from _ic_reference import skip_if_old_imagecodecs  # noqa: E402

pytestmark = skip_if_old_imagecodecs

oc = pytest.importorskip("opencodecs")
_lerc = pytest.importorskip("opencodecs.codecs._lerc")
codec = oc.get_codec("lerc")


def _blob_version(blob: bytes) -> int:
    """Codec version from a Lerc2 header ('Lerc2 ' then int32 version)."""
    assert blob[:6] == b"Lerc2 "
    return int.from_bytes(blob[6:10], "little")


def _cases():
    rng = np.random.default_rng(0)
    u8 = rng.integers(0, 256, (64, 80), dtype=np.uint8)
    mask = rng.random((64, 80)) > 0.2
    return {
        "u8": (u8, {}),
        "u16_depth3": (rng.integers(0, 4000, (20, 30, 3), dtype=np.uint16), {}),
        "i16_1d": (rng.integers(-500, 500, (77,), dtype=np.int16), {}),
        "f64_planar": (rng.random((3, 20, 30)), {"planar": True}),
        "f32_lossy": (rng.random((40, 50)).astype(np.float32), {"level": 0.01}),
        "u8_v6": (u8, {"version": 6}),
        "u8_mask": (u8, {"masks": mask}),
        "u8_zstd": (u8, {"compression": "zstd"}),
        "u8_deflate": (u8, {"compression": "deflate"}),
        # imagecodecs clamps an outer level to the codec's range: deflate
        # to -1..9, zstd to 0..22. 12 and 19 must give level 9's bytes,
        # -3 the default's, and zstd 30 level 22's, not an error.
        "u8_deflate_1": (u8, {"compression": "deflate",
                              "compressionargs": {"level": 1}}),
        "u8_deflate_12": (u8, {"compression": "deflate",
                               "compressionargs": {"level": 12}}),
        "u8_deflate_19": (u8, {"compression": "deflate",
                               "compressionargs": {"level": 19}}),
        "u8_deflate_neg": (u8, {"compression": "deflate",
                                "compressionargs": {"level": -3}}),
        "u8_zstd_30": (u8, {"compression": "zstd",
                            "compressionargs": {"level": 30}}),
    }


_CHILD = textwrap.dedent("""
    import pickle, sys
    import numpy as np
    try:
        import imagecodecs
        imagecodecs.lerc_encode
    except Exception:
        sys.exit(77)
    path = sys.argv[1]
    cases, ours = pickle.load(open(path + ".in", "rb"))
    theirs, decoded = {}, {}
    for name, (arr, kwargs) in cases.items():
        theirs[name] = bytes(imagecodecs.lerc_encode(arr, **kwargs))
        masks = "masks" in kwargs
        result = imagecodecs.lerc_decode(ours[name], masks=masks)
        decoded[name] = result
    pickle.dump((theirs, decoded), open(path + ".out", "wb"))
""")


@pytest.fixture(scope="module")
def imagecodecs_run(tmp_path_factory):
    """Encode every case here, then encode and cross-decode in imagecodecs."""
    cases = _cases()
    ours = {name: codec.encode(arr, **kwargs)
            for name, (arr, kwargs) in cases.items()}
    base = str(tmp_path_factory.mktemp("lerc") / "x")
    with open(base + ".in", "wb") as fh:
        pickle.dump((cases, ours), fh)
    proc = subprocess.run([sys.executable, "-c", _CHILD, base],
                          capture_output=True, text=True)
    if proc.returncode == 77:
        pytest.skip("imagecodecs with LERC is not importable")
    assert proc.returncode == 0, proc.stderr
    with open(base + ".out", "rb") as fh:
        theirs, decoded = pickle.load(fh)
    return cases, ours, theirs, decoded


def test_lerc_default_version_is_4():
    """v4 is imagecodecs' default and what libtiff fixes for TIFF LERC."""
    arr = np.arange(600, dtype=np.float32).reshape(20, 30)
    assert _blob_version(codec.encode(arr)) == 4
    assert _blob_version(_lerc.encode(arr)) == 4
    for version in (2, 3, 4, 5, 6):
        blob = codec.encode(arr, version=version)
        assert _blob_version(blob) == version
        np.testing.assert_array_equal(codec.decode(blob), arr)
    assert _blob_version(codec.encode(arr, version=-1)) == 6
    with pytest.raises(ValueError):
        codec.encode(arr, version=7)


def test_lerc_bytes_match_imagecodecs(imagecodecs_run):
    """Same arguments, same blob, for every keyword imagecodecs defines.

    Lossless float at version 6 is left out: its bytes depend on the
    libLerc release, which the two packages need not share.
    """
    cases, ours, theirs, _ = imagecodecs_run
    for name in cases:
        assert ours[name] == theirs[name], name


def test_lerc_imagecodecs_decodes_ours(imagecodecs_run):
    cases, _, _, decoded = imagecodecs_run
    for name, (arr, kwargs) in cases.items():
        got = decoded[name]
        if "masks" in kwargs:
            got, mask = got
            np.testing.assert_array_equal(mask, kwargs["masks"])
            np.testing.assert_array_equal(got[mask], arr[mask])
            continue
        ref = arr.reshape(1, -1) if arr.ndim == 1 else arr
        if "level" in kwargs:
            assert np.abs(got - ref).max() <= kwargs["level"] + 1e-7
        else:
            np.testing.assert_array_equal(got, ref, err_msg=name)


def test_lerc_decodes_imagecodecs(imagecodecs_run):
    cases, _, theirs, _ = imagecodecs_run
    for name, (arr, kwargs) in cases.items():
        if "masks" in kwargs:
            got, mask = codec.decode(theirs[name], masks=True)
            np.testing.assert_array_equal(mask, kwargs["masks"])
            np.testing.assert_array_equal(got[mask], arr[mask])
            assert not got[~mask].any()
            continue
        got = codec.decode(theirs[name])
        ref = arr.reshape(1, -1) if arr.ndim == 1 else arr
        if "level" in kwargs:
            assert np.abs(got - ref).max() <= kwargs["level"] + 1e-7
        else:
            np.testing.assert_array_equal(got, ref, err_msg=name)


def test_lerc_level_is_max_z_error():
    rng = np.random.default_rng(1)
    arr = rng.random((40, 50)).astype(np.float32)
    lossy = codec.encode(arr, level=0.5)
    assert lossy == codec.encode(arr, max_z_error=0.5)
    assert len(lossy) < len(codec.encode(arr))
    assert np.abs(codec.decode(lossy) - arr).max() <= 0.5
    with pytest.raises(ValueError):
        codec.encode(arr, level=0.5, max_z_error=0.1)


def test_lerc_rejects_unknown_options():
    with pytest.raises(TypeError):
        codec.encode(np.zeros((4, 4), np.uint8), bogus=1)
    # imagecodecs' lerc_decode takes masks and out only; decode used to
    # drop anything else.
    blob = codec.encode(np.arange(16, dtype=np.uint8).reshape(4, 4))
    with pytest.raises(TypeError):
        codec.decode(blob, bogus=1)
    with pytest.raises(TypeError):
        codec.decode(blob, index=0)
    out = np.empty((4, 4), np.uint8)
    assert codec.decode(blob, out=out) is out


def test_lerc_deflate_level_is_clamped_not_an_error():
    """A deflate level past 9 is level 9, as imagecodecs' zlib_encode has it.

    It used to reach zlib.compress unchanged and fail there with a raw
    zlib.error. The deflate stream itself is checked with Python's zlib,
    which is independent of both codecs.
    """
    import zlib
    arr = np.random.default_rng(2).integers(0, 256, (64, 80), dtype=np.uint8)
    bare = codec.encode(arr)
    for level, expect in ((12, 9), (19, 9), (9, 9), (0, 0), (-3, -1)):
        blob = codec.encode(arr, compression="deflate",
                            compressionargs={"level": level})
        assert blob == zlib.compress(bare, expect), level
        np.testing.assert_array_equal(codec.decode(blob), arr)
    with pytest.raises(TypeError):
        codec.encode(arr, compression="deflate",
                     compressionargs={"level": 6, "wbits": 9})


def test_lerc_masked_pixels_decode_as_zero():
    arr = np.full((16, 16), 7, np.uint16)
    mask = np.ones((16, 16), bool)
    mask[3:9, 2:5] = False
    blob = codec.encode(arr, masks=mask)
    out = np.full((16, 16), 99, np.uint16)
    got, got_mask = codec.decode(blob, out=out, masks=True)
    assert got is out
    np.testing.assert_array_equal(got_mask, mask)
    assert not got[~mask].any() and (got[mask] == 7).all()
    # Unmasked blob with masks=True: no mask to return.
    plain = codec.decode(codec.encode(arr), masks=True)
    assert plain[1] is None
