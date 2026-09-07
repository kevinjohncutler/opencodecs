"""The delta predictor against imagecodecs, and against itself.

delta / floatpred / xor were pure numpy, which the kernel audit missed
because it only looked at nogil functions. Racing them found decode
running 5.5x slower than imagecodecs -- np.cumsum carries per-element
dispatch a specialized loop does not, and a prefix sum is serial either
way, so this was never about vectorising.

The correctness half matters more than the speed half: the kernel does
modular arithmetic on eight integer types, and wraparound is what the
format means rather than an accident to be avoided.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

DTYPES = ["u1", "i1", "u2", "i2", "u4", "i4", "u8", "i8"]


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("dist", [1, 2, 3])
def test_round_trip_every_dtype_and_distance(dtype, dist):
    info = np.iinfo(dtype)
    rng = np.random.default_rng(0)
    a = rng.integers(info.min, min(info.max, 2**31 - 1),
                     (7, 61), dtype=dtype)
    enc = oc.get_codec("delta").encode(a, axis=-1, dist=dist)
    back = oc.get_codec("delta").decode(enc, dtype=dtype, shape=a.shape,
                                        axis=-1, dist=dist)
    assert np.array_equal(np.asarray(back), a)


@pytest.mark.parametrize("dtype", DTYPES)
def test_matches_imagecodecs_exactly(dtype):
    """Same bytes, not merely a valid round trip."""
    ic = pytest.importorskip("imagecodecs")
    info = np.iinfo(dtype)
    rng = np.random.default_rng(1)
    a = rng.integers(info.min, min(info.max, 2**31 - 1),
                     (5, 257), dtype=dtype)
    from opencodecs._predictor_codec import DeltaCodec
    ours = DeltaCodec()._apply_along(a, -1, 1, "encode")
    assert np.array_equal(ours, ic.delta_encode(a, axis=-1))

    dec = DeltaCodec()._apply_along(np.ascontiguousarray(ours), -1, 1, "decode")
    assert np.array_equal(dec, ic.delta_decode(
        np.ascontiguousarray(ours), axis=-1))


def test_decode_accepts_a_read_only_buffer():
    """np.frombuffer hands back a read-only array.

    The kernel takes a writable memoryview, so a read-only input used to
    make it raise, and the exception was swallowed into a numpy fallback
    that quietly ran 5x slower. Decoding from bytes is the normal case,
    so this is the path that matters.
    """
    a = np.arange(1000, dtype="u1").reshape(10, 100)
    blob = oc.get_codec("delta").encode(a, axis=-1)
    assert isinstance(blob, (bytes, bytearray))
    ro = np.frombuffer(blob, dtype="u1")
    assert not ro.flags.writeable
    back = oc.get_codec("delta").decode(blob, dtype="u1", shape=a.shape,
                                        axis=-1)
    assert np.array_equal(np.asarray(back), a)


def test_the_compiled_kernel_is_actually_used():
    """Otherwise the fallback silently reinstates the slow path.

    This is the assertion the original wiring lacked: it had a try/except
    that turned "the kernel refused" into "use numpy", so a five-fold
    regression looked exactly like success.
    """
    from opencodecs import _predictor_codec as pc
    kern = pc._delta_decode_kernel()
    if kern is None:
        pytest.skip("_bytetools extension not built")
    seen = []
    original = pc._delta_decode_kernel

    def spy():
        k = original()
        seen.append(k)
        return k

    pc._delta_decode_kernel = spy
    try:
        a = np.arange(600, dtype="u2").reshape(6, 100)
        blob = oc.get_codec("delta").encode(a, axis=-1)
        back = oc.get_codec("delta").decode(blob, dtype="u2", shape=a.shape,
                                            axis=-1)
    finally:
        pc._delta_decode_kernel = original
    assert seen, "decode never consulted the kernel"
    assert np.array_equal(np.asarray(back), a)


def test_wraparound_is_preserved():
    """Delta on unsigned types is modular; that is the format, not a bug."""
    a = np.array([[0, 200, 100, 250]], dtype="u1")
    enc = oc.get_codec("delta").encode(a, axis=-1)
    raw = np.frombuffer(enc, dtype="u1")
    assert raw[1] == np.uint8(200 - 0)
    assert raw[2] == np.uint8((100 - 200) % 256)     # wraps
    back = oc.get_codec("delta").decode(enc, dtype="u1", shape=a.shape,
                                        axis=-1)
    assert np.array_equal(np.asarray(back), a)


# --------------------------------------------------------------------
# xor and packints, found by the same race
# --------------------------------------------------------------------

@pytest.mark.parametrize("dtype", ["u1", "u2", "u4", "u8"])
@pytest.mark.parametrize("dist", [1, 2])
def test_xor_round_trip(dtype, dist):
    rng = np.random.default_rng(2)
    a = rng.integers(0, 255, (5, 71), dtype="u1").astype(dtype)
    enc = oc.get_codec("xor").encode(a, axis=-1, dist=dist)
    back = oc.get_codec("xor").decode(enc, dtype=dtype, shape=a.shape,
                                      axis=-1, dist=dist)
    assert np.array_equal(np.asarray(back), a)


def test_xor_decode_uses_the_kernel():
    """np.bitwise_xor.accumulate is 5x slower; the kernel is the point."""
    from opencodecs import _predictor_codec as pc
    if pc._xor_decode_kernel() is None:
        pytest.skip("_bytetools extension not built")
    a = np.arange(400, dtype="u2").reshape(4, 100)
    blob = oc.get_codec("xor").encode(a, axis=-1)
    back = oc.get_codec("xor").decode(blob, dtype="u2", shape=a.shape,
                                      axis=-1)
    assert np.array_equal(np.asarray(back), a)


@pytest.mark.parametrize("bps", [1, 2, 3, 4, 5, 6, 7, 12, 16, 24, 32])
def test_packints_round_trip_every_width(bps):
    """The widths that matter here are the ones that miss a byte boundary.

    Decode used to be a Python loop with a numpy scalar store per
    element: 81 seconds for a million 4-bit samples. It is vectorized
    now, and these assert the bits are unchanged, which is the only
    thing that made the rewrite safe.
    """
    rng = np.random.default_rng(3)
    n = 997
    src = (rng.integers(0, 1 << min(bps, 20), n, dtype="u4")
           & ((1 << bps) - 1)).astype("u4")
    packed = oc.get_codec("packints").encode(src, bitspersample=bps)
    back = oc.get_codec("packints").decode(
        packed, dtype="u4", bitspersample=bps, n_elements=n)
    assert np.array_equal(np.asarray(back), src)


def test_packints_decode_is_not_quadratic():
    """A guard on the shape of the cost, not on a wall-clock number.

    The old loop took ~81 s for a million samples. Anything still doing
    per-element Python work will not finish this in a few seconds, and
    anything vectorized finishes in milliseconds, so the threshold does
    not need to be tight to be meaningful.
    """
    import time
    n = 1 << 20
    rng = np.random.default_rng(4)
    src = rng.integers(0, 16, n, dtype="u1")
    packed = oc.get_codec("packints").encode(src, bitspersample=4)
    t = time.perf_counter()
    back = oc.get_codec("packints").decode(
        packed, dtype="u1", bitspersample=4, n_elements=n)
    elapsed = time.perf_counter() - t
    assert np.array_equal(np.asarray(back), src)
    assert elapsed < 2.0, (
        f"packints decode took {elapsed:.1f}s for {n} samples; the "
        f"per-element Python loop is back")


def test_packints_rejects_a_short_buffer():
    """The old loop raised mid-walk; the vectorized one checks up front."""
    packed = oc.get_codec("packints").encode(
        np.arange(10, dtype="u1"), bitspersample=4)
    with pytest.raises(ValueError, match="bits"):
        oc.get_codec("packints").decode(
            packed, dtype="u1", bitspersample=4, n_elements=10_000)
