"""The pre-sized gzip inflate must not shorten a concatenated stream.

``GzipCodec.decode`` reads ISIZE from the gzip trailer and hands it to
zlib as the initial allocation, which halves peak memory. ISIZE
describes the LAST member, so on a multi-member stream it can agree
with the length of the FIRST member and the short read looks correct:
two 198-byte members did exactly that during development. These are the
cases that separate a correct fast path from a plausible one.
"""
import gzip
import os
import random

import pytest

import opencodecs as oc


@pytest.fixture(scope="module")
def codec():
    return oc.get_codec("gzip")


# Payloads are BUILT inside each test, never passed as parameters.
# A bytes object used as a parametrize argument becomes the test's id,
# so os.urandom(1 << 20) put a megabyte of raw binary in the node id
# and errored on Windows before the test ran. Parametrize on a name.
def _payloads(kind):
    rng = random.Random(0)
    return {
        "equal-length-different-bytes": (b"aa" * 99, b"bb" * 99),
        "identical-members":            (b"xy" * 50, b"xy" * 50),
        "five-identical-tiny":          (b"q",) * 5,
        "empty-member-in-the-middle":   (b"first", b"", b"third"),
        "random-64k-then-3":            (rng.randbytes(1 << 16),
                                         rng.randbytes(3)),
    }[kind]


@pytest.mark.parametrize("kind", [
    "equal-length-different-bytes",
    "identical-members",              # same crc AND size
    "five-identical-tiny",
    "empty-member-in-the-middle",
    "random-64k-then-3",
])
def test_concatenated_members_decode_whole(codec, kind):
    payloads = _payloads(kind)
    blob = b"".join(gzip.compress(p, 1) for p in payloads)
    assert codec.decode(blob) == b"".join(payloads)


@pytest.mark.parametrize("kind", ["empty", "one-byte", "4k-zeros", "1M-random"])
def test_single_member_round_trips(codec, kind):
    payload = {
        "empty": b"",
        "one-byte": b"z",
        "4k-zeros": b"\x00" * 4096,
        "1M-random": random.Random(1).randbytes(1 << 20),
    }[kind]
    assert codec.decode(gzip.compress(payload, 1)) == payload
    assert codec.decode(codec.encode(payload)) == payload


def test_fast_path_is_actually_taken(codec):
    """A single member must not silently fall back to the stdlib.

    Without this the correctness tests above would still pass with the
    fast path deleted, which is the failure mode that makes a
    performance guard rot quietly.
    """
    payload = os.urandom(1 << 18)
    assert codec._decode_single_member(gzip.compress(payload, 1)) == payload


def test_fast_path_declines_what_it_cannot_do(codec):
    """Multi-member and malformed input must return None, not a guess."""
    assert codec._decode_single_member(
        gzip.compress(b"a" * 10, 1) + gzip.compress(b"b" * 10, 1)) is None
    assert codec._decode_single_member(b"not a gzip stream") is None
    assert codec._decode_single_member(b"") is None
    assert codec._decode_single_member(gzip.compress(b"abc")[:8]) is None


def test_truncated_stream_still_raises(codec):
    blob = gzip.compress(os.urandom(1 << 16), 1)
    with pytest.raises(Exception):
        codec.decode(blob[:len(blob) // 2])


# ---------------------------------------------------------------------------
# libdeflate decode (_deflate.gzip_decode), when the build links it
# ---------------------------------------------------------------------------

def _libdeflate_decode():
    from opencodecs import _gzip_codec
    if not _gzip_codec._LIBDEFLATE_DECODE:
        pytest.skip("this build decodes gzip with the stdlib")
    return _gzip_codec


def _named_member(payload):
    """A member with FNAME and MTIME set, as gzip.GzipFile writes it."""
    import io
    buf = io.BytesIO()
    with gzip.GzipFile(filename="x.bin", mode="wb", fileobj=buf,
                       mtime=123) as f:
        f.write(payload)
    return buf.getvalue()


@pytest.mark.parametrize("kind", [
    "single", "two", "small-last-member", "trailing-zero-padding",
    "zeros-between-members", "named-header", "empty-member-first",
    "same-size-members",
])
def test_valid_input_never_reaches_the_stdlib(codec, monkeypatch, kind):
    """Valid gzip, single or multi-member, decodes through libdeflate.

    The stdlib is the fallback for input libdeflate refuses. If valid
    input reached it, the speedup would vanish with every result still
    correct, so the stdlib is made to fail here.
    """
    gzip_codec = _libdeflate_decode()
    rng = random.Random(2)
    a, b = rng.randbytes(70_000), b"abc" * 9_000
    blob, want = {
        "single": (gzip.compress(a), a),
        "two": (gzip.compress(a) + gzip.compress(b), a + b),
        "small-last-member": (gzip.compress(b) + gzip.compress(b"z"),
                              b + b"z"),
        "trailing-zero-padding": (gzip.compress(a) + bytes(37), a),
        "zeros-between-members": (gzip.compress(a) + bytes(5)
                                  + gzip.compress(b), a + b),
        "named-header": (_named_member(a), a),
        "empty-member-first": (gzip.compress(b"") + gzip.compress(a), a),
        # The last ISIZE sizes the buffer and the first member fills it
        # exactly, so the second starts with no room at all.
        "same-size-members": (gzip.compress(a) + gzip.compress(a), a + a),
    }[kind]

    def stdlib_used(*args, **kwargs):
        raise AssertionError("valid input fell back to the stdlib")

    monkeypatch.setattr(gzip_codec.gzip, "decompress", stdlib_used)
    assert codec.decode(blob) == want


def _malformed(kind):
    good = gzip.compress(random.Random(4).randbytes(50_000))
    return {
        "magic-only": good[:2],
        "short-header": good[:9],
        "truncated-body": good[:len(good) // 2],
        "truncated-trailer": good[:-3],
        "bad-crc": good[:-8] + bytes([good[-8] ^ 1]) + good[-7:],
        "bad-isize": good[:-4] + bytes([good[-4] ^ 1]) + good[-3:],
        "garbage-after-member": good + b"garbage!",
        "leading-zeros": bytes(4) + good,
        "bad-method": good[:2] + b"\x07" + good[3:],
        "corrupt-body": good[:500] + bytes(64) + good[564:],
        "second-member-truncated": good + gzip.compress(b"abc" * 99)[:20],
        "zeros-only": bytes(40),
    }[kind]


@pytest.mark.parametrize("kind", [
    "magic-only", "short-header", "truncated-body", "truncated-trailer",
    "bad-crc", "bad-isize", "garbage-after-member", "leading-zeros",
    "bad-method", "corrupt-body", "second-member-truncated", "zeros-only",
])
def test_malformed_input_raises_what_the_stdlib_raises(codec, kind):
    """Same exception type and message as gzip.decompress, on every build."""
    blob = _malformed(kind)
    with pytest.raises(Exception) as ref:
        gzip.decompress(blob)
    with pytest.raises(type(ref.value)) as got:
        codec.decode(blob)
    assert str(got.value) == str(ref.value)


def test_reserved_flag_bits_still_decode(codec):
    """libdeflate refuses reserved FLG bits; the stdlib ignores them.

    The fallback keeps the stdlib's answer, so a file that decoded
    before still decodes.
    """
    payload = b"reserved" * 1000
    blob = bytearray(gzip.compress(payload))
    blob[3] |= 0x20
    assert codec.decode(bytes(blob)) == payload


def test_native_decode_refuses_rather_than_guesses():
    """gzip_decode raises ZlibError, never returns a partial result."""
    _libdeflate_decode()
    from opencodecs.codecs import _deflate
    good = gzip.compress(b"x" * 5000)
    assert _deflate.gzip_decode(b"") == b""
    for blob in (good[:-1], good + b"junk", b"\x1f\x8b" + bytes(30)):
        with pytest.raises(_deflate.ZlibError):
            _deflate.gzip_decode(blob)
