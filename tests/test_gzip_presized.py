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
