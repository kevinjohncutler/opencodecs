"""A codec claiming streaming_decode must not materialize the file.

The flag means "iter_frames yields without full materialization". That
is a memory property, and nothing else in this suite measures it: a
reader that decodes everything and yields slices passes every
correctness test while making the flag a lie. OIR was exactly that in
reverse -- it streamed and the flag said False, which is the fourth
capability in this manifest found to exist and be recorded as absent.

Measured with tracemalloc against the reader's own bulk read, so the
threshold is a ratio and says something on any machine and any corpus
file.
"""

from __future__ import annotations

import pathlib
import tracemalloc

import numpy as np
import pytest

import opencodecs as oc

DATA = pathlib.Path(__file__).resolve().parent.parent / ".test_data"

# (codec, path relative to .test_data). Only formats where the corpus
# has a file with enough frames for the difference to be visible.
CASES = [
    ("oir", "oir/amy_slice_z_stack.oir"),
    ("eer", "eer/empiar10568_falcon4.eer"),
]


def _peak(fn) -> int:
    tracemalloc.start()
    try:
        fn()
        return tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()


@pytest.mark.parametrize("name,rel", CASES, ids=[c for c, _ in CASES])
def test_iterating_does_not_hold_the_whole_file(name, rel):
    path = DATA / rel
    if not path.is_file():
        pytest.skip(f"fetch the {name} corpus entry first")
    if not oc.has_codec(name):
        pytest.skip(f"{name} not built here")
    codec = oc.get_codec(name)
    if not codec.streaming_decode:
        pytest.skip(f"{name} does not claim streaming_decode")

    with codec.open(str(path)) as r:
        if r.n_frames < 4:
            pytest.skip("too few frames to tell")
        one = r[0].nbytes
        total = one * r.n_frames

        def walk():
            for _ in r.iter_frames():
                pass

        walk()                                  # warm any lazy parse
        peak = _peak(walk)

    assert peak < total * 0.5, (
        f"{name}: iterating peaked at {peak / 1e6:.2f} MB for a "
        f"{total / 1e6:.2f} MB stack; that is materialization, not streaming")


def test_the_flag_implies_a_frame_axis():
    """streaming_decode presupposes something to iterate.

    check_capabilities enforces this against the manifest; asserting it
    here too means a codec added without touching the manifest still
    cannot claim it.
    """
    offenders = [
        info["name"] for info in oc.list_codecs()
        if oc.get_codec(info["name"]).streaming_decode
        and not oc.get_codec(info["name"]).multi_frame
    ]
    assert not offenders, (
        f"these claim streaming_decode with no frame axis: {offenders}")
