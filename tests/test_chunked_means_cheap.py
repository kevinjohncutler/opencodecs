"""``Codec.chunked`` promises cheap indexing, not merely indexing.

Two flags in this package answer questions that sound identical:

    Reader.is_chunked   can you index this reader at all
    Codec.chunked       is indexing CHEAP -- frame N without
                        decoding 0..N-1

Nothing said so, and the gap between them is where a false claim
lives. EER set both: indexing worked, but through the base Reader's
default ``__getitem__``, which walks ``iter_frames()`` to get there --
6.71 ms for frame 0 and 1218 ms for frame 720 of 721. The capability
manifest reads the second flag, so it promised random access that
took longer than reading the file.

GIF is the honest version of the same shape: ``is_chunked = True`` and
``chunked = False``, with a comment saying it replays from frame 0
because disposal state forbids seeking.
"""

from __future__ import annotations

import time

import numpy as np
import pytest

import opencodecs as oc
from opencodecs.core.codec import Codec, Reader


def _readers():
    out = []
    for info in sorted(oc.list_codecs(), key=lambda c: c["name"]):
        cls = type(oc.get_codec(info["name"]))
        for k in cls.__mro__:
            if "open" in k.__dict__:
                if k is not Codec:
                    out.append(info["name"])
                break
    return out


def test_a_chunked_codec_does_not_use_the_walking_default():
    """``chunked = True`` with the inherited ``__getitem__`` is a
    contradiction.

    The default walks ``iter_frames()`` and picks, so a reader using it
    cannot be fetching frame N without decoding the ones before. This
    is the structural half of the claim; the timing half needs a
    multi-frame file and lives in the per-format tests.
    """
    offenders = []
    for name in _readers():
        codec = oc.get_codec(name)
        if not codec.chunked:
            continue
        # Find the class that would answer r[i].
        impl = None
        for k in type(codec).__mro__:
            if "open" in k.__dict__:
                impl = k
                break
        # The reader class is not known without opening a file, so this
        # checks the codec's declared reader where it exposes one.
        reader_cls = getattr(codec, "reader_class", None)
        if reader_cls is None:
            continue
        if not any("__getitem__" in k.__dict__
                   for k in reader_cls.__mro__ if k is not Reader):
            offenders.append(name)
    assert not offenders, (
        f"these declare chunked=True but inherit the walking "
        f"__getitem__: {offenders}")


def test_eer_no_longer_claims_cheap_random_access():
    """The claim this file exists because of.

    EER frames are independent, so real random access is buildable --
    it is simply not built, which the manifest now records as a gap
    rather than as done.
    """
    assert oc.get_codec("eer").chunked is False


def test_gif_offers_indexing_without_claiming_it_is_cheap():
    """The two flags disagreeing is not automatically a bug.

    GIF frames carry disposal state, so frame N genuinely requires the
    ones before it. Indexing is still offered, and saying so through
    is_chunked while chunked stays False is exactly right.
    """
    assert oc.get_codec("gif").chunked is False
    # Asked of an instance: the reader that answers r[i] for GIF is a
    # Cython type, and the flag lives on the wrapper beside it.
    img = (np.arange(32 * 32, dtype="u1") % 251).reshape(32, 32)
    blob = oc.get_codec("gif").encode(img)
    with oc.get_codec("gif").open(blob) as r:
        assert getattr(r, "is_chunked", False) is True, (
            "GIF offers indexing; only its cost is the caveat")


@pytest.mark.parametrize("name", ["mrc", "oir", "vsi"])
def test_indexing_is_flat_across_frames(name, tmp_path):
    """The last frame must cost about what the first does.

    Timing in a test is usually a bad idea, so the threshold is loose
    on purpose: this is not measuring speed, it is separating O(1)
    from walking the whole file, which on these samples differ by two
    orders of magnitude.
    """
    import glob
    import pathlib

    corpus = pathlib.Path(__file__).resolve().parent.parent / ".test_data"
    codec = oc.get_codec(name)

    if name == "mrc":
        vol = (np.arange(24 * 64 * 64, dtype="i2") % 3000).reshape(24, 64, 64)
        path = tmp_path / "v.mrc"
        oc.write_mrc(str(path), vol)
    else:
        hits = [h for e in (codec.file_extensions or ())
                for h in sorted(glob.glob(f"{corpus}/**/*{e}", recursive=True))
                if "/._" not in h]
        if not hits:
            pytest.skip(f"no {name} sample in the corpus")
        path = pathlib.Path(hits[0])

    with codec.open(str(path)) as r:
        n = r.n_frames or 1
        if n < 8:
            pytest.skip(f"{name} sample has only {n} frames")

    def best(idx):
        codec.open(str(path))[idx]
        b = 1e9
        for _ in range(3):
            s = time.perf_counter()
            codec.open(str(path))[idx]
            b = min(b, time.perf_counter() - s)
        return b

    first, last = best(0), best(n - 1)
    assert last < max(first * 8, first + 0.05), (
        f"{name}: frame {n - 1} took {last * 1e3:.1f} ms against "
        f"{first * 1e3:.1f} ms for frame 0, which is walking, not "
        f"random access")
