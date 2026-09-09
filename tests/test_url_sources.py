"""Every codec can be handed an http(s) URL.

``read_src`` is what the codecs that want a complete codestream use to
get their bytes, and a URL fell through it to ``Path(src).read_bytes()``
and failed as a missing file. So about thirty codecs could not open one
at all, which is a usability hole rather than a capability one -- and
one that no per-codec test would have found, because each of them looks
correct on a path.

The URL is fetched in full, deliberately. These formats want the whole
codestream, so a range-reading source would move the same bytes while
putting an `http` tick in capabilities.toml against formats that cannot
use it. The formats that CAN read at offsets take a data source and
never come through here; their byte savings are measured in
test_http_byte_savings.py.
"""

from __future__ import annotations

import numpy as np
import pytest

import opencodecs as oc

from _range_http_server import range_http_server

GRAY = (np.arange(64 * 96, dtype="u1") % 251).reshape(64, 96)
RGB = np.stack([GRAY, GRAY[::-1], GRAY[:, ::-1]], axis=-1)


def _candidates():
    """Codecs that round-trip an image and read through read_src."""
    out = []
    for info in oc.list_codecs():
        name = info["name"]
        try:
            codec = oc.get_codec(name)
        except Exception:                                # noqa: BLE001
            continue
        if not (getattr(codec, "can_encode", False)
                and getattr(codec, "can_decode", False)):
            continue
        out.append(name)
    return sorted(out)


@pytest.mark.parametrize("name", _candidates())
def test_a_url_decodes_to_what_the_path_does(name, tmp_path):
    codec = oc.get_codec(name)
    blob = local = None
    for arr in (GRAY, RGB):
        try:
            candidate = codec.encode(arr)
            # Both halves have to work: several codecs encode a plain
            # array happily and need shape/dtype/itemsize back on
            # decode, and taking `blob` from a successful encode alone
            # left `local` unbound and the failure looking like a URL
            # bug.
            local = codec.decode(candidate)
        except Exception:                                # noqa: BLE001
            continue
        blob = candidate
        break
    if blob is None:
        pytest.skip(f"{name} does not round-trip a plain image unaided")

    p = tmp_path / f"sample_{name}.bin"
    p.write_bytes(blob)
    with range_http_server(str(tmp_path)) as (base, _):
        remote = codec.decode(f"{base}/{p.name}")

    if isinstance(local, np.ndarray):
        assert np.array_equal(np.asarray(remote), local), name
    else:
        assert bytes(remote) == bytes(local), name


def test_a_url_that_is_not_there_reports_itself(tmp_path):
    """A 404 must not look like a decode failure."""
    with range_http_server(str(tmp_path)) as (base, _):
        with pytest.raises(Exception) as exc:
            oc.get_codec("png").decode(f"{base}/nope.png")
    assert "nope.png" in str(exc.value) or "404" in str(exc.value)


def test_a_path_that_looks_like_a_url_is_still_a_path(tmp_path):
    """Only http(s) is special-cased; a file:// or bare path is not."""
    blob = oc.get_codec("png").encode(GRAY)
    p = tmp_path / "httpish.png"
    p.write_bytes(blob)
    assert np.array_equal(oc.get_codec("png").decode(str(p)), GRAY)
