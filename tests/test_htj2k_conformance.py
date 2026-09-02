"""Score our HTJ2K decoder against the JPEG committee's conformance set.

This is a tracking test, not a pass/fail one. We currently decode 7 of
the 42 HT bitstreams; the rest are refused by OpenJPH, almost all of them
for a single reason (it implements one quality layer only). The point is
to pin that number so it cannot drift silently in either direction:

* If a file that used to decode stops decoding, that is our regression.
* If a file that used to be refused starts decoding, an upstream release
  gained a feature and this file should be updated to claim it.

See docs/htj2k_conformance.md for the measured breakdown and what it
would take to close the gap.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import pytest

from opencodecs import get_codec

# OpenJPH is discovered at build time from the system, not vendored or
# built from source, so the extension is simply absent wherever the
# library is not installed (CI included, as of this writing). Every test
# here needs a real backend, not the stub the codec registers with.
try:
    from opencodecs._htj2k_codec import _HAVE_BACKEND as _HAVE_OPENJPH
except Exception:                                     # noqa: BLE001
    _HAVE_OPENJPH = False

pytestmark = pytest.mark.skipif(
    not _HAVE_OPENJPH,
    reason="OpenJPH backend not built (system libopenjph not found)")

CONFORMANCE = (Path(__file__).resolve().parent.parent
               / ".test_data" / "htj2k" / "conformance")

# Files that decode without complaint today.
DECODES = {
    "ds0_ht_01_b11.j2k",
    "ds0_ht_09_b11.j2k",
    "ds0_ht_11_b10.j2k",
    "ds0_ht_12_b11.j2k",
    "ds0_ht_14_b11.j2k",
    "ds1_ht_06_b11.j2k",
    "hifi_ht1_02.j2k",
}

# Decodes, but OpenJPH reports it skipped a marker segment it does not
# implement, so the pixels do not match the reference. We raise rather
# than hand that back; see OpenJphUnsupportedFeature.
UNSUPPORTED_MARKER = {"ds1_ht_04_b9.j2k"}

EXPECTED_TOTAL = 42


def _files():
    if not CONFORMANCE.is_dir():
        pytest.skip("conformance corpus missing (corpus.py fetch "
                    "htj2k_wg1_conformance)")
    found = sorted(p for p in CONFORMANCE.glob("*.j2k")
                   if not p.name.startswith("._"))
    if not found:
        pytest.skip("conformance corpus empty")
    return found


def test_corpus_is_complete():
    files = _files()
    assert len(files) == EXPECTED_TOTAL, (
        f"expected {EXPECTED_TOTAL} conformance codestreams, found "
        f"{len(files)}; the manifest and this test disagree")


def test_decode_score_is_unchanged():
    """The set of files we can decode must be exactly the recorded set.

    Deliberately an equality check rather than ``>=``. An improvement is
    as much a reason to look as a regression: it means a dependency
    changed under us and the recorded score is now a lie.
    """
    from opencodecs.codecs._openjph import OpenJphUnsupportedFeature

    codec = get_codec("htj2k")
    decoded, unsupported, refused = set(), set(), set()
    for path in _files():
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                codec.decode(path.read_bytes())
            decoded.add(path.name)
        except OpenJphUnsupportedFeature:
            unsupported.add(path.name)
        except Exception:                             # noqa: BLE001
            refused.add(path.name)

    assert decoded == DECODES, (
        f"decodable set changed.\n"
        f"  newly decoding: {sorted(decoded - DECODES)}\n"
        f"  no longer decoding: {sorted(DECODES - decoded)}\n"
        f"If this is an upstream improvement, update DECODES and "
        f"docs/htj2k_conformance.md.")
    assert unsupported == UNSUPPORTED_MARKER
    assert len(refused) == EXPECTED_TOTAL - len(DECODES) - len(UNSUPPORTED_MARKER)


def test_multi_layer_refusal_explains_itself():
    """A refusal has to say why, or a user cannot act on it.

    OpenJPH's own handler prints the reason and throws a generic
    exception, so this used to surface as "ojph error (rc=2)". The shim
    installs a collector to keep the text.
    """
    path = CONFORMANCE / "ds1_ht_01_b11.j2k"
    if not path.is_file():
        pytest.skip("conformance file missing")
    with pytest.raises(Exception) as excinfo:
        get_codec("htj2k").decode(path.read_bytes())
    msg = str(excinfo.value)
    assert "quality layer" in msg, f"unhelpful error message: {msg}"
    assert "rc=" in msg


def test_unsupported_marker_raises_rather_than_returning_wrong_pixels():
    """Silent wrong output is the failure mode worth engineering against.

    This file decodes to something 954 levels away from the committee's
    reference because OpenJPH skips a QCD marker inside a tile. It must
    raise by default, and still be reachable for a caller who knows.
    """
    from opencodecs.codecs._openjph import OpenJphUnsupportedFeature
    import opencodecs.codecs._openjph as ojph

    path = CONFORMANCE / "ds1_ht_04_b9.j2k"
    if not path.is_file():
        pytest.skip("conformance file missing")
    blob = path.read_bytes()

    with pytest.raises(OpenJphUnsupportedFeature, match="not supported"):
        get_codec("htj2k").decode(blob)

    arr = ojph.decode(blob, ignore_unsupported=True)
    assert arr.shape == (1024, 1024)


def test_library_does_not_write_to_stdout(capfd):
    """OpenJPH prints warnings to stdout; a library must not do that.

    Left alone it corrupts whatever the calling program was writing
    there, which is how this was noticed: it broke a JSON parse.
    """
    import opencodecs.codecs._openjph as ojph

    path = CONFORMANCE / "ds1_ht_04_b9.j2k"
    if not path.is_file():
        pytest.skip("conformance file missing")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ojph.decode(path.read_bytes(), ignore_unsupported=True)
    out, _ = capfd.readouterr()
    assert "ojph" not in out, f"OpenJPH leaked to stdout: {out[:200]!r}"
