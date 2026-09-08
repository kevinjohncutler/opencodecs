"""The capability manifest's own guards.

capabilities.toml only means anything if the checks around it work.
`verify` passing on a clean tree proves nothing about whether it would
CATCH a wrong entry -- a guard that silently stopped firing would look
exactly like a guard that had nothing to complain about. So these tests
feed it deliberately broken manifests and require it to object, and
they pin the two derivation rules that were wrong before: a docstring
mentioning a class is not an implementation, and reaching storage
through a helper still counts as reaching it.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "ci" / "check_capabilities.py"
MANIFEST = ROOT / "capabilities.toml"


def _run(*args):
    return subprocess.run([sys.executable, str(SCRIPT), *args],
                          capture_output=True, text=True, cwd=ROOT)


def _block_bounds(text: str, name: str) -> tuple[int, int]:
    """Byte range of one codec's block, handling the last one.

    The final entry has no following `[[codec]]` marker, which is the
    kind of detail that makes a test fail for a reason unrelated to
    what it is testing.
    """
    at = text.index(f'name = "{name}"')
    start = text.rindex("[[codec]]", 0, at)
    nxt = text.find("[[codec]]", at)
    return start, (nxt if nxt != -1 else len(text))


@pytest.fixture
def restore_manifest():
    """Edit the real manifest, then put it back.

    The checks re-derive from the working tree, so they have to run
    against the real file in the real repo; a copy in tmp_path would
    not be found. Saving and restoring the bytes keeps that safe.
    """
    original = MANIFEST.read_bytes()
    try:
        yield MANIFEST
    finally:
        MANIFEST.write_bytes(original)


def test_verify_passes_on_the_committed_manifest():
    r = _run("verify")
    assert r.returncode == 0, r.stdout + r.stderr


def test_verify_catches_a_flipped_boolean(restore_manifest):
    s = MANIFEST.read_text()
    # tiff genuinely has pyramid; claiming otherwise must be caught.
    block, end = _block_bounds(s, "tiff")
    patched = (s[:block]
               + s[block:end].replace("pyramid = true", "pyramid = false", 1)
               + s[end:])
    assert patched != s
    MANIFEST.write_text(patched)
    r = _run("verify")
    assert r.returncode != 0
    assert "DRIFTED" in r.stdout and "tiff.pyramid" in r.stdout


def test_verify_catches_a_gap_that_is_already_built(restore_manifest):
    """Claiming credit for finished work is as wrong as hiding work.

    This is not hypothetical: three entries in this file listed `http`
    as a gap for readers that had served range requests all along, and
    this check is what caught them.
    """
    s = MANIFEST.read_text()
    block, end = _block_bounds(s, "tiff")
    seg = s[block:end].replace('feasible = "done"',
                               'feasible = "gap"\ngaps = ["pyramid"]', 1)
    MANIFEST.write_text(s[:block] + seg + s[end:])
    r = _run("verify")
    assert r.returncode != 0
    assert "lists pyramid as a gap, but the code already has it" in r.stdout


def test_verify_catches_a_verdict_that_contradicts_its_gaps(restore_manifest):
    s = MANIFEST.read_text()
    block, end = _block_bounds(s, "zstd")
    seg = s[block:end]
    seg = seg[:seg.index("gaps = [")] + seg[seg.index("note = "):]
    MANIFEST.write_text(s[:block] + seg + s[end:])
    r = _run("verify")
    assert r.returncode != 0
    assert "feasible=gap but nothing listed" in r.stdout


def test_verify_catches_a_judgment_with_no_reason(restore_manifest):
    s = MANIFEST.read_text()
    block, end = _block_bounds(s, "qoi")
    seg = s[block:end]
    seg = seg[:seg.index("note = ")]
    MANIFEST.write_text(s[:block] + seg + s[end:])
    r = _run("verify")
    assert r.returncode != 0
    assert "with no reason" in r.stdout


def test_verify_catches_an_invented_capability(restore_manifest):
    s = MANIFEST.read_text()
    s2 = s.replace('gaps = ["streaming_decode"]',
                   'gaps = ["streaming_decode", "telepathy"]', 1)
    assert s2 != s
    MANIFEST.write_text(s2)
    r = _run("verify")
    assert r.returncode != 0
    assert "is not a capability" in r.stdout


def test_verify_catches_a_removed_codec(restore_manifest):
    s = MANIFEST.read_text()
    start, end = _block_bounds(s, "qoi")
    MANIFEST.write_text(s[:start] + s[end:])
    r = _run("verify")
    assert r.returncode != 0
    assert "UNRECORDED" in r.stdout and "qoi" in r.stdout


def test_sync_preserves_the_judgments(restore_manifest, tmp_path):
    """sync rewrites the derived fields and must not eat the rest.

    feasible, gaps and notes are the part no script can regenerate;
    losing them to a routine re-sync would be unrecoverable without
    going to git.
    """
    before = MANIFEST.read_text()
    r = _run("sync")
    assert r.returncode == 0, r.stdout + r.stderr
    assert MANIFEST.read_text() == before, (
        "sync changed a manifest that verify says is already correct")


def test_report_counts_built_and_feasible_separately():
    r = _run("report")
    assert r.returncode == 0
    assert "+can" in r.stdout
    assert "feasible work left" in r.stdout
    # Every codec is judged; "unassessed" should no longer appear.
    assert "not yet judged" not in r.stdout


# ---- the derivation rules that were wrong before -----


def _derive():
    sys.path.insert(0, str(ROOT / "ci"))
    import check_capabilities
    return check_capabilities.derive()


def test_a_docstring_mention_is_not_an_implementation():
    """jpeg2k's decoder names Jpeg2kPyramidReader in prose.

    The pyramid pattern used to match any occurrence of the name, so
    that sentence alone marked the codec as having a pyramid backend.
    jpeg2k does have one now, so the guard is jpegls: it has no pyramid
    module, and must not acquire one by being adjacent in the tree.
    """
    d = _derive()
    assert d["jpegls"]["pyramid"] is False
    assert d["jpeg2k"]["pyramid"] is True


def test_http_means_range_backed_not_merely_url_openable():
    """jxl opens from a URL with one whole-file GET, which is not this.

    Its own docstring says libjxl wants the whole stream. Recording it
    as `http` claimed a capability the code denies in the same file.
    """
    d = _derive()
    assert d["jxl"]["http"] is False
    assert d["tiff"]["http"] is True


def test_reaching_storage_through_a_helper_still_counts():
    """dicom, mrc and nrrd open through core.io.coerce_data_source.

    That helper turns a URL into an HTTPDataSource itself, so these
    readers serve range requests without naming the class. Attributing
    HTTP by call site alone recorded all three as false.
    """
    d = _derive()
    for name in ("dicom", "mrc", "nrrd"):
        assert d[name]["http"] is True, f"{name} lost its HTTP path"
        assert d[name]["range_reads"] is True
