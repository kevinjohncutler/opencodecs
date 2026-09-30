"""Every place the package or its build chooses by operating system.

A branch on the operating system is a path that only one platform's CI
exercises, so each one has to earn its place: a capability that is
missing (probe it instead, ``hasattr(os, "pread")``), a policy of the
platform, or a measured difference with nothing better to key on. The
catalog below is that list, with the reason for each. A new branch fails
this test until it is either generalized or added here with its reason;
a catalog entry whose code is gone fails it too, so the list stays true.

Scope: src/opencodecs (Python, Cython, C) and setup.py. Probes of a
capability, a file or a tool are not branches and are not listed.
"""
from __future__ import annotations

import io
import pathlib
import re
import tokenize

ROOT = pathlib.Path(__file__).resolve().parents[1]

CATALOG = {
    "setup.py": {
        '_WINDOWS = sys.platform == "win32"':
            "Windows links import libraries and has no rpath; DLLs are found "
            "through os.add_dll_directory or bundled by delvewheel",
        'if sys.platform == "darwin":':
            "the per-user cache folder, where bench/build_codec_libs.sh installs",
        '_GNU_LD = sys.platform.startswith("linux")':
            "GNU ld's -Wl,--disable-new-dtags and -l:FILE for the libjxl link",
    },
    "src/opencodecs/core/io.py": {
        '_COPY_BEATS_READ_ALONE = sys.platform == "darwin"':
            "measured with one reader: one thread copies a file's bytes from a "
            "mapping faster than it reads them on macOS, and reads them faster "
            "on Linux and Windows",
        '_COPY_BEATS_READ_AMONG_OTHERS = sys.platform in ("darwin", "win32")':
            "measured with two to eight readers of one file: copying from a "
            "mapping wins on macOS and Windows, where reads of one file contend "
            "in the kernel, and reading wins on Linux",
    },
    "src/opencodecs/_ndtiff_writer.py": {
        'if sys.platform != "win32":':
            "NTFS zero-fills an ftruncate extension synchronously and the "
            "close-time truncate stalls, so the 4 GiB pre-extension is skipped",
    },
    "src/opencodecs/codecs/__init__.py": {
        'if os.name == "nt":':
            "a DLL loaded from a UNC share holds the shared file open",
        'if sys.platform != "darwin":':
            "only macOS Gatekeeper refuses code on a quarantined network mount",
        'if sys.platform == "darwin":':
            "a copy off such a mount inherits com.apple.quarantine",
    },
    "src/opencodecs/codecs/jpegxr_shim.c": {
        "#ifdef _WIN32":
            "jxrlib's headers use the Windows SDK's SAL annotations there",
    },
}

_C_MACROS = re.compile(r"^\s*#\s*(if|ifdef|ifndef|elif)\b.*\b(_WIN32|_WIN64|_MSC_VER|__APPLE__|__linux__)\b")
_PY_TESTS = (("sys", "platform"), ("os", "name"), ("platform", "system"))


def _python_branch_lines(text: str) -> set[str]:
    """Code lines (not comments or strings) that read the operating system."""
    lines = text.splitlines()
    every = list(tokenize.generate_tokens(io.StringIO(text).readline))
    comment_at = {t.start[0]: t.start[1] for t in every if t.type == tokenize.COMMENT}
    toks = [t for t in every if t.type in (tokenize.NAME, tokenize.OP)]
    found = set()
    for a, dot, b in zip(toks, toks[1:], toks[2:]):
        if dot.string == "." and (a.string, b.string) in _PY_TESTS:
            row = a.start[0]
            found.add(lines[row - 1][:comment_at.get(row)].strip())
    return found


def _sources():
    yield ROOT / "setup.py"
    pkg = ROOT / "src" / "opencodecs"
    for path in sorted(pkg.rglob("*")):
        if path.suffix in (".py", ".pyx", ".pxd"):
            yield path
        elif path.suffix in (".c", ".h", ".cpp"):
            # Cython's output sits beside its .pyx and is not ours to catalog.
            if not path.with_suffix(".pyx").exists():
                yield path


def _branches() -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for path in _sources():
        text = path.read_text(encoding="utf-8", errors="replace")
        if path.suffix in (".py", ".pyx", ".pxd"):
            try:
                lines = _python_branch_lines(text)
            except (tokenize.TokenError, IndentationError, SyntaxError):
                # Cython syntax tokenizes as Python nearly always; fall
                # back to plain matching when it does not.
                lines = {ln.strip() for ln in text.splitlines()
                         if re.search(r"\b(sys\.platform|os\.name|platform\.system)\b", ln)
                         and not ln.lstrip().startswith("#")}
        else:
            lines = {ln.strip() for ln in text.splitlines() if _C_MACROS.search(ln)}
        if lines:
            out[path.relative_to(ROOT).as_posix()] = lines
    return out


def test_every_platform_branch_is_cataloged_with_a_reason():
    found = _branches()
    uncataloged = {f: sorted(ls - set(CATALOG.get(f, {}))) for f, ls in found.items()}
    uncataloged = {f: ls for f, ls in uncataloged.items() if ls}
    assert not uncataloged, (
        "new operating-system branches; generalize them (probe the capability) "
        f"or add them to CATALOG with the reason: {uncataloged}")


def test_every_cataloged_branch_still_exists():
    found = _branches()
    stale = {f: sorted(set(entries) - found.get(f, set()))
             for f, entries in CATALOG.items()}
    stale = {f: ls for f, ls in stale.items() if ls}
    assert not stale, f"catalog entries whose code is gone or changed: {stale}"
