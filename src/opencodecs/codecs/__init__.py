"""Loader + registry for opencodecs's native Cython codec extensions.

Each native codec is one ``.pyx`` extension built into ``opencodecs/codecs/``
(e.g. ``_jxl``, ``_qoi``, ``_zstd``, ``_png``). At import time we:

  1. Locate each extension's ``.so`` (matching the current Python platform
     tag; multiple platform-tagged ``.so`` files can coexist on a NAS).
  2. If the source is on a network mount (smbfs / nfs / afpfs) — which
     macOS Sequoia handles badly with dyld signature checks — shadow-copy
     the ``.so`` to a per-user cache and ``dlopen`` from there.
  3. Add the loaded module to ``sys.modules`` under the FQN so subsequent
     ``from opencodecs.codecs._foo import ...`` works.

After all extensions are loaded, ``_registry.py`` runs and each codec's
``register_codec(...)`` populates the global format registry — making
the codec discoverable via ``opencodecs.read``, ``opencodecs.list_codecs``,
etc.

There is **no** runtime delegation to other libraries: every codec we
expose has a native implementation in this package.
"""

from __future__ import annotations

import importlib.machinery
import os
import shutil
import subprocess
import sys
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent

# Native Cython extensions shipped with opencodecs. Add new entries here
# when implementing a new native codec. Each must have a corresponding
# `_<name>.pyx` source file and a registration in `_registry.py`.
_EXTENSIONS = (
    "_jxl",
    "_qoi",
    "_zstd",
    "_lz4",
    "_brotli",
    "_blosc2",
    "_b2nd",
    "_aec",
    "_lerc",
    "_zfp",
    "_sz3",
    "_sperr",       # SPERR wavelet-based error-bounded lossy (optional)
    "_pcodec",
    "_deflate",
    "_jpeg",
    "_webp",
    "_jpeg2k",
    "_avif",
    "_heif",
    "_png",
    "_bitshuffle",
    "_rcomp",        # cfitsio ricecomp (vendored) — replaces _rcomp_codec.py
    "_rgbe",         # Radiance HDR (.hdr) image format (vendored, no deps)
    "_hcomp",        # H-compress FITS tile decode (vendored cfitsio source)
    "_ultrahdr",     # Ultra HDR (ISO 21496 gainmap JPEG) via libultrahdr
    "_isal",         # Intel ISA-L deflate (x86_64-only; fastest deflate)
    "_bmp",          # Cython BMP encoder — replaces the pure-Python encode path
    "_tiff",
    "_ndtiff",
    "_bytetools",
    "_mozjpeg",     # optional: only present when MozJPEG was found at build
    "_bcdec",       # BC1-7 / DXT / BPTC texture decoder (vendored, no deps)
    "_charls",      # JPEG-LS (optional; built when libcharls is on system)
    "_openjph",     # HTJ2K / JPEG-2000 Part-15 (optional; needs OpenJPH)
    "_eer",         # Thermo Fisher EER cryo-EM event-list decoder (vendored)
    "_brunsli",     # lossless JPEG transcoder (~20% smaller)
    "_gif",         # GIF via giflib
    "_snappy",      # Snappy block compression via libsnappy
)


def _user_cache_dir() -> Path:
    from platformdirs import user_cache_dir
    return Path(user_cache_dir("opencodecs"))


_CACHE_ROOT = _user_cache_dir() / "lib"


def _build_lib_search_paths() -> list[Path]:
    """Return every ``build/lib.*/opencodecs/codecs/`` directory we
    should consult when ``src/opencodecs/codecs/`` is missing an
    extension's ``.so`` (the SMB-stuck-inplace-copy failure mode).

    Looks for ``build/`` next to ``src/`` — the standard ``setup.py``
    layout. There may be multiple ``lib.<platform>-cpython-<ver>``
    subdirs (cross-Python development); we include all of them, and
    ``_find_so`` picks the newest matching artifact by mtime.
    """
    # _THIS_DIR is .../src/opencodecs/codecs/ → repo root is 3 dirs up.
    repo_root = _THIS_DIR.parent.parent.parent
    build_root = repo_root / "build"
    if not build_root.is_dir():
        return []
    return [
        d / "opencodecs" / "codecs"
        for d in sorted(build_root.glob("lib.*"))
        if (d / "opencodecs" / "codecs").is_dir()
    ]


# Search the in-place install dir first (the normal case), then fall
# back to ``build/lib.*/opencodecs/codecs/`` for the cases where
# ``setup.py build_ext --inplace`` silently failed to overwrite an
# existing ``.so`` on an SMB mount. ``_find_so`` ties multiple matches
# by mtime so a fresh build always wins.
_SO_SEARCH_PATHS: list[Path] = [_THIS_DIR, *_build_lib_search_paths()]


def _on_remote_mount(path: Path) -> bool:
    """True if `path` lives on a network filesystem dyld is hostile to."""
    if os.name == "nt":  # pragma: no cover - Windows-only branch
        return path.is_absolute() and path.anchor.startswith("\\\\")
    if sys.platform != "darwin":  # pragma: no cover - Linux test path
        # On Linux NFS works fine for dlopen; only macOS smbfs is hostile.
        return False
    try:
        out = subprocess.check_output(["mount"], text=True)
    except Exception:  # pragma: no cover - mount command never fails on dev mac
        return False
    abs_path = str(path.resolve())
    for line in out.splitlines():
        if " on " not in line or " (" not in line:  # pragma: no cover - malformed mount line
            continue
        mount_point, opts = line.split(" on ", 1)[1].split(" (", 1)
        if abs_path.startswith(mount_point.rstrip()) and (
            "smbfs" in opts or "nfs" in opts or "afpfs" in opts
        ):
            return True
    return False  # pragma: no cover - dev env mounts everything from SMB


def _find_so(basename: str) -> Path | None:
    """Locate the extension's ``.so`` (or ``.pyd`` on Windows).

    Checks two directories in this order:

    1. ``src/opencodecs/codecs/`` — the in-place install target that
       ``setup.py build_ext --inplace`` writes to. Normal users see
       extensions here.
    2. ``build/lib.<platform>/opencodecs/codecs/`` — the intermediate
       ``setup.py`` output directory. The "copy to src/" step that
       ``--inplace`` does last is silently NO-OPed on SMB-mounted dev
       trees when a stale ``.so`` is held open by a different process
       (the failure mode we kept hitting). Falling back here makes the
       build robust to that: a fresh ``setup.py build_ext`` (even
       without ``--inplace``) leaves a loadable artifact on disk.

    When both locations have a copy we prefer whichever has the newer
    ``mtime`` — that's the just-built one. This is what makes
    ``build_ext --inplace`` followed by an in-process reimport pick up
    the new code even when the SMB inplace-copy silently failed.
    """
    candidates: list[Path] = []
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        for root in _SO_SEARCH_PATHS:
            p = root / f"{basename}{suffix}"
            if p.exists():
                candidates.append(p)
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]
    # Prefer the freshest artifact — handles the SMB-stuck-stale-src
    # case where build/ has a newer .so than src/.
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _local_cache_path(src: Path) -> Path:
    st = src.stat()
    return _CACHE_ROOT / f"{st.st_mtime_ns}_{st.st_size}" / src.name


def _copy_off_remote(src: Path, dst: Path) -> None:  # pragma: no cover - first-import-only
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    dst.chmod(0o755)
    if sys.platform == "darwin":
        # cp from an SMB mount inherits com.apple.quarantine; strip it.
        subprocess.run(
            ["xattr", "-d", "com.apple.quarantine", str(dst)],
            check=False, stderr=subprocess.DEVNULL,
        )


def _load_extension(basename: str):
    """Load opencodecs/codecs/<basename>.so, shadowing if on remote mount."""
    src = _find_so(basename)
    if src is None:  # pragma: no cover - extension always present in built env
        # Extension not built — skip silently. _registry.py will see the
        # missing module and skip registration of the corresponding codec.
        return None

    load_path = src
    if _on_remote_mount(src):
        local = _local_cache_path(src)
        if not local.exists():  # pragma: no cover - first-import-only branch
            _copy_off_remote(src, local)
        load_path = local

    fq_name = f"opencodecs.codecs.{basename}"
    spec = spec_from_file_location(fq_name, str(load_path))
    if spec is None or spec.loader is None:  # pragma: no cover - importlib invariant
        raise ImportError(f"failed to build spec for {load_path}")
    mod = module_from_spec(spec)
    spec.loader.exec_module(mod)
    sys.modules[fq_name] = mod
    return mod


# Eagerly load every shipped extension. Failures (missing .so files for
# extensions still being built, OR successfully-built .so files whose
# transitive shared-library deps aren't on the loader path — e.g. an
# editable install of opencodecs that ships ``_sz3.so`` but where the
# host doesn't have ``libSZ3c.so`` installed system-wide) are silent —
# registry will simply skip registering codecs whose backing extension
# didn't load.
_loaded: dict = {}
_load_failures: dict = {}
for _name in _EXTENSIONS:
    try:
        _loaded[_name] = _load_extension(_name)
    except ImportError as _exc:
        # Transitive dlopen failure (e.g. ``libSZ3c.so: cannot open
        # shared object file``). Honour the silent-skip contract so one
        # broken codec doesn't take down the rest of the package.
        _loaded[_name] = None
        _load_failures[_name] = _exc

# Convenient direct attribute access (back-compat).
_jxl = _loaded.get("_jxl")

# Now run codec registrations (after all extensions are in sys.modules).
from . import _registry  # noqa: F401, E402

__all__ = ["_jxl"]
