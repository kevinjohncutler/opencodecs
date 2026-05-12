"""Loader + registry for opencodecs's native Cython codec extensions.

Each native codec is one ``.pyx`` extension built into ``opencodecs/codecs/``
(e.g. ``_jxl``, ``_qoi``, ``_zstd``, ``_png``). At import time we:

  1. Locate each extension's ``.so`` (matching the current Python platform
     tag; multiple platform-tagged ``.so`` files can coexist on a NAS).
  2. If the source is on a network mount (smbfs / nfs / afpfs) — which
     macOS Sequoia handles badly with dyld signature checks — shadow-copy
     the ``.so`` to a per-user cache and ``dlopen`` from there. Same
     pattern used by ``edt`` / ``ncolor`` / ``hiprpy``.
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
    "_pcodec",
    "_deflate",
    "_jpeg",
    "_webp",
    "_jpeg2k",
    "_avif",
    "_heif",
    "_png",
    "_bitshuffle",
    "_tiff",
    "_ndtiff",
    "_bytetools",
    "_mozjpeg",     # optional: only present when MozJPEG was found at build
)


def _user_cache_dir() -> Path:
    from platformdirs import user_cache_dir
    return Path(user_cache_dir("opencodecs"))


_CACHE_ROOT = _user_cache_dir() / "lib"


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
    for suffix in importlib.machinery.EXTENSION_SUFFIXES:
        candidate = _THIS_DIR / f"{basename}{suffix}"
        if candidate.exists():
            return candidate
    return None  # pragma: no cover - extension always present in built env


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
# extensions still being built) are silent — registry will simply skip
# registering codecs whose backing extension didn't load.
_loaded: dict = {}
for _name in _EXTENSIONS:
    _loaded[_name] = _load_extension(_name)

# Convenient direct attribute access (back-compat).
_jxl = _loaded.get("_jxl")

# Now run codec registrations (after all extensions are in sys.modules).
from . import _registry  # noqa: F401, E402

__all__ = ["_jxl"]
