"""opencodecs build script.

Streaming JPEG XL prototype. Single Cython extension (`opencodecs.codecs._jxl`)
that links against libjxl + libjxl_threads.

libjxl resolution, in priority order:

1. **vendored** at ``<repo>/vendor/libjxl/`` — auto-detected if present. This
   is the recommended path on Linux: distro libjxl packages are typically
   ~0.5-0.7x slower than a tuned Release build. Run ``bench/build_libjxl.sh``
   once to populate ``vendor/libjxl/``; this script then picks it up
   automatically and emits an RPATH so the .so loads its deps from there.

2. **imagecodecs bundle** when ``OPENCODECS_USE_IMAGECODECS_LIBJXL=1``. Last-
   resort Linux fast path that piggybacks on imagecodecs's bundled libjxl —
   only useful if you already have imagecodecs installed and don't want to
   build libjxl yourself. Couples our build to imagecodecs's wheel layout;
   prefer (1).

3. **explicit prefix** via ``OPENCODECS_JXL_PREFIX=/some/prefix``.

4. **system probes**: Homebrew (/opt/homebrew, /usr/local), /usr, conda
   ($CONDA_PREFIX). First with ``jxl/types.h`` wins. On Mac this is fine
   (Homebrew libjxl == imagecodecs's libjxl). On Linux this is the slow
   path mentioned above.
"""

from __future__ import annotations

import glob
import os
import re
import sys
import sysconfig
from pathlib import Path

import subprocess

import numpy
from Cython.Build import cythonize
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext as _build_ext

HERE = Path(__file__).resolve().parent


def _find_imagecodecs_libs() -> Path | None:
    """Return path to imagecodecs.libs/ if installed, else None."""
    try:
        import imagecodecs
    except ImportError:
        return None
    pkg_dir = Path(imagecodecs.__file__).resolve().parent
    libs = pkg_dir.parent / "imagecodecs.libs"
    if not libs.is_dir():
        return None
    if not list(libs.glob("libjxl-*.so.*")):
        return None
    return libs


def _imagecodecs_jxl_filenames(libs_dir: Path) -> dict[str, str] | None:
    """Map standard SONAME -> bundled hashed filename in imagecodecs.libs/.

    Returns e.g. {"libjxl.so.0.11" -> "libjxl-bcde3c36.so.0.11.2"}, or None
    if any required lib is missing.
    """
    out: dict[str, str] = {}
    # Each prefix is unique (libjxl-, libjxl_threads-, libjxl_cms-) so we use
    # a strict regex match rather than glob — globs would cross-match.
    patterns = [
        ("libjxl.so.0.11",         re.compile(r"^libjxl-[0-9a-f]+\.so\.\d+\.\d+\.\d+$")),
        ("libjxl_threads.so.0.11", re.compile(r"^libjxl_threads-[0-9a-f]+\.so\.\d+\.\d+\.\d+$")),
        ("libjxl_cms.so.0.11",     re.compile(r"^libjxl_cms-[0-9a-f]+\.so\.\d+\.\d+\.\d+$")),
    ]
    names = sorted(p.name for p in libs_dir.iterdir())
    for std_name, regex in patterns:
        matches = [n for n in names if regex.match(n)]
        if not matches:
            return None
        out[std_name] = matches[0]
    return out


def _user_cache_libjxl() -> Path:
    """Per-user cache install location matching bench/build_libjxl.sh's
    default. Off the source tree so a NAS-mounted repo doesn't trip
    macOS Gatekeeper at runtime when our .so dlopens libjxl."""
    if sys.platform == "darwin":
        return Path.home() / "Library/Caches/opencodecs/libjxl"
    if sys.platform.startswith("linux"):
        xdg = os.environ.get("XDG_CACHE_HOME")
        base = Path(xdg) if xdg else Path.home() / ".cache"
        return base / "opencodecs" / "libjxl"
    if sys.platform == "win32":
        return Path.home() / "AppData/Local/opencodecs/libjxl"
    return Path.home() / ".opencodecs" / "libjxl"


def _candidate_prefixes() -> list[Path]:
    """Ordered candidate prefixes — first one with jxl/types.h wins."""
    prefixes: list[Path] = []
    # 1. user-specified (highest priority)
    env = os.environ.get("OPENCODECS_JXL_PREFIX")
    if env:
        prefixes.append(Path(env))
    # 2. per-user cache install (matches bench/build_libjxl.sh default).
    #    On a NAS-mounted source tree, this is the location that avoids
    #    Gatekeeper hangs when dyld loads our libjxl.
    prefixes.append(_user_cache_libjxl())
    # 3. in-tree vendored — only sensible when the repo is on local disk
    #    (e.g., for wheel builds that bundle the libs in vendor/).
    prefixes.append(HERE / "vendor" / "libjxl")
    # 4. conda env. On Windows, conda-forge installs headers at
    # <env>/Library/include and libs at <env>/Library/lib (the "Library"
    # subdir mirrors a Unix prefix layout). On Mac/Linux the env IS the
    # prefix. Probe both forms so this branch works on every platform.
    conda = os.environ.get("CONDA_PREFIX")
    if conda:
        if sys.platform == "win32":
            prefixes.append(Path(conda) / "Library")
        prefixes.append(Path(conda))
    # 5. system
    if sys.platform == "darwin":
        prefixes += [Path("/opt/homebrew"), Path("/usr/local")]
    elif sys.platform.startswith("linux"):
        prefixes += [Path("/usr"), Path("/usr/local")]
    elif sys.platform == "win32":
        # On Windows we expect the user to set OPENCODECS_JXL_PREFIX explicitly.
        pass
    # de-dup while preserving order
    seen: set[Path] = set()
    out: list[Path] = []
    for p in prefixes:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _find_libjxl() -> tuple[Path | None, list[str], list[str]]:
    """Return (chosen_prefix, include_dirs, library_dirs).

    chosen_prefix is None when nothing matched (let linker complain later).
    """
    for prefix in _candidate_prefixes():
        inc = prefix / "include"
        if not (inc / "jxl" / "types.h").is_file():
            continue
        include_dirs = [str(inc)]
        library_dirs: list[str] = []
        lib = prefix / "lib"
        if lib.is_dir():
            library_dirs.append(str(lib))
        lib64 = prefix / "lib64"
        if lib64.is_dir():
            library_dirs.append(str(lib64))
        # Linux multilib dir (Debian/Ubuntu)
        multi = prefix / "lib" / "x86_64-linux-gnu"
        if multi.is_dir():
            library_dirs.append(str(multi))
        return prefix, include_dirs, library_dirs
    return None, [], []


chosen_prefix, include_dirs, library_dirs = _find_libjxl()

SRC = HERE / "src"
PKG_CODECS = SRC / "opencodecs" / "codecs"
VENDOR_LIBJXL = HERE / "vendor" / "libjxl"
USER_CACHE_LIBJXL = _user_cache_libjxl()

include_dirs.append(numpy.get_include())
include_dirs.append(str(PKG_CODECS))

extra_link_args: list[str] = []
libraries: list[str] = ["jxl", "jxl_threads"]

is_linux = sys.platform.startswith("linux")
is_darwin = sys.platform == "darwin"

# "Vendored" here means: we built libjxl ourselves, so we know its layout
# and need to set RPATH/install_name accordingly. The libs may live in
# the in-tree vendor/ dir OR in the per-user cache (which is the default
# for build_libjxl.sh — keeps libjxl off NAS so dyld doesn't trip on
# Gatekeeper at runtime).
using_vendored = chosen_prefix is not None and chosen_prefix.resolve() in {
    VENDOR_LIBJXL.resolve() if VENDOR_LIBJXL.is_dir() else VENDOR_LIBJXL,
    USER_CACHE_LIBJXL.resolve() if USER_CACHE_LIBJXL.is_dir() else USER_CACHE_LIBJXL,
}
chosen_lib_dir = (
    chosen_prefix / "lib" if (using_vendored and chosen_prefix is not None) else None
)

# Last-resort Linux opt-in: link against imagecodecs's bundled libjxl. The
# vendored path (`bench/build_libjxl.sh`) is the recommended approach now;
# this branch exists for users who already have imagecodecs installed and
# don't want to build libjxl themselves.
USE_IMAGECODECS_LIBJXL = (
    is_linux
    and not using_vendored
    and os.environ.get("OPENCODECS_USE_IMAGECODECS_LIBJXL", "").strip()
    in ("1", "true", "yes", "on")
)

if using_vendored:
    print(f"opencodecs: linking against vendored libjxl at {chosen_prefix}")
    if is_linux:
        # Use --disable-new-dtags so DT_RPATH (not DT_RUNPATH) is emitted —
        # required for libjxl's transitive deps (libjxl_cms, etc.) to be
        # found via our rpath. RUNPATH does not propagate to children.
        extra_link_args.append("-Wl,--disable-new-dtags")
    if is_darwin and chosen_lib_dir is not None:
        # On macOS, setuptools/sysconfig prepends -L/opt/homebrew/lib BEFORE
        # our library_dirs in the link line, so a plain -ljxl resolves to
        # Homebrew's libjxl instead of our vendored one. Pass the vendored
        # dylibs by absolute path via extra_link_args (which appears at the
        # END of the link line) so they take precedence and -ljxl is dropped.
        libraries = []
        for soname in ("libjxl.0.11.dylib", "libjxl_threads.0.11.dylib"):
            dylib = chosen_lib_dir / soname
            if dylib.exists():
                extra_link_args.append(str(dylib))
    for ldir in library_dirs:
        extra_link_args.append(f"-Wl,-rpath,{ldir}")
elif USE_IMAGECODECS_LIBJXL:
    ic_libs = _find_imagecodecs_libs()
    if ic_libs is None:
        raise RuntimeError(
            "OPENCODECS_USE_IMAGECODECS_LIBJXL=1 but imagecodecs.libs/ "
            "with bundled libjxl was not found. Install imagecodecs first "
            "(`pip install imagecodecs`)."
        )
    name_map = _imagecodecs_jxl_filenames(ic_libs)
    if name_map is None:
        raise RuntimeError(
            f"OPENCODECS_USE_IMAGECODECS_LIBJXL=1 but {ic_libs} doesn't "
            "contain libjxl/libjxl_threads/libjxl_cms — imagecodecs install "
            "may be incomplete or a different version."
        )
    # -l:filename links against the EXACT bundled file (with hashed SONAME).
    libraries = []
    for std_soname in ("libjxl.so.0.11", "libjxl_threads.so.0.11"):
        bundled = name_map[std_soname]
        extra_link_args.append(f"-l:{bundled}")
    library_dirs = [str(ic_libs)]
    extra_link_args.append("-Wl,--disable-new-dtags")
    extra_link_args.append(f"-Wl,-rpath,{ic_libs}")
    print(f"opencodecs: linking against imagecodecs bundled libjxl from {ic_libs}")
elif is_darwin and library_dirs:
    for ldir in library_dirs:
        extra_link_args.append(f"-Wl,-rpath,{ldir}")
elif is_linux and library_dirs:
    for ldir in library_dirs:
        extra_link_args.append(f"-Wl,-rpath,{ldir}")

define_macros: list[tuple[str, str | int]] = [
    ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
]

if sys.platform == "win32":
    define_macros.append(("WIN32", 1))


_PROBE_PREFIXES: list[Path] = [
    Path("/opt/homebrew"),
    Path("/opt/homebrew/opt/jpeg-turbo"),
    Path("/opt/homebrew/opt/openjpeg"),
    Path("/usr/local"),
    Path("/usr"),
]

# Per-user opencodecs cache prefixes. Some Tier 1 codec libraries
# (SZ3, pcodec) are not in Homebrew/apt, so we build them once into
# ~/Library/Caches/opencodecs/<lib>/ via bench/build_codec_libs.sh.
# Probe these so the generic header/lib search picks them up.
_OC_USER_CACHE = Path.home() / (
    "Library/Caches/opencodecs" if sys.platform == "darwin"
    else (
        os.environ.get("XDG_CACHE_HOME", str(Path.home() / ".cache"))
        + "/opencodecs"
    )
)
for _libdir in (
    "sz3", "pcodec", "sperr", "brunsli", "lerc", "zstd", "brotli", "giflib",
):
    _p = _OC_USER_CACHE / _libdir
    if (_p / "include").is_dir():
        _PROBE_PREFIXES.insert(0, _p)

# When _find_libjxl() picked up libjxl from a non-standard prefix
# (e.g. /cibw-jxl-prefix in the cibuildwheel manylinux container, or
# the per-user cache dir on a dev machine), make that prefix reachable
# to the generic _has_header() probe too — otherwise the jxl/types.h
# REQUIRED_HEADERS gate fails and _jxl auto-skips even though the lib
# IS present.
if chosen_prefix is not None and Path(chosen_prefix) not in _PROBE_PREFIXES:
    _PROBE_PREFIXES.insert(0, Path(chosen_prefix))

# macOS keeps system headers (zlib.h, etc.) under the active Xcode SDK
# instead of /usr/include/. Probe that too.
if sys.platform == "darwin":
    try:
        _sdk = subprocess.check_output(
            ["xcrun", "--show-sdk-path"], text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        if _sdk:
            _PROBE_PREFIXES.append(Path(_sdk) / "usr")
    except (FileNotFoundError, subprocess.CalledProcessError):
        pass

# conda env: add CONDA_PREFIX to the probe list. On POSIX the env IS
# the prefix (headers at $CONDA_PREFIX/include, libs at $CONDA_PREFIX/lib);
# on Windows conda-forge installs under <prefix>/Library/ to mimic that
# layout. Probe both forms so the same setup.py works in any conda env.
_conda = os.environ.get("CONDA_PREFIX")
if _conda:
    _PROBE_PREFIXES.insert(0, Path(_conda))  # check FIRST so conda wins
    if sys.platform == "win32":
        _PROBE_PREFIXES.insert(0, Path(_conda) / "Library")
# CI build-env sometimes calls setup.py with CONDA_PREFIX scrubbed (e.g.
# pip's PEP 517 isolated-build subprocess on Windows). Accept a fallback
# OPENCODECS_CODEC_LIBS_PREFIX env var that the workflow can set
# explicitly without relying on cibuildwheel propagating CONDA_PREFIX.
_libs_prefix = os.environ.get("OPENCODECS_CODEC_LIBS_PREFIX")
if _libs_prefix:
    _PROBE_PREFIXES.insert(0, Path(_libs_prefix))
    if sys.platform == "win32" and (Path(_libs_prefix) / "Library").is_dir():
        _PROBE_PREFIXES.insert(0, Path(_libs_prefix) / "Library")

# Windows: vcpkg installs to <root>/installed/x64-windows/. Add it too.
if sys.platform == "win32":
    _vcpkg_root = os.environ.get("VCPKG_ROOT")
    if _vcpkg_root:
        _PROBE_PREFIXES.append(Path(_vcpkg_root) / "installed" / "x64-windows")


def _multilib_dirs() -> list[Path]:
    """Linux multilib include dirs (e.g. /usr/include/x86_64-linux-gnu/)."""
    out = []
    base = Path("/usr/include")
    if base.is_dir():
        for child in base.iterdir():
            if child.is_dir() and "linux-gnu" in child.name:
                out.append(child)
    return out


_MULTILIB_INCLUDES = _multilib_dirs()


def _has_header(*relpaths: str) -> bool:
    """True if any of the candidate relative header paths exists under any
    standard prefix or Linux multilib include dir.
    """
    for rel in relpaths:
        for prefix in _PROBE_PREFIXES:
            if (prefix / "include" / rel).is_file():
                return True
        for ml in _MULTILIB_INCLUDES:
            if (ml / rel).is_file():
                return True
    return False


def _header_contains(relpath: str, *needles: str) -> bool:
    """True if the first matching header file contains all given strings.

    Used to gate against API-version mismatches (e.g. older libjpeg-turbo
    headers that lack the TurboJPEG v3 ``tj3Init`` symbol we use).
    """
    for prefix in _PROBE_PREFIXES:
        path = prefix / "include" / relpath
        if path.is_file():
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            return all(n in text for n in needles)
    for ml in _MULTILIB_INCLUDES:
        path = ml / relpath
        if path.is_file():
            try:
                text = path.read_text(errors="ignore")
            except OSError:
                continue
            return all(n in text for n in needles)
    return False


def _resolve_include_dirs(*relpaths: str) -> list[str]:
    """Return all parent include directories for any matched relpath."""
    out: list[str] = []
    for prefix in _PROBE_PREFIXES:
        if any((prefix / "include" / rel).is_file() for rel in relpaths):
            out.append(str(prefix / "include"))
    for ml in _MULTILIB_INCLUDES:
        if any((ml / rel).is_file() for rel in relpaths):
            out.append(str(ml))
    return out


def _lib_dirs_for_probes() -> list[str]:
    """Return the {prefix}/lib, /lib64, /Library/lib paths that exist.

    rhel/AlmaLinux's CMake honours `CMAKE_INSTALL_LIBDIR=lib64` by default
    so source builds inside the manylinux container land in `lib64`, not
    `lib`. Probe both, plus the conda-on-Windows `Library/lib` layout.
    """
    out: list[str] = []
    for p in _PROBE_PREFIXES:
        for sub in ("lib", "lib64", "Library/lib"):
            d = p / sub
            if d.is_dir():
                out.append(str(d))
    return out


def _libname(posix: str, windows: str | None = None) -> str:
    """Pick the right library base name for the current platform.

    conda-forge (Windows) typically prefixes shared library .lib import
    files with ``lib`` (e.g. ``libwebp.lib``) whereas POSIX systems use
    bare names (``libwebp.so`` → ``-lwebp``). Provide both forms so the
    same setup.py works on every host.
    """
    if sys.platform == "win32" and windows is not None:
        return windows
    return posix


def _maybe_build_ext_simple(
    name: str,
    source: str,
    prefixes: list[str],
    probe_header: str,
    libname: str,
    define_macros: list | None = None,
) -> list[Extension]:
    """Build an optional Cython extension when the named header +
    library are found in one of the given prefixes. Used for codecs
    that depend on a single system C library (CharLS, Brunsli, etc.).
    """
    for p in prefixes:
        prefix = Path(p)
        hdr = prefix / "include" / probe_header
        if not hdr.exists():
            continue
        # Find matching dylib/so.
        dlib = None
        for ext in ("dylib", "so", "so.0"):
            cand = prefix / "lib" / f"lib{libname}.{ext}"
            if cand.exists():
                dlib = cand
                break
        if dlib is None:
            for ext in ("so", "so.0"):
                cand = prefix / "lib" / "x86_64-linux-gnu" / f"lib{libname}.{ext}"
                if cand.exists():
                    dlib = cand
                    break
        if dlib is None:
            continue
        # Match the zlib-ng-compat pattern: pass dylib by abs path on
        # macOS so the SDK stub doesn't win the linker lookup.
        extra_link_args = (
            [str(dlib)] if sys.platform == "darwin" else []
        )
        libs = [] if sys.platform == "darwin" else [libname]
        return [Extension(
            name=name,
            sources=[source],
            include_dirs=[
                str(PKG_CODECS),
                numpy.get_include(),
                str(prefix / "include"),
            ],
            library_dirs=[str(prefix / "lib")],
            libraries=libs,
            extra_link_args=extra_link_args,
            define_macros=(define_macros or []) + [
                ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ],
            language="c++" if name.endswith(("_charls", "_brunsli", "_openjph"))
                     else "c",
        )]
    return []


def _maybe_build_mozjpeg_ext() -> list[Extension]:
    """Build the optional ``_mozjpeg`` extension when MozJPEG is
    installed. MozJPEG ships its libturbojpeg under a keg-only prefix
    on macOS (``/opt/homebrew/opt/mozjpeg``) so its symbols don't
    collide with the regular libjpeg-turbo. Linux distros tend to
    have it under ``/usr/include/mozjpeg`` or ``/usr/local/opt/mozjpeg``.
    """
    # Each candidate is (prefix, require_mozjpeg_subdir). MozJPEG-only
    # locations (homebrew keg, an explicit mozjpeg prefix) can be probed
    # by symbol absence alone. For generic system prefixes like /usr or
    # /usr/local we additionally require a ``mozjpeg/`` include subdir
    # so we don't misidentify vanilla libjpeg-turbo as MozJPEG.
    candidates = [
        (Path("/opt/homebrew/opt/mozjpeg"), False),
        (Path("/usr/local/opt/mozjpeg"), False),
        (Path("/usr"), True),
        (Path("/usr/local"), True),
    ]
    prefix = None
    lib_subdir = None  # "lib" or "lib/x86_64-linux-gnu" etc.
    lib_filename = None
    for c, require_mozjpeg_subdir in candidates:
        if require_mozjpeg_subdir and not (c / "include" / "mozjpeg").is_dir():
            continue
        if not (c / "include" / "turbojpeg.h").exists():
            continue
        # Find the actual lib file. Linux multiarch ships under
        # lib/<triple>/, plain /usr/local installs use lib/.
        lib_candidates = [
            ("lib", "libturbojpeg.dylib"),
            ("lib", "libturbojpeg.so"),
            ("lib", "libturbojpeg.so.0"),
            ("lib/x86_64-linux-gnu", "libturbojpeg.so"),
            ("lib/x86_64-linux-gnu", "libturbojpeg.so.0"),
            ("lib/aarch64-linux-gnu", "libturbojpeg.so"),
            ("lib/aarch64-linux-gnu", "libturbojpeg.so.0"),
        ]
        for subdir, name in lib_candidates:
            libpath = c / subdir / name
            if not libpath.exists():
                continue
            # MozJPEG branches off libjpeg-turbo 1.x — it lacks the v3
            # tj3* API. If we're already constrained to a mozjpeg subdir,
            # accept any libturbojpeg under it. Otherwise probe symbols.
            if require_mozjpeg_subdir:
                prefix = c
                lib_subdir = subdir
                lib_filename = name
                break
            try:
                import subprocess
                out = subprocess.run(
                    ["nm", "-gU", str(libpath)],
                    capture_output=True, text=True, timeout=10,
                ).stdout
                if "_tj3Compress8" not in out and "tj3Compress8" not in out:
                    prefix = c
                    lib_subdir = subdir
                    lib_filename = name
                    break
            except (FileNotFoundError, subprocess.SubprocessError):
                continue
        if prefix is not None:
            break
    if prefix is None:
        return []
    return [Extension(
        name="opencodecs.codecs._mozjpeg",
        sources=["src/opencodecs/codecs/_mozjpeg.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            str(prefix / "include"),
        ],
        library_dirs=[str(prefix / lib_subdir)],
        # Use absolute path on macOS to dodge the SDK-stub issue we
        # hit with zlib-ng-compat.
        libraries=[],
        extra_link_args=[str(prefix / lib_subdir / lib_filename)],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
    )]


def _maybe_build_openjph_ext() -> list[Extension]:
    """Build the optional ``_openjph`` HTJ2K extension when OpenJPH is
    installed. OpenJPH (https://github.com/aous72/OpenJPH) provides a
    high-throughput JPEG-2000 (HTJ2K, ISO/IEC 15444-15) codec; we wrap
    its C++ ``ojph::codestream`` API through a small shim in
    ``src/opencodecs/codecs/openjph_shim.cpp``.
    """
    candidates = [
        Path("/opt/homebrew/opt/openjph"),
        Path("/usr/local/opt/openjph"),
        Path(os.environ.get("CONDA_PREFIX", "/dev/null")),
        Path("/usr/local"),
        Path("/usr"),
    ]
    prefix = None
    for c in candidates:
        if not str(c) or not c.is_dir():
            continue
        if (c / "include" / "openjph" / "ojph_codestream.h").exists():
            for ext in ("dylib", "so", "so.0"):
                if (c / "lib" / f"libopenjph.{ext}").exists():
                    prefix = c
                    break
                if (c / "lib" / "x86_64-linux-gnu"
                        / f"libopenjph.{ext}").exists():
                    prefix = c
                    break
        if prefix is not None:
            break
    if prefix is None:
        return []

    # Match the absolute-dylib pattern used for MozJPEG / CharLS on
    # macOS so the linker doesn't bind to an SDK stub.
    if sys.platform == "darwin":
        dylib = prefix / "lib" / "libopenjph.dylib"
        extra_link_args = [str(dylib)]
        libs: list[str] = []
    else:
        extra_link_args = []
        libs = ["openjph"]

    return [Extension(
        name="opencodecs.codecs._openjph",
        sources=[
            "src/opencodecs/codecs/_openjph.pyx",
            "src/opencodecs/codecs/openjph_shim.cpp",
        ],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            str(prefix / "include"),
        ],
        library_dirs=[str(prefix / "lib")],
        libraries=libs,
        extra_link_args=extra_link_args,
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c++",
    )]


def _build_deflate_extension() -> Extension:
    """Pick the deflate backend. Prefer ``zlib-ng-compat`` (drop-in
    zlib replacement, ~1.5-2x faster on most modern CPUs), fall back
    to system zlib.

    Detection:
      1. Look for a Homebrew ``opt/zlib-ng-compat`` directory that
         brews ``libz.dylib`` (replacement) and a ``zlib.h``.
      2. Look for a Linux ``pkg-config --cflags --libs zlib-ng-compat``
         that resolves.
      3. Else: plain system zlib.

    No code changes on the .pyx side — both the compat layer and
    system zlib expose the same ``z*`` symbols.
    """
    zng_compat_prefix = None
    # Homebrew (macOS)
    for cand in (Path("/opt/homebrew/opt/zlib-ng-compat"),
                 Path("/usr/local/opt/zlib-ng-compat")):
        if (cand / "include" / "zlib.h").exists() and (
            cand / "lib" / "libz.dylib"
        ).exists():
            zng_compat_prefix = cand
            break
    # Linux: probe common conda + system paths
    if zng_compat_prefix is None:
        for cand in (Path(os.environ.get("CONDA_PREFIX", "")),
                     Path("/usr/local"), Path("/usr")):
            if str(cand) and (cand / "lib" / "pkgconfig"
                              / "zlib-ng-compat.pc").exists():
                zng_compat_prefix = cand
                break

    include_dirs = [str(PKG_CODECS)]
    library_dirs = list(_lib_dirs_for_probes())
    libraries = [_libname("z", "zlib")]
    extra_link_args: list[str] = []
    define_macros: list[tuple[str, str]] = []
    if zng_compat_prefix is not None:
        include_dirs.insert(0, str(zng_compat_prefix / "include"))
        # macOS distutils prepends -L<SDK>/usr/lib before our paths,
        # which makes a plain ``-lz`` resolve to the system zlib .tbd
        # stub instead of our zlib-ng-compat replacement. Bypass by
        # naming the dylib directly via extra_link_args (always
        # absolute first match) and dropping the "-lz" flag.
        compat_dylib = zng_compat_prefix / "lib" / (
            "libz.dylib" if sys.platform == "darwin"
            else "libz.so"
        )
        if compat_dylib.exists():
            extra_link_args.append(str(compat_dylib))
            libraries = []
        else:
            library_dirs.insert(0, str(zng_compat_prefix / "lib"))
    else:
        include_dirs.extend(_resolve_include_dirs("zlib.h"))

    # libdeflate detection — preferred over zlib (any flavour) for
    # one-shot encode/decode. Probe Homebrew + system paths.
    ld_prefix = _find_libdeflate_prefix()
    if ld_prefix is not None:
        include_dirs.insert(0, str(ld_prefix / "include"))
        define_macros.append(("OPENCODECS_HAVE_LIBDEFLATE", "1"))
        # Same SDK-stub-dodging dance: pass the dylib by absolute path
        # so distutils' implicit -L<SDK>/usr/lib doesn't beat us to
        # it. On Linux just add -ldeflate and let the rpath handle it.
        # On Windows the import-library file is libdeflate.lib (DLL
        # build) or libdeflatestatic.lib (static). Prefer the import
        # lib so we don't bloat the .pyd; the matching DLL must be
        # alongside the .pyd at runtime (or on PATH).
        if sys.platform == "darwin":
            ld_dylib = ld_prefix / "lib" / "libdeflate.dylib"
            if ld_dylib.exists():
                extra_link_args.append(str(ld_dylib))
            else:
                library_dirs.insert(0, str(ld_prefix / "lib"))
                libraries.append("deflate")
        elif sys.platform == "win32":
            library_dirs.insert(0, str(ld_prefix / "lib"))
            # Prefer the import library (.lib paired with .dll).
            if (ld_prefix / "lib" / "libdeflate.lib").exists():
                libraries.append("libdeflate")
            elif (ld_prefix / "lib" / "libdeflatestatic.lib").exists():
                libraries.append("libdeflatestatic")
            else:
                libraries.append("deflate")
        else:
            library_dirs.insert(0, str(ld_prefix / "lib"))
            libraries.append("deflate")

    return Extension(
        name="opencodecs.codecs._deflate",
        sources=["src/opencodecs/codecs/_deflate.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
        define_macros=define_macros,
        language="c",
    )


def _find_libdeflate_prefix() -> Path | None:
    """Locate libdeflate's install prefix across platforms.

    Search order:

      * ``OPENCODECS_LIBDEFLATE_PREFIX`` env var (explicit override —
        most useful on Windows where users typically extract the
        upstream .zip release somewhere like ``C:\\opencodecs_libs``).
      * Homebrew (macOS).
      * The active ``CONDA_PREFIX`` — conda-forge has a ``libdeflate``
        package which installs to ``<prefix>/Library/include`` on
        Windows and ``<prefix>/include`` elsewhere.
      * System ``/usr/local`` / ``/usr``.

    Returns the prefix dir (with ``include/libdeflate.h`` directly
    under it on POSIX, or ``Library/include/libdeflate.h`` on
    conda-Windows). Returns None when libdeflate isn't found —
    callers fall through to zlib-ng-compat or stdlib zlib.
    """
    candidates = []
    env_prefix = os.environ.get("OPENCODECS_LIBDEFLATE_PREFIX")
    if env_prefix:
        candidates.append(Path(env_prefix))
    candidates.extend([
        Path("/opt/homebrew/opt/libdeflate"),
        Path("/usr/local/opt/libdeflate"),
        Path(os.environ.get("CONDA_PREFIX", "")),
        Path("/usr/local"),
        Path("/usr"),
    ])
    # Linux multiarch layout: lib/<triple>/. Probe a few common triples.
    posix_lib_subdirs = (
        "lib",
        "lib/x86_64-linux-gnu",
        "lib/aarch64-linux-gnu",
        "lib64",
    )
    for c in candidates:
        if not str(c) or str(c) == ".":
            continue
        # POSIX layout: <prefix>/include/libdeflate.h
        if (c / "include" / "libdeflate.h").is_file() and any(
            (c / s / name).exists()
            for s in posix_lib_subdirs
            for name in ("libdeflate.dylib", "libdeflate.so", "libdeflate.so.0")
        ):
            return c
        # conda-Windows layout: <prefix>/Library/include/libdeflate.h
        # + <prefix>/Library/lib/libdeflate.lib
        if (c / "Library" / "include" / "libdeflate.h").is_file() and (
            (c / "Library" / "lib" / "libdeflate.lib").exists()
            or (c / "Library" / "lib" / "libdeflatestatic.lib").exists()
        ):
            return c / "Library"
        # Upstream Windows release layout (extracted .zip):
        # <prefix>/include/libdeflate.h + <prefix>/lib/libdeflate.lib.
        if (c / "include" / "libdeflate.h").is_file() and (
            (c / "lib" / "libdeflate.lib").exists()
            or (c / "lib" / "libdeflatestatic.lib").exists()
        ):
            return c
    return None


def _build_png_ext() -> Extension:
    """Build the _png Extension with system-spng if available, else vendored.

    System libspng is preferred when present (Homebrew, Ubuntu's
    libspng-dev) and we don't have a faster zlib alternative — the
    system build is already linked against the distro's zlib by the
    packager and we save build time.

    HOWEVER: when zlib-ng-compat is installed, system libspng is
    typically linked to the SYSTEM libz (not zlib-ng), so PNG-encode
    can't benefit from the 1.3-1.4x deflate speedup the zlib-ng swap
    promised. In that case we fall back to compiling the vendored
    3rdparty/libspng/spng.c ourselves so the inner deflate goes
    through zlib-ng-compat. Measured 1.3x faster PNG encode end-to-
    end on incompressible / large images.

    On Windows the conda-forge channel has no libspng package, so we
    also use the vendored path there.
    """
    have_system_spng = (
        _has_header("spng.h")
        and any((Path(p) / "lib" / "libspng.so").exists()
                or (Path(p) / "lib" / "libspng.dylib").exists()
                or (Path(p) / "Library" / "lib" / "spng.lib").exists()
                for p in (str(x) for x in _PROBE_PREFIXES))
    )
    # Detect zlib-ng-compat the same way _build_deflate_extension does.
    _zng_brew = (Path("/opt/homebrew/opt/zlib-ng-compat").is_dir()
                 or Path("/usr/local/opt/zlib-ng-compat").is_dir())
    _zng_linux = False
    if not _zng_brew:
        for cand in (Path(os.environ.get("CONDA_PREFIX", "")),
                     Path("/usr/local"), Path("/usr")):
            if str(cand) and (cand / "lib" / "pkgconfig"
                              / "zlib-ng-compat.pc").exists():
                _zng_linux = True
                break
    have_zlib_ng_compat = _zng_brew or _zng_linux
    # libdeflate detection — if found, we'll patch the vendored
    # libspng to route its inner deflate calls through libdeflate's
    # one-shot API (~2x faster than zlib-ng for PNG encode).
    ld_prefix = _find_libdeflate_prefix()
    have_libdeflate = ld_prefix is not None
    # Skip the system-libspng fast-path when we have a faster
    # alternative (zlib-ng-compat OR libdeflate); building
    # 3rdparty/libspng/spng.c ourselves routes the inner deflate
    # through our preferred backend.
    prefer_vendored = have_zlib_ng_compat or have_libdeflate
    if have_system_spng and not prefer_vendored:
        return Extension(
            name="opencodecs.codecs._png",
            sources=["src/opencodecs/codecs/_png.pyx"],
            define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
            language="c",
            include_dirs=[str(PKG_CODECS), numpy.get_include(),
                          *_resolve_include_dirs("spng.h")],
            library_dirs=_lib_dirs_for_probes(),
            libraries=["spng"],
        )

    # Vendored fallback. Compile spng.c into the .so/.pyd, link zlib.
    # Note: spng.c uses ``#ifdef SPNG_USE_MINIZ`` (not ``#if``), so
    # defining it as 0 still triggers the miniz path. We just don't
    # define it at all when zlib is the backend.
    #
    # When zlib-ng-compat is available, point the linker directly at
    # its .dylib so macOS's SDK-stub libz.tbd doesn't win the lookup
    # (same trick _build_deflate_extension uses). Without this the
    # vendored libspng would link to /usr/lib/libz.1.dylib instead of
    # the zlib-ng replacement, defeating the whole point of choosing
    # the vendored path.
    include_dirs = [str(PKG_CODECS), numpy.get_include(),
                    str(HERE / "3rdparty" / "libspng")]
    library_dirs = list(_lib_dirs_for_probes())
    libraries: list[str] = []
    extra_link_args: list[str] = []
    zng_compat_prefix = None
    for cand in (Path("/opt/homebrew/opt/zlib-ng-compat"),
                 Path("/usr/local/opt/zlib-ng-compat")):
        if (cand / "include" / "zlib.h").exists() and (
            cand / "lib" / "libz.dylib"
        ).exists():
            zng_compat_prefix = cand
            break
    if zng_compat_prefix is not None and sys.platform == "darwin":
        include_dirs.insert(0, str(zng_compat_prefix / "include"))
        extra_link_args.append(
            str(zng_compat_prefix / "lib" / "libz.dylib")
        )
    else:
        include_dirs.extend(_resolve_include_dirs("zlib.h"))
        libraries = ["z" if not sys.platform == "win32" else "zlib"]
    # libdeflate fast path: patch the vendored libspng to route its
    # IDAT-compress call through libdeflate's one-shot API. ~2x faster
    # PNG encode end-to-end vs zlib-ng. Decode stays on zlib (still
    # needed for tEXt / zTXt / iTXt chunks at minimum).
    define_macros = [
        ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ("SPNG_STATIC", "1"),
    ]
    if have_libdeflate:
        include_dirs.insert(0, str(ld_prefix / "include"))
        define_macros.append(("SPNG_USE_LIBDEFLATE", "1"))
        if sys.platform == "darwin":
            ld_dylib = ld_prefix / "lib" / "libdeflate.dylib"
            if ld_dylib.exists():
                extra_link_args.append(str(ld_dylib))
            else:
                library_dirs.insert(0, str(ld_prefix / "lib"))
                libraries.append("deflate")
        elif sys.platform == "win32":
            library_dirs.insert(0, str(ld_prefix / "lib"))
            if (ld_prefix / "lib" / "libdeflate.lib").exists():
                libraries.append("libdeflate")
            elif (ld_prefix / "lib" / "libdeflatestatic.lib").exists():
                libraries.append("libdeflatestatic")
            else:
                libraries.append("deflate")
        else:
            library_dirs.insert(0, str(ld_prefix / "lib"))
            libraries.append("deflate")
    return Extension(
        name="opencodecs.codecs._png",
        sources=["src/opencodecs/codecs/_png.pyx", "3rdparty/libspng/spng.c"],
        define_macros=define_macros,
        language="c",
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        extra_link_args=extra_link_args,
    )


extensions = [
    Extension(
        name="opencodecs.codecs._jxl",
        sources=["src/opencodecs/codecs/_jxl.pyx"],
        include_dirs=include_dirs,
        library_dirs=library_dirs,
        libraries=libraries,
        define_macros=define_macros,
        extra_link_args=extra_link_args,
        language="c",
    ),
    # QOI: vendored single-header (3rdparty/qoi/qoi.h). Compile the impl
    # via QOI_IMPLEMENTATION; no external library needed.
    Extension(
        name="opencodecs.codecs._qoi",
        sources=["src/opencodecs/codecs/_qoi.pyx"],
        include_dirs=[
            str(HERE / "3rdparty" / "qoi"),
            str(PKG_CODECS),
            numpy.get_include(),
        ],
        define_macros=[
            ("QOI_IMPLEMENTATION", "1"),
            ("QOI_NO_STDIO", "1"),  # we don't use qoi_read/qoi_write
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ],
        language="c",
    ),
    # zstd: per-user cache build preferred (Homebrew libzstd is built
    # without IPO and benches 5% slower than an `-O3 + LTO` rebuild).
    # macOS sysconfig prepends `-L/opt/homebrew/lib` ahead of our
    # library_dirs so `-lzstd` always finds Homebrew; pass the absolute
    # dylib path via extra_link_args to bypass `-l` resolution.
    Extension(
        name="opencodecs.codecs._zstd",
        sources=["src/opencodecs/codecs/_zstd.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            *_resolve_include_dirs("zstd.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=(
            []
            if (_OC_USER_CACHE / "zstd" / "lib" / "libzstd.1.5.7.dylib").exists()
               or (_OC_USER_CACHE / "zstd" / "lib" / "libzstd.so").exists()
            else ["zstd"]
        ),
        extra_link_args=(
            [
                str(_OC_USER_CACHE / "zstd" / "lib" / (
                    "libzstd.1.5.7.dylib" if sys.platform == "darwin"
                    else "libzstd.so"
                )),
                f"-Wl,-rpath,{_OC_USER_CACHE / 'zstd' / 'lib'}",
            ]
            if (_OC_USER_CACHE / "zstd" / "lib").is_dir() else []
        ),
        language="c",
    ),
    # lz4: links against system liblz4 (Homebrew on Mac, liblz4-dev on
    # Ubuntu). LZ4 frame format (.lz4 file format) — uses lz4frame.h.
    Extension(
        name="opencodecs.codecs._lz4",
        sources=["src/opencodecs/codecs/_lz4.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            *_resolve_include_dirs("lz4frame.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["lz4"],
        language="c",
    ),
    # brotli: per-user cache build preferred (same reason as zstd above).
    # Homebrew currently ships brotli 1.2.0 but imagecodecs and most
    # consumers stick to 1.1.0; the cache build pins 1.1.0 with
    # `-O3 + LTO + apple-m1` tuning.
    Extension(
        name="opencodecs.codecs._brotli",
        sources=["src/opencodecs/codecs/_brotli.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            *_resolve_include_dirs("brotli/encode.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=(
            []
            if (_OC_USER_CACHE / "brotli" / "lib" / "libbrotlienc.1.1.0.dylib").exists()
               or (_OC_USER_CACHE / "brotli" / "lib" / "libbrotlienc.so").exists()
            else ["brotlienc", "brotlidec", "brotlicommon"]
        ),
        extra_link_args=(
            [
                str(_OC_USER_CACHE / "brotli" / "lib" / (
                    "libbrotlienc.1.1.0.dylib" if sys.platform == "darwin"
                    else "libbrotlienc.so"
                )),
                str(_OC_USER_CACHE / "brotli" / "lib" / (
                    "libbrotlidec.1.1.0.dylib" if sys.platform == "darwin"
                    else "libbrotlidec.so"
                )),
                str(_OC_USER_CACHE / "brotli" / "lib" / (
                    "libbrotlicommon.1.1.0.dylib" if sys.platform == "darwin"
                    else "libbrotlicommon.so"
                )),
                f"-Wl,-rpath,{_OC_USER_CACHE / 'brotli' / 'lib'}",
            ]
            if (_OC_USER_CACHE / "brotli" / "lib").is_dir() else []
        ),
        language="c",
    ),
    # blosc2: links against system c-blosc2.
    Extension(
        name="opencodecs.codecs._blosc2",
        sources=["src/opencodecs/codecs/_blosc2.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            *_resolve_include_dirs("blosc2.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["blosc2"],
        language="c",
    ),
    # blosc2 NDim (b2nd): exposes c-blosc2's multidimensional layer via a
    # tiny C shim (b2nd_helpers.c). Same library; we just call b2nd_*
    # entry points instead of blosc2_* ones.
    Extension(
        name="opencodecs.codecs._b2nd",
        sources=[
            "src/opencodecs/codecs/_b2nd.pyx",
            "src/opencodecs/codecs/b2nd_helpers.c",
        ],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            *_resolve_include_dirs("blosc2.h"),
            *_resolve_include_dirs("b2nd.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["blosc2"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
    ),
    # JPEG via libjpeg-turbo (TurboJPEG v3 API).
    Extension(
        name="opencodecs.codecs._jpeg",
        sources=["src/opencodecs/codecs/_jpeg.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            *_resolve_include_dirs("turbojpeg.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["turbojpeg"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
    ),
    # MozJPEG (Mozilla's libjpeg-turbo fork — smaller files at the
    # same quality). Optional; built only when ``mozjpeg`` is found
    # on the system. Both Mac homebrew (keg-only) and Linux distros
    # ship it in a separate prefix to avoid colliding with the regular
    # libjpeg-turbo libs.
    *_maybe_build_mozjpeg_ext(),
    # CharLS / JPEG-LS via libcharls. Optional — built only when CharLS
    # is on the system.
    *_maybe_build_ext_simple(
        name="opencodecs.codecs._charls",
        source="src/opencodecs/codecs/_charls.pyx",
        prefixes=[
            "/opt/homebrew/opt/charls",
            "/usr/local/opt/charls",
            "/usr/local", "/usr",
        ],
        probe_header="charls/charls.h",
        libname="charls",
    ),
    # HTJ2K (high-throughput JPEG-2000) via OpenJPH. Optional — built
    # only when libopenjph is on the system.
    *_maybe_build_openjph_ext(),
    # EER (Thermo Fisher Electron Event Representation) — cryo-EM
    # event-list decoder, vendored from imagecodecs imcd.c (BSD-3).
    # No external deps; always built.
    Extension(
        name="opencodecs.codecs._eer",
        sources=[
            "src/opencodecs/codecs/_eer.pyx",
            "3rdparty/imcd_eer/eer.c",
        ],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            "3rdparty/imcd_eer",
        ],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
        ],
        language="c",
    ),
    # BC1-7 / DXT / BPTC GPU texture decoder via the vendored single-
    # header ``bcdec.h`` (MIT). No external deps; the implementation
    # gets compiled into our .so via BCDEC_IMPLEMENTATION.
    Extension(
        name="opencodecs.codecs._bcdec",
        sources=["src/opencodecs/codecs/_bcdec.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            "3rdparty/bcdec",
        ],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("BCDEC_IMPLEMENTATION", "1"),
            ("BCDEC_STATIC", "1"),
            # Enables the 4-arg BC4/BC5 entry points (with isSigned)
            # and slightly slower-but-bit-exact decode.
            ("BCDEC_BC4BC5_PRECISE", "1"),
        ],
        language="c",
    ),
    # WebP via libwebp. We add a tiny C shim (webp_shim.c) so we can
    # expose WebPConfig.thread_level — libwebp's simple WebPEncode*
    # API doesn't take it.
    Extension(
        name="opencodecs.codecs._webp",
        sources=[
            "src/opencodecs/codecs/_webp.pyx",
            "src/opencodecs/codecs/webp_shim.c",
        ],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            *_resolve_include_dirs("webp/encode.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=[_libname("webp", "libwebp")],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
    ),
    # JPEG-2000 via OpenJPEG. The header lives in a versioned subdir on
    # most platforms (openjpeg-2.5/), but conda-forge installs it directly
    # at <prefix>/Library/include/openjpeg.h. Probe both layouts.
    Extension(
        name="opencodecs.codecs._jpeg2k",
        sources=["src/opencodecs/codecs/_jpeg2k.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            # Versioned subdir on Mac/Linux
            *([str(p / "include" / "openjpeg-2.5") for p in _PROBE_PREFIXES
               if (p / "include" / "openjpeg-2.5" / "openjpeg.h").is_file()]),
            *([str(p / "include" / "openjpeg-2.4") for p in _PROBE_PREFIXES
               if (p / "include" / "openjpeg-2.4" / "openjpeg.h").is_file()]),
            # Direct (conda-forge Windows): include/openjpeg.h
            *_resolve_include_dirs("openjpeg.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["openjp2"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
    ),
    # AVIF via libavif (system; pulls in libaom/libdav1d/libsvtav1).
    Extension(
        name="opencodecs.codecs._avif",
        sources=["src/opencodecs/codecs/_avif.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            *_resolve_include_dirs("avif/avif.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["avif"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
    ),
    # HEIF/HEIC via libheif (system; depends on libde265/x265).
    # On Ubuntu the header lives under /usr/include/x86_64-linux-gnu/libheif/.
    Extension(
        name="opencodecs.codecs._heif",
        sources=["src/opencodecs/codecs/_heif.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            *_resolve_include_dirs("libheif/heif.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["heif"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
    ),
    # zlib / deflate. Prefers zlib-ng-compat when available (it's a
    # drop-in zlib replacement that ships ~1.5-2x faster on most
    # x86_64 / arm64 hardware). Fall back to system zlib otherwise.
    _build_deflate_extension(),
    # Tight nogil byte-shuffling helpers (used by the CZI reader).
    Extension(
        name="opencodecs.codecs._bytetools",
        sources=["src/opencodecs/codecs/_bytetools.pyx"],
        include_dirs=[str(PKG_CODECS)],
        language="c",
    ),
    # Native TIFF IFD walker — pure-Python parsing logic in Cython for
    # speed on huge multi-IFD files (e.g. 10000-page tiled TIFFs).
    # No external library; tile decompression dispatches to the
    # already-built compression extensions at runtime.
    Extension(
        name="opencodecs.codecs._tiff",
        sources=[
            "src/opencodecs/codecs/_tiff.pyx",
            "3rdparty/imcd_lzw/lzw.c",
        ],
        include_dirs=[str(PKG_CODECS), "3rdparty/imcd_lzw"],
        language="c",
    ),
    # NDTiff index parser — Cython nogil walk of the Micro-Manager
    # NDTiff.index binary side-file. No external library; frame
    # pixel reads go through os.pread + opencodecs._tiff.
    Extension(
        name="opencodecs.codecs._ndtiff",
        sources=["src/opencodecs/codecs/_ndtiff.pyx"],
        include_dirs=[str(PKG_CODECS)],
        language="c",
    ),
    # pcodec (cpcodec): Rust cdylib built via cargo into per-user cache.
    Extension(
        name="opencodecs.codecs._pcodec",
        sources=["src/opencodecs/codecs/_pcodec.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            *_resolve_include_dirs("cpcodec.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["cpcodec"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_link_args=(
            [f"-Wl,-rpath,{_OC_USER_CACHE / 'pcodec' / 'lib'}"]
            if (_OC_USER_CACHE / "pcodec" / "lib").is_dir() else []
        ),
        language="c",
    ),
    # SZ3 (error-bounded lossy compressor): system SZ3c, or per-user
    # cache when built via bench/build_codec_libs.sh (no system package).
    Extension(
        name="opencodecs.codecs._sz3",
        sources=["src/opencodecs/codecs/_sz3.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            *_resolve_include_dirs("SZ3c/sz3c.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["SZ3c"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        # Embed RPATH to the per-user cache so dyld finds libSZ3c at
        # runtime even though the lib lives outside of standard search
        # paths. On Linux + cibuildwheel this becomes a no-op (the lib
        # is built to /cibw-jxl-prefix and bundled via auditwheel).
        extra_link_args=(
            [f"-Wl,-rpath,{_OC_USER_CACHE / 'sz3' / 'lib'}"]
            if (_OC_USER_CACHE / "sz3" / "lib").is_dir() else []
        ),
        language="c",
    ),
    # GIF via giflib. Per-user cache build preferred (Homebrew's
    # libgif 6.x is built portable / -O2; a tuned 5.2.2 -O3+LTO build
    # is measurably faster on LZW). Absolute-dylib link bypasses macOS
    # sysconfig's -L/opt/homebrew/lib prepend.
    Extension(
        name="opencodecs.codecs._gif",
        sources=["src/opencodecs/codecs/_gif.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            *_resolve_include_dirs("gif_lib.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=(
            []
            if (_OC_USER_CACHE / "giflib" / "lib" / "libgif.7.2.0.dylib").exists()
               or (_OC_USER_CACHE / "giflib" / "lib" / "libgif.so").exists()
            else ["gif"]
        ),
        extra_link_args=(
            [
                str(_OC_USER_CACHE / "giflib" / "lib" / (
                    "libgif.7.2.0.dylib" if sys.platform == "darwin"
                    else "libgif.so"
                )),
                f"-Wl,-rpath,{_OC_USER_CACHE / 'giflib' / 'lib'}",
            ]
            if (_OC_USER_CACHE / "giflib" / "lib").is_dir() else []
        ),
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
    ),
    # Brunsli (lossless JPEG transcoder, ~20% smaller storage): per-user
    # cache when built via bench/build_codec_libs.sh.
    Extension(
        name="opencodecs.codecs._brunsli",
        sources=["src/opencodecs/codecs/_brunsli.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            *_resolve_include_dirs("brunsli/encode.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["brunslienc-c", "brunslidec-c"],
        extra_link_args=(
            [f"-Wl,-rpath,{_OC_USER_CACHE / 'brunsli' / 'lib'}"]
            if (_OC_USER_CACHE / "brunsli" / "lib").is_dir() else []
        ),
        language="c",
    ),
    # SPERR (wavelet-based error-bounded lossy compressor): system
    # libSPERR, or per-user cache when built via bench/build_codec_libs.sh.
    Extension(
        name="opencodecs.codecs._sperr",
        sources=["src/opencodecs/codecs/_sperr.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            *_resolve_include_dirs("SPERR_C_API.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["SPERR"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_link_args=(
            [f"-Wl,-rpath,{_OC_USER_CACHE / 'sperr' / 'lib'}"]
            if (_OC_USER_CACHE / "sperr" / "lib").is_dir() else []
        ),
        language="c",
    ),
    # ZFP (lossy floating-point compression): system libzfp.
    Extension(
        name="opencodecs.codecs._zfp",
        sources=["src/opencodecs/codecs/_zfp.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            *_resolve_include_dirs("zfp.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["zfp"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
    ),
    # LERC (Esri Limited Error Raster Compression): per-user cache
    # build preferred (Homebrew's libLerc is built `-O2` portable and
    # measurably 15% slower than an `-O3 + LTO` rebuild; see
    # bench/build_codec_libs.sh::build_lerc). Falls back to system
    # liblerc when the cache copy isn't present.
    #
    # macOS link-order trap: sysconfig prepends `-L/opt/homebrew/lib`
    # AHEAD of our library_dirs, so `-lLerc` always resolves to
    # Homebrew. Pass the absolute dylib path via extra_link_args
    # (libjxl uses the same trick).
    Extension(
        name="opencodecs.codecs._lerc",
        sources=["src/opencodecs/codecs/_lerc.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            *_resolve_include_dirs("Lerc_c_api.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=(
            []
            if (_OC_USER_CACHE / "lerc" / "lib" / "libLerc.4.dylib").exists()
               or (_OC_USER_CACHE / "lerc" / "lib" / "libLerc.so").exists()
            else ["Lerc"]
        ),
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        extra_link_args=(
            [
                str(_OC_USER_CACHE / "lerc" / "lib" / (
                    "libLerc.4.dylib" if sys.platform == "darwin"
                    else "libLerc.so"
                )),
                f"-Wl,-rpath,{_OC_USER_CACHE / 'lerc' / 'lib'}",
            ]
            if (_OC_USER_CACHE / "lerc" / "lib").is_dir() else []
        ),
        language="c",
    ),
    # AEC (CCSDS 121.0-B-2 adaptive entropy coding): system libaec.
    Extension(
        name="opencodecs.codecs._aec",
        sources=["src/opencodecs/codecs/_aec.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            *_resolve_include_dirs("libaec.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["aec"],
        language="c",
    ),
    # Bitshuffle: vendored single-purpose filter (~3 small C files). No
    # external dep — bitshuffle.h is the LZ4-coupled API which we don't
    # use, but bshuf_compress_lz4 references LZ4_compress_HC, so we
    # include only bitshuffle_core.{c,h} + iochain.{c,h} sources here.
    # Pure transpose (encode/decode) needs no compressor library.
    Extension(
        name="opencodecs.codecs._bitshuffle",
        sources=[
            "src/opencodecs/codecs/_bitshuffle.pyx",
            "3rdparty/bitshuffle/bitshuffle_core.c",
            "3rdparty/bitshuffle/iochain.c",
        ],
        include_dirs=[
            str(PKG_CODECS),
            str(HERE / "3rdparty" / "bitshuffle"),
        ],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
        language="c",
    ),
    # PNG via libspng (system; Homebrew on Mac, libspng-dev on Ubuntu).
    # libspng is a clean C reimplementation of libpng with a much smaller
    # API surface — ideal for our wrapper. Falls back to vendored sources
    # when no system spng is found (Windows, where conda-forge has no libspng).
    _build_png_ext(),
]


# Filter out extensions whose required system header isn't installed on this
# host. This lets us ship a single setup.py that builds whatever's available
# (e.g. on Ubuntu without liblz4-dev, just skip _lz4 instead of failing).
_REQUIRED_HEADERS = {
    "opencodecs.codecs._jxl":    ("jxl/types.h",),
    "opencodecs.codecs._zstd":   ("zstd.h",),
    "opencodecs.codecs._deflate": ("zlib.h",),
    "opencodecs.codecs._lz4":    ("lz4frame.h",),
    "opencodecs.codecs._brotli": ("brotli/encode.h",),
    "opencodecs.codecs._blosc2": ("blosc2.h",),
    "opencodecs.codecs._b2nd":   ("b2nd.h",),
    "opencodecs.codecs._aec":    ("libaec.h",),
    "opencodecs.codecs._lerc":   ("Lerc_c_api.h",),
    "opencodecs.codecs._zfp":    ("zfp.h",),
    "opencodecs.codecs._sz3":    ("SZ3c/sz3c.h",),
    "opencodecs.codecs._sperr":  ("SPERR_C_API.h",),
    "opencodecs.codecs._brunsli": ("brunsli/encode.h",),
    "opencodecs.codecs._gif":    ("gif_lib.h",),
    "opencodecs.codecs._pcodec": ("cpcodec.h",),
    "opencodecs.codecs._jpeg":   ("turbojpeg.h",),
    "opencodecs.codecs._webp":   ("webp/encode.h",),
    "opencodecs.codecs._jpeg2k": ("openjpeg-2.5/openjpeg.h", "openjpeg-2.4/openjpeg.h"),
    "opencodecs.codecs._avif":   ("avif/avif.h",),
    "opencodecs.codecs._heif":   ("libheif/heif.h",),
    # _png falls back to vendored 3rdparty/libspng/, so don't gate on a
    # system spng.h. (The vendored fallback also requires zlib.h, which
    # system zlib provides on every supported platform.)
    "opencodecs.codecs._png":    ("zlib.h",),
    # _qoi vendored, _deflate uses system zlib (always present).
}

_kept_extensions = []
_skipped_extensions: list[str] = []
for ext in extensions:
    headers = _REQUIRED_HEADERS.get(ext.name)
    if headers is None or _has_header(*headers):
        # Extra version gates for API-incompatible older builds.
        if ext.name == "opencodecs.codecs._jpeg" and not _header_contains(
            "turbojpeg.h", "tj3Init"
        ):
            _skipped_extensions.append(ext.name + " (TurboJPEG v3 API not found)")
            continue
        _kept_extensions.append(ext)
    else:
        _skipped_extensions.append(ext.name)
extensions = _kept_extensions
if _skipped_extensions:
    print(f"opencodecs: skipping extensions (missing system headers): "
          f"{', '.join(_skipped_extensions)}")

class build_ext(_build_ext):
    """Custom build_ext that strips conflicting RPATHs on macOS so the
    vendored libjxl wins over Homebrew at runtime.

    sysconfig prepends -L/opt/homebrew/lib + -Wl,-rpath,/opt/homebrew/lib to
    every Mac extension link, regardless of what we put in
    library_dirs/extra_link_args. Result: dyld searches Homebrew before
    our vendor/libjxl/lib for @rpath/libjxl.0.11.dylib, and silently uses
    the system libjxl 0.11.1 instead of the vendored 0.11.2 we just built.
    Strip the Homebrew RPATH from the built .so as a post-build step.
    """

    def run(self):
        super().run()
        if not (sys.platform == "darwin" and using_vendored):
            return
        for ext in self.extensions:
            so_path = self.get_ext_fullpath(ext.name)
            if not os.path.exists(so_path):
                continue
            try:
                subprocess.run(
                    ["install_name_tool", "-delete_rpath",
                     "/opt/homebrew/lib", so_path],
                    check=True,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except (FileNotFoundError, subprocess.CalledProcessError):
                # Either install_name_tool missing or rpath not present —
                # not fatal; the vendored libs will still be tried (just
                # after Homebrew) and the wrong libjxl might load.
                pass


setup(
    cmdclass={"build_ext": build_ext},
    package_dir={"": "src"},
    packages=["opencodecs", "opencodecs.core", "opencodecs.codecs"],
    ext_modules=cythonize(
        extensions,
        include_path=[str(PKG_CODECS)],
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
            "cdivision": True,
            "nonecheck": False,
        },
        annotate=bool(os.environ.get("OPENCODECS_ANNOTATE")),
    ),
)
