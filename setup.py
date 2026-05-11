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
for _libdir in ("sz3", "pcodec"):
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


def _build_png_ext() -> Extension:
    """Build the _png Extension with system-spng if available, else vendored.

    System libspng is preferred when present (Homebrew, Ubuntu's
    libspng-dev, etc.) — it's already linked against zlib by the distro.
    On Windows the conda-forge channel has no libspng package, so we
    fall back to compiling 3rdparty/libspng/spng.c into the extension
    and link against zlib (which conda-forge does provide).
    """
    have_system_spng = (
        _has_header("spng.h")
        and any((Path(p) / "lib" / "libspng.so").exists()
                or (Path(p) / "lib" / "libspng.dylib").exists()
                or (Path(p) / "Library" / "lib" / "spng.lib").exists()
                for p in (str(x) for x in _PROBE_PREFIXES))
    )

    if have_system_spng:
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
    # Setuptools requires sources to be /-separated paths *relative* to
    # setup.py. Use the relative path even though we know the absolute
    # one — modern setuptools (>=80) refuses absolute paths in
    # Extension.sources outright.
    return Extension(
        name="opencodecs.codecs._png",
        sources=["src/opencodecs/codecs/_png.pyx", "3rdparty/libspng/spng.c"],
        define_macros=[
            ("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION"),
            ("SPNG_STATIC", "1"),
        ],
        language="c",
        include_dirs=[str(PKG_CODECS), numpy.get_include(),
                      str(HERE / "3rdparty" / "libspng"),
                      *_resolve_include_dirs("zlib.h")],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["z" if not sys.platform == "win32" else "zlib"],
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
    # zstd: links against system libzstd (Homebrew on Mac, libzstd-dev on
    # Ubuntu, $CONDA_PREFIX otherwise). No vendoring; zstd's ABI is
    # stable across versions.
    Extension(
        name="opencodecs.codecs._zstd",
        sources=["src/opencodecs/codecs/_zstd.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            *_resolve_include_dirs("zstd.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["zstd"],
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
    # brotli: links against system libbrotli (also a libjxl transitive dep,
    # so already present on any system that built libjxl).
    Extension(
        name="opencodecs.codecs._brotli",
        sources=["src/opencodecs/codecs/_brotli.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            *_resolve_include_dirs("brotli/encode.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["brotlienc", "brotlidec", "brotlicommon"],
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
    # WebP via libwebp.
    Extension(
        name="opencodecs.codecs._webp",
        sources=["src/opencodecs/codecs/_webp.pyx"],
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
    # zlib / deflate (system zlib).
    Extension(
        name="opencodecs.codecs._deflate",
        sources=["src/opencodecs/codecs/_deflate.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            *_resolve_include_dirs("zlib.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=[_libname("z", "zlib")],
        language="c",
    ),
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
        sources=["src/opencodecs/codecs/_tiff.pyx"],
        include_dirs=[str(PKG_CODECS)],
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
    # LERC (Esri Limited Error Raster Compression): system liblerc.
    Extension(
        name="opencodecs.codecs._lerc",
        sources=["src/opencodecs/codecs/_lerc.pyx"],
        include_dirs=[
            str(PKG_CODECS),
            numpy.get_include(),
            *_resolve_include_dirs("Lerc_c_api.h"),
        ],
        library_dirs=_lib_dirs_for_probes(),
        libraries=["Lerc"],
        define_macros=[("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")],
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
