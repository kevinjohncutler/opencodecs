#!/usr/bin/env bash
# Build EVERY system C library opencodecs links against, from source, into
# a single prefix. Used by cibuildwheel BEFORE_ALL inside the manylinux
# container (where the dnf/EPEL versions are too old, missing, or both).
# Also useful locally to produce a self-contained tree for benchmarking
# against a known set of versions, or to cache /usr/local in CI.
#
# All-source builds are what imagecodecs ships — it's the only way to
# guarantee the same versions across Linux/Mac/Windows wheels.
#
# Each library is built independently and *idempotently*: if the install
# fingerprint is already present at $PREFIX, that lib is skipped. This
# makes the script cache-friendly: ``actions/cache`` keyed on the script's
# hash gets you a ~30-second warm restore vs a ~25-minute cold rebuild.
#
# ----------------------------------------------------------------------
# Usage
# ----------------------------------------------------------------------
#   bash bench/build_codec_libs.sh                  # builds all into $PREFIX
#   bash bench/build_codec_libs.sh --only=zstd,lz4  # subset
#   bash bench/build_codec_libs.sh --skip=heif      # build everything except
#
# Env vars (with defaults):
#   PREFIX           Install root.  Default: $OPENCODECS_LIBS_PREFIX
#                    or /usr/local on root, ~/.cache/opencodecs/libs else.
#   JOBS             Parallel jobs.  Default: $(nproc).
#   USE_LTO          1 to enable link-time optimization on cmake builds.
#                    Default: 1.
#   MARCH            -march flag.  Default: '' (portable; pass 'native'
#                    for max-speed on the build host only).
#   ENABLE_AOM       Build libaom (for libavif AV1 encode).  Default: 1.
#                    Off-by-default on tiny CI runners — aom is the
#                    single biggest build (~3 min).
#   ENABLE_X265      Build x265 (for libheif HEVC encode).  Default: 1.
#
# ----------------------------------------------------------------------
# Library version pins
# ----------------------------------------------------------------------
# These are bumped together. CI cache invalidates on any change to this
# script, so a single edit cycles the whole stack.

set -euo pipefail

VERSIONS=(
    # Compression / archival (small, fast to build)
    "zlib            1.3.1"
    "zstd            1.5.7"
    "lz4             1.10.0"
    "brotli          1.1.0"
    "giflib          5.2.2"
    "libdeflate      1.23"

    # Image (small to medium)
    "libpng          1.6.50"
    "libjpeg-turbo   3.1.2"
    "libwebp         1.6.0"
    "openjpeg        2.5.5"
    "mozjpeg         4.1.5"

    # Container / multi-codec (medium)
    # c-blosc2 pinned at 2.x because 3.x changed the default filter chain
    # — same data round-trips but the 3.x defaults are tuned for size at
    # the cost of CPU (2x slower encode at zstd-level-1, ~9% smaller
    # output). imagecodecs bundles 2.23.0 and beats us by 2x with the
    # 3.x brew bottle. Pinning the cache build to the latest 2.x branch
    # matches their wire format and closes the perf gap.
    "c-blosc2        2.23.0"

    # AV1 / HEVC (largest builds)
    "libaom          3.13.0"
    "dav1d           1.5.1"
    "libavif         1.3.0"
    "libde265        1.0.16"
    "x265            4.1"
    "libheif         1.21.0"
    "libultrahdr     1.4.0"

    # Tier 1 scientific compressors (small / medium)
    "libaec          1.1.6"
    "lerc            4.1.0"
    "zfp             1.0.1"
    "SZ3             3.3.1"
    "SPERR           0.8.5"
    "pcodec          1.0.2"
    "brunsli         master"

    # JPEG-LS (CharLS): system package builds are typically -O2 with no
    # vector tuning; imagecodecs bundles a custom build that runs ~2x
    # faster. Same pattern as zfp — source build with -O3 -march=native
    # closes the gap.
    "CharLS          2.4.3"

    # Marquee codec — delegated to the dedicated script for parity with
    # the per-developer flow (some users only want to source-build libjxl
    # and rely on system libs for the rest).
    "libjxl          v0.11.2"
)

# ----------------------------------------------------------------------
# Argument parsing
# ----------------------------------------------------------------------

ONLY=""
SKIP=""
for arg in "$@"; do
    case "$arg" in
        --only=*)  ONLY="${arg#--only=}" ;;
        --skip=*)  SKIP="${arg#--skip=}" ;;
        --help|-h) sed -n '2,40p' "$0"; exit 0 ;;
        *)         echo "unknown arg: $arg"; exit 2 ;;
    esac
done

want() {
    local name="$1"
    if [ -n "$ONLY" ]; then
        case ",$ONLY," in *",$name,"*) return 0 ;; esac
        return 1
    fi
    if [ -n "$SKIP" ]; then
        case ",$SKIP," in *",$name,"*) return 1 ;; esac
    fi
    return 0
}

# ----------------------------------------------------------------------
# Environment
# ----------------------------------------------------------------------

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)

if [ -n "${OPENCODECS_LIBS_PREFIX:-}" ]; then
    PREFIX="$OPENCODECS_LIBS_PREFIX"
elif [ "$(id -u)" = "0" ]; then
    PREFIX="/usr/local"
elif [ "$(uname)" = "Darwin" ]; then
    PREFIX="${HOME}/Library/Caches/opencodecs/libs"
else
    PREFIX="${XDG_CACHE_HOME:-$HOME/.cache}/opencodecs/libs"
fi

JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
USE_LTO="${USE_LTO:-1}"
MARCH="${MARCH:-}"
ENABLE_AOM="${ENABLE_AOM:-1}"
ENABLE_X265="${ENABLE_X265:-1}"

# Build dirs OFF the source tree so SMB/NFS mounts don't break ninja.
WORKDIR="${OPENCODECS_LIBS_WORKDIR:-${TMPDIR:-/tmp}/opencodecs-libs-build}"
mkdir -p "$PREFIX" "$WORKDIR"

# Make freshly-installed libs visible to dependent builds (libheif needs
# libde265/x265 already installed; libavif needs libaom; etc.).
export PKG_CONFIG_PATH="$PREFIX/lib/pkgconfig:$PREFIX/lib64/pkgconfig:${PKG_CONFIG_PATH:-}"
export CMAKE_PREFIX_PATH="$PREFIX:${CMAKE_PREFIX_PATH:-}"
case "$(uname -s)" in
    MINGW*|MSYS*|CYGWIN*)
        # Windows / MSVC: do NOT export GCC/clang-style -L/-Wl,-rpath
        # in LDFLAGS or -I in CPPFLAGS. CMake on Windows picks up those
        # env vars and passes them through to link.exe (via vs_link_dll),
        # which can't parse them and crashes with STATUS_ACCESS_VIOLATION
        # (exit code 3221225477) when linking SHARED libs.
        # CMAKE_PREFIX_PATH alone is enough — find_package() and the
        # generic header/lib probe walk PREFIX correctly without
        # POSIX-style flag scaffolding.
        ;;
    *)
        export CPPFLAGS="-I$PREFIX/include ${CPPFLAGS:-}"
        export LDFLAGS="-L$PREFIX/lib -L$PREFIX/lib64 -Wl,-rpath,$PREFIX/lib -Wl,-rpath,$PREFIX/lib64 ${LDFLAGS:-}"
        ;;
esac
if [ "$(uname)" = "Linux" ]; then
    export LD_LIBRARY_PATH="$PREFIX/lib:$PREFIX/lib64:${LD_LIBRARY_PATH:-}"
fi

# Common compile flags — portable by default; opt-in via MARCH=native.
COMMON_CFLAGS="-O3 -DNDEBUG"
[ -n "$MARCH" ] && COMMON_CFLAGS="$COMMON_CFLAGS -march=$MARCH -mtune=$MARCH"
export CFLAGS="$COMMON_CFLAGS ${CFLAGS:-}"
export CXXFLAGS="$COMMON_CFLAGS ${CXXFLAGS:-}"

CMAKE_COMMON=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
    -DCMAKE_POSITION_INDEPENDENT_CODE=ON
    -DBUILD_SHARED_LIBS=ON
    # cmake 4.x dropped support for old policies (CMP0025 etc.) used by
    # x265's pre-3.5 CMakeLists. This baseline is a no-op for projects
    # already declaring cmake_minimum_required >= 3.5.
    -DCMAKE_POLICY_VERSION_MINIMUM=3.5
)
[ "$USE_LTO" = "1" ] && CMAKE_COMMON+=(-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON)
# Windows note: this script picks the Ninja generator (preferred) or
# falls back to Make. On Windows, CMake's compiler auto-detection
# scans PATH — so the **caller must source vcvars64.bat first** to
# put cl.exe + the MSVC INCLUDE/LIB env in scope. Otherwise CMake
# picks whichever compiler is first on PATH (often conda's gcc),
# producing gnu-format import libraries that MSVC link.exe can't
# consume from cibuildwheel later. The Visual Studio generator
# self-discovers MSVC without vcvars but trips SZ3's multi-config
# install bug; Ninja-with-vcvars is the working combination.
if command -v ninja >/dev/null 2>&1; then
    CMAKE_GEN=(-G Ninja)
    BUILD_TOOL=(ninja -j"$JOBS")
    INSTALL_TOOL=(ninja install)
else
    CMAKE_GEN=()
    BUILD_TOOL=(make -j"$JOBS")
    INSTALL_TOOL=(make install)
fi

# Cache fingerprint per lib — if the file exists at $PREFIX/.opencodecs/<name>
# AND its content matches the requested version, we skip the rebuild.
#
# Marker format (two lines):
#   <version>
#   <install_dir>      # optional; recipes that install OUTSIDE $PREFIX
#                      # (e.g. lerc/mozjpeg/sperr/brunsli land in
#                      # ~/.cache/opencodecs/<lib>) pass it so is_built
#                      # can verify the actual install survived.
#
# Why the install_dir line exists: CI caches $PREFIX (= /cibw-jxl-prefix
# inside cibuildwheel's manylinux container), so the marker file survives
# across runs. But per-user-cache install dirs (~/.cache/opencodecs/<lib>)
# are NOT cached. Without the install-dir check, a cache hit would short-
# circuit the recipe with "already built" while the actual library is
# gone — and setup.py's header probe silently drops the extension,
# producing wheels missing _lerc / _mozjpeg.
HASHDIR="$PREFIX/.opencodecs"
mkdir -p "$HASHDIR"

is_built() {
    local name="$1" version="$2"
    local marker="$HASHDIR/$name"
    [ -f "$marker" ] || return 1
    local recorded_version recorded_dir
    recorded_version="$(sed -n '1p' "$marker" 2>/dev/null || true)"
    recorded_dir="$(sed -n '2p' "$marker" 2>/dev/null || true)"
    [ "$recorded_version" = "$version" ] || return 1
    # If recipe recorded its install dir, verify it's still on disk.
    if [ -n "$recorded_dir" ] && [ ! -d "$recorded_dir" ]; then
        return 1
    fi
    return 0
}

mark_built() {
    local name="$1" version="$2" install_dir="${3:-}"
    if [ -n "$install_dir" ]; then
        printf '%s\n%s\n' "$version" "$install_dir" > "$HASHDIR/$name"
    else
        printf '%s\n' "$version" > "$HASHDIR/$name"
    fi
}

# ----------------------------------------------------------------------
# Per-library build helpers
# ----------------------------------------------------------------------

fetch_tar() {
    # fetch_tar <name> <version> <url> <strip>
    # Returns the source dir on stdout — keep ALL diagnostics on stderr
    # so callers can use `src=$(fetch_tar ...)` cleanly. Earlier version
    # printed "fetch <url>" to stdout which then poisoned cmake's
    # source-dir argument.
    #
    # Use a sentinel file (.fetched) rather than mere directory existence
    # to signal a complete extract: an interrupted ``curl | tar -xz`` (CI
    # job killed, ssh disconnected, disk full mid-extract) leaves an
    # empty/partial dir on disk, and a naive ``[ ! -d "$src" ]`` check
    # would then skip the fetch on the next run — leading to confusing
    # "Cargo.toml not found" / "CMakeLists.txt missing" errors when the
    # caller cd's into the empty dir.
    local name="$1" version="$2" url="$3" strip="${4:-1}"
    local src="$WORKDIR/$name-$version"
    if [ ! -f "$src/.fetched" ]; then
        rm -rf "$src"
        mkdir -p "$src"
        echo "    fetch  $url" >&2
        # --retry / --retry-connrefused / --retry-delay: gitlab.dkrz.de
        # (libaec mirror) times out intermittently. cibuildwheel's
        # manylinux_2_28 (AlmaLinux 8) ships curl 7.61, so we can't use
        # --retry-all-errors (that's 7.71+, and an unknown flag aborts
        # curl immediately, which is exactly how v0.1.6 attempt #3 broke).
        # --retry alone covers 5xx + 408 + 429 + timeouts (all the
        # actual flake modes we've hit) on every supported curl.
        # --max-time 300: hard cap at 5 min per attempt so a stuck
        # connection doesn't burn the CI runner.
        curl --retry 5 --retry-delay 4 --retry-connrefused \
             --max-time 300 -fsSL "$url" \
             | tar -xz --strip-components="$strip" -C "$src"
        # Belt-and-braces: tar on an empty stream returns 0, so a silent
        # curl flake can still leave the dir empty. Verify there's at
        # least one entry before declaring the fetch good.
        if [ -z "$(ls -A "$src" 2>/dev/null)" ]; then
            echo "fetch_tar: extracted dir $src is empty; $url likely flaked" >&2
            rm -rf "$src"
            return 1
        fi
        touch "$src/.fetched"
    fi
    echo "$src"
}

cmake_build() {
    # cmake_build <src> [cmake_args...]
    local src="$1"; shift
    local build="$src/_build"
    rm -rf "$build"
    mkdir -p "$build"
    ( cd "$build" && cmake "${CMAKE_GEN[@]}" "${CMAKE_COMMON[@]}" "$@" "$src" \
      && "${BUILD_TOOL[@]}" && "${INSTALL_TOOL[@]}" )
}

autotools_build() {
    # autotools_build <src> [configure_args...]
    local src="$1"; shift
    ( cd "$src" \
      && ./configure --prefix="$PREFIX" --enable-shared --disable-static "$@" \
      && make -j"$JOBS" && make install )
}

# ---- zlib ---------------------------------------------------------------
build_zlib() {
    local v="$(get_version zlib)"
    is_built zlib "$v" && { echo "  zlib $v already built"; return; }
    echo "==> zlib $v"
    local src
    src=$(fetch_tar zlib "$v" "https://zlib.net/zlib-$v.tar.gz")
    autotools_build "$src"
    mark_built zlib "$v"
}

# ---- zstd ---------------------------------------------------------------
# Install into the per-lib cache (`_OC_USER_CACHE/zstd`) with `-O3 + LTO`
# so the wrapper picks it up via setup.py's absolute-dylib link. zstd's
# Makefile build is preferred over CMake because it picks up the
# upstream-tuned flags (`-fomit-frame-pointer`, etc.). Patches the
# dylib install_name to `@rpath/` so the loader can resolve it after
# we copy the .so off the SMB mount.
build_zstd() {
    local v="$(get_version zstd)"
    is_built zstd "$v" && { echo "  zstd $v already built"; return; }
    echo "==> zstd $v"
    local src
    src=$(fetch_tar zstd "$v" "https://github.com/facebook/zstd/releases/download/v$v/zstd-$v.tar.gz")
    local zstd_prefix
    if [ "$(uname)" = "Darwin" ]; then
        zstd_prefix="${HOME}/Library/Caches/opencodecs/zstd"
    else
        zstd_prefix="${XDG_CACHE_HOME:-$HOME/.cache}/opencodecs/zstd"
    fi
    local cflags="-O3 -DNDEBUG -fomit-frame-pointer -flto"
    if [ "$(uname)" = "Darwin" ]; then
        cflags="$cflags -mcpu=apple-m1"
    fi
    ( cd "$src/lib" && make clean >/dev/null 2>&1 || true \
        && make -j"$JOBS" CFLAGS="$cflags" libzstd \
        && make PREFIX="$zstd_prefix" install )
    if [ "$(uname)" = "Darwin" ]; then
        install_name_tool -id @rpath/libzstd.1.dylib \
            "$zstd_prefix/lib/libzstd.${v}.dylib"
    fi
    mark_built zstd "$v" "$zstd_prefix"
}

# ---- lz4 ----------------------------------------------------------------
build_lz4() {
    local v="$(get_version lz4)"
    is_built lz4 "$v" && { echo "  lz4 $v already built"; return; }
    echo "==> lz4 $v"
    local src
    src=$(fetch_tar lz4 "$v" "https://github.com/lz4/lz4/releases/download/v$v/lz4-$v.tar.gz")
    cmake_build "$src/build/cmake" -DBUILD_SHARED_LIBS=ON -DBUILD_STATIC_LIBS=OFF
    mark_built lz4 "$v"
}

# ---- brotli -------------------------------------------------------------
# Same pattern as zstd above — per-lib cache prefix + -O3 + LTO. Pinned
# at brotli 1.1.0 (current upstream stable) for ABI consistency with
# imagecodecs.
build_giflib() {
    # giflib 5.2.2 (matches what imagecodecs vendors). The 6.x branch on
    # Homebrew is API-compatible but Homebrew builds with -O2 portable
    # flags; we want -O3 + LTO + hidden-visibility on the same source
    # to close the encode gap vs imagecodecs.
    local v="$(get_version giflib)"
    is_built giflib "$v" && { echo "  giflib $v already built"; return; }
    echo "==> giflib $v"
    local src
    src=$(fetch_tar giflib "$v" \
        "https://sourceforge.net/projects/giflib/files/giflib-$v.tar.gz/download")
    local prefix
    if [ "$(uname)" = "Darwin" ]; then
        prefix="${HOME}/Library/Caches/opencodecs/giflib"
    else
        prefix="${XDG_CACHE_HOME:-$HOME/.cache}/opencodecs/giflib"
    fi
    local oflags="-O3 -DNDEBUG -fomit-frame-pointer -fvisibility=hidden -flto"
    if [ "$(uname)" = "Darwin" ]; then
        oflags="$oflags -mcpu=apple-m1"
    fi
    ( cd "$src" && make clean >/dev/null 2>&1 || true \
        && OFLAGS="$oflags" make -j"$JOBS" all \
        && make PREFIX="$prefix" install-include install-lib )
    if [ "$(uname)" = "Darwin" ]; then
        install_name_tool -id @rpath/libgif.7.dylib \
            "$prefix/lib/libgif.7.2.0.dylib"
    fi
    mark_built giflib "$v" "$prefix"
}

build_brotli() {
    local v="$(get_version brotli)"
    is_built brotli "$v" && { echo "  brotli $v already built"; return; }
    echo "==> brotli $v"
    local src
    src=$(fetch_tar brotli "$v" "https://github.com/google/brotli/archive/refs/tags/v$v.tar.gz")
    local brotli_prefix
    if [ "$(uname)" = "Darwin" ]; then
        brotli_prefix="${HOME}/Library/Caches/opencodecs/brotli"
    else
        brotli_prefix="${XDG_CACHE_HOME:-$HOME/.cache}/opencodecs/brotli"
    fi
    local cflags="-O3 -DNDEBUG"
    if [ "$(uname)" = "Darwin" ]; then
        cflags="$cflags -mcpu=apple-m1"
    fi
    local build="$src/_build"
    rm -rf "$build"
    mkdir -p "$build"
    ( cd "$build" && cmake "${CMAKE_GEN[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS_RELEASE="$cflags" \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DBROTLI_DISABLE_TESTS=ON \
        -DCMAKE_INSTALL_PREFIX="$brotli_prefix" \
        "$src" \
      && "${BUILD_TOOL[@]}" && "${INSTALL_TOOL[@]}" )
    mark_built brotli "$v" "$brotli_prefix"
}

# ---- libdeflate ---------------------------------------------------------
build_libdeflate() {
    local v="$(get_version libdeflate)"
    is_built libdeflate "$v" && { echo "  libdeflate $v already built"; return; }
    echo "==> libdeflate $v"
    local src
    src=$(fetch_tar libdeflate "$v" "https://github.com/ebiggers/libdeflate/archive/refs/tags/v$v.tar.gz")
    cmake_build "$src" -DLIBDEFLATE_BUILD_GZIP=OFF
    mark_built libdeflate "$v"
}

# ---- MozJPEG (libjpeg-turbo fork; ~10-15% smaller JPEGs) ---------------
# Installs into a dedicated "keg-style" subdir so its libturbojpeg /
# libjpeg don't collide with the libjpeg-turbo 3.x install at $PREFIX
# (both share the same .so/.dll/.lib names). setup.py's mozjpeg probe
# expects exactly this keg-style layout.
build_mozjpeg() {
    local v="$(get_version mozjpeg)"
    is_built mozjpeg "$v" && { echo "  mozjpeg $v already built"; return; }
    echo "==> mozjpeg $v"
    local src
    src=$(fetch_tar mozjpeg "$v" \
        "https://github.com/mozilla/mozjpeg/archive/refs/tags/v$v.tar.gz")
    local mozjpeg_prefix
    case "$(uname -s)" in
        Darwin)
            mozjpeg_prefix="${HOME}/Library/Caches/opencodecs/mozjpeg"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            mozjpeg_prefix="${PREFIX}/mozjpeg"
            ;;
        *)
            mozjpeg_prefix="${XDG_CACHE_HOME:-$HOME/.cache}/opencodecs/mozjpeg"
            ;;
    esac
    install -d "$mozjpeg_prefix"
    local build="$src/_build"
    rm -rf "$build"
    mkdir -p "$build"
    ( cd "$build" \
      && cmake "${CMAKE_GEN[@]}" "${CMAKE_COMMON[@]}" \
          -DCMAKE_INSTALL_PREFIX="$mozjpeg_prefix" \
          -DENABLE_STATIC=OFF -DENABLE_SHARED=ON \
          -DWITH_TURBOJPEG=ON -DWITH_JPEG8=ON \
          -DPNG_SUPPORTED=OFF \
          "$src" \
      && "${BUILD_TOOL[@]}" && "${INSTALL_TOOL[@]}" )
    mark_built mozjpeg "$v" "$mozjpeg_prefix"
}

# ---- libpng (depends on zlib) ------------------------------------------
build_libpng() {
    local v="$(get_version libpng)"
    is_built libpng "$v" && { echo "  libpng $v already built"; return; }
    echo "==> libpng $v"
    local src
    src=$(fetch_tar libpng "$v" "https://download.sourceforge.net/libpng/libpng-$v.tar.gz")
    cmake_build "$src" -DPNG_TESTS=OFF -DPNG_TOOLS=OFF -DPNG_STATIC=OFF
    mark_built libpng "$v"
}

# ---- libjpeg-turbo (TJv3 — required for opencodecs._jpeg) -------------
build_libjpeg_turbo() {
    local v="$(get_version libjpeg-turbo)"
    is_built libjpeg-turbo "$v" && { echo "  libjpeg-turbo $v already built"; return; }
    echo "==> libjpeg-turbo $v"
    if ! command -v nasm >/dev/null 2>&1 && ! command -v yasm >/dev/null 2>&1; then
        echo "    NOTE: nasm/yasm not found — libjpeg-turbo will skip SIMD."
    fi
    local src
    src=$(fetch_tar libjpeg-turbo "$v" "https://github.com/libjpeg-turbo/libjpeg-turbo/releases/download/$v/libjpeg-turbo-$v.tar.gz")
    cmake_build "$src" -DENABLE_STATIC=OFF -DWITH_TURBOJPEG=ON -DWITH_JAVA=OFF
    mark_built libjpeg-turbo "$v"
}

# ---- libwebp (depends on libpng, libjpeg) -------------------------------
build_libwebp() {
    local v="$(get_version libwebp)"
    is_built libwebp "$v" && { echo "  libwebp $v already built"; return; }
    echo "==> libwebp $v"
    local src
    src=$(fetch_tar libwebp "$v" "https://github.com/webmproject/libwebp/archive/refs/tags/v$v.tar.gz")
    cmake_build "$src" -DWEBP_BUILD_ANIM_UTILS=OFF -DWEBP_BUILD_CWEBP=OFF \
        -DWEBP_BUILD_DWEBP=OFF -DWEBP_BUILD_EXTRAS=OFF -DWEBP_BUILD_GIF2WEBP=OFF \
        -DWEBP_BUILD_IMG2WEBP=OFF -DWEBP_BUILD_VWEBP=OFF -DWEBP_BUILD_WEBPINFO=OFF \
        -DWEBP_BUILD_WEBPMUX=OFF -DBUILD_SHARED_LIBS=ON
    mark_built libwebp "$v"
}

# ---- openjpeg (jpeg2000) ------------------------------------------------
build_openjpeg() {
    local v="$(get_version openjpeg)"
    is_built openjpeg "$v" && { echo "  openjpeg $v already built"; return; }
    echo "==> openjpeg $v"
    local src
    src=$(fetch_tar openjpeg "$v" "https://github.com/uclouvain/openjpeg/archive/refs/tags/v$v.tar.gz")
    cmake_build "$src" -DBUILD_TESTING=OFF -DBUILD_CODEC=OFF
    mark_built openjpeg "$v"
}

# ---- c-blosc2 (depends on zstd, lz4) -----------------------------------
build_c_blosc2() {
    local v="$(get_version c-blosc2)"
    is_built c-blosc2 "$v" && { echo "  c-blosc2 $v already built"; return; }
    echo "==> c-blosc2 $v"
    local src
    src=$(fetch_tar c-blosc2 "$v" "https://github.com/Blosc/c-blosc2/archive/refs/tags/v$v.tar.gz")
    # Use the bundled zstd / lz4 sources instead of system libs. On Linux
    # the system libzstd.a ships as non-PIC, which breaks c-blosc2's
    # shared-library link with "relocation R_X86_64_PC32 against symbol
    # ... can not be used when making a shared object; recompile with
    # -fPIC". Even when system zstd is PIC, the bundled zstd is what
    # imagecodecs's wheel ships against and what we match for wire-format
    # parity, so there's no upside to preferring external here.
    cmake_build "$src" -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_FUZZERS=OFF \
        -DBUILD_EXAMPLES=OFF -DPREFER_EXTERNAL_ZSTD=OFF -DPREFER_EXTERNAL_LZ4=OFF
    mark_built c-blosc2 "$v"
}

# ---- libaom (slowest single build; AV1 encoder for libavif) ------------
build_libaom() {
    local v="$(get_version libaom)"
    [ "$ENABLE_AOM" = "1" ] || { echo "  libaom: ENABLE_AOM=0, skipping"; return; }
    is_built libaom "$v" && { echo "  libaom $v already built"; return; }
    echo "==> libaom $v (slow — ~3 min)"
    local src
    src=$(fetch_tar libaom "$v" "https://storage.googleapis.com/aom-releases/libaom-$v.tar.gz")
    cmake_build "$src" -DENABLE_TESTS=OFF -DENABLE_DOCS=OFF -DENABLE_TOOLS=OFF \
        -DENABLE_EXAMPLES=OFF -DCONFIG_RUNTIME_CPU_DETECT=1
    mark_built libaom "$v"
}

# ---- dav1d (AV1 decoder; faster than libaom decode) --------------------
build_dav1d() {
    local v="$(get_version dav1d)"
    is_built dav1d "$v" && { echo "  dav1d $v already built"; return; }
    if ! command -v meson >/dev/null 2>&1; then
        echo "  dav1d: meson not found — skipping (libavif will use libaom decode)"
        return
    fi
    echo "==> dav1d $v"
    local src build
    src=$(fetch_tar dav1d "$v" "https://code.videolan.org/videolan/dav1d/-/archive/$v/dav1d-$v.tar.gz")
    build="$src/_build"
    rm -rf "$build"
    meson setup "$build" "$src" --prefix="$PREFIX" --buildtype=release \
        --default-library=shared
    ninja -C "$build" install
    mark_built dav1d "$v"
}

# ---- libavif (AV1 image; depends on libaom + dav1d) --------------------
build_libavif() {
    local v="$(get_version libavif)"
    is_built libavif "$v" && { echo "  libavif $v already built"; return; }
    echo "==> libavif $v"
    local src
    src=$(fetch_tar libavif "$v" "https://github.com/AOMediaCodec/libavif/archive/refs/tags/v$v.tar.gz")
    local args=(-DAVIF_BUILD_TESTS=OFF -DAVIF_BUILD_APPS=OFF
                -DAVIF_LIBYUV=OFF)
    [ "$ENABLE_AOM" = "1" ] && args+=(-DAVIF_CODEC_AOM=SYSTEM)
    if [ -f "$PREFIX/lib/pkgconfig/dav1d.pc" ] || [ -f "$PREFIX/lib64/pkgconfig/dav1d.pc" ]; then
        args+=(-DAVIF_CODEC_DAV1D=SYSTEM)
    fi
    cmake_build "$src" "${args[@]}"
    mark_built libavif "$v"
}

# ---- libde265 (HEVC decoder for libheif) -------------------------------
build_libde265() {
    local v="$(get_version libde265)"
    is_built libde265 "$v" && { echo "  libde265 $v already built"; return; }
    echo "==> libde265 $v"
    local src
    src=$(fetch_tar libde265 "$v" "https://github.com/strukturag/libde265/releases/download/v$v/libde265-$v.tar.gz")
    cmake_build "$src" -DENABLE_DECODER=ON -DENABLE_ENCODER=OFF
    mark_built libde265 "$v"
}

# ---- x265 (HEVC encoder for libheif — large C++ build) -----------------
build_x265() {
    local v="$(get_version x265)"
    [ "$ENABLE_X265" = "1" ] || { echo "  x265: ENABLE_X265=0, skipping"; return; }
    is_built x265 "$v" && { echo "  x265 $v already built"; return; }
    echo "==> x265 $v (slow — ~2 min)"
    local src
    src=$(fetch_tar x265 "$v" "https://bitbucket.org/multicoreware/x265_git/downloads/x265_$v.tar.gz")
    cmake_build "$src/source" -DENABLE_CLI=OFF -DENABLE_SHARED=ON
    mark_built x265 "$v"
}

# ---- libultrahdr (Google's gainmap JPEG; depends on libjpeg-turbo) ------
build_libultrahdr() {
    local v="$(get_version libultrahdr)"
    is_built libultrahdr "$v" && { echo "  libultrahdr $v already built"; return; }
    echo "==> libultrahdr $v"
    local src
    src=$(fetch_tar libultrahdr "$v" "https://github.com/google/libultrahdr/archive/refs/tags/v$v.tar.gz")
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            # libultrahdr's CMakeLists.txt gates the install() rules on
            # `NOT WIN32` (see CMakeLists.txt around line 470 in v1.4.0),
            # so `ninja install` fails with "unknown target 'install'"
            # on Windows even when the build itself succeeds. Configure
            # + build only, then copy the artifacts into $PREFIX manually
            # — same pattern as brunsli's Windows install branch above.
            local build="$src/_build"
            rm -rf "$build"
            mkdir -p "$build"
            ( cd "$build" && cmake "${CMAKE_GEN[@]}" "${CMAKE_COMMON[@]}" \
                -DUHDR_BUILD_DEPS=OFF \
                -DUHDR_BUILD_EXAMPLES=OFF \
                -DUHDR_BUILD_TESTS=OFF \
                -DUHDR_BUILD_FUZZERS=OFF \
                -DUHDR_BUILD_BENCHMARK=OFF \
                "$src" \
              && "${BUILD_TOOL[@]}" )
            # MSVC ninja drops uhdr.dll + uhdr.lib alongside the build
            # objects in _build/. Header lives in src/ultrahdr_api.h.
            install -d "$PREFIX/include" "$PREFIX/lib" "$PREFIX/bin"
            cp -v "$src/ultrahdr_api.h" "$PREFIX/include/"
            cp -v "$build/uhdr.dll" "$PREFIX/bin/"
            cp -v "$build/uhdr.lib" "$PREFIX/lib/"
            # Loud post-check fails the build if any expected file is
            # missing — better to fail here than to silently produce a
            # wheel without _uhdr.
            ls "$PREFIX/include/ultrahdr_api.h" \
               "$PREFIX/bin/uhdr.dll" \
               "$PREFIX/lib/uhdr.lib" >&2
            ;;
        *)
            cmake_build "$src" \
                -DUHDR_BUILD_DEPS=OFF \
                -DUHDR_BUILD_EXAMPLES=OFF \
                -DUHDR_BUILD_TESTS=OFF \
                -DUHDR_BUILD_FUZZERS=OFF \
                -DUHDR_BUILD_BENCHMARK=OFF
            ;;
    esac
    mark_built libultrahdr "$v"
}

# ---- libheif (HEIC/HEIF; depends on x265 for encode, libde265 for decode) -
build_libheif() {
    local v="$(get_version libheif)"
    is_built libheif "$v" && { echo "  libheif $v already built"; return; }
    echo "==> libheif $v"
    local src
    src=$(fetch_tar libheif "$v" "https://github.com/strukturag/libheif/releases/download/v$v/libheif-$v.tar.gz")
    local args=(-DBUILD_TESTING=OFF -DBUILD_GDK_PIXBUF_LOADER=OFF
                -DWITH_EXAMPLES=OFF)
    [ "$ENABLE_X265" = "1" ] && args+=(-DWITH_X265=ON)
    # libheif's CMake auto-detects libde265 via pkg-config when present
    # at $PKG_CONFIG_PATH (we exported that earlier in this script).
    cmake_build "$src" "${args[@]}"
    mark_built libheif "$v"
}

# ---- libaec (CCSDS adaptive entropy coding) ----------------------------
build_libaec() {
    local v="$(get_version libaec)"
    is_built libaec "$v" && { echo "  libaec $v already built"; return; }
    echo "==> libaec $v"
    local src
    # libaec releases use a YYYYMMDD-tagged tarball on its gitlab; the
    # GitHub mirror has clean version tags.
    src=$(fetch_tar libaec "$v" "https://gitlab.dkrz.de/k202009/libaec/-/archive/v$v/libaec-v$v.tar.gz")
    cmake_build "$src" -DBUILD_TESTING=OFF
    mark_built libaec "$v"
}

# ---- lerc (Esri Limited Error Raster Compression) ----------------------
build_lerc() {
    local v="$(get_version lerc)"
    is_built lerc "$v" && { echo "  lerc $v already built"; return; }
    echo "==> lerc $v"
    local src
    src=$(fetch_tar lerc "$v" "https://github.com/Esri/lerc/archive/refs/tags/v$v.tar.gz")
    # Build with -O3 + LTO into a per-lib cache subdir the setup.py
    # probe (`_OC_USER_CACHE/lerc`) will pick up. Homebrew's libLerc
    # is built -O2 portable and benches 15% slower on decode.
    # ``set -u`` is on at script scope, so source defaults for any
    # var we may not have set yet.
    local prev_prefix="${CMAKE_INSTALL_PREFIX:-}"
    local prev_cflags="${CMAKE_C_FLAGS_RELEASE_OVERRIDE:-}"
    local lerc_prefix
    if [ "$(uname)" = "Darwin" ]; then
        lerc_prefix="${HOME}/Library/Caches/opencodecs/lerc"
    else
        lerc_prefix="${XDG_CACHE_HOME:-$HOME/.cache}/opencodecs/lerc"
    fi
    local build="$src/_build"
    rm -rf "$build"
    mkdir -p "$build"
    ( cd "$build" && cmake "${CMAKE_GEN[@]}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_C_FLAGS_RELEASE="-O3 -DNDEBUG" \
        -DCMAKE_CXX_FLAGS_RELEASE="-O3 -DNDEBUG" \
        -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON \
        -DBUILD_SHARED_LIBS=ON \
        -DLERC_BUILD_TESTING=OFF \
        -DCMAKE_INSTALL_PREFIX="$lerc_prefix" \
        "$src" \
      && "${BUILD_TOOL[@]}" && "${INSTALL_TOOL[@]}" )
    mark_built lerc "$v" "$lerc_prefix"
}

# ---- zfp (lossy float compression) -------------------------------------
build_zfp() {
    local v="$(get_version zfp)"
    is_built zfp "$v" && { echo "  zfp $v already built"; return; }
    echo "==> zfp $v"
    local src
    src=$(fetch_tar zfp "$v" "https://github.com/LLNL/zfp/archive/refs/tags/$v.tar.gz")
    cmake_build "$src" -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_UTILITIES=OFF
    mark_built zfp "$v"
}

# ---- SZ3 (error-bounded lossy scientific) ------------------------------
build_SZ3() {
    local v="$(get_version SZ3)"
    is_built SZ3 "$v" && { echo "  SZ3 $v already built"; return; }
    echo "==> SZ3 $v"
    local src
    src=$(fetch_tar SZ3 "$v" "https://github.com/szcompressor/SZ3/archive/refs/tags/v$v.tar.gz")
    # SZ3 ships with bundled zstd; force external to share the zstd we
    # already built earlier in this script.
    cmake_build "$src" -DBUILD_SZ3_TESTS=OFF -DSZ3_USE_BUNDLED_ZSTD=OFF
    mark_built SZ3 "$v"
}

# ---- SPERR (wavelet-based scientific compressor) ----------------------
build_SPERR() {
    local v="$(get_version SPERR)"
    is_built SPERR "$v" && { echo "  SPERR $v already built"; return; }
    echo "==> SPERR $v"
    local src
    src=$(fetch_tar SPERR "$v" \
        "https://github.com/NCAR/SPERR/archive/refs/tags/v$v.tar.gz")
    # USE_OMP=OFF for portable wheels. macOS clang lacks an OpenMP
    # runtime by default (find_package(OpenMP) fails without libomp),
    # and MSVC's OpenMP 2.0 impl rejects SPERR's ``size_t`` loop
    # indices (C3016 errors on sperr_helper.cpp). OpenMP only
    # parallelizes a handful of preprocessing loops; correctness is
    # unaffected. Local builds wanting it can pass --extra-cmake-args.
    #
    # BUILD_UNIT_TESTS=OFF — SPERR's unit tests use ``std::iota``
    # without including <numeric>. GCC/Clang transitively pull it in
    # via other STL headers; MSVC doesn't (sperr_helper_unit_test.cpp
    # line 257: ``error C2039: 'iota': is not a member of 'std'``).
    # We don't need the tests anyway — just the library + headers.
    #
    # On Windows the same pattern as brunsli applies: SPERR's C-API
    # header (SPERR_C_API.h) has no __declspec(dllexport), so the
    # SHARED build produces SPERR.dll with no exports and no import
    # SPERR.lib. Pass WINDOWS_EXPORT_ALL_SYMBOLS=ON for the .lib +
    # CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF to dodge the MSVC
    # /LTCG + /DEF + SHARED link.exe crash (STATUS_ACCESS_VIOLATION).
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            # SPERR's src/CMakeLists.txt line 55 unconditionally
            # force-enables IPO/LTO per-target:
            #   set_property(TARGET SPERR PROPERTY
            #       INTERPROCEDURAL_OPTIMIZATION TRUE)
            # This overrides our global
            # -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF flag, so
            # MSVC compiles SPERR.dll's objects with /GL and link.exe
            # gets /LTCG + /DEF (from WINDOWS_EXPORT_ALL_SYMBOLS),
            # which crashes (STATUS_ACCESS_VIOLATION). Patch the
            # override out on Windows only — Linux/macOS keep IPO
            # for the perf they were tuned for.
            # Match the indented line: `    set_property(TARGET SPERR
            # PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)`. Earlier
            # exact-match sed missed the 4-space indent. Use a lenient
            # any-content pattern; verified on the Windows VM (MSVC 14.44 — same
            # version as CI): after patch, no IPO line remains and
            # ninja produces both SPERR.dll + SPERR.lib cleanly.
            sed -i \
                's|.*INTERPROCEDURAL_OPTIMIZATION TRUE.*|    # Patched: IPO disabled for Windows MSVC + WINDOWS_EXPORT_ALL_SYMBOLS|' \
                "$src/src/CMakeLists.txt"
            # Belt + suspenders: also override CMAKE_INTERPROCEDURAL_OPTIMIZATION
            # globally. CMAKE_COMMON sets it ON via USE_LTO=1 default; that
            # would still enable IPO on SPERR even with the per-target
            # override sed'd out.
            cmake_build "$src" \
                -DBUILD_CLI_UTILITIES=OFF \
                -DBUILD_UNIT_TESTS=OFF \
                -DUSE_OMP=OFF \
                -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON \
                -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
            ;;
        *)
            cmake_build "$src" \
                -DBUILD_CLI_UTILITIES=OFF \
                -DBUILD_UNIT_TESTS=OFF \
                -DUSE_OMP=OFF
            ;;
    esac
    mark_built SPERR "$v"
}

# ---- CharLS (JPEG-LS reference impl) ----------------------------------
build_CharLS() {
    local v="$(get_version CharLS)"
    is_built CharLS "$v" && { echo "  CharLS $v already built"; return; }
    echo "==> CharLS $v"
    local src
    src=$(fetch_tar CharLS "$v" \
        "https://github.com/team-charls/charls/archive/refs/tags/$v.tar.gz")
    # CharLS ships unit tests + a CLI binary by default; we just want
    # the shared library. CHARLS_INSTALL=ON puts the .so + headers
    # under PREFIX (the install layout downstream codecs probe for).
    cmake_build "$src" \
        -DCHARLS_BUILD_TESTS=OFF \
        -DCHARLS_BUILD_FUZZ_TEST=OFF \
        -DCHARLS_BUILD_SAMPLES=OFF \
        -DCHARLS_INSTALL=ON
    mark_built CharLS "$v"
}

# ---- Brunsli (lossless JPEG transcoder) -------------------------------
# Brunsli's top-level CMake pulls in vintage googletest; modern CMake
# refuses it without an explicit policy floor.
build_brunsli() {
    local v="$(get_version brunsli)"
    is_built brunsli "$v" && { echo "  brunsli $v already built"; return; }
    echo "==> brunsli $v"
    local src
    # Brunsli has no release tags; tar of the master branch is the
    # supported source.
    src=$(fetch_tar brunsli "$v" \
        "https://github.com/google/brunsli/archive/refs/heads/$v.tar.gz")
    # CMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON tells CMake to enumerate
    # every symbol from the SHARED target's object files, synthesize
    # a .def, and pass /DEF to link.exe so an import .lib gets
    # generated alongside the .dll. Without it brunsli's C-API DLLs
    # have zero exports (the headers don't use __declspec(dllexport))
    # and no .lib materializes, leaving every downstream consumer
    # unable to link. No-op on Linux/macOS (Unix ELF/Mach-O export
    # all symbols by default). Locally validated on the Windows host
    # (MSVC 14.41, Ninja, vcvars-sourced): produces brunsli{dec,enc}-c.lib
    # in artifacts/ alongside the .dlls.
    #
    # CMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF on Windows — MSVC 14.44's
    # link.exe crashes with STATUS_ACCESS_VIOLATION (exit code
    # 3221225477) when combining /LTCG with /DEF (from
    # WINDOWS_EXPORT_ALL_SYMBOLS) on brunsli's SHARED target. Local
    # the Windows host builds with MSVC 14.41 succeed because we didn't
    # enable LTO there; the failure is specific to the LTO+def+SHARED
    # combination in the newer MSVC's link. Override the script's
    # global USE_LTO=1 default for this one target.
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            cmake_build "$src" \
                -DBUILD_TESTING=OFF \
                -DBRUNSLI_EMSCRIPTEN=OFF \
                -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON \
                -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
                -DCMAKE_INTERPROCEDURAL_OPTIMIZATION=OFF
            ;;
        *)
            cmake_build "$src" \
                -DBUILD_TESTING=OFF \
                -DBRUNSLI_EMSCRIPTEN=OFF \
                -DCMAKE_WINDOWS_EXPORT_ALL_SYMBOLS=ON \
                -DCMAKE_POLICY_VERSION_MINIMUM=3.5
            ;;
    esac
    # brunsli.cmake's install() rule lacks RUNTIME DESTINATION so the
    # .dll isn't installed automatically; the import .lib is supposed
    # to come via ARCHIVE but doesn't fire in the Ninja+MSVC SHARED
    # path. Copy both manually from _build/artifacts/ into PREFIX
    # ({bin,lib}) — delvewheel then picks up the .dlls during
    # repair-wheel-command.
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            install -d "$PREFIX/lib" "$PREFIX/bin"
            cp -v "$src/_build/artifacts/brunslidec-c.dll" \
                  "$src/_build/artifacts/brunslienc-c.dll" "$PREFIX/bin/"
            cp -v "$src/_build/artifacts/brunslidec-c.lib" \
                  "$src/_build/artifacts/brunslienc-c.lib" "$PREFIX/lib/"
            # Loud post-check fails the build if any file went missing.
            ls "$PREFIX/bin/brunslidec-c.dll" "$PREFIX/bin/brunslienc-c.dll" \
               "$PREFIX/lib/brunslidec-c.lib" "$PREFIX/lib/brunslienc-c.lib" >&2
            ;;
    esac
    mark_built brunsli "$v"
}

# ---- pcodec (Rust cdylib via cargo) ------------------------------------
build_pcodec() {
    local v="$(get_version pcodec)"
    is_built pcodec "$v" && { echo "  pcodec $v already built"; return; }
    if ! command -v cargo >/dev/null 2>&1; then
        echo "  pcodec: cargo not found — skipping (codec auto-disables)"
        return
    fi
    echo "==> pcodec $v (cargo build)"
    local src
    src=$(fetch_tar pcodec "$v" "https://github.com/pcodec/pcodec/archive/refs/tags/v$v.tar.gz")
    # On Windows, rustc resolves ``link.exe`` via PATH and the GitHub
    # runner's PATH puts ``C:\Program Files\Git\usr\bin`` ahead of MSVC's
    # bin dir. That folder ships GNU coreutils' ``link.exe`` (a 2-arg
    # symlink utility), which rejects rustc's command line with
    # "/usr/bin/link: extra operand ...". Pin the linker explicitly to
    # MSVC's link.exe via vcvars's $VCToolsInstallDir so cargo doesn't
    # PATH-search for it.
    case "$(uname -s)" in
        MINGW*|MSYS*|CYGWIN*)
            if [ -n "$VCToolsInstallDir" ]; then
                export CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER=$(
                    cygpath -m "$VCToolsInstallDir/bin/Hostx64/x64/link.exe"
                )
                echo "  cargo linker: $CARGO_TARGET_X86_64_PC_WINDOWS_MSVC_LINKER"
            else
                echo "  pcodec: VCToolsInstallDir not set; cargo may pick GNU link" >&2
            fi
            ;;
    esac
    ( cd "$src" && cargo build --release -p cpcodec )
    # Copy the cdylib + header into the prefix layout opencodecs expects.
    install -d "$PREFIX/include" "$PREFIX/lib"
    cp "$src/pco_c/include/cpcodec.h" "$src/pco_c/include/cpcodec_generated.h" \
        "$PREFIX/include/"
    case "$(uname -s)" in
        Darwin)
            cp "$src/target/release/libcpcodec.dylib" "$PREFIX/lib/"
            # Install_name fix so RPATH-based loading works:
            install_name_tool -id "@rpath/libcpcodec.dylib" \
                "$PREFIX/lib/libcpcodec.dylib"
            ;;
        MINGW*|MSYS*|CYGWIN*)
            # Windows: cargo emits ``cpcodec.dll`` in
            # ``target/release/`` plus an MSVC import library — named
            # ``cpcodec.dll.lib`` on modern rust (>= 1.61), or just
            # ``cpcodec.lib`` on older builds. setup.py links via
            # ``-lcpcodec`` so MSVC's link.exe searches for
            # ``cpcodec.lib`` in ``$PREFIX/lib/`` — strip the
            # ``.dll`` infix when copying. Install the dll itself
            # to ``$PREFIX/bin/`` for delvewheel to scoop into the
            # wheel.
            install -d "$PREFIX/bin"
            cp "$src/target/release/cpcodec.dll" "$PREFIX/bin/"
            if [ -f "$src/target/release/cpcodec.dll.lib" ]; then
                cp "$src/target/release/cpcodec.dll.lib" \
                    "$PREFIX/lib/cpcodec.lib"
            elif [ -f "$src/target/release/cpcodec.lib" ]; then
                cp "$src/target/release/cpcodec.lib" "$PREFIX/lib/"
            else
                echo "  pcodec: no MSVC import lib found in target/release/" >&2
                ls -la "$src/target/release/" >&2 || true
                return 1
            fi
            ;;
        *)
            cp "$src/target/release/libcpcodec.so" "$PREFIX/lib/"
            ;;
    esac
    mark_built pcodec "$v"
}

# ---- libjxl (delegate to dedicated script for parity) ------------------
build_libjxl() {
    local v="$(get_version libjxl)"
    is_built libjxl "$v" && { echo "  libjxl $v already built"; return; }
    echo "==> libjxl $v (delegating to bench/build_libjxl.sh)"
    LIBJXL_VERSION="$v" \
    OPENCODECS_LIBJXL_PREFIX="$PREFIX" \
    LIBJXL_WORKDIR="$WORKDIR/libjxl" \
        bash "$HERE/build_libjxl.sh"
    mark_built libjxl "$v"
}

# ----------------------------------------------------------------------
# Main: version lookup helper, run builds in dependency order
# ----------------------------------------------------------------------

# Linear scan over VERSIONS; bash-3.2-compatible (macOS ships /bin/bash
# 3.2 which lacks ``declare -A``). 20 entries — the scan is unmeasurably
# fast next to a 200 MB cmake build.
get_version() {
    local target="$1" line
    for line in "${VERSIONS[@]}"; do
        if [ "${line%% *}" = "$target" ]; then
            echo "${line##* }"
            return 0
        fi
    done
    echo "(unknown codec: $target)" >&2
    return 1
}

echo "================================================================"
echo "opencodecs codec library builder"
echo "  PREFIX=$PREFIX  JOBS=$JOBS  USE_LTO=$USE_LTO  MARCH=${MARCH:-(portable)}"
[ -n "$ONLY" ] && echo "  --only=$ONLY"
[ -n "$SKIP" ] && echo "  --skip=$SKIP"
echo "================================================================"
echo ""

# Ordered build list (deps come first).
ORDERED=(
    zlib
    zstd
    lz4
    brotli
    libdeflate
    giflib
    libpng
    libjpeg-turbo
    libwebp
    openjpeg
    mozjpeg
    c-blosc2
    libaom
    dav1d
    libavif
    libde265
    x265
    libheif
    libultrahdr
    libaec
    lerc
    zfp
    SZ3
    SPERR
    pcodec
    brunsli
    CharLS
    libjxl
)

for name in "${ORDERED[@]}"; do
    if want "$name"; then
        case "$name" in
            zlib)            build_zlib ;;
            zstd)            build_zstd ;;
            lz4)             build_lz4 ;;
            brotli)          build_brotli ;;
            libdeflate)      build_libdeflate ;;
            giflib)          build_giflib ;;
            libpng)          build_libpng ;;
            libjpeg-turbo)   build_libjpeg_turbo ;;
            mozjpeg)         build_mozjpeg ;;
            libwebp)         build_libwebp ;;
            openjpeg)        build_openjpeg ;;
            c-blosc2)        build_c_blosc2 ;;
            libaom)          build_libaom ;;
            dav1d)           build_dav1d ;;
            libavif)         build_libavif ;;
            libde265)        build_libde265 ;;
            x265)            build_x265 ;;
            libheif)         build_libheif ;;
            libultrahdr)     build_libultrahdr ;;
            libaec)          build_libaec ;;
            lerc)            build_lerc ;;
            zfp)             build_zfp ;;
            SZ3)             build_SZ3 ;;
            SPERR)           build_SPERR ;;
            pcodec)          build_pcodec ;;
            brunsli)         build_brunsli ;;
            CharLS)          build_CharLS ;;
            libjxl)          build_libjxl ;;
        esac
    fi
done

echo ""
echo "================================================================"
echo "All requested codec libs installed under $PREFIX"
echo "  set OPENCODECS_JXL_PREFIX=$PREFIX before \`pip install opencodecs\`"
echo "  (or pass \`--config-settings setup-args=...\` if using PEP 517)"
echo "================================================================"
