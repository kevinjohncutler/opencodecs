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

    # Container / multi-codec (medium)
    "c-blosc2        2.16.0"

    # AV1 / HEVC (largest builds)
    "libaom          3.13.0"
    "dav1d           1.5.1"
    "libavif         1.3.0"
    "libde265        1.0.16"
    "x265            4.1"
    "libheif         1.21.0"

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
export CPPFLAGS="-I$PREFIX/include ${CPPFLAGS:-}"
export LDFLAGS="-L$PREFIX/lib -L$PREFIX/lib64 -Wl,-rpath,$PREFIX/lib -Wl,-rpath,$PREFIX/lib64 ${LDFLAGS:-}"
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
)
[ "$USE_LTO" = "1" ] && CMAKE_COMMON+=(-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON)
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
HASHDIR="$PREFIX/.opencodecs"
mkdir -p "$HASHDIR"

is_built() {
    local name="$1" version="$2"
    [ "$(cat "$HASHDIR/$name" 2>/dev/null || true)" = "$version" ]
}

mark_built() {
    echo "$2" > "$HASHDIR/$1"
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
    local name="$1" version="$2" url="$3" strip="${4:-1}"
    local src="$WORKDIR/$name-$version"
    if [ ! -d "$src" ]; then
        mkdir -p "$src"
        echo "    fetch  $url" >&2
        curl -fsSL "$url" | tar -xz --strip-components="$strip" -C "$src"
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
    local v="${VERSIONS_MAP[zlib]}"
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
    local v="${VERSIONS_MAP[zstd]}"
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
    mark_built zstd "$v"
}

# ---- lz4 ----------------------------------------------------------------
build_lz4() {
    local v="${VERSIONS_MAP[lz4]}"
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
    local v="${VERSIONS_MAP[giflib]}"
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
    mark_built giflib "$v"
}

build_brotli() {
    local v="${VERSIONS_MAP[brotli]}"
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
    mark_built brotli "$v"
}

# ---- libdeflate ---------------------------------------------------------
build_libdeflate() {
    local v="${VERSIONS_MAP[libdeflate]}"
    is_built libdeflate "$v" && { echo "  libdeflate $v already built"; return; }
    echo "==> libdeflate $v"
    local src
    src=$(fetch_tar libdeflate "$v" "https://github.com/ebiggers/libdeflate/archive/refs/tags/v$v.tar.gz")
    cmake_build "$src" -DLIBDEFLATE_BUILD_GZIP=OFF
    mark_built libdeflate "$v"
}

# ---- libpng (depends on zlib) ------------------------------------------
build_libpng() {
    local v="${VERSIONS_MAP[libpng]}"
    is_built libpng "$v" && { echo "  libpng $v already built"; return; }
    echo "==> libpng $v"
    local src
    src=$(fetch_tar libpng "$v" "https://download.sourceforge.net/libpng/libpng-$v.tar.gz")
    cmake_build "$src" -DPNG_TESTS=OFF -DPNG_TOOLS=OFF -DPNG_STATIC=OFF
    mark_built libpng "$v"
}

# ---- libjpeg-turbo (TJv3 — required for opencodecs._jpeg) -------------
build_libjpeg_turbo() {
    local v="${VERSIONS_MAP[libjpeg-turbo]}"
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
    local v="${VERSIONS_MAP[libwebp]}"
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
    local v="${VERSIONS_MAP[openjpeg]}"
    is_built openjpeg "$v" && { echo "  openjpeg $v already built"; return; }
    echo "==> openjpeg $v"
    local src
    src=$(fetch_tar openjpeg "$v" "https://github.com/uclouvain/openjpeg/archive/refs/tags/v$v.tar.gz")
    cmake_build "$src" -DBUILD_TESTING=OFF -DBUILD_CODEC=OFF
    mark_built openjpeg "$v"
}

# ---- c-blosc2 (depends on zstd, lz4) -----------------------------------
build_c_blosc2() {
    local v="${VERSIONS_MAP[c-blosc2]}"
    is_built c-blosc2 "$v" && { echo "  c-blosc2 $v already built"; return; }
    echo "==> c-blosc2 $v"
    local src
    src=$(fetch_tar c-blosc2 "$v" "https://github.com/Blosc/c-blosc2/archive/refs/tags/v$v.tar.gz")
    cmake_build "$src" -DBUILD_TESTS=OFF -DBUILD_BENCHMARKS=OFF -DBUILD_FUZZERS=OFF \
        -DBUILD_EXAMPLES=OFF -DPREFER_EXTERNAL_ZSTD=ON -DPREFER_EXTERNAL_LZ4=ON
    mark_built c-blosc2 "$v"
}

# ---- libaom (slowest single build; AV1 encoder for libavif) ------------
build_libaom() {
    local v="${VERSIONS_MAP[libaom]}"
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
    local v="${VERSIONS_MAP[dav1d]}"
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
    local v="${VERSIONS_MAP[libavif]}"
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
    local v="${VERSIONS_MAP[libde265]}"
    is_built libde265 "$v" && { echo "  libde265 $v already built"; return; }
    echo "==> libde265 $v"
    local src
    src=$(fetch_tar libde265 "$v" "https://github.com/strukturag/libde265/releases/download/v$v/libde265-$v.tar.gz")
    cmake_build "$src" -DENABLE_DECODER=ON -DENABLE_ENCODER=OFF
    mark_built libde265 "$v"
}

# ---- x265 (HEVC encoder for libheif — large C++ build) -----------------
build_x265() {
    local v="${VERSIONS_MAP[x265]}"
    [ "$ENABLE_X265" = "1" ] || { echo "  x265: ENABLE_X265=0, skipping"; return; }
    is_built x265 "$v" && { echo "  x265 $v already built"; return; }
    echo "==> x265 $v (slow — ~2 min)"
    local src
    src=$(fetch_tar x265 "$v" "https://bitbucket.org/multicoreware/x265_git/downloads/x265_$v.tar.gz")
    cmake_build "$src/source" -DENABLE_CLI=OFF -DENABLE_SHARED=ON
    mark_built x265 "$v"
}

# ---- libheif (HEIC/HEIF; depends on x265 for encode, libde265 for decode) -
build_libheif() {
    local v="${VERSIONS_MAP[libheif]}"
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
    local v="${VERSIONS_MAP[libaec]}"
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
    local v="${VERSIONS_MAP[lerc]}"
    is_built lerc "$v" && { echo "  lerc $v already built"; return; }
    echo "==> lerc $v"
    local src
    src=$(fetch_tar lerc "$v" "https://github.com/Esri/lerc/archive/refs/tags/v$v.tar.gz")
    # Build with -O3 + LTO into a per-lib cache subdir the setup.py
    # probe (`_OC_USER_CACHE/lerc`) will pick up. Homebrew's libLerc
    # is built -O2 portable and benches 15% slower on decode.
    local prev_prefix="$CMAKE_INSTALL_PREFIX"
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
    mark_built lerc "$v"
}

# ---- zfp (lossy float compression) -------------------------------------
build_zfp() {
    local v="${VERSIONS_MAP[zfp]}"
    is_built zfp "$v" && { echo "  zfp $v already built"; return; }
    echo "==> zfp $v"
    local src
    src=$(fetch_tar zfp "$v" "https://github.com/LLNL/zfp/archive/refs/tags/$v.tar.gz")
    cmake_build "$src" -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF -DBUILD_UTILITIES=OFF
    mark_built zfp "$v"
}

# ---- SZ3 (error-bounded lossy scientific) ------------------------------
build_SZ3() {
    local v="${VERSIONS_MAP[SZ3]}"
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
    local v="${VERSIONS_MAP[SPERR]}"
    is_built SPERR "$v" && { echo "  SPERR $v already built"; return; }
    echo "==> SPERR $v"
    local src
    src=$(fetch_tar SPERR "$v" \
        "https://github.com/NCAR/SPERR/archive/refs/tags/v$v.tar.gz")
    cmake_build "$src" -DBUILD_CLI_UTILITIES=OFF -DUSE_OMP=ON
    mark_built SPERR "$v"
}

# ---- CharLS (JPEG-LS reference impl) ----------------------------------
build_CharLS() {
    local v="${VERSIONS_MAP[CharLS]}"
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
    local v="${VERSIONS_MAP[brunsli]}"
    is_built brunsli "$v" && { echo "  brunsli $v already built"; return; }
    echo "==> brunsli $v"
    local src
    # Brunsli has no release tags; tar of the master branch is the
    # supported source.
    src=$(fetch_tar brunsli "$v" \
        "https://github.com/google/brunsli/archive/refs/heads/$v.tar.gz")
    cmake_build "$src" \
        -DBUILD_TESTING=OFF \
        -DBRUNSLI_EMSCRIPTEN=OFF \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5
    mark_built brunsli "$v"
}

# ---- pcodec (Rust cdylib via cargo) ------------------------------------
build_pcodec() {
    local v="${VERSIONS_MAP[pcodec]}"
    is_built pcodec "$v" && { echo "  pcodec $v already built"; return; }
    if ! command -v cargo >/dev/null 2>&1; then
        echo "  pcodec: cargo not found — skipping (codec auto-disables)"
        return
    fi
    echo "==> pcodec $v (cargo build)"
    local src
    src=$(fetch_tar pcodec "$v" "https://github.com/pcodec/pcodec/archive/refs/tags/v$v.tar.gz")
    ( cd "$src" && cargo build --release -p cpcodec )
    # Copy the cdylib + header into the prefix layout opencodecs expects.
    install -d "$PREFIX/include" "$PREFIX/lib"
    cp "$src/pco_c/include/cpcodec.h" "$src/pco_c/include/cpcodec_generated.h" \
        "$PREFIX/include/"
    if [ "$(uname)" = "Darwin" ]; then
        cp "$src/target/release/libcpcodec.dylib" "$PREFIX/lib/"
        # Install_name fix so RPATH-based loading works:
        install_name_tool -id "@rpath/libcpcodec.dylib" "$PREFIX/lib/libcpcodec.dylib"
    else
        cp "$src/target/release/libcpcodec.so" "$PREFIX/lib/"
    fi
    mark_built pcodec "$v"
}

# ---- libjxl (delegate to dedicated script for parity) ------------------
build_libjxl() {
    local v="${VERSIONS_MAP[libjxl]}"
    is_built libjxl "$v" && { echo "  libjxl $v already built"; return; }
    echo "==> libjxl $v (delegating to bench/build_libjxl.sh)"
    LIBJXL_VERSION="$v" \
    OPENCODECS_LIBJXL_PREFIX="$PREFIX" \
    LIBJXL_WORKDIR="$WORKDIR/libjxl" \
        bash "$HERE/build_libjxl.sh"
    mark_built libjxl "$v"
}

# ----------------------------------------------------------------------
# Main: parse VERSIONS into a map, run builds in dependency order
# ----------------------------------------------------------------------

declare -A VERSIONS_MAP
for line in "${VERSIONS[@]}"; do
    name="${line%% *}"
    version="${line##* }"
    VERSIONS_MAP[$name]="$version"
done

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
    c-blosc2
    libaom
    dav1d
    libavif
    libde265
    x265
    libheif
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
            libwebp)         build_libwebp ;;
            openjpeg)        build_openjpeg ;;
            c-blosc2)        build_c_blosc2 ;;
            libaom)          build_libaom ;;
            dav1d)           build_dav1d ;;
            libavif)         build_libavif ;;
            libde265)        build_libde265 ;;
            x265)            build_x265 ;;
            libheif)         build_libheif ;;
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
