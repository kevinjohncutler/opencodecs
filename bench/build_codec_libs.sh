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
    "x265            4.1"
    "libheif         1.21.0"

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
    local name="$1" version="$2" url="$3" strip="${4:-1}"
    local src="$WORKDIR/$name-$version"
    if [ ! -d "$src" ]; then
        mkdir -p "$src"
        echo "    fetch  $url"
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
build_zstd() {
    local v="${VERSIONS_MAP[zstd]}"
    is_built zstd "$v" && { echo "  zstd $v already built"; return; }
    echo "==> zstd $v"
    local src
    src=$(fetch_tar zstd "$v" "https://github.com/facebook/zstd/releases/download/v$v/zstd-$v.tar.gz")
    cmake_build "$src/build/cmake" -DZSTD_BUILD_PROGRAMS=OFF -DZSTD_BUILD_STATIC=OFF
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
build_brotli() {
    local v="${VERSIONS_MAP[brotli]}"
    is_built brotli "$v" && { echo "  brotli $v already built"; return; }
    echo "==> brotli $v"
    local src
    src=$(fetch_tar brotli "$v" "https://github.com/google/brotli/archive/refs/tags/v$v.tar.gz")
    cmake_build "$src" -DBROTLI_DISABLE_TESTS=ON
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
    cmake_build "$src" "${args[@]}"
    mark_built libheif "$v"
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
    libpng
    libjpeg-turbo
    libwebp
    openjpeg
    c-blosc2
    libaom
    dav1d
    libavif
    x265
    libheif
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
            libpng)          build_libpng ;;
            libjpeg-turbo)   build_libjpeg_turbo ;;
            libwebp)         build_libwebp ;;
            openjpeg)        build_openjpeg ;;
            c-blosc2)        build_c_blosc2 ;;
            libaom)          build_libaom ;;
            dav1d)           build_dav1d ;;
            libavif)         build_libavif ;;
            x265)            build_x265 ;;
            libheif)         build_libheif ;;
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
