#!/usr/bin/env bash
# Build a tuned libjxl into <repo>/vendor/libjxl/.
#
# Why: distro libjxl packages on Ubuntu / Debian are built generically and
# decode 0.5-0.7x as fast as the libjxl shipped inside the imagecodecs wheel
# (which Christoph Gohlke builds with optimization flags). This script gives
# us our OWN tuned build, so we don't have to depend on imagecodecs at runtime.
#
# After this completes:
#   pip install -e . --no-build-isolation     # auto-detects vendor/libjxl/
#
# Required tools: git, cmake (>= 3.16), a C++17 compiler, and either ninja
# or make. On Ubuntu: `sudo apt install -y cmake ninja-build g++ git`.
#
# Tunables:
#   LIBJXL_VERSION   tag to check out (default v0.11.2 — matches imagecodecs)
#   JOBS             parallel jobs (default $(nproc))
#   USE_LTO          1 to enable link-time optimization (default 1)
#   MARCH            target arch flag (default '' — empty for portable; pass
#                    'native' for max-speed on this exact CPU only).

set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
REPO=$(cd "$HERE/.." && pwd)
VENDOR="$REPO/vendor"

# Install location for the built libjxl. Defaults to a per-user cache OFF
# the source tree — this is the same "shadow to local disk" pattern that
# edt / ncolor / hiprpy use for their .so files: when the source lives on
# a network mount (smbfs / nfs), macOS Gatekeeper queues a notarization
# check on every fresh dlopen and blocks the load. Installing libjxl to a
# stable local-disk path lets dyld load it without prompting and keeps
# the install persistent across rebuilds (unlike $TMPDIR).
#
# Override with $OPENCODECS_LIBJXL_PREFIX if you want the libs vendored
# in-tree (e.g., for a wheel build that bundles them).
if [ -n "${OPENCODECS_LIBJXL_PREFIX:-}" ]; then
    PREFIX="$OPENCODECS_LIBJXL_PREFIX"
elif [ "$(uname)" = "Darwin" ]; then
    PREFIX="${HOME}/Library/Caches/opencodecs/libjxl"
else
    PREFIX="${XDG_CACHE_HOME:-$HOME/.cache}/opencodecs/libjxl"
fi

# Build OUTSIDE the project tree if it's on a network mount — git pack index
# files break on macOS smbfs (AppleDouble ._<file> companions get mistaken for
# real git objects), and CMake/ninja are slow to rebuild incrementally on SMB.
# Default to a local-disk build dir; users can override with $LIBJXL_WORKDIR.
detect_remote_mount() {
    local target="$1"
    # Use `df` to find the mount point for $target, then check its fs type.
    local fs
    if [ "$(uname)" = "Darwin" ]; then
        fs=$(df "$target" 2>/dev/null | awk 'NR==2 {print $1}')
        case "$fs" in
            //*|*"smbfs"*|*"nfs"*|*"afpfs"*) return 0 ;;
        esac
        # macOS: also check via mount output for the source containing $target
        mount 2>/dev/null | awk -v t="$target" '
            { mp = $0; sub(/^.* on /, "", mp); sub(/ \(.*$/, "", mp);
              fs = $0; sub(/^.*\(/, "", fs); sub(/,.*/, "", fs);
              if (index(t, mp) == 1 && (fs == "smbfs" || fs == "nfs" || fs == "afpfs"))
                  exit 0;
            }
            END { exit 1 }
        ' && return 0
    elif [ "$(uname)" = "Linux" ]; then
        # findmnt is most reliable; fall back to /proc/mounts
        local fst
        if command -v findmnt >/dev/null 2>&1; then
            fst=$(findmnt -no FSTYPE -T "$target" 2>/dev/null)
        else
            fst=$(awk -v t="$target" '$2 == t || index(t, $2 "/") == 1 {fs=$3} END{print fs}' /proc/mounts)
        fi
        case "$fst" in
            cifs|nfs*|smbfs|afpfs|fuse.smbnetfs) return 0 ;;
        esac
    fi
    return 1
}

if [ -n "${LIBJXL_WORKDIR:-}" ]; then
    WORKDIR="$LIBJXL_WORKDIR"
elif detect_remote_mount "$REPO"; then
    WORKDIR="${TMPDIR:-/tmp}/opencodecs-libjxl-build"
else
    WORKDIR="$VENDOR"
fi
SRC="$WORKDIR/libjxl-src"
BUILD="$WORKDIR/libjxl-build"

LIBJXL_VERSION="${LIBJXL_VERSION:-v0.11.2}"
JOBS="${JOBS:-$(getconf _NPROCESSORS_ONLN 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)}"
USE_LTO="${USE_LTO:-1}"
MARCH="${MARCH:-}"

echo "==> opencodecs: building libjxl $LIBJXL_VERSION"
echo "    workdir : $WORKDIR  (clone + build)"
echo "    install : $PREFIX   (final lib location)"
echo "    JOBS=$JOBS  USE_LTO=$USE_LTO  MARCH=${MARCH:-(portable)}"

# Idempotency: if the requested version is already installed at $PREFIX,
# exit early so cache restores translate to genuine zero-work runs.
SENTINEL="$PREFIX/.opencodecs-libjxl-version"
if [ -f "$SENTINEL" ] && [ "$(cat "$SENTINEL" 2>/dev/null)" = "$LIBJXL_VERSION" ] \
   && ( [ -e "$PREFIX/include/jxl/types.h" ] ); then
    echo "==> libjxl $LIBJXL_VERSION already installed at $PREFIX (cache hit). Skipping."
    exit 0
fi

mkdir -p "$(dirname "$PREFIX")" "$WORKDIR"

# --- clone (with submodules; libjxl vendors libhwy, libbrotli, skcms in submodules)
if [ ! -d "$SRC/.git" ]; then
    rm -rf "$SRC"
    git clone --depth 1 --branch "$LIBJXL_VERSION" \
        https://github.com/libjxl/libjxl.git "$SRC"
fi
cd "$SRC"
git submodule update --init --depth 1 --recursive

# --- configure
rm -rf "$BUILD"
mkdir -p "$BUILD"
cd "$BUILD"

# libjxl 0.11.2 with newer CMake (>=4.x) sometimes fails to propagate the
# INTERFACE include path that exposes the public jxl/ headers from the
# build dir. Forcing both source-side public includes onto the compile
# line avoids "JXL_PARALLEL_RET_SUCCESS undeclared" build failures.
CMAKE_CXX_FLAGS="-O3 -DNDEBUG -I$SRC/lib/include -I$BUILD/lib/include"
CMAKE_C_FLAGS="-O3 -DNDEBUG -I$SRC/lib/include -I$BUILD/lib/include"
if [ -n "$MARCH" ]; then
    CMAKE_CXX_FLAGS="$CMAKE_CXX_FLAGS -march=$MARCH -mtune=$MARCH"
    CMAKE_C_FLAGS="$CMAKE_C_FLAGS -march=$MARCH -mtune=$MARCH"
fi

CMAKE_ARGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DCMAKE_INSTALL_PREFIX="$PREFIX"
    -DCMAKE_C_FLAGS_RELEASE="$CMAKE_C_FLAGS"
    -DCMAKE_CXX_FLAGS_RELEASE="$CMAKE_CXX_FLAGS"
    -DBUILD_SHARED_LIBS=ON
    -DBUILD_TESTING=OFF
    -DJPEGXL_ENABLE_TOOLS=OFF
    -DJPEGXL_ENABLE_DEVTOOLS=OFF
    -DJPEGXL_ENABLE_BENCHMARK=OFF
    -DJPEGXL_ENABLE_EXAMPLES=OFF
    -DJPEGXL_ENABLE_DOXYGEN=OFF
    -DJPEGXL_ENABLE_MANPAGES=OFF
    -DJPEGXL_ENABLE_PLUGINS=OFF
    -DJPEGXL_ENABLE_OPENEXR=OFF
    -DJPEGXL_ENABLE_JPEGLI=OFF
    -DJPEGXL_ENABLE_SKCMS=ON
    -DJPEGXL_BUNDLE_LIBPNG=OFF
)

if [ "$USE_LTO" = "1" ]; then
    CMAKE_ARGS+=(-DCMAKE_INTERPROCEDURAL_OPTIMIZATION=ON)
fi

# Prefer Ninja; fall back to Make.
if command -v ninja >/dev/null 2>&1; then
    CMAKE_ARGS=(-G Ninja "${CMAKE_ARGS[@]}")
    BUILD_CMD=(ninja -j"$JOBS")
    INSTALL_CMD=(ninja install)
else
    BUILD_CMD=(make -j"$JOBS")
    INSTALL_CMD=(make install)
fi

cmake "$SRC" "${CMAKE_ARGS[@]}"
"${BUILD_CMD[@]}"
"${INSTALL_CMD[@]}"

# Drop the version sentinel so the next run can short-circuit if nothing
# changed. Cleared automatically by `rm -rf $PREFIX` on a forced rebuild.
echo "$LIBJXL_VERSION" > "$PREFIX/.opencodecs-libjxl-version"

echo ""
echo "==> opencodecs: libjxl installed to $PREFIX"
echo "==> libs:"
ls -la "$PREFIX/lib"/libjxl* 2>/dev/null | head -10 || ls -la "$PREFIX/lib64"/libjxl* 2>/dev/null | head -10
echo ""
echo "Now build opencodecs against this:"
echo "    cd $REPO"
echo "    pip install -e . --no-build-isolation --force-reinstall --no-deps"
echo "    python setup.py build_ext --inplace --force"
