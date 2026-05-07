<!-- markdownlint-disable MD060 -->

# Installing opencodecs

opencodecs is a Cython package that links against system C libraries
for most codecs. The build is **conditional** — extensions whose
required system header is missing are skipped cleanly with a one-line
notice. So you don't need every dependency listed below; install only
the codecs you need.

## TL;DR

```sh
# macOS
brew install jpeg-turbo webp libavif libheif openjpeg libtiff hdf5 c-blosc2

# Ubuntu / Debian (24.04+)
sudo apt install -y \
    libturbojpeg0-dev libwebp-dev libavif-dev libheif-dev \
    libopenjp2-7-dev libblosc2-dev libcharls-dev \
    liblz4-dev libspng-dev libtiff-dev libhdf5-dev \
    libbrotli-dev libzstd-dev

# Build libjxl (~3 minutes; cached for subsequent installs)
./bench/build_libjxl.sh

# Build all extensions
pip install -e .
# or
python setup.py build_ext --inplace
```

Test the install:

```sh
python -m pytest tests/test_native_parity.py
```

## What each system library enables

| System library | apt package | brew formula | Codec built |
| --- | --- | --- | --- |
| libzstd | libzstd-dev | zstd | `zstd` |
| liblz4 (frame format) | liblz4-dev | lz4 | `lz4` |
| libbrotli | libbrotli-dev | brotli | `brotli` |
| c-blosc2 | libblosc2-dev | c-blosc2 | `blosc2` |
| zlib | libz-dev | (system) | `deflate` |
| libspng | libspng-dev | libspng | `png` |
| libjpeg-turbo (TJ v3) | libturbojpeg0-dev | jpeg-turbo | `jpeg` |
| libwebp | libwebp-dev | webp | `webp` |
| openjpeg | libopenjp2-7-dev | openjpeg | `jpeg2k` |
| libavif | libavif-dev | libavif | `avif` |
| libheif | libheif-dev | libheif | `heif` |
| HDF5 | libhdf5-dev | hdf5 | (read-only via h5py) |
| libtiff | libtiff-dev | libtiff | (transitive) |

The TurboJPEG entry is specifically the **v3 API**. Older
libjpeg-turbo on Ubuntu 22.04 ships only the v2 API; the build will
detect this (probes for `tj3Init` in `turbojpeg.h`) and skip `_jpeg`
cleanly. The fix on those systems is `apt install libturbojpeg0-dev`
from the 22.04+ repo, or upgrade to 24.04.

## libjxl (vendored)

JPEG-XL needs libjxl ≥ 0.10. Distro / Homebrew builds are typically a
generic build; the libjxl shipped inside the imagecodecs wheel is
`-O3` with LTO enabled and ~30% faster. We ship `bench/build_libjxl.sh` which clones
libjxl 0.11.2, builds it `-O3 -DNDEBUG` + LTO, and installs it into a
per-user cache:

| OS | Default install location |
| --- | --- |
| macOS | `~/Library/Caches/opencodecs/libjxl/` |
| Linux | `~/.cache/opencodecs/libjxl/` (or `$XDG_CACHE_HOME/opencodecs/libjxl/`) |
| Windows | `~/AppData/Local/opencodecs/libjxl/` |

Install path is **off the source tree** by design: when the repo lives
on a network mount, macOS Sequoia Gatekeeper blocks fresh dlopens of
.dylib files there. The same shadow-cache pattern as `edt`, `ncolor`,
and `hiprpy`.

```sh
./bench/build_libjxl.sh         # ~3 minutes
python setup.py build_ext --inplace --force
```

setup.py auto-detects the cache as the highest-priority libjxl prefix.
On Linux it emits `DT_RPATH` (via `-Wl,--disable-new-dtags`) so
transitive libs (libjxl_cms, libbrotli*) load from the cache. On macOS
it links by absolute path and post-build runs `install_name_tool
-delete_rpath /opt/homebrew/lib` to keep dyld from preferring the
Homebrew libjxl at runtime.

Tunables on the build script:

```sh
LIBJXL_VERSION=v0.11.2  \
JOBS=16                 \
USE_LTO=1               \
MARCH=native            \
./bench/build_libjxl.sh
```

`MARCH=native` produces a host-specific binary (~5-10% faster on most
CPUs) that won't run elsewhere — leave empty for portable wheels.

Build prerequisites: `git`, `cmake` (≥ 3.16), a C++17 compiler,
`ninja` (optional but faster). On Ubuntu without root: `pip install
--user cmake ninja && export PATH="$HOME/.local/bin:$PATH"`.

## Optional Python dependencies

opencodecs uses `numpy` and `platformdirs` as core dependencies. A few
codecs require additional Python packages:

| Codec | Python dep | Install |
| --- | --- | --- |
| `hdf5` | `h5py` | `pip install h5py` |
| `OcZstd` etc. (zarr v3 wrappers) | `zarr` ≥ 3.0 | `pip install zarr` |

The HDF5 reader is registered only when `h5py` is importable. The zarr
wrappers in `opencodecs._zarr_codecs` import lazily; you only pay for
zarr when you import the wrappers.

## Running tests

```sh
pip install pytest
python -m pytest tests/test_native_parity.py
```

Some tests require `imagecodecs` as a parity reference. Install with
`pip install imagecodecs` if not already present. Tests for codecs
whose system library is absent are auto-skipped.

The CZI tests look for a specific lab file at
`/Volumes/HiprDrive/...` and skip if not mounted. They're for in-house
verification only and don't fail external installs.

## Platform notes

### macOS Sequoia + NAS-mounted source

Newly-built `.so` files on SMB-mounted volumes trip the "Apple could
not verify" Gatekeeper dialog on first dlopen. opencodecs ships a
shadow-cache loader (`opencodecs/codecs/__init__.py:_load_extension`)
that copies the .so to `~/Library/Caches/opencodecs/lib/` before
loading. Same pattern hiprpy and edt use. Nothing to configure.

### Linux x86_64 + multilib include dirs

Ubuntu installs library headers under
`/usr/include/x86_64-linux-gnu/` (e.g. `libheif`). setup.py probes
both `/usr/include/` and the multilib dirs automatically.

### Linux without root

If you can't `apt install` the dev packages, conda is a viable
alternative for most of them:

```sh
conda install -c conda-forge libzstd libwebp libavif libheif openjpeg libspng \
    c-blosc2 libtiff hdf5 brotli
export OPENCODECS_JXL_PREFIX=$CONDA_PREFIX
pip install -e .
```

setup.py probes `$CONDA_PREFIX` for headers when set.

### Windows

Build verified on Windows 10 (build 26100) with MSVC 2022 BuildTools and
pyenv-win Python 3.11.9. The conditional-skip mechanism does the right
thing — codecs whose system libraries aren't installed skip cleanly,
and the package imports successfully even when most extensions are
absent.

#### Recommended build flow

```pwsh
# 1. Copy source off the NAS to local disk first. Building on a NAS-
#    mounted UNC path or PSDrive is unreliable on Windows — cmd.exe
#    (used by setup.py at points) doesn't accept UNC working dirs, and
#    the build is full of small file ops where SMB latency hurts.
Copy-Item -Recurse \\server\share\imagecodecs\opencodecs C:\Users\you\opencodecs

# 2. Build, calling the python EXE directly (NOT the pyenv-win .bat shim,
#    which spawns cmd.exe internally and chokes on UNC).
Set-Location C:\Users\you\opencodecs
& C:\Users\you\.pyenv\pyenv-win\versions\3.11.9\python311.exe setup.py build_ext --inplace
```

To re-sync the source from a NAS mount, use `robocopy` (PSDrive paths
like `HiprDrive:\` aren't accepted by `robocopy` directly — use the
underlying UNC):

```pwsh
robocopy \\server\share\imagecodecs\opencodecs\src\opencodecs `
         C:\Users\you\opencodecs\src\opencodecs `
         /MIR /XF *.pyd *.so   # don't sync stale platform binaries
```

#### Out-of-the-box codecs (no system library install needed)

Without installing any C library, you get these codecs:

| Codec | Source |
| --- | --- |
| `qoi` | vendored `qoi.h` |
| `bmp` | pure Python + numpy |
| `hdf5` | h5py wheel (ships its own libhdf5) |

Plus 5 zarr v3 wrappers (`OcZstd` etc.) — those import cleanly even
when their backing extension didn't build, and only fail if you
actually try to use them.

#### For full codec coverage on Windows

The system libraries can be installed via either **vcpkg** or
**conda-forge**. setup.py probes both:

+ `$env:VCPKG_ROOT\installed\x64-windows\` for vcpkg
+ `$env:CONDA_PREFIX\Library\` for conda environments

vcpkg path:

```pwsh
git clone https://github.com/microsoft/vcpkg.git C:\vcpkg
C:\vcpkg\bootstrap-vcpkg.bat
$env:VCPKG_ROOT = "C:\vcpkg"
& $env:VCPKG_ROOT\vcpkg.exe install zstd lz4 brotli libwebp libavif libheif `
                                    openjpeg libspng libjpeg-turbo zlib c-blosc2
```

conda-forge path (simpler if you already have conda):

```pwsh
conda create -n oc python=3.11 -c conda-forge libzstd lz4-c brotli `
    libwebp libavif libheif openjpeg libspng libjpeg-turbo c-blosc2 zlib hdf5
conda activate oc
& python.exe setup.py build_ext --inplace
```

libjxl on Windows isn't yet covered by the bench/build_libjxl.sh script
(bash-only). For now, Windows users either skip JXL (everything else
still works) or pre-install libjxl manually and set
`OPENCODECS_JXL_PREFIX=C:\path\to\libjxl\prefix`.

#### PowerShell quoting footguns

When invoking PowerShell from another shell (e.g. via SSH), three
patterns to know:

+ `powershell -File <path>` does NOT accept PowerShell-drive paths
  (`HiprDrive:\foo.ps1`). It needs a real Windows path. Copy first.
+ This VM has Windows PowerShell 5.1 (`powershell.exe`), not
  PowerShell 7 (`pwsh.exe`). The instructions in this file use
  `powershell` because it's universal.
+ For complex commands, write a `.ps1` script and invoke with
  `powershell -ExecutionPolicy Bypass -File C:\path\to.ps1` rather
  than fighting nested-quote escaping.

### Conda-installed binaries on macOS / Apple Silicon

If you also have miniforge / conda installed and `python` on your
PATH points there rather than your pyenv, the build's library-search
path order may pick up Conda's libjxl / libwebp / etc. before
Homebrew's. Either:

+ invoke the build via the explicit pyenv shim (`~/.pyenv/shims/python
  setup.py build_ext --inplace`), or
+ set `OPENCODECS_JXL_PREFIX=/opt/homebrew` (or wherever your tuned
  libjxl lives).

## Troubleshooting

**`fatal error: 'jxl/types.h' file not found`** — libjxl headers
weren't found. Run `./bench/build_libjxl.sh` or set
`OPENCODECS_JXL_PREFIX=/path/to/libjxl/prefix`.

**`ImportError: dlopen ... library not loaded: @rpath/libjxl.0.11.dylib`** —
the .so was built linking against a libjxl that's no longer
findable. Most often happens when you upgrade Homebrew libjxl.
Rebuild with `--force`:

```sh
python setup.py build_ext --inplace --force
```

**`ImportError: ... code signature ... not valid for use in process: library load disallowed by system policy`** —
macOS Gatekeeper blocked the .so. The shadow-cache loader handles
this automatically; if you see this it means the loader path got
bypassed (e.g. directly importing `opencodecs.codecs._jxl` rather
than going through `opencodecs.codecs.__init__`). Re-import via the
package, not by file path.

**TurboJPEG v3 API not found** — Ubuntu 22.04's `libturbojpeg0-dev`
is the v2 API. Either upgrade to 24.04 or apt-pin from a newer repo.
The `_jpeg` extension will skip cleanly; everything else builds.

**Build succeeds but tests fail with "no codec named X"** — make sure
the .so files are in `src/opencodecs/codecs/`. If your `pip install
-e .` builds them in `build/` but doesn't copy back, run `python
setup.py build_ext --inplace` directly.
