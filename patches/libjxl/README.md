# Vendored libjxl patches

`bench/build_libjxl.sh` applies every `*.diff` file in this directory to the
freshly-cloned libjxl source tree before configuring + building it. Patches
are applied in lexical filename order; prefix with a number (`01-foo.diff`,
`02-bar.diff`) if order matters.

Each patch must apply cleanly with `git apply --directory=<libjxl-clone>`
against the pinned `LIBJXL_VERSION` (currently v0.11.2). The build script
records a hash of all patches in the install sentinel so editing or adding
a patch triggers a full libjxl rebuild on the next run.

## Why patches live here

libjxl's public C API doesn't expose every decoder feature its C++ internals
support. The most prominent gap is **`JXL_DEC_FRAME_PROGRESSION` emission
for modular streams** — modular images with the Squeeze transform have a
DC pyramid that the decoder reads but never surfaces through the public
event API. The same is what Apple's CoreGraphics JXL decoder taps into
(they vendor libjxl with internal-header access). Patches here exist to
expose those internals on the public surface.

Upstream cadence is slow — see the libjxl GitHub releases page: there was
exactly one coordinated release across all supported branches in the 15
months between Nov 2024 and Feb 2026. So even a clean upstream PR realistically
ships to wheel users 12-18 months later. Vendoring the patch here keeps
opencodecs users on the fixed behavior immediately, with no maintenance
once the patch is written (because we pin `LIBJXL_VERSION` ourselves).

This mirrors the pattern that Christoph Gohlke used for `imagecodecs` from
2018 to 2022 (`imagecodecs/patches/*.diff`); his patches typically lived
for 2-4 years against pinned upstream versions, with near-zero maintenance.

## How to add a patch

1. Create the patch from a libjxl source tree using `git diff > patches/libjxl/NN-name.diff`.
2. Verify it applies cleanly against the pinned `LIBJXL_VERSION`:
   ```sh
   cd /tmp/libjxl-test
   git clone --branch v0.11.2 https://github.com/libjxl/libjxl.git
   cd libjxl
   git apply <repo>/patches/libjxl/NN-name.diff
   ```
3. Drop it in this directory and bump `LIBJXL_VERSION` only if the patch
   needs a newer base. Reinstall opencodecs — `setup.py` runs
   `bench/build_libjxl.sh` automatically when the cached install is stale.

## Wheel distribution

When opencodecs publishes wheels:

- **Linux / macOS**: the wheel builder runs `bench/build_libjxl.sh` (which
  applies these patches) as a pre-build step, then `auditwheel repair` /
  `delocate-wheel` bundles the patched `libjxl.dylib` / `libjxl.so` inside
  the wheel. Wheel users get the patched libjxl automatically — no system
  libjxl required, no patch application at install time.
- **Windows**: `bench/build_libjxl.sh` doesn't run on Windows directly;
  the equivalent step would be a `build_libjxl.ps1` invoked from CI.

Source installs (`pip install -e .` from a checkout) apply patches by
running `bench/build_libjxl.sh` from `setup.py`. The user only needs
`cmake`, `ninja`, and a C++17 compiler.

## Maintenance cost

- **Pinned libjxl version** = patches don't drift. We only rebase patches
  when we deliberately bump `LIBJXL_VERSION`.
- **Idempotent install** = the build script's sentinel file short-circuits
  to zero-work when the version+patches hash matches the installed library.
- **Patch removal** = delete the `.diff` and bump `LIBJXL_VERSION` to the
  release that includes the upstream fix. Build script's sentinel triggers
  a rebuild on the next install.
